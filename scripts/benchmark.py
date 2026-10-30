"""Benchmark generation throughput, latency, and memory across configurations.

Examples (run from project folder):
    python scripts/benchmark.py
    python scripts/benchmark.py --checkpoint checkpoints/v3/best.pt --num-tokens 200 --runs 5
    python scripts/benchmark.py --device cpu
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.performance import benchmark_generation_performance
from model.configuration import get_device
from model.tokenizer import CharTokenizer
from training.checkpoint import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Mini GPT inference performance.")
    default_ckpt = ROOT / "checkpoints" / "v3" / "best.pt"
    if not default_ckpt.is_file():
        default_ckpt = ROOT / "checkpoints" / "best.pt"

    parser.add_argument(
        "--checkpoint",
        default=str(default_ckpt),
        help=f"Path to model checkpoint (.pt) (default: {default_ckpt}).",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(ROOT / "checkpoints" / "tokenizer.json"),
        help="Path to tokenizer.json (default: checkpoints/tokenizer.json).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use ('auto', 'cpu', 'cuda') (default: auto).",
    )
    parser.add_argument(
        "--num-tokens",
        "--num_tokens",
        type=int,
        default=200,
        dest="num_tokens",
        help="Number of tokens to generate per run (default: 200).",
    )
    parser.add_argument(
        "--warmup-tokens",
        "--warmup_tokens",
        type=int,
        default=10,
        dest="warmup_tokens",
        help="Warmup tokens before timing (default: 10).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs (default: 3).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of simulated concurrent clients (default: 1).",
    )
    return parser.parse_args()


def benchmark_concurrent_clients(model, device, num_tokens=50, concurrency=4):
    """Benchmark model generation under concurrent thread load."""
    import concurrent.futures
    import time
    import torch
    from model.generation import generate

    prompt_tokens = torch.tensor([[0]], dtype=torch.long, device=device)
    model.eval()

    def single_worker():
        t0 = time.time()
        with torch.no_grad():
            out = generate(model, prompt_tokens, max_new_tokens=num_tokens, temperature=0.8, top_k=40)
        t1 = time.time()
        return t1 - t0, out.size(1) - 1

    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_worker) for _ in range(concurrency)]
        results = [f.result() for f in futures]
    total_wall_time = time.time() - start_all

    total_tokens = sum(r[1] for r in results)
    avg_latency = sum(r[0] for r in results) / len(results)
    aggregate_tokens_per_sec = total_tokens / max(1e-6, total_wall_time)

    return {
        "concurrency": concurrency,
        "total_wall_time": total_wall_time,
        "avg_client_latency": avg_latency,
        "total_tokens": total_tokens,
        "aggregate_tokens_per_second": aggregate_tokens_per_sec,
    }


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    tok_path = Path(args.tokenizer)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not tok_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    device = get_device(args.device)
    print("=" * 60)
    print("Mini GPT Performance Benchmark")
    print("=" * 60)
    print(f"Checkpoint:    {ckpt_path}")
    print(f"Device:        {device}")
    print(f"Target tokens: {args.num_tokens}")
    print(f"Runs:          {args.runs} (warmup: {args.warmup_tokens} tokens)")

    model, _ = load_model_from_checkpoint(ckpt_path, device)
    model.eval()

    perf = benchmark_generation_performance(
        model,
        device,
        num_tokens=args.num_tokens,
        warmup_tokens=args.warmup_tokens,
        runs=args.runs,
    )

    print("\nBenchmark Results:")
    print("-" * 60)
    print(f"Parameters:         {perf['parameters']:,}")
    print(f"Model Size:         {perf['parameter_mb']:.2f} MB")
    print(f"Throughput:         {perf['tokens_per_second']:.2f} tokens/second")
    print(f"Latency per token:  {perf['ms_per_token']:.2f} ms/token")
    print(f"Total time (min):   {perf['min_latency_seconds']:.4f} s")
    print(f"Total time (avg):   {perf['avg_latency_seconds']:.4f} s")
    if "gpu_peak_mb" in perf:
        print(f"Peak GPU Memory:    {perf['gpu_peak_mb']:.2f} MB")
    print("=" * 60)

    if args.concurrency > 1:
        print(f"\nRunning Concurrency Benchmark ({args.concurrency} parallel clients)...")
        c_perf = benchmark_concurrent_clients(
            model, device, num_tokens=args.num_tokens, concurrency=args.concurrency
        )
        print("-" * 60)
        print(f"Concurrency level:           {c_perf['concurrency']} clients")
        print(f"Total tokens generated:      {c_perf['total_tokens']}")
        print(f"Aggregate throughput:        {c_perf['aggregate_tokens_per_second']:.2f} tokens/second")
        print(f"Average client latency:      {c_perf['avg_client_latency']:.4f} s")
        print(f"Total wall time:             {c_perf['total_wall_time']:.4f} s")
        print("=" * 60)


if __name__ == "__main__":
    main()

