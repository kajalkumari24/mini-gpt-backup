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
    return parser.parse_args()


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


if __name__ == "__main__":
    main()

