"""CLI Runner for Mini GPT Intelligence Evaluation.

Usage:
    python scripts/evaluate_intelligence.py
    python scripts/evaluate_intelligence.py --checkpoint checkpoints/v3/best.pt
    python scripts/evaluate_intelligence.py --output evaluation_report.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.service import ModelService
from evaluation.benchmark_intelligence import run_intelligence_benchmark


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mini GPT intelligence and reasoning.")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "v3" / "best.pt"),
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(ROOT / "checkpoints" / "tokenizer.json"),
        help="Path to tokenizer JSON file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling threshold (default: 40)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON benchmark results",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    tok_path = Path(args.tokenizer)
    if not ckpt_path.is_file():
        # Fall back to root checkpoints/best.pt if v3 not present
        fallback = ROOT / "checkpoints" / "best.pt"
        if fallback.is_file():
            ckpt_path = fallback
        else:
            print(f"Error: checkpoint not found at {ckpt_path}")
            sys.exit(1)

    print(f"Loading model from {ckpt_path}...")
    service = ModelService.from_checkpoint(ckpt_path, tok_path)

    def model_fn(messages):
        return service.chat(
            messages,
            max_new_tokens=150,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=42,
        )

    print("Running intelligence benchmark on held-out questions...")
    report = run_intelligence_benchmark(model_fn)

    print("\n" + "=" * 60)
    print("MINI GPT INTELLIGENCE EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Questions: {report['total_questions']}")
    print(f"Total Passed:    {report['total_passed']} / {report['total_questions']}")
    print(f"Overall Accuracy: {report['overall_pass_rate'] * 100:.1f}%\n")

    print(f"{'Category':<15} | {'Passed':<8} | {'Accuracy':<10} | {'3-Gram Repetition':<18}")
    print("-" * 60)
    for cat, stats in report["category_summary"].items():
        pass_pct = stats["pass_rate"] * 100
        rep_pct = stats["avg_3gram_repetition"] * 100
        print(f"{cat:<15} | {stats['passed']}/{stats['total']:<6} | {pass_pct:>6.1f}%    | {rep_pct:>6.1f}%")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nDetailed report saved to {out_path}")


if __name__ == "__main__":
    main()
