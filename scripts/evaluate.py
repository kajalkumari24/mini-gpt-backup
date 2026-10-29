"""Evaluate a Mini GPT checkpoint on a text dataset.

Examples (run from project folder):
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/v3/best.pt --data data/mini_gpt_conversations.txt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/input_shakespeare.txt --device cpu
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from evaluation.evaluate import (
    bits_per_char,
    evaluate_full,
    measure_generation_speed,
    parameter_megabytes,
    peak_memory_during_generation,
    perplexity,
    unigram_baseline,
)
from evaluation.generate_samples import generate_evaluation_samples
from model.configuration import get_device
from model.tokenizer import CharTokenizer
from training.checkpoint import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Mini GPT checkpoint.")
    default_ckpt = ROOT / "checkpoints" / "v3" / "best.pt"
    if not default_ckpt.is_file():
        default_ckpt = ROOT / "checkpoints" / "best.pt"

    default_data = ROOT / "data" / "mini_gpt_conversations.txt"
    if not default_data.is_file():
        default_data = ROOT / "data" / "input_shakespeare.txt"

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
        "--data",
        default=str(default_data),
        help=f"Path to evaluation text dataset (default: {default_data}).",
    )
    parser.add_argument(
        "--train-data",
        "--train_data",
        default=None,
        dest="train_data",
        help="Optional path to training data to build the unigram baseline distribution.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use ('auto', 'cpu', 'cuda') (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=16,
        dest="batch_size",
        help="Batch size for loss calculation (default: 16).",
    )
    parser.add_argument(
        "--num-tokens",
        "--num_tokens",
        type=int,
        default=100,
        dest="num_tokens",
        help="Number of tokens to generate for speed test (default: 100).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    tok_path = Path(args.tokenizer)
    data_path = Path(args.data)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not tok_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    device = get_device(args.device)
    print("=" * 60)
    print("Mini GPT Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Tokenizer:  {tok_path}")
    print(f"Dataset:    {data_path}")
    print(f"Device:     {device}")

    tokenizer = CharTokenizer.load(tok_path)
    model, ckpt_info = load_model_from_checkpoint(ckpt_path, device)
    model.eval()

    num_params = model.num_parameters()
    param_mb = parameter_megabytes(model)
    print(f"Parameters: {num_params:,} ({param_mb:.2f} MB)")
    if "step" in ckpt_info:
        print(f"Trained steps: {ckpt_info['step']}")
    if "best_val_loss" in ckpt_info:
        print(f"Recorded best val loss: {ckpt_info['best_val_loss']:.4f}")

    # Dataset evaluation
    text = data_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, errors="ignore")
    data_tensor = torch.tensor(token_ids, dtype=torch.long)
    print(f"Evaluation text:   {len(text):,} characters")
    print(f"Evaluation tokens: {len(data_tensor):,} tokens")

    if len(data_tensor) > model.config.context_length:
        n_windows = (len(data_tensor) - 1) // model.config.context_length
        eval_tokens_count = n_windows * model.config.context_length
        eval_text = tokenizer.decode(data_tensor[:eval_tokens_count].tolist())
        eval_chars_count = len(eval_text)

        loss = evaluate_full(model, data_tensor, batch_size=args.batch_size)
        ppl = perplexity(loss)
        bits_token = loss / math.log(2)
        bpc = bits_per_char(loss, num_tokens=eval_tokens_count, num_chars=eval_chars_count)
        print(f"Evaluation Loss:  {loss:.4f} nats/token")
        print(f"Perplexity:       {ppl:.2f}")
        print(f"Bits per token:   {bits_token:.2f} bits/token")
        print(f"Bits per char:    {bpc:.2f} bits/char (evaluated over {eval_chars_count:,} characters / {eval_tokens_count:,} tokens)")

        # Baseline comparison
        if args.train_data and Path(args.train_data).is_file():
            train_text = Path(args.train_data).read_text(encoding="utf-8")
            train_tokens = torch.tensor(tokenizer.encode(train_text, errors="ignore"), dtype=torch.long)
            baseline = unigram_baseline(train_tokens, data_tensor, tokenizer.vocab_size)
            print(f"Unigram Baseline (from {args.train_data}): {baseline:.4f}")
        else:
            half = len(data_tensor) // 2
            train_slice, val_slice = data_tensor[:half], data_tensor[half:]
            baseline = unigram_baseline(train_slice, val_slice, tokenizer.vocab_size)
            print(f"Unigram Baseline (split): {baseline:.4f}")

        improvement = ((baseline - loss) / baseline) * 100.0 if baseline > 0 else 0.0
        print(f"Context Gain:     {improvement:+.1f}% vs random unigram")
    else:
        print("Dataset shorter than model context length; skipping full loss evaluation.")

    # Speed & Memory
    speed = measure_generation_speed(model, num_tokens=args.num_tokens, warmup_tokens=10)
    print(f"Generation Speed: {speed:.1f} tokens/second")
    peak_mem = peak_memory_during_generation(model, num_tokens=args.num_tokens)
    if peak_mem is not None:
        print(f"Peak GPU Memory:  {peak_mem:.2f} MB")

    # Sample Generation
    print("\nSample Outputs:")
    print("-" * 60)
    samples = generate_evaluation_samples(
        model,
        tokenizer,
        prompts=["YOU:\nHi\n\nBOT:\n", "YOU:\nWhat is your name?\n\nBOT:\n"],
        max_new_tokens=80,
    )
    for p, out in samples.items():
        print(f"[Prompt]: {repr(p)}")
        print(f"[Output]: {repr(out)}")
        print("-" * 60)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
