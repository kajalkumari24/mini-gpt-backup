"""Generate text from a trained checkpoint.

Examples (run from the project folder):
    python scripts/generate.py --prompt "ROMEO:"
    python scripts/generate.py --prompt "First Citizen:" --temperature 0.5 --max_new_tokens 300
    python scripts/generate.py --temperature 0            # greedy
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.configuration import get_device, set_seed
from model.generation import generate_text
from model.tokenizer import CharTokenizer
from training.checkpoint import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text with the Mini GPT.")
    default_ckpt = ROOT / "checkpoints" / "v3" / "best.pt"
    if not default_ckpt.is_file():
        default_ckpt = ROOT / "checkpoints" / "best.pt"
    parser.add_argument("--checkpoint", default=str(default_ckpt))
    parser.add_argument("--tokenizer", default=str(ROOT / "checkpoints" / "tokenizer.json"))
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    parser.add_argument("--top_k", type=int, default=40, help="0 = no top-k filtering")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device("auto")
    tokenizer = CharTokenizer.load(args.tokenizer)
    model, _ = load_model_from_checkpoint(args.checkpoint, device)

    print(generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
    ))


if __name__ == "__main__":
    main()
