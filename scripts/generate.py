"""Generate text from a trained checkpoint.

Examples (run from the project folder):
    python scripts/generate.py --prompt "ROMEO:"
    python scripts/generate.py --prompt "YOU:\nHi\n\nBOT:\n"
    python scripts/generate.py --temperature 0            # greedy
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.configuration import get_device, load_generation_config, set_seed
from model.generation import generate_text
from model.tokenizer import CharTokenizer
from training.checkpoint import load_model_from_checkpoint


def main():
    try:
        gen_cfg = load_generation_config()
        default_max_new_tokens = gen_cfg.max_new_tokens
        default_temperature = gen_cfg.temperature
        default_top_k = gen_cfg.top_k
        default_seed = gen_cfg.seed or 42
    except Exception:
        default_max_new_tokens = 200
        default_temperature = 0.8
        default_top_k = 40
        default_seed = 42

    parser = argparse.ArgumentParser(description="Generate text with the Mini GPT.")
    default_ckpt = ROOT / "checkpoints" / "v3" / "best.pt"
    if not default_ckpt.is_file():
        default_ckpt = ROOT / "checkpoints" / "best.pt"
    parser.add_argument("--checkpoint", default=str(default_ckpt))
    parser.add_argument("--tokenizer", default=str(ROOT / "checkpoints" / "tokenizer.json"))
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("--temperature", type=float, default=default_temperature, help="0 = greedy")
    parser.add_argument("--top_k", "--top-k", type=int, default=default_top_k, help="0 = no top-k filtering")
    parser.add_argument("--seed", type=int, default=default_seed)
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
