"""Train the Mini GPT from a text file.

Examples (run from the project folder):
    python scripts/train.py
    python scripts/train.py --data data/mini_gpt_conversations.txt --max-steps 500
    python scripts/train.py --data data/input_shakespeare.txt --max-steps 1000 --device cpu
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.configuration import get_device, load_model_config, load_training_config, set_seed
from model.gpt import GPT
from model.tokenizer import CharTokenizer
from training.data import split_data
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Mini GPT.")
    default_data = ROOT / "data" / "mini_gpt_conversations.txt"
    parser.add_argument(
        "--data",
        default=str(default_data),
        help="Path to training text file (default: data/mini_gpt_conversations.txt).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=str(ROOT / "checkpoints"),
        help="Directory to save checkpoints and tokenizer (default: checkpoints).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint (.pt file) to resume training from.",
    )
    parser.add_argument(
        "--max-steps",
        "--max_steps",
        type=int,
        default=None,
        dest="max_steps",
        help="Maximum training steps (overrides training config).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device: 'auto', 'cpu', 'cuda', etc. (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=None,
        dest="batch_size",
        help="Batch size for training (overrides training config).",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        "--learning_rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Learning rate (overrides training config).",
    )
    parser.add_argument(
        "--eval-interval",
        "--eval_interval",
        type=int,
        default=None,
        dest="eval_interval",
        help="Evaluation interval in steps (overrides training config).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input dataset
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training data file not found: {data_path}. "
            "Please check the path or use data/mini_gpt_conversations.txt or data/input_shakespeare.txt."
        )
    if data_path.stat().st_size == 0:
        raise ValueError(f"Training data file is empty: {data_path}")

    train_cfg = load_training_config()
    model_cfg = load_model_config()

    # Apply command-line overrides
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.eval_interval is not None:
        train_cfg.eval_interval = args.eval_interval

    set_seed(train_cfg.random_seed)
    device = get_device(args.device)

    print(f"Loading data from: {data_path}")
    text = data_path.read_text(encoding="utf-8")
    print(f"Data size: {len(text):,} characters")

    tokenizer = CharTokenizer.from_text(text)
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")
    model_cfg = model_cfg.with_vocab_size(tokenizer.vocab_size)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(checkpoint_dir / "tokenizer.json")

    import torch
    if "YOU:" in text:
        from training.data import split_conversations
        train_text, val_text = split_conversations(text, train_cfg.train_split)
        train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
        val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    else:
        token_ids = tokenizer.encode(text)
        train_data, val_data = split_data(token_ids, train_cfg.train_split)
    print(f"Split into {len(train_data):,} training and {len(val_data):,} validation tokens")

    model = GPT(model_cfg)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Training on device: {device}")

    trainer = Trainer(
        model, train_data, val_data, train_cfg, model_cfg, device, checkpoint_dir, tokenizer=tokenizer
    )
    trainer.train(max_steps=train_cfg.max_steps, resume_from=args.resume)
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
