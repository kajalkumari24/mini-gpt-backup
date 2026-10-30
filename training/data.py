"""Turn a long list of token IDs or text into training batches."""
import re
from typing import Tuple
import torch


def deduplicate_conversations(text: str) -> str:
    """Remove duplicate conversation blocks while preserving order."""
    blocks = re.split(r"\n\s*\n(?=YOU:)", text.strip())
    seen = set()
    unique = []
    for block in blocks:
        clean = block.strip()
        if clean and clean not in seen:
            seen.add(clean)
            unique.append(clean)
    return "\n\n".join(unique) + "\n\n"


def split_conversations(text: str, train_split: float, deduplicate: bool = True) -> Tuple[str, str]:
    """Split complete dialogue blocks into train and validation sets without cutting conversations.

    Guarantees:
    - Entire conversation turns remain together (no train/val bleed mid-sentence).
    - Validation data is genuinely held-out.
    - Optional deduplication prevents identical conversations in both sets.
    """
    if deduplicate:
        text = deduplicate_conversations(text)

    blocks = [b.strip() for b in re.split(r"\n\s*\n(?=YOU:)", text.strip()) if b.strip()]
    if not blocks:
        # Fallback to double newline splitting
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    n_train = max(1, int(len(blocks) * train_split))
    train_blocks = blocks[:n_train]
    val_blocks = blocks[n_train:]

    train_text = "\n\n".join(train_blocks) + "\n\n"
    val_text = "\n\n".join(val_blocks) + "\n\n" if val_blocks else train_text
    return train_text, val_text


def split_data(token_ids, train_split):
    """Split into train and validation tensors. Order is kept (no shuffling),
    so validation text is genuinely unseen during training."""
    data = torch.tensor(token_ids, dtype=torch.long)
    n_train = int(len(data) * train_split)
    return data[:n_train], data[n_train:]


def get_batch(data, batch_size, context_length):
    """Random windows of the text.

    x = tokens[i : i + T]
    y = tokens[i + 1 : i + T + 1]   (the same window shifted one step ahead)
    At every position the model must predict the next token, so y is the answer key.
    """
    if len(data) <= context_length:
        raise ValueError(
            f"Data has {len(data)} tokens, but context_length is {context_length}. "
            "Need at least context_length + 1 tokens."
        )
    starts = torch.randint(len(data) - context_length, (batch_size,), device=data.device)
    offsets = torch.arange(context_length, device=data.device)
    index = starts[:, None] + offsets[None, :]          # (B, T)
    return data[index], data[index + 1]
