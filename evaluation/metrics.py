"""Evaluation metrics: perplexity, bits-per-char, n-gram repetition, and distinctness."""
import math
from collections import Counter
from typing import List, Optional, Union

import torch


def perplexity(loss: float) -> float:
    """exp(loss): measures uncertainty in next token prediction."""
    return math.exp(loss)


def bits_per_token(loss: float) -> float:
    """Convert per-token cross entropy (nats) to bits per token (base-2 logarithm)."""
    return loss / math.log(2)


def bits_per_char(
    loss: float,
    num_tokens: Optional[int] = None,
    num_chars: Optional[int] = None,
) -> float:
    """Convert loss to bits per character (BPC).

    If num_tokens and num_chars are provided (e.g. for tokenizers where 1 token != 1 character),
    BPC = (total_nll_in_nats / ln(2)) / num_chars = (loss * num_tokens / ln(2)) / num_chars.
    If omitted or equal, assumes 1 token per character: loss / ln(2).
    """
    if num_tokens is not None and num_chars is not None:
        if num_chars <= 0:
            raise ValueError("num_chars must be positive")
        total_bits = (loss * num_tokens) / math.log(2)
        return total_bits / num_chars
    return loss / math.log(2)


def loss_metrics(
    loss: float,
    num_tokens: Optional[int] = None,
    num_chars: Optional[int] = None,
) -> dict:
    """Return common loss-derived metrics as a dictionary."""
    return {
        "loss": loss,
        "perplexity": perplexity(loss),
        "bits_per_token": bits_per_token(loss),
        "bits_per_char": bits_per_char(loss, num_tokens=num_tokens, num_chars=num_chars),
    }


def unigram_baseline(
    train_tokens: torch.Tensor,
    eval_tokens: torch.Tensor,
    vocab_size: int,
    smoothing: float = 1.0,
) -> float:
    """Cross-entropy loss of a static unigram distribution evaluated on held-out data.

    The unigram distribution is estimated exclusively from `train_tokens` using Laplace smoothing,
    and then evaluated on held-out `eval_tokens`.
    Handles empty tensors and filters out-of-vocabulary token IDs safely.
    """
    if len(train_tokens) == 0:
        raise ValueError("train_tokens cannot be empty")
    if len(eval_tokens) == 0:
        raise ValueError("eval_tokens cannot be empty")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    # Filter out-of-range token IDs in eval data for safety
    valid_mask = (eval_tokens >= 0) & (eval_tokens < vocab_size)
    if not valid_mask.any():
        raise ValueError(f"eval_tokens contains no tokens within vocabulary range [0, {vocab_size - 1}]")
    eval_tokens_clean = eval_tokens[valid_mask]

    counts = torch.bincount(train_tokens.cpu().clamp(0, vocab_size - 1), minlength=vocab_size).float() + smoothing
    probs = counts / counts.sum()
    return -torch.log(probs[eval_tokens_clean.cpu()]).mean().item()


def calculate_distinct_ngrams(sequence: Union[str, List[int]], n: int = 2) -> float:
    """Calculate the ratio of distinct n-grams to total n-grams (diversity metric).

    Returns 1.0 for completely distinct text, lower values for repetitive text.
    """
    if len(sequence) < n:
        return 1.0
    ngrams = [tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def calculate_repetition_rate(sequence: Union[str, List[int]], n: int = 3) -> float:
    """Calculate the fraction of n-grams that occur more than once."""
    if len(sequence) < n:
        return 0.0
    ngrams = [tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated_count = sum(count for count in counts.values() if count > 1)
    return repeated_count / len(ngrams)
