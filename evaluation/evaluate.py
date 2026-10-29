"""Measure how good (and how fast) a trained model is."""
import math
import time

import torch

from model.generation import generate


def perplexity(loss):
    """exp(loss): roughly 'how many equally likely characters the model is choosing between'."""
    return math.exp(loss)


def bits_per_char(loss, num_tokens=None, num_chars=None):
    if num_tokens is not None and num_chars is not None:
        if num_chars <= 0:
            raise ValueError("num_chars must be positive")
        return (loss * num_tokens / math.log(2)) / num_chars
    return loss / math.log(2)


@torch.no_grad()
def evaluate_full(model, data, batch_size):
    """Average loss over EVERY non-overlapping window of `data`.

    Unlike the random batches used during training, this is deterministic,
    so two models (or two checkpoints) can be compared fairly.
    """
    context = model.config.context_length
    if len(data) <= context:
        raise ValueError(f"Data has {len(data)} tokens; need more than context_length ({context}).")
    device = next(model.parameters()).device
    data = data.to(device)

    n_windows = (len(data) - 1) // context
    x = data[: n_windows * context].view(n_windows, context)
    y = data[1 : n_windows * context + 1].view(n_windows, context)

    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(0, n_windows, batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        _, loss = model(xb, yb)
        total += loss.item() * xb.size(0)          # weight by rows, last batch may be smaller
    model.train(was_training)
    return total / n_windows


def unigram_baseline(train_data, val_data, vocab_size):
    """Loss of a 'model' that only knows how common each character is (no context).
    A real language model must beat this to prove it uses context."""
    if len(train_data) == 0:
        raise ValueError("train_data cannot be empty")
    if len(val_data) == 0:
        raise ValueError("val_data cannot be empty")
    valid_mask = (val_data >= 0) & (val_data < vocab_size)
    if not valid_mask.any():
        raise ValueError(f"val_data contains no tokens within vocabulary range [0, {vocab_size - 1}]")
    val_clean = val_data[valid_mask]
    counts = torch.bincount(train_data.cpu().clamp(0, vocab_size - 1), minlength=vocab_size).float() + 1.0   # +1 smoothing
    probs = counts / counts.sum()
    return -torch.log(probs[val_clean.cpu()]).mean().item()


def parameter_megabytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def measure_generation_speed(model, num_tokens=200, warmup_tokens=10):
    """Tokens per second when generating one sequence."""
    device = next(model.parameters()).device
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    generate(model, start, warmup_tokens, temperature=1.0, top_k=40)       # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    began = time.perf_counter()
    generate(model, start, num_tokens, temperature=1.0, top_k=40)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return num_tokens / (time.perf_counter() - began)


def peak_memory_during_generation(model, num_tokens=200):
    """Peak GPU memory in MB while generating, or None when not on a GPU."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    generate(model, start, num_tokens, temperature=1.0, top_k=40)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6
