"""Text generation: the model predicts one token, we append it, and repeat.

Four ways to choose the next token from the model's scores (logits):
  greedy       (temperature = 0): always the single most likely token.
  temperature  (> 0): divide the logits first. Below 1 = safer, above 1 = more random.
  top-k        keep only the k most likely tokens, then sample among them.
  top-p        (nucleus sampling) keep the smallest set of tokens whose cumulative probability >= p.
"""
from typing import List, Optional, Union
import torch
import torch.nn.functional as F


def _get_eos_id(tokenizer, eos_token_id: Optional[int] = None) -> Optional[int]:
    if eos_token_id is not None:
        return eos_token_id
    if tokenizer is not None:
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            return tokenizer.eos_token_id
        if hasattr(tokenizer, "stoi") and "[EOS]" in tokenizer.stoi:
            return tokenizer.stoi["[EOS]"]
        if hasattr(tokenizer, "special_to_id") and "[EOS]" in tokenizer.special_to_id:
            return tokenizer.special_to_id["[EOS]"]
    return None


def _decode_token(tokenizer, token_id: int) -> str:
    if not tokenizer:
        return ""
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode([token_id])
        except Exception:
            pass
    if hasattr(tokenizer, "itos"):
        return tokenizer.itos.get(token_id, "")
    return ""


@torch.no_grad()
def generate(
    model,
    token_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    stop_strings=None,
    tokenizer=None,
    eos_token_id=None,
):
    """token_ids: (batch, prompt_len) integers. Returns (batch, prompt_len + num_new_tokens)."""
    if token_ids.dim() != 2 or token_ids.size(1) < 1:
        raise ValueError("token_ids must have shape (batch, prompt_len) with prompt_len >= 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be 0 or more")
    if temperature < 0:
        raise ValueError("temperature must be 0 or more (0 means greedy)")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 (or None to turn it off)")
    if top_p is not None and (top_p <= 0.0 or top_p > 1.0):
        raise ValueError("top_p must be between 0.0 (exclusive) and 1.0 (inclusive), or None")

    was_training = model.training
    model.eval()                                    # dropout off while generating
    context_length = model.config.context_length
    new_text = ""
    actual_eos = _get_eos_id(tokenizer, eos_token_id)

    for _ in range(max_new_tokens):
        context = token_ids[:, -context_length:]    # the model only sees its last window
        logits, _ = model(context)
        logits = logits[:, -1, :].clone()           # scores for the NEXT token only

        if repetition_penalty != 1.0:
            recent_tokens = set(token_ids[0, -context_length:].tolist())
            for tid in recent_tokens:
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth_best = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above top_p threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right by 1 to always preserve at least the top candidate token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_ids = torch.cat([token_ids, next_id], dim=1)

        # Immediate EOS stop check
        if actual_eos is not None and int(next_id[0, 0]) == actual_eos:
            break

        if stop_strings and tokenizer:
            char = _decode_token(tokenizer, int(next_id[0, 0]))
            new_text += char
            stopped = False
            for s in stop_strings:
                if isinstance(s, str):
                    if s in new_text:
                        stopped = True
                        break
                elif hasattr(s, "search"):
                    if s.search(new_text):
                        stopped = True
                        break
            if stopped:
                break

    model.train(was_training)
    return token_ids


@torch.no_grad()
def generate_stream(
    model,
    token_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    stop_strings=None,
    tokenizer=None,
    eos_token_id=None,
):
    """Yields each generated token text in real-time as it is predicted."""
    if token_ids.dim() != 2 or token_ids.size(1) < 1:
        raise ValueError("token_ids must have shape (batch, prompt_len) with prompt_len >= 1")
    if top_p is not None and (top_p <= 0.0 or top_p > 1.0):
        raise ValueError("top_p must be between 0.0 (exclusive) and 1.0 (inclusive), or None")

    was_training = model.training
    model.eval()
    context_length = model.config.context_length
    buffer = ""
    actual_eos = _get_eos_id(tokenizer, eos_token_id)

    for _ in range(max_new_tokens):
        context = token_ids[:, -context_length:]
        logits, _ = model(context)
        logits = logits[:, -1, :].clone()

        if repetition_penalty != 1.0:
            recent_tokens = set(token_ids[0, -context_length:].tolist())
            for tid in recent_tokens:
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth_best = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_ids = torch.cat([token_ids, next_id], dim=1)

        # Immediate EOS stop check
        if actual_eos is not None and int(next_id[0, 0]) == actual_eos:
            break

        char = _decode_token(tokenizer, int(next_id[0, 0])) if tokenizer else ""
        buffer += char

        stopped = False
        if stop_strings:
            for s in stop_strings:
                if isinstance(s, str):
                    if s in buffer:
                        stopped = True
                        break
                elif hasattr(s, "search"):
                    if s.search(buffer):
                        stopped = True
                        break

        if stopped:
            break

        yield char

    model.train(was_training)


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=40,
    top_p=None,
    repetition_penalty=1.0,
    stop_strings=None,
    eos_token_id=None,
):
    """Prompt text in, prompt + continuation out."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, errors="ignore")
    if not ids:
        ids = tokenizer.encode("\n", errors="ignore") or [0]
    start = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model,
        start,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_strings=stop_strings,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
    )
    return tokenizer.decode(out[0].tolist())
