"""Text generation: the model predicts one token, we append it, and repeat.

Three ways to choose the next token from the model's scores (logits):
  greedy       (temperature = 0): always the single most likely token. Repetitive.
  temperature  (> 0): divide the logits first. Below 1 = safer, above 1 = more random.
  top-k        keep only the k most likely tokens, then sample among them.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    token_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    repetition_penalty=1.0,
    stop_strings=None,
    tokenizer=None,
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

    was_training = model.training
    model.eval()                                    # dropout off while generating
    context_length = model.config.context_length
    new_text = ""

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
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_ids = torch.cat([token_ids, next_id], dim=1)

        if stop_strings and tokenizer:
            char = tokenizer.itos.get(int(next_id[0, 0]), "")
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
    repetition_penalty=1.0,
    stop_strings=None,
    tokenizer=None,
):
    """Yields each generated character in real-time as it is predicted."""
    if token_ids.dim() != 2 or token_ids.size(1) < 1:
        raise ValueError("token_ids must have shape (batch, prompt_len) with prompt_len >= 1")

    was_training = model.training
    model.eval()
    context_length = model.config.context_length
    buffer = ""

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
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_ids = torch.cat([token_ids, next_id], dim=1)

        char = tokenizer.itos.get(int(next_id[0, 0]), "") if tokenizer else ""
        buffer += char

        # Check if buffer ends with or contains any stop token
        stopped = False
        if stop_strings:
            for s in stop_strings:
                if s in buffer:
                    stopped = True
                    break

        if stopped:
            break

        yield char

    model.train(was_training)


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40, repetition_penalty=1.0, stop_strings=None):
    """Prompt text in, prompt + continuation out."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, errors="ignore")
    if not ids:
        ids = tokenizer.encode("\n", errors="ignore") or [0]
    start = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model, start, max_new_tokens, temperature, top_k,
        repetition_penalty=repetition_penalty,
        stop_strings=stop_strings,
        tokenizer=tokenizer,
    )
    return tokenizer.decode(out[0].tolist())
