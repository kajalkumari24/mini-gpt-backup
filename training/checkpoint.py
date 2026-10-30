"""Save and load training checkpoints (weights + optimizer + settings + history)."""
import os
from dataclasses import asdict
from pathlib import Path

import torch

from model.configuration import ModelConfig
from model.gpt import GPT


def save_checkpoint(
    path, model, optimizer, step, best_val_loss, model_cfg, train_cfg, history, tokenizer=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")        # write first, then swap in,
    data = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": model_cfg.to_dict(),
        "training_config": asdict(train_cfg),
        "history": history,
    }
    if tokenizer is not None and hasattr(tokenizer, "get_config"):
        data["tokenizer_config"] = tokenizer.get_config()

    torch.save(data, tmp)                           # so a crash never leaves a half-written file
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def load_model_from_checkpoint(path, device="cpu", tokenizer=None):
    """Rebuild the GPT from a checkpoint. Returns (model in eval mode, checkpoint dict).

    If tokenizer is supplied, verifies vocabulary and mapping compatibility against the model & checkpoint.
    """
    checkpoint = load_checkpoint(path, device)
    config = ModelConfig(**checkpoint["model_config"])

    if tokenizer is not None:
        from model.tokenizer import verify_compatibility
        verify_compatibility(tokenizer, config, checkpoint.get("tokenizer_config"))

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
