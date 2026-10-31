"""The training loop.

Each step:  get a batch -> forward pass -> loss -> backward pass -> clip gradients -> AdamW update.
Every eval_interval steps: measure loss on train AND validation text and save checkpoints.
"""
import math
import time
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import get_batch


def get_lr(step, max_steps, base_lr, warmup_steps, min_ratio=0.1):
    """Linear warmup, then cosine decay down to min_ratio * base_lr."""
    min_lr = base_lr * min_ratio
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, eval_iters):
    """Average loss over several random batches, with dropout off."""
    was_training = model.training
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train(was_training)
    return losses.mean().item()


class Trainer:
    def __init__(
        self, model, train_data, val_data, train_cfg, model_cfg, device, checkpoint_dir, tokenizer=None
    ):
        if len(train_data) <= model_cfg.context_length:
            raise ValueError(
                f"train_data has {len(train_data)} tokens, which is <= context_length {model_cfg.context_length}."
            )
        if len(val_data) <= model_cfg.context_length:
            raise ValueError(
                f"val_data has {len(val_data)} tokens, which is <= context_length {model_cfg.context_length}."
            )

        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device)
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = tokenizer
        self.best_val_loss = float("inf")
        self.history = []

        # Weight decay only on matrices (2-D+), not on biases and LayerNorm.
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": train_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=train_cfg.learning_rate,
        )

    def _save(self, name, step, best_val, history):
        save_checkpoint(
            self.checkpoint_dir / name,
            self.model,
            self.optimizer,
            step,
            best_val,
            self.model_cfg,
            self.train_cfg,
            history,
            tokenizer=self.tokenizer,
        )

    def train(self, max_steps=None, resume_from=None, log=print):
        cfg = self.train_cfg
        max_steps = max_steps or cfg.max_steps
        warmup = min(100, max(1, max_steps // 10))
        context = self.model_cfg.context_length

        start_step, best_val, history = 0, float("inf"), []
        if resume_from is not None:
            ckpt = load_checkpoint(resume_from, self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = ckpt["step"]
            best_val = ckpt["best_val_loss"]
            history = [h for h in ckpt["history"] if h["step"] < start_step]
            log(f"Resuming from step {start_step} (best val loss so far {best_val:.4f})")

        self.model.train()
        began = time.time()

        for step in range(start_step, max_steps + 1):
            lr = get_lr(step, max_steps, cfg.learning_rate, warmup)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            if step % cfg.eval_interval == 0 or step == max_steps:
                train_loss = estimate_loss(self.model, self.train_data, cfg.batch_size, context, cfg.eval_iters)
                val_loss = estimate_loss(self.model, self.val_data, cfg.batch_size, context, cfg.eval_iters)
                history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
                marker = ""
                if val_loss < best_val:
                    best_val = val_loss
                    self._save("best.pt", step, best_val, history)
                    marker = "  <- best"
                self._save("last.pt", step, best_val, history)
                msg = (
                    f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} "
                    f"| lr {lr:.2e} | {time.time() - began:5.0f}s{marker}"
                )
                try:
                    log(msg, flush=True)
                except TypeError:
                    log(msg)

            if step == max_steps:
                break

            x, y = get_batch(self.train_data, cfg.batch_size, context)
            _, loss = self.model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss became {loss.item()} at step {step}. Lower the learning rate.")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

        self.best_val_loss = best_val
        self.history = history
        return history
