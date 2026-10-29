"""Compare two or more Mini-GPT checkpoints side by side."""
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

from model.configuration import get_device
from model.tokenizer import CharTokenizer
from training.checkpoint import load_model_from_checkpoint
from .evaluate import evaluate_full, perplexity, bits_per_char
from .performance import benchmark_generation_performance
from .generate_samples import generate_evaluation_samples


def compare_models(
    checkpoints: Dict[str, str],
    tokenizer_path: str,
    eval_data_path: Optional[str] = None,
    prompts: Optional[List[str]] = None,
    device: str = "auto",
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Compare multiple model checkpoints on loss, perplexity, speed, and sample generation.

    Args:
        checkpoints: Dict mapping model alias to checkpoint path (.pt file).
        tokenizer_path: Path to tokenizer.json.
        eval_data_path: Optional path to validation text file.
        prompts: Optional list of prompt strings.
        device: Device to run evaluation on.
        batch_size: Batch size for full loss evaluation.

    Returns:
        Dictionary containing comparison results for each model.
    """
    dev = get_device(device)
    tokenizer = CharTokenizer.load(tokenizer_path)

    eval_tokens = None
    if eval_data_path and Path(eval_data_path).is_file():
        text = Path(eval_data_path).read_text(encoding="utf-8")
        eval_tokens = torch.tensor(tokenizer.encode(text, errors="ignore"), dtype=torch.long)

    results = {}
    for alias, ckpt_path in checkpoints.items():
        if not Path(ckpt_path).is_file():
            results[alias] = {"error": f"Checkpoint file not found: {ckpt_path}"}
            continue

        model, ckpt_data = load_model_from_checkpoint(ckpt_path, dev)
        perf = benchmark_generation_performance(model, dev, num_tokens=50, runs=2)

        entry = {
            "checkpoint": str(ckpt_path),
            "step": ckpt_data.get("step"),
            "best_val_loss": ckpt_data.get("best_val_loss"),
            "parameters": model.num_parameters(),
            "parameter_mb": perf["parameter_mb"],
            "tokens_per_second": perf["tokens_per_second"],
        }

        if eval_tokens is not None and len(eval_tokens) > model.config.context_length:
            try:
                loss = evaluate_full(model, eval_tokens, batch_size=batch_size)
                entry["eval_loss"] = loss
                entry["eval_perplexity"] = perplexity(loss)
                entry["eval_bits_per_char"] = bits_per_char(loss)
            except Exception as e:
                entry["eval_loss_error"] = str(e)

        if prompts:
            entry["samples"] = generate_evaluation_samples(model, tokenizer, prompts=prompts)

        results[alias] = entry

    return results

