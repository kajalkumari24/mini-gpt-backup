"""Benchmark Transformer Model Capacity, Parameters, and Inference Speed across Tiers."""
import sys
from pathlib import Path
import time
from typing import Dict, List
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.configuration import ModelConfig
from model.gpt import GPT

TIERS = {
    "nano": {
        "name": "Nano (Edge / Quick Test)",
        "context_length": 64,
        "embedding_dim": 64,
        "num_heads": 2,
        "num_layers": 2,
        "dropout": 0.0,
    },
    "micro": {
        "name": "Micro (Standard Mini GPT)",
        "context_length": 128,
        "embedding_dim": 128,
        "num_heads": 4,
        "num_layers": 4,
        "dropout": 0.1,
    },
    "mini": {
        "name": "Mini (High Capacity / Extended Context)",
        "context_length": 256,
        "embedding_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "dropout": 0.1,
    },
}


def benchmark_tier(name: str, spec: dict, vocab_size: int = 256) -> Dict[str, any]:
    cfg = ModelConfig(
        vocab_size=vocab_size,
        context_length=spec["context_length"],
        embedding_dim=spec["embedding_dim"],
        num_heads=spec["num_heads"],
        num_layers=spec["num_layers"],
        dropout=spec["dropout"],
    )
    model = GPT(cfg).eval()
    params = model.num_parameters()
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # Benchmark forward pass latency
    dummy_input = torch.randint(0, vocab_size, (1, spec["context_length"]))
    # Warmup
    for _ in range(5):
        _ = model(dummy_input)

    n_runs = 20
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(dummy_input)
    elapsed = time.perf_counter() - start
    ms_per_step = (elapsed / n_runs) * 1000.0

    return {
        "tier": name,
        "title": spec["name"],
        "context_length": spec["context_length"],
        "embedding_dim": spec["embedding_dim"],
        "num_heads": spec["num_heads"],
        "num_layers": spec["num_layers"],
        "parameters": params,
        "size_mb": param_mb,
        "forward_latency_ms": ms_per_step,
    }


def main():
    print("Benchmarking Mini GPT Capacity Configurations...\n")
    print(f"{'Tier':<8} | {'Params':<10} | {'Size (MB)':<10} | {'Context':<8} | {'Dim/Heads/Layers':<18} | {'Latency (ms)':<12}")
    print("-" * 80)

    for tier_id, spec in TIERS.items():
        res = benchmark_tier(tier_id, spec)
        arch = f"{res['embedding_dim']}/{res['num_heads']}/{res['num_layers']}"
        print(
            f"{tier_id:<8} | "
            f"{res['parameters']:<10,d} | "
            f"{res['size_mb']:<10.2f} | "
            f"{res['context_length']:<8} | "
            f"{arch:<18} | "
            f"{res['forward_latency_ms']:<12.2f}"
        )


if __name__ == "__main__":
    main()
