"""Performance benchmarking: throughput (tokens/sec), latency, and memory."""
import time
from typing import Dict, Any
import torch

from model.generation import generate


def benchmark_generation_performance(
    model,
    device,
    num_tokens: int = 100,
    warmup_tokens: int = 10,
    runs: int = 3,
) -> Dict[str, Any]:
    """Measure generation latency, throughput, and memory consumption."""
    device = torch.device(device)
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    if warmup_tokens > 0:
        generate(model, start, warmup_tokens, temperature=0.0)
        sync()

    times = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        generate(model, start, num_tokens, temperature=0.0)
        sync()
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    throughput = num_tokens / min_time

    param_count = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    result = {
        "parameters": param_count,
        "parameter_mb": param_mb,
        "tokens_per_second": throughput,
        "avg_latency_seconds": avg_time,
        "min_latency_seconds": min_time,
        "ms_per_token": (min_time / num_tokens) * 1000.0,
    }

    if device.type == "cuda":
        result["gpu_peak_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

    return result

