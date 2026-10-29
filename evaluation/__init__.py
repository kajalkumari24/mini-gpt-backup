"""Mini-GPT Evaluation Package.

Provides tools for measuring loss, perplexity, generation speed, memory,
n-gram repetition/diversity, and comparing models.
"""

from .evaluate import (
    bits_per_char,
    evaluate_full,
    measure_generation_speed,
    parameter_megabytes,
    peak_memory_during_generation,
    perplexity,
    unigram_baseline,
)
from .metrics import (
    calculate_distinct_ngrams,
    calculate_repetition_rate,
    loss_metrics,
)

__all__ = [
    "bits_per_char",
    "evaluate_full",
    "measure_generation_speed",
    "parameter_megabytes",
    "peak_memory_during_generation",
    "perplexity",
    "unigram_baseline",
    "calculate_distinct_ngrams",
    "calculate_repetition_rate",
    "loss_metrics",
]
