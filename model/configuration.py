"""Load YAML settings, validate them, pick the device and set the random seed."""
import random
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    vocab_size: Optional[int] = None   # filled in from the tokenizer
    context_length: int = 128
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("context_length", "embedding_dim", "num_heads", "num_layers"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible "
                f"by num_heads ({self.num_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.vocab_size is not None and self.vocab_size < 1:
            raise ValueError("vocab_size must be at least 1 when set")

    @property
    def head_dim(self):
        return self.embedding_dim // self.num_heads

    def with_vocab_size(self, vocab_size):
        """Return a copy with the vocabulary size filled in."""
        return replace(self, vocab_size=int(vocab_size))

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingConfig:
    learning_rate: float = 0.0003
    batch_size: int = 16
    max_steps: int = 1000
    eval_interval: int = 100
    random_seed: int = 42
    train_split: float = 0.9
    weight_decay: float = 0.1     # optional in the YAML
    grad_clip: float = 1.0        # optional in the YAML
    eval_iters: int = 50          # batches averaged per evaluation (optional in the YAML)

    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)   # guards against "3e-4" read as text
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
        for name in ("batch_size", "max_steps", "eval_interval", "eval_iters"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: Optional[int] = 40
    repetition_penalty: float = 1.1
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be at least 1 or None")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1.0")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative or None")

    def to_dict(self):
        return asdict(self)


def _read_yaml(relative_path):
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(path="configs/model.yaml"):
    data = _read_yaml(path).get("model")
    if data is None:
        raise ValueError(f"{path} must contain a top-level 'model:' section")
    return ModelConfig(**data)


def load_training_config(path="configs/training.yaml"):
    data = _read_yaml(path).get("training")
    if data is None:
        raise ValueError(f"{path} must contain a top-level 'training:' section")
    return TrainingConfig(**data)


def load_generation_config(path="configs/generation.yaml"):
    data = _read_yaml(path).get("generation")
    if data is None:
        raise ValueError(f"{path} must contain a top-level 'generation:' section")
    return GenerationConfig(**data)


def get_device(preference="auto"):
    """'auto' picks cuda, then mps, then cpu. Returns a torch.device."""
    import torch
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """Make random numbers repeatable."""
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
