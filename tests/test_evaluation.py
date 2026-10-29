import math

import pytest
import torch

from evaluation.evaluate import (
    bits_per_char, evaluate_full, measure_generation_speed, parameter_megabytes,
    peak_memory_during_generation, perplexity, unigram_baseline,
)
from model.configuration import ModelConfig
from model.gpt import GPT


def make_model():
    cfg = ModelConfig(vocab_size=8, context_length=16, embedding_dim=32,
                      num_heads=4, num_layers=2, dropout=0.0)
    return GPT(cfg).eval()


def test_perplexity_and_bits():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(math.log(8)) == pytest.approx(8.0)
    assert bits_per_char(math.log(8)) == pytest.approx(3.0)


def test_full_loss_does_not_depend_on_batch_size():
    torch.manual_seed(0)
    model = make_model()
    data = torch.randint(0, 8, (200,))
    small = evaluate_full(model, data, batch_size=5)
    large = evaluate_full(model, data, batch_size=100)
    assert small == pytest.approx(large, abs=1e-5)


def test_full_loss_matches_a_manual_calculation():
    torch.manual_seed(0)
    model = make_model()
    data = torch.randint(0, 8, (200,))
    x = data[:192].view(12, 16)
    y = data[1:193].view(12, 16)
    manual = model(x, y)[1].item()
    assert evaluate_full(model, data, batch_size=100) == pytest.approx(manual, abs=1e-5)


def test_full_loss_is_deterministic():
    model = make_model()
    data = torch.randint(0, 8, (100,))
    assert evaluate_full(model, data, 4) == evaluate_full(model, data, 4)


def test_too_little_data_rejected():
    with pytest.raises(ValueError):
        evaluate_full(make_model(), torch.arange(16) % 8, 4)


def test_unigram_baseline():
    even = torch.arange(8).repeat(500)
    assert unigram_baseline(even, even, 8) == pytest.approx(math.log(8), abs=0.01)
    zeros = torch.zeros(1000, dtype=torch.long)
    assert unigram_baseline(zeros, zeros, 8) < 0.1

    # Empty data rejected
    with pytest.raises(ValueError):
        unigram_baseline(torch.tensor([], dtype=torch.long), even, 8)

    # Out of vocabulary token in eval handled safely
    oov_eval = torch.tensor([0, 1, 999], dtype=torch.long)
    assert unigram_baseline(even, oov_eval, 8) > 0


def test_bits_per_char_normalized():
    # 100 tokens, 200 chars -> BPC is halved relative to bits_per_token
    loss = math.log(8)  # 3.0 bits per token
    bpc = bits_per_char(loss, num_tokens=100, num_chars=200)
    assert bpc == pytest.approx(1.5)


def test_parameter_megabytes():
    model = make_model()
    expected = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    assert parameter_megabytes(model) == pytest.approx(expected)


def test_speed_is_positive_and_cpu_has_no_gpu_memory_figure():
    model = make_model()
    assert measure_generation_speed(model, num_tokens=10, warmup_tokens=2) > 0
    assert peak_memory_during_generation(model, num_tokens=5) is None
