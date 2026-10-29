"""Unit tests for TransformerBlock and FeedForward sublayers."""
import pytest
import torch

from model.configuration import ModelConfig
from model.transformer_block import FeedForward, TransformerBlock


@pytest.fixture
def small_config():
    return ModelConfig(
        vocab_size=16,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )


def test_feedforward_shape(small_config):
    ffn = FeedForward(small_config)
    x = torch.randn(2, 8, 64)
    out = ffn(x)
    assert out.shape == (2, 8, 64)


def test_transformer_block_shape(small_config):
    block = TransformerBlock(small_config)
    x = torch.randn(3, 10, 64)
    out = block(x)
    assert out.shape == (3, 10, 64)


def test_residual_gradient_flow(small_config):
    block = TransformerBlock(small_config)
    x = torch.randn(2, 4, 64, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

