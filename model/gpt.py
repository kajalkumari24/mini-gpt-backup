"""The full GPT model.

token IDs -> embeddings (+ positions) -> N transformer blocks
          -> final LayerNorm -> output layer -> logits (one score per vocabulary entry)
"""
import math

import torch.nn as nn
import torch.nn.functional as F

from .embeddings import InputEmbedding
from .transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.vocab_size is None:
            raise ValueError(
                "config.vocab_size is not set. "
                "Use config.with_vocab_size(tokenizer.vocab_size)."
            )
        self.config = config
        self.input_embedding = InputEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Weight tying: the output layer reuses the token embedding table.
        # Saves parameters and usually helps small models.
        self.lm_head.weight = self.input_embedding.token_embedding.embedding.weight

        self.apply(self._init_weights)
        # Shrink the layers that write into the residual stream, so deep
        # models start out stable.
        scale = 0.02 / math.sqrt(2 * config.num_layers)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ffn.net.2.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, token_ids, targets=None):
        """token_ids: (batch, seq_len) integers.
        Returns (logits, loss). logits: (batch, seq_len, vocab_size).
        loss is None unless targets (same shape as token_ids) are given.
        """
        x = self.input_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must match "
                    f"token_ids shape {tuple(token_ids.shape)}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss
