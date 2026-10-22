"""Embeddings, feed-forward network and the stack of pre-LayerNorm Transformer blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TwinSimAttention
from .config import TwinSimConfig


class TwinSimEmbedding(nn.Module):
    """Token embeddings + learned absolute positional embeddings."""

    def __init__(self, cfg: TwinSimConfig):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position = nn.Embedding(cfg.context_length, cfg.embedding_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.context_length = cfg.context_length

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.context_length:
            raise ValueError(f"sequence length {T} exceeds context length {self.context_length}")
        pos = torch.arange(T, device=idx.device)
        return self.dropout(self.token(idx) + self.position(pos)[None, :, :])


class TwinSimMLP(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU(tanh) -> Linear."""

    def __init__(self, cfg: TwinSimConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.embedding_dim, cfg.mlp_dim)
        self.proj = nn.Linear(cfg.mlp_dim, cfg.embedding_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class TwinSimTransformerBlock(nn.Module):
    """Pre-LN block:  x + Attn(LN(x)),  then  x + MLP(LN(x))."""

    def __init__(self, cfg: TwinSimConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embedding_dim, eps=cfg.layer_norm_eps)
        self.attn = TwinSimAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embedding_dim, eps=cfg.layer_norm_eps)
        self.mlp = TwinSimMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TwinSimTransformer(nn.Module):
    """N blocks followed by the final LayerNorm."""

    def __init__(self, cfg: TwinSimConfig):
        super().__init__()
        self.blocks = nn.ModuleList([TwinSimTransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embedding_dim, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)
