"""Masked multi-head self-attention, written out explicitly (no nn.MultiheadAttention,
no scaled_dot_product_attention) so every step of the computation is visible."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TwinSimConfig


class TwinSimAttention(nn.Module):
    def __init__(self, cfg: TwinSimConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.embedding_dim, 3 * cfg.embedding_dim)
        self.proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular causal mask: position t may only attend to positions <= t.
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_length, cfg.context_length),
                             persistent=False)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, Dh)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        y = self.attn_dropout(weights) @ v  # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return (y, weights) if return_weights else y
