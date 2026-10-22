"""TwinSimLanguageModel: the complete decoder-only Transformer (PyTorch).

    token ids -> TwinSimEmbedding (token + position)
              -> TwinSimTransformer (N x [LN -> masked MHSA -> +res, LN -> MLP -> +res], final LN)
              -> TwinSimLMHead (linear projection to vocabulary logits)

Weights are randomly initialised (GPT-2 style: N(0, 0.02), residual projections
scaled by 1/sqrt(2N)). Nothing is loaded from any pretrained model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TwinSimConfig
from .transformer import TwinSimEmbedding, TwinSimTransformer


class TwinSimLMHead(nn.Module):
    """Projects hidden states to vocabulary logits. When `tied`, it reuses the token-embedding matrix."""

    def __init__(self, cfg: TwinSimConfig, embedding: TwinSimEmbedding):
        super().__init__()
        self.tied = cfg.tie_embeddings
        if self.tied:
            # keep a non-registered reference so the weight appears once in state_dict()
            object.__setattr__(self, "_embedding", embedding)
        else:
            self.proj = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.tied:
            return F.linear(h, self._embedding.token.weight)
        return self.proj(h)


class TwinSimLanguageModel(nn.Module):
    def __init__(self, cfg: TwinSimConfig):
        super().__init__()
        self.config = cfg
        self.embedding = TwinSimEmbedding(cfg)
        self.transformer = TwinSimTransformer(cfg)
        self.lm_head = TwinSimLMHead(cfg, self.embedding)
        self.apply(self._init_weights)
        resid_std = cfg.init_std / math.sqrt(2 * cfg.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_weights: torch.Tensor | None = None):
        """Returns (logits, loss). `loss_weights` (B, T) lets us ignore prompt tokens / weight examples."""
        h = self.transformer(self.embedding(idx))
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none")
            if loss_weights is None:
                loss = ce.mean()
            else:
                w = loss_weights.reshape(-1).to(ce.dtype)
                loss = (ce * w).sum() / w.sum().clamp_min(1e-8)
        return logits, loss
