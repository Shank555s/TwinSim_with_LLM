"""Autoregressive decoding with temperature, top-k, top-p, repetition penalty and seeding.

The sampler works on the logits returned by the selected backend, so the same code
(and therefore the same randomness for a given seed) serves PyTorch and NumPy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .backend import ModelRunner
from .config import GenerationConfig


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    tokens_generated: int
    latency_ms: float
    stop_reason: str  # "stop_token" | "max_tokens" | "context_full"


def apply_repetition_penalty(logits: np.ndarray, previous: list[int], penalty: float) -> np.ndarray:
    """CTRL-style penalty: divide positive / multiply negative logits of tokens already generated."""
    if penalty == 1.0 or not previous:
        return logits
    logits = logits.copy()
    idx = np.unique(np.asarray(previous))
    vals = logits[idx]
    logits[idx] = np.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def filter_top_k_top_p(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    logits = logits.astype(np.float64).copy()
    V = logits.shape[0]
    if 0 < top_k < V:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    if 0.0 < top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        probs = np.exp(sorted_logits - sorted_logits[0])
        probs /= probs.sum()
        cum = np.cumsum(probs)
        cut = np.searchsorted(cum, top_p) + 1  # keep the smallest prefix with mass >= top_p
        logits[order[cut:]] = -np.inf
    return logits


def sample_next(logits: np.ndarray, cfg: GenerationConfig, rng: np.random.Generator, previous: list[int],
                banned: list[int] | None = None) -> int:
    logits = apply_repetition_penalty(logits, previous, cfg.repetition_penalty)
    if banned:
        logits = logits.copy()
        logits[banned] = -np.inf
    if cfg.deterministic or cfg.temperature <= 0:
        return int(np.argmax(logits))
    logits = filter_top_k_top_p(logits / max(cfg.temperature, 1e-5), cfg.top_k, cfg.top_p)
    m = np.max(logits)
    probs = np.exp(logits - m)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(runner: ModelRunner, prompt_ids: list[int], cfg: GenerationConfig, stop_ids: list[int],
             banned_ids: list[int] | None = None, decode=None, on_token=None) -> GenerationResult:
    """`on_token(new_ids)` is called after every generated token (used for live streaming to the UI)."""
    t0 = time.perf_counter()
    ctx = runner.config.context_length
    if len(prompt_ids) >= ctx:
        raise ValueError(f"prompt is {len(prompt_ids)} tokens; context length is {ctx}")
    rng = np.random.default_rng(cfg.seed if cfg.seed is not None else None)
    ids = list(prompt_ids)
    new: list[int] = []
    reason = "max_tokens"
    for _ in range(cfg.max_new_tokens):
        if len(ids) >= ctx:
            reason = "context_full"
            break
        banned = list(banned_ids or [])
        if len(new) < cfg.min_new_tokens:
            banned += list(stop_ids)
        nxt = sample_next(runner.next_logits(ids), cfg, rng, new, banned)
        if nxt in stop_ids:
            reason = "stop_token"
            break
        ids.append(nxt)
        new.append(nxt)
        if on_token is not None:
            on_token(new)
    text = decode(new) if decode else ""
    return GenerationResult(new, text, len(new), (time.perf_counter() - t0) * 1000.0, reason)
