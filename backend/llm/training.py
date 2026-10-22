"""Causal-LM training loop: batching, train/val split, AdamW, warmup+cosine LR, gradient
accumulation, clipping, periodic validation, early stopping, cancellation and resume.

Every step performs a real gradient update through the selected backend.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .backend import ModelRunner
from .config import TrainingConfig


# --------------------------------------------------------------------------- data
class SequenceSource:
    """Something that can produce (x, y, w) batches. x/y are int64 [B, T], w is float32 [B, T]."""

    def __len__(self) -> int:  # number of sequences
        raise NotImplementedError

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (token_ids [L], loss_weights [L-1]) for sequence i."""
        raise NotImplementedError

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        idx = rng.integers(0, len(self), size=batch_size)
        return collate([self.get(int(i)) for i in idx])

    def eval_batches(self, batch_size: int, max_batches: int):
        n = min(len(self), batch_size * max_batches)
        for s in range(0, n, batch_size):
            yield collate([self.get(i) for i in range(s, min(n, s + batch_size))])


def collate(items: Sequence[tuple[np.ndarray, np.ndarray]], pad_id: int = 0):
    """Dynamic padding to the longest sequence in the batch (pads get weight 0)."""
    L = max(len(ids) for ids, _ in items)
    B = len(items)
    x = np.full((B, L - 1), pad_id, dtype=np.int64)
    y = np.full((B, L - 1), pad_id, dtype=np.int64)
    w = np.zeros((B, L - 1), dtype=np.float32)
    for b, (ids, wt) in enumerate(items):
        n = len(ids)
        x[b, : n - 1] = ids[:-1]
        y[b, : n - 1] = ids[1:]
        w[b, : n - 1] = wt
    return x, y, w


class StreamSource(SequenceSource):
    """Language-modelling stream: documents joined with <eos>, cut into fixed windows (Stage 1)."""

    def __init__(self, token_stream: np.ndarray, window: int):
        self.tokens = np.asarray(token_stream, dtype=np.int64)
        self.window = window
        self.n = max(0, (len(self.tokens) - 1) // window)

    def __len__(self):
        return self.n

    def get(self, i):
        s = i * self.window
        ids = self.tokens[s: s + self.window + 1]
        return ids, np.ones(len(ids) - 1, dtype=np.float32)

    def sample_batch(self, rng, batch_size):
        # random offsets (not only window-aligned) for more variety
        hi = len(self.tokens) - self.window - 1
        starts = rng.integers(0, max(1, hi), size=batch_size)
        return collate([(self.tokens[s: s + self.window + 1],
                         np.ones(self.window, dtype=np.float32)) for s in starts])


class ExampleSource(SequenceSource):
    """Conditional examples (prompt + response). Loss only on the response, scaled by example weight."""

    def __init__(self, examples: list[tuple[list[int], int, float]]):
        # each example: (token ids, prompt_length, weight)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def get(self, i):
        ids, plen, weight = self.examples[i]
        ids = np.asarray(ids, dtype=np.int64)
        w = np.zeros(len(ids) - 1, dtype=np.float32)
        w[max(0, plen - 1):] = weight  # predict tokens from position plen onward
        return ids, w


class MixtureSource(SequenceSource):
    """Samples each batch element from a component chosen with the given probabilities (replay mixing)."""

    def __init__(self, components: list[tuple[str, SequenceSource, float]]):
        comps = [(n, s, r) for n, s, r in components if len(s) > 0 and r > 0]
        if not comps:
            raise ValueError("mixture has no non-empty component")
        total = sum(r for _, _, r in comps)
        self.components = [(n, s, r / total) for n, s, r in comps]

    def __len__(self):
        return sum(len(s) for _, s, _ in self.components)

    @property
    def effective_ratios(self) -> dict[str, float]:
        return {n: r for n, _, r in self.components}

    def get(self, i):
        for _, s, _ in self.components:
            if i < len(s):
                return s.get(i)
            i -= len(s)
        raise IndexError(i)

    def sample_batch(self, rng, batch_size):
        probs = np.array([r for _, _, r in self.components])
        choice = rng.choice(len(self.components), size=batch_size, p=probs)
        items = []
        for c in choice:
            s = self.components[int(c)][1]
            items.append(s.get(int(rng.integers(0, len(s)))))
        return collate(items)


def train_val_split(items: list, val_fraction: float, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_val = max(1, int(len(items) * val_fraction)) if len(items) > 1 else 0
    val = [items[i] for i in order[:n_val]]
    train = [items[i] for i in order[n_val:]]
    return train, val


# --------------------------------------------------------------------------- schedule
def lr_at(step: int, cfg: TrainingConfig, total_steps: int | None = None) -> float:
    total = total_steps or cfg.max_steps
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.learning_rate * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


# --------------------------------------------------------------------------- trainer
@dataclass
class TrainResult:
    steps_completed: int
    final_train_loss: float | None
    best_val_loss: float | None
    final_val_loss: float | None
    tokens_seen: int
    stopped_early: bool
    cancelled: bool
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


class TrainingCancelled(Exception):
    pass


class Trainer:
    def __init__(self, runner: ModelRunner, cfg: TrainingConfig, train: SequenceSource,
                 val: SequenceSource | None = None,
                 on_event: Callable[[dict], None] | None = None,
                 should_stop: Callable[[], bool] | None = None):
        if len(train) == 0:
            raise ValueError("insufficient training data: the training set is empty")
        self.runner, self.cfg, self.train, self.val = runner, cfg, train, val
        self.on_event = on_event or (lambda e: None)
        self.should_stop = should_stop or (lambda: False)

    def evaluate(self) -> float | None:
        if self.val is None or len(self.val) == 0:
            return None
        tot, n = 0.0, 0.0
        for x, y, w in self.val.eval_batches(self.cfg.batch_size, self.cfg.eval_batches):
            ws = float(w.sum())
            if ws == 0:
                continue
            tot += self.runner.eval_loss(x, y, w) * ws
            n += ws
        return tot / n if n else None

    def run(self, start_step: int = 0) -> TrainResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed + start_step)
        t0 = time.time()
        history: list[dict] = []
        best_val, best_weights, bad_evals = None, None, 0
        tokens = 0
        last_loss = None
        stopped_early = cancelled = False
        val_loss = self.evaluate()
        if val_loss is not None:
            best_val, best_weights = val_loss, self.runner.get_weights()
            history.append({"step": start_step, "val_loss": val_loss, "kind": "eval"})
            self.on_event({"type": "eval", "step": start_step, "val_loss": val_loss})
        step = start_step
        ema = None
        for step in range(start_step, cfg.max_steps):
            if self.should_stop():
                cancelled = True
                break
            lr = lr_at(step, cfg)
            micro = [self.train.sample_batch(rng, cfg.batch_size) for _ in range(cfg.grad_accum_steps)]
            out = self.runner.train_step(micro, lr)
            if not math.isfinite(out["loss"]):
                raise FloatingPointError(f"training diverged at step {step} (loss={out['loss']})")
            tokens += int(sum(float((w > 0).sum()) for _, _, w in micro))
            last_loss = out["loss"]
            ema = last_loss if ema is None else 0.9 * ema + 0.1 * last_loss
            if step % cfg.log_interval == 0 or step == cfg.max_steps - 1:
                rec = {"step": step + 1, "train_loss": last_loss, "train_loss_ema": ema, "lr": lr,
                       "grad_norm": out["grad_norm"], "tokens_seen": tokens, "kind": "train",
                       "elapsed_s": time.time() - t0}
                history.append(rec)
                self.on_event({"type": "train", **rec, "max_steps": cfg.max_steps})
            if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
                val_loss = self.evaluate()
                if val_loss is not None:
                    history.append({"step": step + 1, "val_loss": val_loss, "kind": "eval"})
                    self.on_event({"type": "eval", "step": step + 1, "val_loss": val_loss})
                    if best_val is None or val_loss < best_val - 1e-4:
                        best_val, best_weights, bad_evals = val_loss, self.runner.get_weights(), 0
                    else:
                        bad_evals += 1
                        if cfg.early_stopping_patience and bad_evals >= cfg.early_stopping_patience:
                            stopped_early = True
                            break
        steps_done = (step + 1 - start_step) if not cancelled else (step - start_step)
        final_val = self.evaluate()
        if cfg.restore_best and best_weights is not None and best_val is not None and final_val is not None \
                and best_val < final_val:
            # keep the best checkpoint seen (standard early-stopping behaviour)
            self.runner.set_weights(best_weights)
            final_val = best_val
        return TrainResult(steps_done, last_loss, best_val, final_val, tokens, stopped_early, cancelled,
                           history, time.time() - t0)
