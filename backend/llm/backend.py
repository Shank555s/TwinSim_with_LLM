"""Compute-backend selection.

TwinSim's model is defined once (architecture, weight names, equations) and has two
interchangeable executors:

* ``torch`` : the primary implementation (``llm/model.py``). Used whenever PyTorch is
               installed. Supports CUDA, mixed precision and larger research configs.
* ``numpy`` : a reference implementation of the *same* network with hand-written
               back-propagation (``llm/numpy_backend.py``). Used automatically when PyTorch
               is unavailable (CPU-only machines, CI, the sandbox this project was built in).

Both read/write the same portable checkpoint format, and ``tests/test_model_torch.py``
checks that they produce identical logits for identical weights.

Override with the environment variable ``TWINSIM_BACKEND=torch|numpy``.
"""
from __future__ import annotations

import importlib.util
import os
from abc import ABC, abstractmethod

import numpy as np

from .config import TwinSimConfig


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def selected_backend() -> str:
    forced = os.environ.get("TWINSIM_BACKEND", "").strip().lower()
    if forced in ("torch", "numpy"):
        if forced == "torch" and not torch_available():
            raise RuntimeError("TWINSIM_BACKEND=torch but PyTorch is not installed (pip install torch)")
        return forced
    return "torch" if torch_available() else "numpy"


class ModelRunner(ABC):
    """Uniform interface the trainer, generator and checkpoint code talk to."""

    backend: str = "?"
    config: TwinSimConfig
    device: str = "cpu"

    @abstractmethod
    def num_parameters(self) -> int: ...

    @abstractmethod
    def next_logits(self, ids: list[int]) -> np.ndarray:
        """Logits (float32, shape [vocab]) for the token following ``ids``."""

    @abstractmethod
    def train_step(self, micro_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]], lr: float) -> dict:
        """One optimiser step accumulated over ``micro_batches`` of (x, y, w). Returns loss & grad norm."""

    @abstractmethod
    def eval_loss(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float: ...

    @abstractmethod
    def get_weights(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def set_weights(self, weights: dict[str, np.ndarray]) -> None: ...

    @abstractmethod
    def optimizer_state(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def load_optimizer_state(self, state: dict[str, np.ndarray]) -> None: ...

    def reset_optimizer(self) -> None:
        self.load_optimizer_state({})

    def describe(self) -> dict:
        c = self.config
        return {
            "backend": self.backend,
            "device": self.device,
            "parameters": self.num_parameters(),
            "layers": c.num_layers,
            "heads": c.num_heads,
            "embedding_dim": c.embedding_dim,
            "context_length": c.context_length,
            "vocab_size": c.vocab_size,
        }


def create_runner(cfg: TwinSimConfig, seed: int = 0, backend: str | None = None,
                  weights: dict[str, np.ndarray] | None = None, train_options: dict | None = None) -> ModelRunner:
    backend = backend or selected_backend()
    train_options = train_options or {}
    if backend == "torch":
        from .torch_backend import TorchRunner
        runner: ModelRunner = TorchRunner(cfg, seed=seed, **train_options)
    elif backend == "numpy":
        from .numpy_backend import NumpyRunner
        runner = NumpyRunner(cfg, seed=seed, **train_options)
    else:
        raise ValueError(f"unknown backend {backend}")
    if weights is not None:
        runner.set_weights(weights)
    return runner


def format_parameter_count(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def banner(runner: ModelRunner) -> str:
    d = runner.describe()
    return (
        "TwinSim LLM\n"
        f"Parameters: {format_parameter_count(d['parameters'])} ({d['parameters']:,})\n"
        f"Layers: {d['layers']}\nHeads: {d['heads']}\nEmbedding: {d['embedding_dim']}\n"
        f"Context: {d['context_length']}\nVocabulary: {d['vocab_size']}\n"
        f"Backend: {d['backend']}\nDevice: {d['device'].upper()}"
    )
