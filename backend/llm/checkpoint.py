"""Checkpoint save/load. One directory per model version; never overwritten.

checkpoints/TwinSim-v3/
    weights.npz          model weights (portable across the torch and numpy backends)
    optimizer.npz        AdamW first/second moments + step
    tokenizer.json       the exact tokenizer used for training (with checksum)
    config.json          TwinSimConfig
    training_state.json  step, LR-schedule state, training config, seed
    dataset.json         dataset version / composition that produced these weights
    metrics.json         training + validation metrics
    meta.json            version, parent, timestamps, backend, code version, weight checksum
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backend import ModelRunner, create_runner
from .config import TwinSimConfig
from .tokenizer import TokenizerError, TwinSimTokenizer


class CheckpointError(Exception):
    pass


BACKEND_DIR = Path(__file__).resolve().parent.parent


def code_version() -> str:
    """Hash of the backend source code (git may not be available on the target machine)."""
    h = hashlib.sha256()
    for sub in ("llm", "twin", "evaluation", "learning", "data"):
        for f in sorted((BACKEND_DIR / sub).glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return "code-" + h.hexdigest()[:10]


def _weights_digest(weights: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(weights):
        h.update(k.encode())
        h.update(np.ascontiguousarray(weights[k], dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


@dataclass
class LoadedCheckpoint:
    runner: ModelRunner
    tokenizer: TwinSimTokenizer
    meta: dict
    training_state: dict
    metrics: dict
    dataset: dict
    path: Path


def save_checkpoint(path: str | Path, runner: ModelRunner, tokenizer: TwinSimTokenizer, *,
                    meta: dict, training_state: dict | None = None, metrics: dict | None = None,
                    dataset: dict | None = None, include_optimizer: bool = True) -> Path:
    path = Path(path)
    if path.exists():
        raise CheckpointError(f"refusing to overwrite existing research checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=path.name + ".tmp-", dir=path.parent))
    try:
        weights = runner.get_weights()
        np.savez(tmp / "weights.npz", **weights)
        if include_optimizer:
            np.savez(tmp / "optimizer.npz", **runner.optimizer_state())
        tokenizer.save(tmp / "tokenizer.json")
        (tmp / "config.json").write_text(json.dumps(runner.config.to_dict(), indent=2))
        (tmp / "training_state.json").write_text(json.dumps(training_state or {}, indent=2, default=str))
        (tmp / "metrics.json").write_text(json.dumps(metrics or {}, indent=2, default=str))
        (tmp / "dataset.json").write_text(json.dumps(dataset or {}, indent=2, default=str))
        full_meta = {
            **meta,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": runner.backend,
            "device": runner.device,
            "parameters": runner.num_parameters(),
            "tokenizer_version": tokenizer.version,
            "code_version": code_version(),
            "weights_sha256": _weights_digest(weights),
        }
        (tmp / "meta.json").write_text(json.dumps(full_meta, indent=2, default=str))
        os.replace(tmp, path)  # atomic on the same filesystem
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path


def load_checkpoint(path: str | Path, backend: str | None = None, with_optimizer: bool = False,
                    train_options: dict | None = None) -> LoadedCheckpoint:
    path = Path(path)
    if not path.is_dir():
        raise CheckpointError(f"checkpoint missing: {path}")
    try:
        meta = json.loads((path / "meta.json").read_text())
        cfg = TwinSimConfig.from_dict(json.loads((path / "config.json").read_text()))
        with np.load(path / "weights.npz") as z:
            weights = {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError) as e:
        raise CheckpointError(f"corrupted checkpoint {path}: {e}") from e
    if _weights_digest(weights) != meta.get("weights_sha256"):
        raise CheckpointError(f"corrupted checkpoint {path}: weight checksum mismatch")
    try:
        tokenizer = TwinSimTokenizer.load(path / "tokenizer.json")
    except TokenizerError as e:
        raise CheckpointError(f"invalid tokenizer in {path}: {e}") from e
    if tokenizer.vocab_size != cfg.vocab_size:
        raise CheckpointError(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {cfg.vocab_size}")
    runner = create_runner(cfg, backend=backend, weights=weights, train_options=train_options)
    if with_optimizer and (path / "optimizer.npz").exists():
        with np.load(path / "optimizer.npz") as z:
            runner.load_optimizer_state({k: z[k] for k in z.files})

    def _j(name):
        f = path / name
        return json.loads(f.read_text()) if f.exists() else {}

    return LoadedCheckpoint(runner, tokenizer, meta, _j("training_state.json"), _j("metrics.json"),
                            _j("dataset.json"), path)
