"""PyTorch backend parity (skipped when torch is not installed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm.backend import create_runner  # noqa: E402
from llm.config import TwinSimConfig  # noqa: E402

CFG = TwinSimConfig(vocab_size=64, context_length=16, embedding_dim=32, num_layers=2, num_heads=4)


def test_torch_matches_numpy_logits():
    np_r = create_runner(CFG, seed=1, backend="numpy")
    t_r = create_runner(CFG, seed=1, backend="torch", weights=np_r.get_weights())
    ids = [1, 5, 9, 3, 7]
    np.testing.assert_allclose(np_r.next_logits(ids), t_r.next_logits(ids), atol=1e-3)


def test_torch_train_step_runs():
    r = create_runner(CFG, seed=0, backend="torch")
    x = np.random.default_rng(0).integers(0, 64, (2, 16))
    out = r.train_step([(x[:, :-1], x[:, 1:], np.ones((2, 15), dtype=np.float32))], lr=1e-3)
    assert np.isfinite(out["loss"])
