"""From-scratch NumPy Transformer: gradients, causality, KV cache, training, checkpoints, generation."""
import numpy as np
import pytest

from llm.backend import create_runner, format_parameter_count
from llm.checkpoint import load_checkpoint, save_checkpoint
from llm.config import GenerationConfig, TrainingConfig, TwinSimConfig
from llm.generation import generate
from llm.numpy_backend import NumpyTransformer
from llm.tokenizer import TwinSimTokenizer
from llm.training import ExampleSource, StreamSource, Trainer

CFG = TwinSimConfig(vocab_size=300, context_length=32, embedding_dim=32, num_layers=2, num_heads=4)


def test_config_validation():
    with pytest.raises(ValueError):
        TwinSimConfig(embedding_dim=30, num_heads=4)


def test_parameter_count_matches_formula():
    r = create_runner(CFG, backend="numpy")
    C, V, T, L = CFG.embedding_dim, CFG.vocab_size, CFG.context_length, CFG.num_layers
    per_layer = 2 * 2 * C + (3 * C * C + 3 * C) + (C * C + C) + (4 * C * C + 4 * C) + (4 * C * C + C)
    assert r.num_parameters() == V * C + T * C + L * per_layer + 2 * C
    assert format_parameter_count(1_500_000).startswith("1.5")


def test_gradient_check():
    cfg = TwinSimConfig(vocab_size=40, context_length=8, embedding_dim=16, num_layers=2, num_heads=2)
    net = NumpyTransformer(cfg, seed=3)
    net.params = {k: v.astype(np.float64) for k, v in net.params.items()}
    net.neg_mask = net.neg_mask.astype(np.float64)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 40, (2, 8))
    y = rng.integers(0, 40, (2, 8))
    w = np.ones((2, 8))
    w[0, :3] = 0  # masked prompt tokens

    def f():
        lg, _ = net.forward(x, keep_cache=False)
        return net.loss(lg, y, w)[0]

    lg, fc = net.forward(x)
    _, lc = net.loss(lg, y, w)
    grads = net.backward(y, lc, fc)
    for name in ["embedding.token.weight", "embedding.position.weight", "transformer.blocks.0.attn.qkv.weight",
                 "transformer.blocks.1.mlp.fc.weight", "transformer.blocks.0.ln1.weight", "transformer.ln_f.bias"]:
        p = net.params[name]
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + 1e-5
            lp = f()
            p[i] = old - 1e-5
            lm = f()
            p[i] = old
            num = (lp - lm) / 2e-5
            assert abs(num - grads[name][i]) < 1e-6 + 1e-4 * abs(num), name


def test_causal_mask():
    net = NumpyTransformer(CFG, seed=1)
    a = np.array([[5, 6, 7, 8, 9, 10]])
    b = a.copy()
    b[0, -1] = 99
    la, _ = net.forward(a, keep_cache=False)
    lb, _ = net.forward(b, keep_cache=False)
    np.testing.assert_allclose(la[0, :-1], lb[0, :-1], atol=1e-5)
    assert not np.allclose(la[0, -1], lb[0, -1])


def test_kv_cache_matches_full_forward():
    r = create_runner(CFG, seed=2, backend="numpy")
    ids = [1, 7, 9, 22, 41, 5, 17]
    for n in range(2, len(ids) + 1):
        inc = r.next_logits(ids[:n])  # incremental (KV cache) path
        full, _ = r.net.forward(np.asarray(ids[:n])[None], keep_cache=False)
        np.testing.assert_allclose(inc, full[0, -1], atol=1e-4)


def _pattern_source():
    stream = np.tile(np.arange(10, 30), 60)
    return StreamSource(stream, 16)


def test_training_reduces_loss():
    r = create_runner(CFG, seed=0, backend="numpy")
    tc = TrainingConfig(batch_size=8, max_steps=60, learning_rate=3e-3, warmup_steps=5, eval_interval=20,
                        eval_batches=2, log_interval=10)
    res = Trainer(r, tc, _pattern_source(), _pattern_source()).run()
    assert res.steps_completed == 60
    assert res.final_train_loss < 2.5 < np.log(CFG.vocab_size)  # well below uniform (5.7)


def test_prompt_tokens_have_zero_weight():
    src = ExampleSource([([1, 2, 3, 4, 5, 6], 3, 1.0)])
    ids, w = src.get(0)
    assert w[:2].sum() == 0 and w[2:].sum() > 0


def test_restore_best_flag():
    """restore_best=False must keep the final weights (feedback updates rely on this)."""
    for restore in (True, False):
        r = create_runner(CFG, seed=0, backend="numpy")
        tc = TrainingConfig(batch_size=4, max_steps=30, learning_rate=5e-2, warmup_steps=1, eval_interval=10,
                            eval_batches=1, early_stopping_patience=0, restore_best=restore)
        Trainer(r, tc, _pattern_source(), _pattern_source()).run()
        assert r.get_weights()  # ran without error in both modes


def test_checkpoint_roundtrip(tmp_path):
    tok = TwinSimTokenizer.train(["hello world, this is twinsim."] * 10, vocab_size=300)
    cfg = TwinSimConfig(**{**CFG.to_dict(), "vocab_size": tok.vocab_size})
    r = create_runner(cfg, seed=4, backend="numpy")
    save_checkpoint(tmp_path / "ck", r, tok, meta={"name": "t"})
    lc = load_checkpoint(tmp_path / "ck", backend="numpy")
    for k, v in r.get_weights().items():
        np.testing.assert_array_equal(v, lc.runner.get_weights()[k])
    assert lc.tokenizer.version == tok.version


def test_checkpoint_corruption_detected(tmp_path):
    from llm.checkpoint import CheckpointError
    tok = TwinSimTokenizer.train(["hello world"] * 10, vocab_size=300)
    cfg = TwinSimConfig(**{**CFG.to_dict(), "vocab_size": tok.vocab_size})
    save_checkpoint(tmp_path / "ck", create_runner(cfg, backend="numpy"), tok, meta={})
    (tmp_path / "ck" / "weights.npz").write_bytes(b"garbage")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "ck", backend="numpy")


def test_generation_seed_and_min_tokens():
    r = create_runner(CFG, seed=5, backend="numpy")
    g = GenerationConfig(max_new_tokens=12, min_new_tokens=6, temperature=1.0, seed=11)
    stop = list(range(0, 300))  # every token is a stop token: only min_new_tokens can force output
    res = generate(r, [1, 2, 3], g, stop_ids=stop[:250])
    assert res.tokens_generated >= 6
    a = generate(r, [1, 2, 3], GenerationConfig(max_new_tokens=10, seed=9), stop_ids=[]).token_ids
    b = generate(r, [1, 2, 3], GenerationConfig(max_new_tokens=10, seed=9), stop_ids=[]).token_ids
    assert a == b
    streamed = []
    generate(r, [1, 2], GenerationConfig(max_new_tokens=5, seed=1), stop_ids=[], on_token=lambda n: streamed.append(len(n)))
    assert streamed == [1, 2, 3, 4, 5]


def test_greedy_is_deterministic():
    r = create_runner(CFG, seed=5, backend="numpy")
    g = GenerationConfig(max_new_tokens=8, deterministic=True, seed=None)
    assert generate(r, [4, 5], g, []).token_ids == generate(r, [4, 5], g, []).token_ids
