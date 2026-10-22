"""NumPy reference executor for the TwinSim Transformer, forward AND backward written by hand.

Same architecture, same parameter names and the same equations as ``llm/model.py``:
pre-LN blocks, masked multi-head self-attention, tanh-GELU MLP, learned positional
embeddings, tied LM head, weighted cross-entropy, AdamW with decoupled weight decay.

It exists so TwinSim trains and runs where PyTorch is not installed, and as an
executable specification of every gradient (checked by finite differences in
``tests/test_model_numpy.py``).
"""
from __future__ import annotations

import math

import numpy as np

from .backend import ModelRunner
from .config import TwinSimConfig

DT = np.float32
_GELU_C = math.sqrt(2.0 / math.pi)


# --------------------------------------------------------------------------- primitives
def layer_norm(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    return xhat * g + b, (xhat, rstd)


def layer_norm_back(dy, cache, g):
    xhat, rstd = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, dy.shape[-1]).sum(0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def gelu(x):
    u = _GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * x * (1.0 + t), t


def gelu_back(dy, x, t):
    du = _GELU_C * (1.0 + 3 * 0.044715 * x * x)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# --------------------------------------------------------------------------- model
class NumpyTransformer:
    def __init__(self, cfg: TwinSimConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        d, V, T, m = cfg.embedding_dim, cfg.vocab_size, cfg.context_length, cfg.mlp_dim
        std, rstd = cfg.init_std, cfg.init_std / math.sqrt(2 * cfg.num_layers)
        n = lambda *s, sd=std: (rng.standard_normal(s) * sd).astype(DT)  # noqa: E731
        p: dict[str, np.ndarray] = {
            "embedding.token.weight": n(V, d),
            "embedding.position.weight": n(T, d),
        }
        for i in range(cfg.num_layers):
            pre = f"transformer.blocks.{i}."
            p[pre + "ln1.weight"] = np.ones(d, DT)
            p[pre + "ln1.bias"] = np.zeros(d, DT)
            p[pre + "attn.qkv.weight"] = n(3 * d, d)
            p[pre + "attn.qkv.bias"] = np.zeros(3 * d, DT)
            p[pre + "attn.proj.weight"] = n(d, d, sd=rstd)
            p[pre + "attn.proj.bias"] = np.zeros(d, DT)
            p[pre + "ln2.weight"] = np.ones(d, DT)
            p[pre + "ln2.bias"] = np.zeros(d, DT)
            p[pre + "mlp.fc.weight"] = n(m, d)
            p[pre + "mlp.fc.bias"] = np.zeros(m, DT)
            p[pre + "mlp.proj.weight"] = n(d, m, sd=rstd)
            p[pre + "mlp.proj.bias"] = np.zeros(d, DT)
        p["transformer.ln_f.weight"] = np.ones(d, DT)
        p["transformer.ln_f.bias"] = np.zeros(d, DT)
        if not cfg.tie_embeddings:
            p["lm_head.proj.weight"] = n(V, d)
        self.params = p
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # True where attention is forbidden
        self.neg_mask = np.where(mask, -1e9, 0.0).astype(DT)

    def head_weight(self):
        return self.params["embedding.token.weight"] if self.cfg.tie_embeddings else self.params["lm_head.proj.weight"]

    # ------------------------------------------------------------------ forward
    def forward(self, idx: np.ndarray, keep_cache: bool = True):
        cfg, p = self.cfg, self.params
        B, T = idx.shape
        if T > cfg.context_length:
            raise ValueError(f"sequence length {T} exceeds context length {cfg.context_length}")
        H, Dh, C = cfg.num_heads, cfg.head_dim, cfg.embedding_dim
        h = p["embedding.token.weight"][idx] + p["embedding.position.weight"][:T][None]
        caches = []
        scale = 1.0 / math.sqrt(Dh)
        for i in range(cfg.num_layers):
            pre = f"transformer.blocks.{i}."
            a_in, ln1c = layer_norm(h, p[pre + "ln1.weight"], p[pre + "ln1.bias"], cfg.layer_norm_eps)
            qkv = a_in.reshape(B * T, C) @ p[pre + "attn.qkv.weight"].T + p[pre + "attn.qkv.bias"]
            qkv = qkv.reshape(B, T, 3, H, Dh).transpose(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
            q, k, v = qkv[0], qkv[1], qkv[2]
            s = (q @ k.transpose(0, 1, 3, 2)) * scale + self.neg_mask[:T, :T]
            att = softmax(s)
            o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
            proj = o.reshape(B * T, C) @ p[pre + "attn.proj.weight"].T + p[pre + "attn.proj.bias"]
            h = h + proj.reshape(B, T, C)
            m_in, ln2c = layer_norm(h, p[pre + "ln2.weight"], p[pre + "ln2.bias"], cfg.layer_norm_eps)
            f = m_in.reshape(B * T, C) @ p[pre + "mlp.fc.weight"].T + p[pre + "mlp.fc.bias"]
            g, t = gelu(f)
            mo = g @ p[pre + "mlp.proj.weight"].T + p[pre + "mlp.proj.bias"]
            h = h + mo.reshape(B, T, C)
            if keep_cache:
                caches.append(dict(a_in=a_in, ln1c=ln1c, q=q, k=k, v=v, att=att, o=o,
                                   m_in=m_in, ln2c=ln2c, f=f, g=g, t=t))
        hf, lnfc = layer_norm(h, p["transformer.ln_f.weight"], p["transformer.ln_f.bias"], cfg.layer_norm_eps)
        logits = (hf.reshape(B * T, C) @ self.head_weight().T).reshape(B, T, -1)
        return logits, (idx, caches, hf, lnfc)

    # ----------------------------------------------------------- loss & backward
    @staticmethod
    def loss(logits, y, w):
        B, T, V = logits.shape
        lg = logits.reshape(-1, V)
        m = lg.max(-1, keepdims=True)
        lse = np.log(np.exp(lg - m).sum(-1)) + m[:, 0]
        nll = lse - lg[np.arange(lg.shape[0]), y.reshape(-1)]
        wf = w.reshape(-1).astype(DT)
        wsum = max(float(wf.sum()), 1e-8)
        return float((nll * wf).sum() / wsum), (lg, lse, wf, wsum)

    def backward(self, y, loss_cache, fwd_cache) -> dict[str, np.ndarray]:
        cfg, p = self.cfg, self.params
        lg, lse, wf, wsum = loss_cache
        idx, caches, hf, lnfc = fwd_cache
        B, T = idx.shape
        H, Dh, C = cfg.num_heads, cfg.head_dim, cfg.embedding_dim
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        probs = np.exp(lg - lse[:, None])
        probs[np.arange(lg.shape[0]), y.reshape(-1)] -= 1.0
        dlogits = (probs * (wf / wsum)[:, None]).astype(DT)  # (B*T, V)

        hw_name = "embedding.token.weight" if cfg.tie_embeddings else "lm_head.proj.weight"
        grads[hw_name] += dlogits.T @ hf.reshape(B * T, C)
        dhf = (dlogits @ self.head_weight()).reshape(B, T, C)
        dh, dg_, db_ = layer_norm_back(dhf, lnfc, p["transformer.ln_f.weight"])
        grads["transformer.ln_f.weight"] += dg_
        grads["transformer.ln_f.bias"] += db_
        scale = 1.0 / math.sqrt(Dh)

        for i in reversed(range(cfg.num_layers)):
            pre = f"transformer.blocks.{i}."
            c = caches[i]
            # --- MLP branch
            dmo = dh.reshape(B * T, C)
            grads[pre + "mlp.proj.weight"] += dmo.T @ c["g"]
            grads[pre + "mlp.proj.bias"] += dmo.sum(0)
            dg = dmo @ p[pre + "mlp.proj.weight"]
            df = gelu_back(dg, c["f"], c["t"])
            grads[pre + "mlp.fc.weight"] += df.T @ c["m_in"].reshape(B * T, C)
            grads[pre + "mlp.fc.bias"] += df.sum(0)
            dm_in = (df @ p[pre + "mlp.fc.weight"]).reshape(B, T, C)
            dx, dg_, db_ = layer_norm_back(dm_in, c["ln2c"], p[pre + "ln2.weight"])
            grads[pre + "ln2.weight"] += dg_
            grads[pre + "ln2.bias"] += db_
            dh = dh + dx
            # --- attention branch
            dproj = dh.reshape(B * T, C)
            grads[pre + "attn.proj.weight"] += dproj.T @ c["o"].reshape(B * T, C)
            grads[pre + "attn.proj.bias"] += dproj.sum(0)
            do = (dproj @ p[pre + "attn.proj.weight"]).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
            datt = do @ c["v"].transpose(0, 1, 3, 2)
            dv = c["att"].transpose(0, 1, 3, 2) @ do
            ds = c["att"] * (datt - (datt * c["att"]).sum(-1, keepdims=True))
            ds *= scale
            dq = ds @ c["k"]
            dk = ds.transpose(0, 1, 3, 2) @ c["q"]
            dqkv = np.stack([dq, dk, dv], 0).transpose(1, 3, 0, 2, 4).reshape(B * T, 3 * C)
            grads[pre + "attn.qkv.weight"] += dqkv.T @ c["a_in"].reshape(B * T, C)
            grads[pre + "attn.qkv.bias"] += dqkv.sum(0)
            da_in = (dqkv @ p[pre + "attn.qkv.weight"]).reshape(B, T, C)
            dx, dg_, db_ = layer_norm_back(da_in, c["ln1c"], p[pre + "ln1.weight"])
            grads[pre + "ln1.weight"] += dg_
            grads[pre + "ln1.bias"] += db_
            dh = dh + dx

        np.add.at(grads["embedding.token.weight"], idx.reshape(-1), dh.reshape(B * T, C))
        grads["embedding.position.weight"][:T] += dh.sum(0)
        return grads


# --------------------------------------------------------------------------- runner
class NumpyRunner(ModelRunner):
    backend = "numpy"

    def __init__(self, cfg: TwinSimConfig, seed: int = 0, weight_decay: float = 0.01,
                 betas: tuple[float, float] = (0.9, 0.98), grad_clip: float = 1.0,
                 mixed_precision: bool = False, device: str | None = None):
        self.config = cfg
        self.device = "cpu"
        self.net = NumpyTransformer(cfg, seed=seed)
        self.weight_decay = weight_decay
        self.betas = betas
        self.grad_clip = grad_clip
        self._kv = None  # incremental-decoding cache: (prefix ids, per-layer K/V)
        self.load_optimizer_state({})

    def num_parameters(self) -> int:
        return int(sum(v.size for v in self.net.params.values()))

    # -------------------------------------------------------------- inference
    def next_logits(self, ids: list[int]) -> np.ndarray:
        ids = list(ids)[-self.config.context_length:]
        cache = self._kv
        if cache is not None and len(cache[0]) == len(ids) - 1 and cache[0] == ids[:-1]:
            return self._decode_one(ids)
        return self._prefill(ids)

    def _prefill(self, ids: list[int]) -> np.ndarray:
        """Full forward over the prompt; stores K/V for every layer for fast incremental decoding."""
        logits, (_, caches, _, _) = self.net.forward(np.asarray(ids)[None, :], keep_cache=True)
        self._kv = (list(ids), [(c["k"][0].copy(), c["v"][0].copy()) for c in caches])
        return logits[0, -1].astype(np.float32)

    def _decode_one(self, ids: list[int]) -> np.ndarray:
        cfg, p = self.config, self.net.params
        _, kv = self._kv
        pos = len(ids) - 1
        H, Dh, C = cfg.num_heads, cfg.head_dim, cfg.embedding_dim
        h = p["embedding.token.weight"][ids[-1]] + p["embedding.position.weight"][pos]
        h = h[None, :]
        new_kv = []
        for i in range(cfg.num_layers):
            pre = f"transformer.blocks.{i}."
            a_in, _ = layer_norm(h, p[pre + "ln1.weight"], p[pre + "ln1.bias"], cfg.layer_norm_eps)
            qkv = (a_in @ p[pre + "attn.qkv.weight"].T + p[pre + "attn.qkv.bias"]).reshape(3, H, 1, Dh)
            k_all = np.concatenate([kv[i][0], qkv[1]], axis=1)  # (H, t+1, Dh)
            v_all = np.concatenate([kv[i][1], qkv[2]], axis=1)
            s = (qkv[0] @ k_all.transpose(0, 2, 1)) / math.sqrt(Dh)  # (H, 1, t+1)
            o = (softmax(s) @ v_all).transpose(1, 0, 2).reshape(1, C)
            h = h + o @ p[pre + "attn.proj.weight"].T + p[pre + "attn.proj.bias"]
            m_in, _ = layer_norm(h, p[pre + "ln2.weight"], p[pre + "ln2.bias"], cfg.layer_norm_eps)
            g, _ = gelu(m_in @ p[pre + "mlp.fc.weight"].T + p[pre + "mlp.fc.bias"])
            h = h + g @ p[pre + "mlp.proj.weight"].T + p[pre + "mlp.proj.bias"]
            new_kv.append((k_all, v_all))
        hf, _ = layer_norm(h, p["transformer.ln_f.weight"], p["transformer.ln_f.bias"], cfg.layer_norm_eps)
        self._kv = (list(ids), new_kv)
        return (hf @ self.net.head_weight().T)[0].astype(np.float32)

    # --------------------------------------------------------------- training
    def loss_and_grads(self, x, y, w):
        logits, fcache = self.net.forward(x)
        loss, lcache = self.net.loss(logits, y, w)
        return loss, self.net.backward(y, lcache, fcache)

    def train_step(self, micro_batches, lr: float) -> dict:
        self._kv = None
        total_w = sum(float(np.sum(w)) for _, _, w in micro_batches) or 1.0
        acc = None
        loss_acc = 0.0
        for x, y, w in micro_batches:
            share = float(np.sum(w)) / total_w
            loss, grads = self.loss_and_grads(x, y, w)
            loss_acc += loss * share
            if acc is None:
                acc = {k: g * share for k, g in grads.items()}
            else:
                for k, g in grads.items():
                    acc[k] += g * share
        gnorm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in acc.values()))
        clip = min(1.0, self.grad_clip / (gnorm + 1e-6)) if self.grad_clip > 0 else 1.0
        self._step += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self._step, 1 - b2 ** self._step
        for k, prm in self.net.params.items():
            g = acc[k] * clip
            m, v = self._m[k], self._v[k]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            if prm.ndim >= 2 and self.weight_decay > 0:
                prm *= (1 - lr * self.weight_decay)
            prm -= (lr * (m / bc1) / (np.sqrt(v / bc2) + 1e-8)).astype(DT)
        return {"loss": loss_acc, "grad_norm": gnorm}

    def eval_loss(self, x, y, w) -> float:
        logits, _ = self.net.forward(x, keep_cache=False)
        return self.net.loss(logits, y, w)[0]

    # ------------------------------------------------------------- state I/O
    def get_weights(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.net.params.items()}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        own = self.net.params
        missing, unexpected = set(own) - set(weights), set(weights) - set(own)
        if missing or unexpected:
            raise ValueError(f"weight mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
        for k, v in weights.items():
            if own[k].shape != v.shape:
                raise ValueError(f"shape mismatch for {k}: {own[k].shape} vs {v.shape}")
            own[k] = np.ascontiguousarray(v, dtype=DT)
        self._kv = None

    def optimizer_state(self) -> dict[str, np.ndarray]:
        out = {f"m::{k}": v.copy() for k, v in self._m.items()}
        out.update({f"v::{k}": v.copy() for k, v in self._v.items()})
        out["__step__"] = np.asarray(self._step)
        return out

    def load_optimizer_state(self, state: dict[str, np.ndarray]) -> None:
        self._m = {k: np.zeros_like(v) for k, v in self.net.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.net.params.items()}
        self._step = int(state.get("__step__", 0)) if state else 0
        for k in self.net.params:
            if state and f"m::{k}" in state:
                self._m[k] = state[f"m::{k}"].astype(DT)
                self._v[k] = state[f"v::{k}"].astype(DT)
