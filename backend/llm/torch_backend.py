"""PyTorch executor for TwinSimLanguageModel (primary backend)."""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from .backend import ModelRunner
from .config import TwinSimConfig
from .model import TwinSimLanguageModel


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchRunner(ModelRunner):
    backend = "torch"

    def __init__(self, cfg: TwinSimConfig, seed: int = 0, weight_decay: float = 0.01,
                 betas: tuple[float, float] = (0.9, 0.98), grad_clip: float = 1.0,
                 mixed_precision: bool = True, device: str | None = None):
        self.config = cfg
        self.device = device or pick_device()
        torch.manual_seed(seed)
        self.model = TwinSimLanguageModel(cfg).to(self.device)
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.betas = betas
        self.use_amp = bool(mixed_precision and self.device == "cuda")
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self._build_optimizer()

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self) -> None:
        decay = [p for n, p in self.model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in self.model.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=1e-3, betas=self.betas, eps=1e-8,
        )

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def num_parameters(self) -> int:
        return self.model.num_parameters()

    def _t(self, a: np.ndarray, dtype=torch.long) -> torch.Tensor:
        return torch.as_tensor(a, dtype=dtype, device=self.device)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def next_logits(self, ids: list[int]) -> np.ndarray:
        self.model.eval()
        ids = ids[-self.config.context_length:]
        x = self._t(np.asarray(ids)[None, :])
        with self._autocast():
            logits, _ = self.model(x)
        return logits[0, -1].float().cpu().numpy()

    # --------------------------------------------------------------- training
    def train_step(self, micro_batches, lr: float) -> dict:
        self.model.train()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        total_w = sum(float(np.sum(w)) for _, _, w in micro_batches) or 1.0
        loss_acc = 0.0
        for x, y, w in micro_batches:
            share = float(np.sum(w)) / total_w
            with self._autocast():
                _, loss = self.model(self._t(x), self._t(y), self._t(w, torch.float32))
            self.scaler.scale(loss * share).backward()
            loss_acc += float(loss.detach()) * share
        self.scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {"loss": loss_acc, "grad_norm": grad_norm}

    @torch.no_grad()
    def eval_loss(self, x, y, w) -> float:
        self.model.eval()
        with self._autocast():
            _, loss = self.model(self._t(x), self._t(y), self._t(w, torch.float32))
        return float(loss)

    # ------------------------------------------------------------- state I/O
    def get_weights(self) -> dict[str, np.ndarray]:
        return {k: v.detach().float().cpu().numpy() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        own = self.model.state_dict()
        missing = set(own) - set(weights)
        unexpected = set(weights) - set(own)
        if missing or unexpected:
            raise ValueError(f"weight mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
        for k, v in weights.items():
            if tuple(own[k].shape) != tuple(v.shape):
                raise ValueError(f"shape mismatch for {k}: {tuple(own[k].shape)} vs {tuple(v.shape)}")
        self.model.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})

    def optimizer_state(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        name_of = {id(p): n for n, p in self.model.named_parameters()}
        step = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                st = self.optimizer.state.get(p)
                if not st:
                    continue
                n = name_of[id(p)]
                out[f"m::{n}"] = st["exp_avg"].float().cpu().numpy()
                out[f"v::{n}"] = st["exp_avg_sq"].float().cpu().numpy()
                step = int(st["step"])
        out["__step__"] = np.asarray(step)
        return out

    def load_optimizer_state(self, state: dict[str, np.ndarray]) -> None:
        self._build_optimizer()
        if not state:
            return
        step = int(state.get("__step__", 0))
        params = dict(self.model.named_parameters())
        for n, p in params.items():
            if f"m::{n}" in state:
                self.optimizer.state[p] = {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.as_tensor(state[f"m::{n}"], device=self.device),
                    "exp_avg_sq": torch.as_tensor(state[f"v::{n}"], device=self.device),
                }
