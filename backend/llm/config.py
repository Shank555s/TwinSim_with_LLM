"""TwinSim model + training configuration.

All sizes are configurable; the architecture code never hard-codes a size.
Parameter counts are always computed from the instantiated weights, never typed in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class TwinSimConfig:
    """Architecture hyper-parameters of the decoder-only Transformer."""

    vocab_size: int = 1024
    context_length: int = 384
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True  # LM head shares weights with the token embedding
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        for name in ("vocab_size", "context_length", "embedding_dim", "num_layers", "num_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TwinSimConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainingConfig:
    """Optimisation hyper-parameters shared by pre-training and feedback training."""

    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 1500
    learning_rate: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 8
    early_stopping_patience: int = 5  # in evaluations; 0 disables
    restore_best: bool = True  # keep the weights with the best validation loss (off for feedback updates)
    mixed_precision: bool = True  # only used by the PyTorch backend on CUDA
    seed: int = 1337
    log_interval: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Named profiles
# ---------------------------------------------------------------------------
MODEL_PROFILES: dict[str, dict] = {
    # Trains in under an hour on a laptop CPU. Used for demos, CI and the sandbox run.
    "dev": dict(vocab_size=1024, context_length=384, embedding_dim=128, num_layers=4, num_heads=4, dropout=0.0),
    # Even smaller: unit tests and smoke checks.
    "tiny": dict(vocab_size=320, context_length=64, embedding_dim=32, num_layers=2, num_heads=2, dropout=0.0),
    # A single consumer GPU (~8 GB): roughly 16M parameters.
    "research": dict(vocab_size=4096, context_length=512, embedding_dim=384, num_layers=8, num_heads=6, dropout=0.1),
    # Cloud GPU (16 GB+): roughly 42M parameters.
    "research-large": dict(vocab_size=8192, context_length=1024, embedding_dim=512, num_layers=12, num_heads=8, dropout=0.1),
}

TRAINING_PROFILES: dict[str, dict] = {
    "tiny": dict(batch_size=8, max_steps=60, learning_rate=5e-3, warmup_steps=5, eval_interval=20, eval_batches=2),
    "dev": dict(batch_size=16, max_steps=1500, learning_rate=3e-3, warmup_steps=100, eval_interval=100),
    "research": dict(batch_size=32, grad_accum_steps=4, max_steps=20000, learning_rate=6e-4, warmup_steps=1000,
                     eval_interval=500, eval_batches=20),
    "research-large": dict(batch_size=16, grad_accum_steps=8, max_steps=60000, learning_rate=3e-4,
                           warmup_steps=2000, eval_interval=1000, eval_batches=40),
}


def model_config(profile: str = "dev", **overrides) -> TwinSimConfig:
    if profile not in MODEL_PROFILES:
        raise KeyError(f"Unknown model profile '{profile}'. Choose from {sorted(MODEL_PROFILES)}")
    return TwinSimConfig(**{**MODEL_PROFILES[profile], **overrides})


def training_config(profile: str = "dev", **overrides) -> TrainingConfig:
    if profile not in TRAINING_PROFILES:
        raise KeyError(f"Unknown training profile '{profile}'. Choose from {sorted(TRAINING_PROFILES)}")
    return TrainingConfig(**{**TRAINING_PROFILES[profile], **overrides})


def count_parameters_analytic(cfg: TwinSimConfig) -> int:
    """Closed-form parameter count (used only to *recommend* a profile before a model exists).

    The number shown at runtime always comes from summing the real weight tensors.
    """
    d, v, t, L, m = cfg.embedding_dim, cfg.vocab_size, cfg.context_length, cfg.num_layers, cfg.mlp_dim
    per_block = (2 * d) + (d * 3 * d + 3 * d) + (d * d + d) + (2 * d) + (d * m + m) + (m * d + d)
    head = 0 if cfg.tie_embeddings else v * d
    return v * d + t * d + L * per_block + 2 * d + head


@dataclass
class GenerationConfig:
    """Sampling controls (section 11). `deterministic=True` means greedy decoding."""

    max_new_tokens: int = 96
    min_new_tokens: int = 4  # stop tokens are blocked until this many tokens exist (prevents empty answers)
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    deterministic: bool = False
    seed: int | None = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "GenerationConfig":
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known and v is not None})


@dataclass
class LearningConfig:
    """Feedback-learning behaviour (sections 14–17). Persisted in the DB and editable from the UI."""

    learning_mode: str = "batch"  # "batch" | "immediate" ("immediate" = learn after every single experiment)
    batch_trigger_trials: int = 10  # automatic learning starts once this many experiment signals are pending
    samples_per_prompt: int = 4  # best-of-N sampling used to harvest feedback
    feedback_prompts_per_iteration: int = 128
    sample_temperature: float = 1.0
    min_fidelity_for_positive: float = 80.0  # overall fidelity (0–100) needed to become a positive example
    require_clean_positive: bool = True  # positives must also be free of every failure mode
    new_feedback_ratio: float = 0.3  # fraction of each training batch drawn from *new* feedback
    replay_feedback_ratio: float = 0.3  # fraction from *older* feedback in the replay buffer
    # the remainder (here 0.4) comes from the baseline persona corpus (anchor data)
    finetune_steps: int = 300  # upper bound; see max_epochs_per_new_example
    finetune_lr: float = 5e-4
    max_epochs_per_new_example: float = 8.0  # caps steps so a few new examples are not memorised
    micro_update_steps: int = 30
    # acceptance criteria (section 17)
    # benchmark fidelity points (0-100) the candidate must gain. The paired standard error on the
    # default benchmark was ~0.6-1.4 points in the first build run, so 1.0 filters most pure noise.
    min_improvement: float = 1.0
    # relative increase allowed in validation loss on the (synthetic) persona corpus. Feedback training
    # deliberately moves the model away from the template corpus, so this is a guard against
    # catastrophic forgetting (e.g. loss +50 %), not a requirement to keep fitting the templates.
    max_val_loss_increase: float = 0.5
    # the gain must also be at least this many standard errors on the paired benchmark items (0 = off), so a
    # candidate that is only better by noise is not promoted
    min_paired_z: float = 1.0
    max_regressions: int = -1  # max benchmark items allowed to drop by > regression_threshold (-1 = ignore)
    regression_threshold: float = 10.0
    benchmark_repetitions: int = 2
    use_revision_examples: bool = False  # also train on critique->revision examples (longer sequences)
    harvest_user_personas: bool = True  # include user-created (non-benchmark) personas when harvesting
    # ---- automatic continual learning (learning/auto.py) and bounded storage (learning/retention.py)
    auto_learning: bool = True  # background learning after experiments; there is no manual "train" action
    auto_cooldown_s: float = 60.0  # minimum pause between two automatic cycles
    corrective_samples: int = 4  # best-of-N resamples drawn for each failed / weak experiment
    max_corrective_per_cycle: int = 12  # experiments given a corrective harvest in one cycle
    corrective_min_fidelity: float = 75.0  # a corrective answer must be failure-free and reach this fidelity
    max_signal_attempts: int = 3  # a signal whose cycle keeps failing is skipped after this many tries
    max_failure_rate_increase: float = 0.05  # candidate may not raise the benchmark failure rate by more (absolute)
    max_recent_experiments: int = 15  # MAX_RECENT_EXPERIMENTS: experiments whose full raw payload is kept
    max_replay_examples: int = 400  # MAX_REPLAY_EXAMPLES: bounded replay buffer (training_examples)
    max_memory_signals: int = 500  # compact per-experiment learning records kept (older ones are rolled up)
    max_active_checkpoints: int = 5  # MAX_ACTIVE_CHECKPOINTS: model versions that keep their weights on disk
    max_history_events: int = 3000  # history log rows kept

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LearningConfig":
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EvaluatorConfig:
    """Weights for the overall fidelity score (section 12)."""

    weights: dict = field(default_factory=lambda: {
        "openness": 1.0,
        "conscientiousness": 1.0,
        "extraversion": 1.0,
        "agreeableness": 1.0,
        "neuroticism": 1.0,
        "political": 2.0,
        "style": 1.0,
    })
