"""Learning signals: one compact record per experiment, derived from its fidelity report.

An experiment's raw payload (response, evaluator evidence, explanation text, ...) is large and is kept
only for the latest `max_recent_experiments`. What learning needs from it is small, and that is what
this module extracts:

    what was done well      -> successes / reinforce   (behaviour to keep)
    what went wrong         -> failures / reduce       (behaviour to suppress)
    which traits matter     -> traits                  (persona traits that define this twin)
    how valuable is it      -> training_weight         (priority for replay / corrective harvest)

`kind` decides how the signal is used by the next learning cycle:

    reinforce   clean, high fidelity   -> already a positive training example (learning/feedback.py)
    correct     failure mode(s) fired  -> corrective best-of-N harvest (learning/auto.py)
    monitor     clean but mediocre     -> weak axes are counted; corrected only if a weak axis is severe

Nothing here changes a fidelity score or invents a metric: every number is read from the evaluation.
"""
from __future__ import annotations

import time

from database.db import Database
from evaluation.failure_modes import FAILURE_LABELS
from llm.config import LearningConfig

from .feedback import quality_weight

# Heuristic seriousness of each failure mode in [0,1]. Combined as a noisy-OR into `failure_score`.
FAILURE_SEVERITY = {"degeneration": 1.0, "inconsistency": 0.6, "blue_shift": 0.6, "red_shift": 0.6,
                    "exaggeration": 0.5, "stereotype": 0.5}
WEAK_AXIS = 7.0  # axis fidelity (0-10) below this is "represented incorrectly"
STRONG_AXIS = 8.5  # at or above this is "represented well"
AXIS_NAMES = {"political": "political alignment", "style": "communication style"}
TRAIT_WORDS = {"openness": "openness", "conscientiousness": "conscientiousness", "extraversion": "extraversion",
               "agreeableness": "agreeableness", "neuroticism": "neuroticism"}


def failure_score(codes: list[str]) -> float:
    keep = 1.0
    for c in set(codes):
        keep *= 1.0 - FAILURE_SEVERITY.get(c, 0.5)
    return round(1.0 - keep, 4)


def important_traits(intended: dict[str, float], limit: int = 3) -> list[str]:
    """Persona traits that deviate most from neutral, e.g. 'high conscientiousness', 'low extraversion'."""
    out = []
    for k, v in sorted(intended.items(), key=lambda kv: -abs(kv[1] - 5)):
        if abs(v - 5) < 1.5:
            continue
        if k == "political":
            out.append("left-leaning politics" if v < 5 else "right-leaning politics")
        elif k in TRAIT_WORDS:
            out.append(f"{'high' if v > 5 else 'low'} {k}")
        if len(out) == limit:
            break
    return out


def expected_behavior(persona: dict, traits: list[str]) -> str:
    style = (persona.get("commStyle") or "").strip()
    parts = []
    if traits:
        parts.append("express " + ", ".join(traits))
    if style:
        parts.append(f"communicate in a {style} manner")
    return "; ".join(parts) or "respond consistently with the persona"


def _direction(axis: str, drift: dict[str, float]) -> str:
    d = drift.get(axis)
    if d is None or abs(d) < 0.5:
        return ""
    if axis == "political":
        return " (too left)" if d < 0 else " (too right)"
    return " (over-expressed)" if d > 0 else " (under-expressed)"


def build_signal(persona: dict, scenario: dict, response: str, report: dict, cfg: LearningConfig,
                 model_version: str | None = None, created_at: float | None = None) -> dict:
    """Pure function: (persona, scenario, response, fidelity report) -> signal row (without trial_id/state)."""
    overall = float(report["overall"])
    codes = [f["code"] for f in report["failures"]]
    axes = {k: float(v) for k, v in report["axes"].items()}
    drift = report.get("drift") or {}
    weak = sorted([a for a, v in axes.items() if v < WEAK_AXIS], key=lambda a: axes[a])
    strong = [a for a, v in axes.items() if v >= STRONG_AXIS]
    fscore = failure_score(codes)
    traits = important_traits(report.get("intended") or {})

    if codes:
        kind = "correct"
        weight = 0.5 + 0.5 * fscore
    elif overall >= cfg.min_fidelity_for_positive:
        kind = "reinforce"
        weight = quality_weight(overall, cfg.min_fidelity_for_positive)
    else:
        kind = "monitor"
        weight = 0.2 + 0.2 * min(1.0, len(weak) / 3)

    successes = [f"{AXIS_NAMES.get(a, a)} matched ({axes[a]:.1f}/10)" for a in strong]
    if not codes and report.get("valid", True):
        successes.append("no failure modes")
    reduce_ = [f"{FAILURE_LABELS.get(f['code'], f['code'])}: {f['detail']}" for f in report["failures"]]
    reduce_ += [f"{AXIS_NAMES.get(a, a)} mismatch{_direction(a, drift)} ({axes[a]:.1f}/10)" for a in weak[:3]]
    return {
        "kind": kind, "domain": scenario.get("domain"), "issue": scenario.get("issue"), "model_version": model_version,
        "created_at": created_at or time.time(), "fidelity": round(overall, 2), "failure_score": fscore,
        "training_weight": round(weight, 4), "traits": traits, "axes": axes, "weak_axes": weak,
        "failures": [{"code": f["code"], "severity": FAILURE_SEVERITY.get(f["code"], 0.5)} for f in report["failures"]],
        "successes": successes, "reinforce": [s for s in successes if s != "no failure modes"][:4], "reduce": reduce_[:5],
        "expected_behavior": expected_behavior(persona, traits),
        # payload: needed until the signal is consumed, dropped when compacted
        "persona": persona, "scenario": {k: scenario.get(k) for k in ("id", "domain", "issue", "text", "split")},
        "response": response,
    }


def record_signal(db: Database, trial: dict, report: dict, cfg: LearningConfig) -> int | None:
    """Store the learning signal of one finished experiment. Idempotent: the same experiment never
    produces a second signal (UNIQUE trial_id), so it can never be trained on twice through this path."""
    if db.one("SELECT id FROM learning_signals WHERE trial_id=?", (trial["id"],)):
        return None
    sig = build_signal(trial["persona_snapshot"], trial["scenario_snapshot"], trial["response"], report, cfg,
                       trial["model_version"], trial.get("created_at"))
    # benchmark personas / scenarios are held out of training, so they never create work for the learner
    from .feedback import is_protected
    protected = is_protected(db, trial.get("persona_id"), trial["scenario_snapshot"]["text"])
    state = "consumed" if protected else "pending"
    if protected:
        sig.update(kind="held_out", training_weight=0.0, persona=None, scenario=None, response=None, compacted=1)
    row = {"trial_id": trial["id"], "state": state, **sig}
    try:
        return db.insert("learning_signals", row)
    except Exception:  # noqa: BLE001 - lost a race with another writer on the UNIQUE key: already recorded
        return None


def pending_count(db: Database) -> int:
    return int(db.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='pending'") or 0)


def failure_frequency(db: Database, limit: int = 200) -> dict[str, float]:
    """Share of recent signals in which each failure mode occurred (used to prioritise corrective work)."""
    rows = db.query("SELECT failures FROM learning_signals WHERE kind!='held_out' ORDER BY id DESC LIMIT ?", (limit,))
    if not rows:
        return {}
    cnt: dict[str, int] = {}
    for r in rows:
        for c in {f["code"] for f in (r["failures"] or [])}:
            cnt[c] = cnt.get(c, 0) + 1
    return {c: n / len(rows) for c, n in cnt.items()}
