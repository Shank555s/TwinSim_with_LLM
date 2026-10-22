"""Model acceptance (section 17). A candidate is promoted only if ALL configured criteria hold:

  1. benchmark fidelity  >  previous active fidelity + min_improvement  (strictly better)
  2. validation loss     <= previous validation loss * (1 + max_val_loss_increase)
  3. regression count    <= max_regressions  (benchmark items whose fidelity fell by more than
                                              regression_threshold points; -1 disables the check)
  4. paired significance gain >= min_paired_z standard errors over the identical benchmark items
                         (skipped when no paired statistics exist, or min_paired_z = 0)
  5. failure rate        <= previous failure rate + max_failure_rate_increase  (share of benchmark answers
                                              with any failure mode; skipped when either side is unknown)

Otherwise the candidate is kept on disk with status `rejected` and the active model is unchanged.
"""
from __future__ import annotations

from llm.config import LearningConfig


def decide(prev: dict, new: dict, cfg: LearningConfig) -> dict:
    reasons, passed = [], True
    prev_score, new_score = prev.get("fidelity"), new.get("fidelity")
    improvement = None if prev_score is None or new_score is None else new_score - prev_score
    if improvement is None:
        passed = False
        reasons.append("missing benchmark score")
    elif improvement <= cfg.min_improvement:
        passed = False
        pct = 100 * improvement / prev_score if prev_score else 0
        reasons.append(f"benchmark fidelity changed by {improvement:+.2f} points ({pct:+.1f}%), "
                       f"needs > {cfg.min_improvement:+.2f}")
    else:
        reasons.append(f"benchmark fidelity improved by {improvement:+.2f} points")
    paired = new.get("paired")
    if cfg.min_paired_z > 0 and paired and improvement is not None and improvement > cfg.min_improvement:
        if paired["z"] < cfg.min_paired_z:
            passed = False
            reasons.append(f"gain is within benchmark noise: paired z = {paired['z']:.2f}, needs >= {cfg.min_paired_z:.2f}")
        else:
            reasons.append(f"gain is significant on paired items: z = {paired['z']:.2f}")
    pv, nv = prev.get("val_loss"), new.get("val_loss")
    if pv is not None and nv is not None:
        if nv > pv * (1 + cfg.max_val_loss_increase):
            passed = False
            reasons.append(f"validation loss rose from {pv:.4f} to {nv:.4f} (limit +{cfg.max_val_loss_increase:.0%})")
        else:
            reasons.append(f"validation loss {pv:.4f} -> {nv:.4f} (within limit)")
    reg = new.get("regression_count")
    if cfg.max_regressions >= 0 and reg is not None and reg > cfg.max_regressions:
        passed = False
        reasons.append(f"{reg} benchmark items regressed (limit {cfg.max_regressions})")
    elif reg is not None:
        reasons.append(f"{reg} benchmark items regressed by > {cfg.regression_threshold:.0f} points")
    pf, nf = prev.get("failure_rate"), new.get("failure_rate")
    if pf is not None and nf is not None:
        if nf > pf + cfg.max_failure_rate_increase:
            passed = False
            reasons.append(f"failure rate rose from {pf:.1%} to {nf:.1%} (limit +{cfg.max_failure_rate_increase:.0%})")
        else:
            reasons.append(f"failure rate {pf:.1%} -> {nf:.1%} (within limit)")
    return {
        "accepted": passed,
        "previous_failure_rate": pf,
        "failure_rate": nf,
        "previous_score": prev_score,
        "new_score": new_score,
        "improvement": improvement,
        "previous_val_loss": pv,
        "validation_loss": nv,
        "benchmark_score": new_score,
        "regression_count": reg,
        "reasons": reasons,
        "criteria": {"min_improvement": cfg.min_improvement, "max_val_loss_increase": cfg.max_val_loss_increase,
                     "max_regressions": cfg.max_regressions, "regression_threshold": cfg.regression_threshold,
                     "max_failure_rate_increase": cfg.max_failure_rate_increase, "min_paired_z": cfg.min_paired_z},
    }
