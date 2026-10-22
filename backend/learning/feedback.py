"""Feedback processor (section 13): evaluation results -> labelled feedback -> training examples.

A fidelity number alone is not a training target. We turn it into one of these, each with an
explicit provenance label that is stored and shown in the UI:

  source                meaning
  --------------------  ---------------------------------------------------------------------
  evaluator_feedback    a response produced in an experiment that the rubric evaluator scored
                        above the acceptance threshold -> reused as a (weighted) positive example
  auto_best_of_n        automatically generated: N samples drawn for one prompt, the best one
                        (by evaluator score) becomes the target ("rejection-sampling fine-tuning")
  human_rating          a researcher rated the response >= 7/10 in Human Evaluation
  human_correction      a researcher wrote a corrected response (strongest signal, weight 1.0)
  synthetic             the template-generated development corpus (anchor / replay baseline)

Responses below the threshold, or showing any failure mode, are stored as feedback but never
trained on as positives. Benchmark personas/scenarios are *never* converted into training data.
"""
from __future__ import annotations

import time

from database.db import Database
from database.repo import benchmark_keys
from llm.config import LearningConfig

FEEDBACK_SOURCES = {
    "evaluator": "Evaluator feedback (rubric, automatic)",
    "human": "Human feedback (researcher)",
    "auto_generated": "Automatically generated (best-of-N ranking)",
}
EXAMPLE_SOURCES = ["synthetic", "evaluator_feedback", "auto_best_of_n", "human_rating", "human_correction"]


def quality_weight(overall: float, threshold: float) -> float:
    """Map fidelity in [threshold, 100] to a loss weight in [0.5, 1.0]."""
    return round(0.5 + 0.5 * max(0.0, overall - threshold) / max(1e-6, 100 - threshold), 4)


def is_protected(db: Database, persona_id: str | None, scenario_text: str) -> bool:
    bench_p, bench_s = benchmark_keys(db)
    return (persona_id in bench_p) or (scenario_text in bench_s)


def record_evaluator_feedback(db: Database, trial: dict, report: dict, cfg: LearningConfig,
                              source: str = "evaluator", make_example: bool = True) -> dict:
    """Store feedback for a trial and, if eligible, a weighted training example. Returns status info."""
    fid = db.insert("feedback", {"trial_id": trial["id"], "source": source, "overall": report["overall"],
                                 "axes": report["axes"], "failures": report["failure_codes"],
                                 "explanation": "\n".join(report["explanation"]), "created_at": time.time()})
    status = "not_eligible"
    if is_protected(db, trial["persona_id"], trial["scenario_snapshot"]["text"]):
        status = "excluded_benchmark"
    elif trial.get("purpose") == "benchmark":
        status = "excluded_benchmark"
    elif report["overall"] >= cfg.min_fidelity_for_positive and "degeneration" not in report["failure_codes"] \
            and not (cfg.require_clean_positive and report["failure_codes"]):
        if make_example:
            db.insert("training_examples", {
                "source": "evaluator_feedback", "feedback_id": fid, "trial_id": trial["id"],
                "persona": trial["persona_snapshot"], "scenario": trial["scenario_snapshot"],
                "response": trial["response"], "weight": quality_weight(report["overall"], cfg.min_fidelity_for_positive),
                "example_type": "response", "created_at": time.time()})
            status = "pending"
        else:
            status = "eligible"
    db.update("trials", "id", trial["id"], {"training_status": status})
    return {"feedback_id": fid, "training_status": status}


def record_human_evaluation(db: Database, trial: dict, score: int, blind: bool, rater: str = "researcher",
                            corrected_response: str | None = None, notes: str = "") -> dict:
    if not 1 <= int(score) <= 10:
        raise ValueError("score must be 1-10")
    hid = db.insert("human_evaluations", {"trial_id": trial["id"], "rater": rater, "score": int(score),
                                          "blind": int(bool(blind)), "corrected_response": corrected_response,
                                          "notes": notes, "created_at": time.time()})
    fid = db.insert("feedback", {"trial_id": trial["id"], "source": "human", "overall": float(score) * 10.0,
                                 "axes": {}, "failures": [], "explanation": notes or "", "created_at": time.time()})
    added = []
    if not is_protected(db, trial["persona_id"], trial["scenario_snapshot"]["text"]):
        if corrected_response and corrected_response.strip():
            db.insert("training_examples", {
                "source": "human_correction", "feedback_id": fid, "trial_id": trial["id"],
                "persona": trial["persona_snapshot"], "scenario": trial["scenario_snapshot"],
                "response": corrected_response.strip(), "weight": 1.0, "example_type": "response",
                "created_at": time.time()})
            added.append("human_correction")
        elif int(score) >= 7:
            db.insert("training_examples", {
                "source": "human_rating", "feedback_id": fid, "trial_id": trial["id"],
                "persona": trial["persona_snapshot"], "scenario": trial["scenario_snapshot"],
                "response": trial["response"], "weight": round(int(score) / 10.0, 2), "example_type": "response",
                "created_at": time.time()})
            added.append("human_rating")
    db.log_event("human_evaluation", f"Human rating {score}/10 for trial {trial['id']}" + (" (blind)" if blind else ""),
                 "trial", trial["id"], {"score": score, "examples_added": added})
    return {"human_evaluation_id": hid, "feedback_id": fid, "examples_added": added}


def new_feedback_count(db: Database) -> int:
    return int(db.scalar("SELECT COUNT(*) FROM training_examples WHERE used_count=0 AND example_type='response'") or 0)


def retire_examples_below_criteria(db: Database, cfg: LearningConfig) -> int:
    """Automatic (evaluator / best-of-N) examples that no longer satisfy the current positive-example
    criteria are retired from the replay buffer (kept in the table, never trained on again)."""
    rows = db.query("SELECT te.id, f.overall, f.failures FROM training_examples te JOIN feedback f ON f.id=te.feedback_id "
                    "WHERE te.example_type IN ('response','revision') AND te.source IN ('evaluator_feedback','auto_best_of_n') "
                    "AND f.source IN ('evaluator','auto_generated')")
    bad = [r["id"] for r in rows if r["overall"] < cfg.min_fidelity_for_positive
           or (cfg.require_clean_positive and r["failures"])]
    for i in range(0, len(bad), 500):
        chunk = bad[i:i + 500]
        db.execute(f"UPDATE training_examples SET example_type='retired' WHERE id IN ({','.join('?' for _ in chunk)})", chunk)
    if bad:
        db.log_event("examples_retired", f"{len(bad)} automatic examples retired: below fidelity "
                     f"{cfg.min_fidelity_for_positive} or showing a failure mode", "training_examples", None,
                     {"count": len(bad)})
    return len(bad)
