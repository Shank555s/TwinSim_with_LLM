"""Read-side research metrics: dashboard, learning curve, failure trends, Twin DNA, drift,
model comparison. Everything is computed from stored experiments, no placeholder numbers."""
from __future__ import annotations

from collections import defaultdict

from database.db import Database
from database.repo import active_model, baseline_model, get_persona, latest_metrics
from evaluation.failure_modes import FAILURE_LABELS
from evaluation.fidelity import AXES

TRAIT_AXES = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "political"]


def _bench_exp(db: Database, model: str) -> dict | None:
    return db.one("SELECT * FROM experiments WHERE kind='benchmark' AND model_version=? AND status='completed' "
                  "ORDER BY created_at DESC LIMIT 1", (model,))


def _paired(db: Database, ref: str, new: str):
    from learning.benchmark import reference_item_means
    from learning.trainer import paired_difference
    return paired_difference(reference_item_means(db, ref), reference_item_means(db, new))


def paired_vs_baseline(db: Database, model: str):
    b = baseline_model(db)
    return _paired(db, b["name"], model) if b and b["name"] != model else None


def learning_curve(db: Database) -> list[dict]:
    rows = []
    for m in db.query("SELECT * FROM model_versions ORDER BY version"):
        met = latest_metrics(db, m["name"]) or {}
        run = db.one("SELECT * FROM training_runs WHERE output_model=? ORDER BY id DESC LIMIT 1", (m["name"],)) or {}
        n_examples = db.scalar("SELECT COUNT(*) FROM training_examples WHERE created_at <= ? AND example_type!='retired'",
                               (m["created_at"],))
        rows.append({
            "model": m["name"], "version": m["version"], "status": m["status"], "lineage": m["lineage"],
            "kind": "bootstrap" if m["lineage"] == "baseline" else (db.scalar(
                "SELECT kind FROM training_runs WHERE id=?", (m["training_run_id"],)) or "?"),
            "parent": m["parent"], "created_at": m["created_at"],
            "fidelity": met.get("fidelity"), "stability": met.get("stability"), "consistency": met.get("consistency"),
            "failure_rate": met.get("failure_rate"), "val_loss": met.get("val_loss") or run.get("final_val_loss"),
            "train_loss": met.get("train_loss") or run.get("final_train_loss"), "avg_length": met.get("avg_length"),
            "regression_count": met.get("regression_count"), "training_examples": n_examples,
            "failure_rates": (met.get("details") or {}).get("failure_rates", {}),
            "axes": (met.get("details") or {}).get("axes", {}),
            "decision": m.get("decision"),
            "paired_vs_parent": _paired(db, m["parent"], m["name"]) if m["parent"] else None,
            "parameters": m["parameters"],
        })
    return rows


def failure_analysis(db: Database) -> dict:
    curve = learning_curve(db)
    codes = sorted({c for r in curve for c in (r["failure_rates"] or {})} | set(FAILURE_LABELS))
    trend = {c: [{"model": r["model"], "rate": (r["failure_rates"] or {}).get(c, 0.0) if r["fidelity"] is not None else None,
                  "status": r["status"]} for r in curve] for c in codes}
    am = active_model(db)
    current = {}
    if am:
        met = latest_metrics(db, am["name"])
        current = (met.get("details") or {}).get("failure_rates", {}) if met else {}
    counts, n = defaultdict(int), 0
    for r in db.query("SELECT e.failures FROM evaluations e JOIN trials t ON t.id=e.trial_id WHERE t.purpose!='benchmark'"):
        n += 1
        for c in {f["code"] for f in (r["failures"] or [])}:
            counts[c] += 1
    examples = {}
    for c in codes:
        r = db.one("SELECT t.id, t.response, t.model_version, t.persona_snapshot, e.failures FROM evaluations e "
                   "JOIN trials t ON t.id=e.trial_id WHERE e.failures LIKE ? ORDER BY e.id DESC LIMIT 1",
                   (f'%"code": "{c}"%',))
        if r:
            detail = next((f["detail"] for f in r["failures"] if f["code"] == c), "")
            examples[c] = {"trial_id": r["id"], "model": r["model_version"], "response": r["response"],
                           "persona": r["persona_snapshot"].get("label"), "detail": detail}
    return {"labels": FAILURE_LABELS, "active_model": am["name"] if am else None, "current": current,
            "trend": trend, "all_trials": {"n": n, "rates": {c: counts[c] / n for c in counts} if n else {}},
            "examples": examples}


def twin_dna(db: Database, model: str, persona_id: str) -> dict:
    """Intended persona vs generated twin behaviour (mean over every evaluated trial of that model x persona)."""
    rows = db.query("SELECT e.* FROM evaluations e JOIN trials t ON t.id=e.trial_id "
                    "WHERE t.model_version=? AND t.persona_id=?", (model, persona_id))
    persona = get_persona(db, persona_id)
    if persona is None:
        raise KeyError(persona_id)
    intended = {a: persona.get(a, 50) / 10.0 for a in TRAIT_AXES[:-1]}
    intended["political"] = persona.get("political", 50) / 10.0
    if not rows:
        return {"model": model, "persona": persona, "n": 0, "intended": intended, "observed": None,
                "drift": None, "axes": None}
    valid = [r for r in rows if r["overall"] > 0]
    obs = {a: sum(r["observed"][a] for r in valid) / len(valid) for a in TRAIT_AXES} if valid else None
    axes = {a: sum(r["axes"][a] for r in rows) / len(rows) for a in AXES}
    style_obs = defaultdict(float)
    for r in valid:
        for k, v in ((r["style_detail"] or {}).get("observed") or {}).items():
            style_obs[k] += v / len(valid)
    style_exp = (valid[0]["style_detail"] or {}).get("expected") if valid else None
    return {"model": model, "persona": persona, "n": len(rows), "valid": len(valid), "intended": intended,
            "observed": obs, "drift": {a: round(obs[a] - intended[a], 3) for a in TRAIT_AXES} if obs else None,
            "axes": axes, "style": {"expected": style_exp, "observed": dict(style_obs)},
            "overall": sum(r["overall"] for r in rows) / len(rows)}


def persona_drift(db: Database, model: str) -> dict:
    exp = _bench_exp(db, model)
    return (exp["summary"] or {}).get("drift", {}) if exp else {}


def compare_models(db: Database, a: str, b: str, limit: int = 24) -> dict:
    curve = {r["model"]: r for r in learning_curve(db)}
    ea, eb = _bench_exp(db, a), _bench_exp(db, b)
    pairs = []
    if ea and eb:
        ta = {(r["persona_id"], r["scenario_id"], r["repetition"]): r for r in db.query(
            "SELECT t.*, e.overall, e.failures FROM trials t JOIN evaluations e ON e.trial_id=t.id WHERE t.experiment_id=?",
            (ea["id"],))}
        tb = db.query("SELECT t.*, e.overall, e.failures FROM trials t JOIN evaluations e ON e.trial_id=t.id "
                      "WHERE t.experiment_id=? ORDER BY t.persona_id, t.scenario_id, t.repetition", (eb["id"],))
        for r in tb:
            k = (r["persona_id"], r["scenario_id"], r["repetition"])
            if k in ta and r["repetition"] == 0:
                x = ta[k]
                pairs.append({"persona": r["persona_snapshot"].get("label"), "scenario": r["scenario_snapshot"]["text"],
                              "domain": r["scenario_snapshot"]["domain"], "seed": r["seed"],
                              "a": {"response": x["response"], "fidelity": x["overall"],
                                    "failures": [f["code"] for f in x["failures"]]},
                              "b": {"response": r["response"], "fidelity": r["overall"],
                                    "failures": [f["code"] for f in r["failures"]]}})
    pairs.sort(key=lambda p: -(p["b"]["fidelity"] - p["a"]["fidelity"]))
    return {"a": curve.get(a), "b": curve.get(b), "pairs": pairs[:limit], "n_pairs": len(pairs),
            "paired": _paired(db, a, b)}


def dashboard(db: Database) -> dict:
    am, bm = active_model(db), baseline_model(db)
    am_met = latest_metrics(db, am["name"]) if am else None
    bm_met = latest_metrics(db, bm["name"]) if bm else None
    counts = {r["purpose"]: r["n"] for r in db.query("SELECT purpose, COUNT(*) AS n FROM trials GROUP BY purpose")}
    recent = db.query("SELECT t.id, t.created_at, t.model_version, t.purpose, t.persona_snapshot, t.scenario_snapshot, "
                      "e.overall, e.failures FROM trials t JOIN evaluations e ON e.trial_id=t.id "
                      "WHERE t.purpose IN ('user','study','demo') ORDER BY t.created_at DESC LIMIT 8")
    return {
        "active_model": am, "baseline_model": bm["name"] if bm else None,
        "active_metrics": am_met, "baseline_metrics": bm_met,
        "improvement": (am_met["fidelity"] - bm_met["fidelity"]) if am_met and bm_met else None,
        "paired_vs_baseline": paired_vs_baseline(db, am["name"]) if am else None,
        "trial_counts": counts, "experiments": sum(counts.values()),
        "training_examples": db.scalar("SELECT COUNT(*) FROM training_examples WHERE example_type!='retired'") or 0,
        "training_examples_by_source": {r["source"]: r["n"] for r in db.query(
            "SELECT source, COUNT(*) AS n FROM training_examples WHERE example_type!='retired' GROUP BY source")},
        "models": db.scalar("SELECT COUNT(*) FROM model_versions") or 0,
        "curve": learning_curve(db),
        "drift": persona_drift(db, am["name"]) if am else {},
        "recent": [{"id": r["id"], "created_at": r["created_at"], "model": r["model_version"], "purpose": r["purpose"],
                    "persona": r["persona_snapshot"].get("label"), "domain": r["scenario_snapshot"]["domain"],
                    "scenario": r["scenario_snapshot"]["text"], "fidelity": r["overall"],
                    "failures": [f["code"] for f in r["failures"]]} for r in recent],
    }
