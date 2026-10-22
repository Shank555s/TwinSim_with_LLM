"""Fixed benchmark (sections 18-19): benchmark personas x benchmark scenarios x repetitions,
with per-item seeds that depend only on (persona, scenario, repetition), so every model
version is tested on exactly the same prompts with the same sampling randomness.

Default: 6 original research personas x 8 scenarios (the 4 original study prompts + 4 more)
x 2 repetitions = 96 trials. Benchmark items are never used as training data.
"""
from __future__ import annotations

import time
from collections import defaultdict

from database.db import Database, new_id
from database.repo import get_model, get_persona
from experiments.runner import run_trial, stable_seed, summarize_trials
from llm.config import GenerationConfig

BENCHMARK_GENERATION = GenerationConfig(max_new_tokens=96, min_new_tokens=4, temperature=0.7, top_k=40, top_p=0.95,
                                        repetition_penalty=1.15, deterministic=False, seed=0)


def benchmark_items(db: Database) -> tuple[list[dict], list[dict]]:
    personas = [get_persona(db, r["id"]) for r in
                db.query("SELECT id FROM personas WHERE is_benchmark=1 AND deleted=0 ORDER BY id")]
    scenarios = db.query("SELECT * FROM scenarios WHERE split='benchmark' AND deleted=0 ORDER BY domain, id")
    return personas, scenarios


def run_benchmark(db: Database, ctx, model_name: str, repetitions: int = 2, reference_model: str | None = None,
                  regression_threshold: float = 10.0, progress_base: float = 0.0, progress_span: float = 1.0) -> dict:
    model = get_model(db, model_name)
    if not model:
        raise KeyError(model_name)
    personas, scenarios = benchmark_items(db)
    if not personas or not scenarios:
        raise ValueError("benchmark is empty")
    exp_id = new_id("bench")
    total = len(personas) * len(scenarios) * repetitions
    db.insert("experiments", {"id": exp_id, "kind": "benchmark", "name": f"Benchmark {model_name}",
                              "model_version": model_name,
                              "config": {"personas": [p["id"] for p in personas], "scenarios": [s["id"] for s in scenarios],
                                         "repetitions": repetitions, "generation": BENCHMARK_GENERATION.to_dict()},
                              "status": "running", "created_at": time.time(), "job_id": ctx.id if ctx else None})
    rows, done = [], 0
    for p in personas:
        for s in scenarios:
            for r in range(repetitions):
                if ctx:
                    ctx.check()
                g = GenerationConfig(**{**BENCHMARK_GENERATION.to_dict(), "seed": stable_seed("bench", p["id"], s["id"], r)})
                out = run_trial(db, persona=p, scenario=s, model=model, gen=g, purpose="benchmark",
                                experiment_id=exp_id, repetition=r, store_feedback=False)
                rows.append(out)
                done += 1
                if ctx:
                    ctx.stage(f"benchmark {model_name}", progress_base + progress_span * done / total,
                              f"{done}/{total} benchmark trials",
                              {"model": model_name, "done": done, "total": total,
                               "fidelity": out["evaluation"]["overall"], "persona": p["label"],
                               "domain": s["domain"]}, force=False)
    summary = summarize_trials(rows)
    items = defaultdict(list)
    for o in rows:
        items[f"{o['trial']['persona_id']}|{o['trial']['scenario_id']}"].append(o["evaluation"]["overall"])
    item_means = {k: sum(v) / len(v) for k, v in items.items()}
    summary["items"] = {k: round(v, 3) for k, v in item_means.items()}
    regressions = None
    if reference_model:
        ref = reference_item_means(db, reference_model)
        if ref:
            regressions = sum(1 for k, v in item_means.items() if k in ref and ref[k] - v > regression_threshold)
    summary["regression_count"] = regressions
    summary["reference_model"] = reference_model
    summary["drift"] = drift_by_persona(rows)
    run = db.one("SELECT final_val_loss, final_train_loss FROM training_runs WHERE output_model=? ORDER BY id DESC LIMIT 1",
                 (model_name,))
    db.update("experiments", "id", exp_id, {"status": "completed", "finished_at": time.time(), "summary": summary})
    db.insert("model_metrics", {
        "model_name": model_name, "experiment_id": exp_id, "fidelity": summary["fidelity"],
        "stability": summary["stability"], "consistency": summary["consistency"],
        "failure_rate": summary["failure_rate"], "val_loss": run["final_val_loss"] if run else None,
        "train_loss": run["final_train_loss"] if run else None, "avg_length": summary["avg_length"],
        "regression_count": regressions,
        "details": {k: summary[k] for k in ("axes", "failure_rates", "by_domain", "drift", "n")},
        "created_at": time.time()})
    db.log_event("benchmark_completed", f"Benchmark {model_name}: fidelity {summary['fidelity']:.2f}, "
                 f"stability {summary['stability'] or 0:.1f}, failure rate {summary['failure_rate']:.0%}",
                 "model", model_name, {"experiment_id": exp_id, "fidelity": summary["fidelity"]})
    summary["experiment_id"] = exp_id
    summary["val_loss"] = run["final_val_loss"] if run else None
    return summary


def reference_item_means(db: Database, model_name: str) -> dict[str, float]:
    exp = db.one("SELECT summary FROM experiments WHERE kind='benchmark' AND model_version=? AND status='completed' "
                 "ORDER BY created_at DESC LIMIT 1", (model_name,))
    return (exp["summary"] or {}).get("items", {}) if exp else {}


def drift_by_persona(rows: list[dict]) -> dict:
    """Mean intended vs observed per persona and axis (Twin DNA / persona drift)."""
    acc: dict[str, dict] = {}
    for o in rows:
        pid = o["trial"]["persona_id"]
        e = o["evaluation"]
        a = acc.setdefault(pid, {"label": o["trial"]["persona_snapshot"].get("label"), "n": 0,
                                 "intended": e["intended"], "observed": defaultdict(float),
                                 "axes": defaultdict(float)})
        a["n"] += 1
        for k, v in e["observed"].items():
            a["observed"][k] += v
        for k, v in e["axes"].items():
            a["axes"][k] += v
    out = {}
    for pid, a in acc.items():
        obs = {k: round(v / a["n"], 3) for k, v in a["observed"].items()}
        out[pid] = {"label": a["label"], "intended": a["intended"], "observed": obs,
                    "drift": {k: round(obs[k] - a["intended"][k], 3) for k in obs},
                    "axes": {k: round(v / a["n"], 3) for k, v in a["axes"].items()}}
    return out
