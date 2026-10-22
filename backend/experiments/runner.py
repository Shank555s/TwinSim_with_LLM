"""Experiment engine: single trials, multi-trial studies (N personas x N scenarios x N reps),
and replay of a recorded trial. Every trial is stored with full reproducibility metadata."""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict

from database.db import Database, new_id
from database.repo import active_model, get_model, get_persona, get_scenario
from evaluation.fidelity import EVALUATOR_VERSION, evaluate
from evaluation.stability import stability_metrics
from learning.feedback import record_evaluator_feedback
from learning.signals import record_signal
from llm.config import EvaluatorConfig, GenerationConfig, LearningConfig
from twin.twin_engine import ModelNotTrained, engine


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def learning_config(db: Database) -> LearningConfig:
    return LearningConfig.from_dict(db.get_setting("learning_config"))


def evaluator_config(db: Database) -> EvaluatorConfig:
    w = db.get_setting("evaluator_weights")
    return EvaluatorConfig(weights=w) if w else EvaluatorConfig()


def resolve_model(db: Database, name: str | None) -> dict:
    if name in (None, "", "active"):
        m = active_model(db)
    elif name == "baseline":
        m = db.one("SELECT * FROM model_versions WHERE lineage='baseline' ORDER BY version LIMIT 1")
    else:
        m = get_model(db, name)
    if not m:
        raise ModelNotTrained("No trained TwinSim model is available yet. Start the bootstrap training "
                              "(Learning page or `python scripts/train.py bootstrap`).")
    return m


def run_trial(db: Database, *, persona: dict, scenario: dict, model: dict, gen: GenerationConfig,
              purpose: str = "user", experiment_id: str | None = None, repetition: int = 0,
              store_feedback: bool = True, make_example: bool = True, on_stage=None, on_text=None) -> dict:
    """`on_stage(name, data)` reports real pipeline stages: generating -> evaluating -> feedback."""
    stage = on_stage or (lambda *a, **k: None)
    stage("generating", {"model": model["name"]})
    res, info = engine.generate(model, persona, scenario, gen, on_text=on_text)
    stage("evaluating", {"tokens": res.tokens_generated, "latency_ms": round(res.latency_ms, 1)})
    report = evaluate(persona, scenario["text"], res.text, evaluator_config(db)).to_dict()
    tid = new_id("trial")
    frozen = model.get("lineage") == "baseline"
    trial = {
        "id": tid, "experiment_id": experiment_id, "persona_id": persona.get("id"),
        "persona_snapshot": persona, "scenario_id": scenario.get("id"),
        "scenario_snapshot": {k: scenario.get(k) for k in ("id", "domain", "issue", "text", "split")},
        "model_version": model["name"], "seed": gen.seed, "gen_params": gen.to_dict(), "response": res.text,
        "tokens_generated": res.tokens_generated, "latency_ms": round(res.latency_ms, 1),
        "repetition": repetition, "purpose": purpose, "created_at": time.time(),
        "training_status": "excluded_benchmark" if purpose == "benchmark" else ("pending" if store_feedback else "not_recorded"),
        "tokenizer_version": info["tokenizer_version"], "backend": info["backend"],
    }
    db.insert("trials", trial)
    db.insert("evaluations", {
        "trial_id": tid, "evaluator": EVALUATOR_VERSION, "evaluator_kind": report["evaluator_kind"],
        "overall": report["overall"], "axes": report["axes"], "observed": report["observed"],
        "intended": report["intended"], "drift": report["drift"], "failures": report["failures"],
        "explanation": report["explanation"], "style_detail": report["style_detail"], "created_at": time.time()})
    fb = {"training_status": trial["training_status"]}
    stage("feedback", {"overall": report["overall"]})
    if store_feedback and purpose != "benchmark":
        cfg = learning_config(db)
        fb = record_evaluator_feedback(db, trial, report, cfg, make_example=make_example)
        if purpose in ("user", "study", "demo"):  # every real experiment leaves a compact learning signal
            record_signal(db, trial, report, cfg)
    trial["training_status"] = fb["training_status"]
    return {"trial": trial, "evaluation": report, "generation": {
        "tokens_generated": res.tokens_generated, "latency_ms": round(res.latency_ms, 1),
        "stop_reason": res.stop_reason, "model_version": model["name"], "baseline_frozen": frozen, **info}}


def run_study(db: Database, ctx, *, persona_ids: list[str], scenario_ids: list[str], repetitions: int,
              model_name: str | None, gen: GenerationConfig, name: str = "Study", purpose: str = "study") -> dict:
    model = resolve_model(db, model_name)
    personas = [get_persona(db, p) for p in persona_ids]
    scenarios = [get_scenario(db, s) for s in scenario_ids]
    if any(p is None for p in personas) or any(s is None for s in scenarios):
        raise ValueError("unknown persona or scenario id in study configuration")
    exp_id = new_id("exp")
    total = len(personas) * len(scenarios) * repetitions
    db.insert("experiments", {"id": exp_id, "kind": "study", "name": name, "model_version": model["name"],
                              "config": {"persona_ids": persona_ids, "scenario_ids": scenario_ids,
                                         "repetitions": repetitions, "generation": gen.to_dict()},
                              "status": "running", "created_at": time.time(), "job_id": ctx.id if ctx else None})
    db.log_event("experiment_started", f"{name}: {len(personas)} personas x {len(scenarios)} scenarios x "
                 f"{repetitions} reps = {total} trials on {model['name']}", "experiment", exp_id)
    done = 0
    rows = []
    for p in personas:
        for s in scenarios:
            for r in range(repetitions):
                if ctx:
                    ctx.check()
                    ctx.stage("generate", done / total, f"{p['label']} x {s['domain']} (rep {r + 1})",
                              {"persona": p["label"], "scenario": s["text"][:80], "done": done, "total": total},
                              force=False)
                g = GenerationConfig(**{**gen.to_dict(), "seed": stable_seed(gen.seed, p["id"], s["id"], r)})
                out = run_trial(db, persona=p, scenario=s, model=model, gen=g, purpose=purpose,
                                experiment_id=exp_id, repetition=r)
                rows.append(out)
                done += 1
                if ctx:
                    ctx.stage("evaluate", done / total, f"fidelity {out['evaluation']['overall']:.1f}",
                              {"trial_id": out["trial"]["id"], "fidelity": out["evaluation"]["overall"],
                               "failures": out["evaluation"]["failure_codes"], "done": done, "total": total},
                              force=False)
    summary = summarize_trials(rows)
    db.update("experiments", "id", exp_id, {"status": "completed", "finished_at": time.time(), "summary": summary})
    db.log_event("experiment_completed", f"{name} finished: fidelity {summary['fidelity']:.1f}, "
                 f"failure rate {summary['failure_rate']:.0%}", "experiment", exp_id, summary)
    return {"experiment_id": exp_id, "summary": summary}


def summarize_trials(rows: list[dict]) -> dict:
    """Aggregate metrics over trials (each row = run_trial output or {trial, evaluation})."""
    if not rows:
        return {"n": 0, "fidelity": None}
    ev = [r["evaluation"] for r in rows]
    n = len(ev)
    axes = {a: round(sum(e["axes"][a] for e in ev) / n, 3) for a in ev[0]["axes"]}
    fail_counts: dict[str, int] = defaultdict(int)
    for e in ev:
        for c in set(e["failure_codes"]):
            fail_counts[c] += 1
    groups = defaultdict(list)
    for r in rows:
        t = r["trial"]
        groups[(t["persona_id"], t["scenario_id"])].append(r)
    stabs, cons = [], []
    for g in groups.values():
        if len(g) >= 2:
            sm = stability_metrics([x["trial"]["response"] for x in g],
                                   ["inconsistency" in x["evaluation"]["failure_codes"] for x in g])
            stabs.append(sm["stability"])
            cons.append(sm["consistency"])
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["trial"]["scenario_snapshot"]["domain"]].append(r["evaluation"]["overall"])
    return {
        "n": n,
        "fidelity": round(sum(e["overall"] for e in ev) / n, 3),
        "axes": axes,
        "failure_rate": round(sum(1 for e in ev if e["failure_codes"]) / n, 4),
        "failure_rates": {k: round(v / n, 4) for k, v in sorted(fail_counts.items())},
        "stability": round(sum(stabs) / len(stabs), 3) if stabs else None,
        "consistency": round(sum(cons) / len(cons), 3) if cons else None,
        "avg_length": round(sum(len(r["trial"]["response"].split()) for r in rows) / n, 2),
        "by_domain": {k: round(sum(v) / len(v), 3) for k, v in by_domain.items()},
    }


def replay_trial(db: Database, trial_id: str) -> dict:
    """Re-run a recorded trial with the exact stored configuration (section 41)."""
    t = db.one("SELECT * FROM trials WHERE id=?", (trial_id,))
    if not t:
        raise KeyError(trial_id)
    model = get_model(db, t["model_version"])
    if not model:
        raise KeyError(f"model {t['model_version']} no longer registered")
    gen = GenerationConfig.from_dict(t["gen_params"])
    out = run_trial(db, persona=t["persona_snapshot"], scenario=t["scenario_snapshot"], model=model, gen=gen,
                    purpose="replay", store_feedback=False)
    same = out["trial"]["response"] == t["response"]
    db.log_event("experiment_replayed", f"Replayed trial {trial_id}: {'identical' if same else 'DIFFERENT'} output",
                 "trial", out["trial"]["id"], {"original": trial_id, "identical": same})
    return {"original": {"id": t["id"], "response": t["response"]}, "replay": out, "identical": same}
