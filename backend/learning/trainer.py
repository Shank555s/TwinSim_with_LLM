"""Training worker: bootstrap (tokenizer + stage-1 pre-training + stage-2 persona specialisation
-> TwinSim-v0) and feedback-driven updates (-> TwinSim-v1, v2, ...), plus the one-click
learning experiment. All functions run inside a background job (`ctx`)."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from data.ingestion import load_extra_persona_examples, load_extra_pretraining_texts
from data.synthetic import DATASET_VERSION, build_persona_examples, build_pretraining_texts
from database.db import HOME, Database, new_id
from database.repo import (active_model, baseline_model, checkpoint_dir, get_model, get_persona, latest_metrics,
                           next_version)
from experiments.runner import learning_config, run_trial, stable_seed
from llm.backend import create_runner
from llm.checkpoint import load_checkpoint, save_checkpoint
from llm.config import GenerationConfig, LearningConfig, TrainingConfig, model_config, training_config
from llm.tokenizer import TwinSimTokenizer
from llm.training import ExampleSource, StreamSource, Trainer
from twin.persona_serializer import SerializationFormat, serialize_example
from twin.personas import sample_persona
from twin.twin_engine import engine

from .benchmark import reference_item_means, run_benchmark
from .feedback import quality_weight, retire_examples_below_criteria
from .promotion import decide
from .replay_buffer import ReplayBuffer, encode_rows, load_jsonl

CHECKPOINT_DIR = HOME / "checkpoints"
DATASET_DIR = HOME / "datasets"


def _dataset_file(ds: dict, key: str) -> Path:
    """Dataset file path, falling back to <TWINSIM_HOME>/datasets/<name>/ when runtime/ was moved."""
    p = Path(ds["composition"]["files"][key])
    return p if p.exists() else DATASET_DIR / ds["name"] / p.name


def model_name(v: int) -> str:
    return f"TwinSim-v{v}"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def register_dataset(db: Database, name: str, kind: str, path: Path | None, n: int, composition: dict,
                     description: str) -> str:
    version = int(db.scalar("SELECT COUNT(*) FROM datasets") or 0) + 1
    did = f"dataset-v{version}"
    db.insert("datasets", {"id": did, "name": name, "kind": kind, "version": version, "n_examples": n,
                           "sha": _sha_file(path) if path and path.exists() else
                           hashlib.sha256(json.dumps(composition, sort_keys=True, default=str).encode()).hexdigest()[:12],
                           "composition": composition, "path": str(path) if path else None,
                           "description": description, "created_at": time.time()})
    db.log_event("dataset_created", f"{did} ({kind}): {n} examples", "dataset", did, composition)
    return did


def _on_train_event(ctx, run_id: int, db: Database, log: list, label: str, p0: float, p1: float):
    def cb(ev: dict):
        log.append(ev)
        if ctx is None:
            return
        ctx.check()
        if ev["type"] == "train":
            frac = ev["step"] / max(1, ev["max_steps"])
            ctx.stage(label, p0 + (p1 - p0) * frac,
                      f"step {ev['step']}/{ev['max_steps']} loss {ev['train_loss']:.3f}",
                      {"run_id": run_id, **{k: ev[k] for k in ("step", "max_steps", "train_loss", "train_loss_ema",
                                                               "lr", "tokens_seen", "grad_norm")}}, force=False)
        elif ev["type"] == "eval":
            ctx.stage(label, None, f"validation loss {ev['val_loss']:.3f}", {"run_id": run_id, **ev}, force=True)
        if len(log) % 20 == 0:
            db.update("training_runs", "id", run_id, {"log": log[-500:]})
    return cb


def _start_run(db: Database, kind: str, base: str | None, output: str, dataset_id: str | None, config: dict,
               ctx, runner) -> int:
    return db.insert("training_runs", {"kind": kind, "status": "running", "base_model": base, "output_model": output,
                                       "dataset_id": dataset_id, "config": config, "started_at": time.time(),
                                       "log": [], "job_id": ctx.id if ctx else None, "device": runner.device,
                                       "backend": runner.backend})


def _finish_run(db: Database, run_id: int, res, log: list, status: str = "completed") -> None:
    db.update("training_runs", "id", run_id, {
        "status": status, "finished_at": time.time(), "steps": res.steps_completed, "tokens_seen": res.tokens_seen,
        "final_train_loss": res.final_train_loss, "final_val_loss": res.final_val_loss,
        "best_val_loss": res.best_val_loss, "log": log[-1500:]})


# =========================================================================== bootstrap
def bootstrap(db: Database, ctx, profile: str = "dev", pretrain_steps: int | None = None,
              persona_steps: int | None = None, n_pretrain_docs: int = 6000, n_persona_examples: int = 8000,
              n_val_examples: int = 400, seed: int = 1337, batch_size: int | None = None,
              model_overrides: dict | None = None) -> dict:
    """Create TwinSim-v0 from random initialisation."""
    if db.scalar("SELECT COUNT(*) FROM model_versions"):
        raise RuntimeError("a baseline model already exists; bootstrap only creates TwinSim-v0 "
                           "(delete the runtime/ folder to start a fresh research run)")
    t_cfg = training_config(profile if profile in ("tiny", "dev", "research", "research-large") else "dev")
    if batch_size:
        t_cfg.batch_size = batch_size
    stage = (lambda *a, **k: ctx.stage(*a, **k)) if ctx else (lambda *a, **k: None)

    # ---- 1. datasets
    stage("data", 0.01, "building the development corpus (synthetic, clearly labelled)")
    docs = build_pretraining_texts(n_pretrain_docs, seed)
    extra_docs = load_extra_pretraining_texts()
    train_rows = build_persona_examples(n_persona_examples, "train", seed + 1)
    train_rows += load_extra_persona_examples()
    val_rows = build_persona_examples(n_val_examples, "validation", seed + 2)
    ddir = DATASET_DIR / DATASET_VERSION
    _write_jsonl(ddir / "pretrain.jsonl", [{"text": d} for d in docs + extra_docs])
    _write_jsonl(ddir / "persona_train.jsonl", train_rows)
    _write_jsonl(ddir / "persona_val.jsonl", val_rows)
    dataset_id = register_dataset(db, DATASET_VERSION, "bootstrap", ddir / "persona_train.jsonl", len(train_rows), {
        "pretrain_docs": len(docs), "extra_pretrain_docs": len(extra_docs), "persona_train": len(train_rows),
        "persona_val": len(val_rows), "generator": "data/synthetic.py", "seed": seed,
        "files": {"pretrain": str(ddir / "pretrain.jsonl"), "persona_train": str(ddir / "persona_train.jsonl"),
                  "persona_val": str(ddir / "persona_val.jsonl")},
        "note": "SYNTHETIC development corpus generated from templates (not human data)"},
        "Synthetic persona corpus (dev) + optional ingested corpora")

    # ---- 2. tokenizer
    fmt = SerializationFormat()
    stage("tokenizer", 0.03, "training byte-level BPE tokenizer on the training corpus")
    mcfg = model_config(profile, **(model_overrides or {}))
    tok_texts = docs + extra_docs + ["".join(serialize_example(r["persona"], r["scenario"], r["response"], fmt))
                                     for r in train_rows[:3000]]
    tok = TwinSimTokenizer.train(tok_texts, vocab_size=mcfg.vocab_size)
    mcfg.vocab_size = tok.vocab_size
    stage("tokenizer", 0.05, f"tokenizer ready: {tok.vocab_size} tokens ({len(tok.merges)} merges), {tok.version}")

    # ---- 3. model (random init)
    runner = create_runner(mcfg, seed=seed, train_options={"weight_decay": t_cfg.weight_decay,
                                                           "betas": (t_cfg.beta1, t_cfg.beta2),
                                                           "grad_clip": t_cfg.grad_clip,
                                                           "mixed_precision": t_cfg.mixed_precision})
    desc = runner.describe()
    stage("model", 0.06, f"initialised {desc['parameters']:,} parameters (random) on {desc['backend']}/{desc['device']}",
          desc)

    # ---- 4. stage 1: general LM pre-training
    window = min(128, mcfg.context_length)
    stream = []
    for d in docs + extra_docs:
        stream.extend(tok.encode(d) + [tok.eos_id])
    stream = np.asarray(stream)
    n_val_tok = max(window * 8, len(stream) // 50)
    s1_cfg = TrainingConfig(**{**t_cfg.to_dict(), "max_steps": pretrain_steps or max(50, t_cfg.max_steps // 4),
                               "warmup_steps": min(t_cfg.warmup_steps, 50), "eval_interval": 100,
                               "early_stopping_patience": 0})
    run1 = _start_run(db, "pretrain", None, model_name(0), dataset_id, {"stage": 1, **s1_cfg.to_dict(),
                                                                          "window": window}, ctx, runner)
    log1: list = []
    tr = Trainer(runner, s1_cfg, StreamSource(stream[n_val_tok:], window), StreamSource(stream[:n_val_tok], window),
                 on_event=_on_train_event(ctx, run1, db, log1, "stage 1: language pre-training", 0.07, 0.30))
    res1 = tr.run()
    _finish_run(db, run1, res1, log1)

    # ---- 5. stage 2: persona specialisation
    train_enc, sk1 = encode_rows(train_rows, tok, fmt, mcfg.context_length)
    val_enc, sk2 = encode_rows(val_rows, tok, fmt, mcfg.context_length)
    if not train_enc:
        raise ValueError(f"insufficient training data: all {len(train_rows)} persona examples exceed the context "
                         f"length {mcfg.context_length}; use a larger context_length")
    runner.reset_optimizer()
    s2_cfg = TrainingConfig(**{**t_cfg.to_dict(), "max_steps": persona_steps or t_cfg.max_steps})
    run2 = _start_run(db, "persona", None, model_name(0), dataset_id,
                      {"stage": 2, **s2_cfg.to_dict(), "skipped_too_long": sk1 + sk2,
                       "serialization_format": fmt.to_dict()}, ctx, runner)
    log2: list = []
    tr = Trainer(runner, s2_cfg, ExampleSource(train_enc), ExampleSource(val_enc),
                 on_event=_on_train_event(ctx, run2, db, log2, "stage 2: persona specialisation", 0.30, 0.80))
    res2 = tr.run()
    _finish_run(db, run2, res2, log2)

    # ---- 6. checkpoint + registry
    name = model_name(0)
    path = CHECKPOINT_DIR / name
    save_checkpoint(path, runner, tok, meta={"name": name, "version": 0, "parent": None, "lineage": "baseline",
                                              "dataset_id": dataset_id},
                    training_state={"step": res2.steps_completed, "stage1_steps": res1.steps_completed,
                                    "scheduler": {"type": "warmup_cosine", **s2_cfg.to_dict()},
                                    "training_config": s2_cfg.to_dict(), "seed": seed,
                                    "serialization_format": fmt.to_dict()},
                    metrics={"stage1": {"final_train_loss": res1.final_train_loss, "final_val_loss": res1.final_val_loss},
                             "stage2": {"final_train_loss": res2.final_train_loss, "final_val_loss": res2.final_val_loss,
                                        "best_val_loss": res2.best_val_loss}},
                    dataset={"dataset_id": dataset_id, "version": DATASET_VERSION})
    register_model(db, name=name, version=0, parent=None, lineage="baseline", status="baseline", path=path,
                   dataset_id=dataset_id, runner=runner, tok=tok, training_cfg=s2_cfg.to_dict(), run_id=run2,
                   notes="Baseline Twin: trained from random initialisation, frozen (never updated by feedback).")
    db.set_setting("active_model", name)
    db.update("training_runs", "id", run1, {"output_model": name})

    # ---- 7. benchmark
    lc = learning_config(db)
    bench = run_benchmark(db, ctx, name, repetitions=lc.benchmark_repetitions, progress_base=0.8, progress_span=0.2)
    db.log_event("model_baseline", f"{name} created and benchmarked: fidelity {bench['fidelity']:.2f}", "model", name)
    return {"model": name, "parameters": desc["parameters"], "tokenizer_version": tok.version,
            "stage1": {"steps": res1.steps_completed, "val_loss": res1.final_val_loss},
            "stage2": {"steps": res2.steps_completed, "val_loss": res2.final_val_loss},
            "benchmark": {k: bench[k] for k in ("fidelity", "stability", "consistency", "failure_rate")}}


def register_model(db: Database, *, name, version, parent, lineage, status, path, dataset_id, runner, tok,
                   training_cfg, run_id, notes="", decision=None) -> None:
    meta = json.loads((Path(path) / "meta.json").read_text())
    db.insert("model_versions", {
        "name": name, "version": version, "parent": parent, "lineage": lineage, "status": status,
        "created_at": time.time(), "checkpoint_path": str(path), "dataset_id": dataset_id,
        "tokenizer_version": tok.version, "code_version": meta["code_version"], "training_config": training_cfg,
        "model_config": runner.config.to_dict(), "parameters": runner.num_parameters(), "backend": runner.backend,
        "notes": notes, "decision": decision, "training_run_id": run_id})
    size = sum(f.stat().st_size for f in Path(path).iterdir())
    db.insert("checkpoints", {"model_name": name, "path": str(path), "weights_sha": meta["weights_sha256"],
                              "size_bytes": size, "created_at": time.time()})
    db.log_event("checkpoint_saved", f"Checkpoint {name} saved ({size / 1e6:.1f} MB)", "model", name,
                 {"path": str(path), "weights_sha": meta["weights_sha256"]})


# =========================================================================== harvesting
def harvest_one(db: Database, model: dict, persona: dict, scenario: dict, cfg: LearningConfig, exp_id: str,
                seed_parts: tuple, previous: dict | None = None, min_fidelity: float | None = None,
                example_weight_boost: float = 0.0, source: str = "auto_best_of_n") -> tuple[dict, bool, list[dict]]:
    """Best-of-N sampling for ONE prompt. The evaluator ranks the samples; the best becomes a training example
    if it passes the gate (fidelity >= `min_fidelity`, no failure mode, not degenerate). `previous` is an
    earlier failing answer {"response", "overall", "axes", "failures"}: when given, a critique -> revision
    example (previous answer + its feedback -> best answer) is stored as well. Returns (best, accepted, samples)."""
    gate = cfg.min_fidelity_for_positive if min_fidelity is None else min_fidelity
    samples = []
    for k in range(cfg.samples_per_prompt):
        g = GenerationConfig(max_new_tokens=96, temperature=cfg.sample_temperature, top_k=50, top_p=0.95,
                             repetition_penalty=1.15, seed=stable_seed(*seed_parts, k))
        out = run_trial(db, persona=persona, scenario=scenario, model=model, gen=g, purpose="harvest",
                        experiment_id=exp_id, repetition=k, store_feedback=False)
        samples.append(out)
    samples.sort(key=lambda o: -o["evaluation"]["overall"])
    best = samples[0]
    for o in samples:
        ev = o["evaluation"]
        o["feedback_id"] = db.insert("feedback", {"trial_id": o["trial"]["id"], "source": "auto_generated",
                                                  "overall": ev["overall"], "axes": ev["axes"],
                                                  "failures": ev["failure_codes"],
                                                  "explanation": "\n".join(ev["explanation"]),
                                                  "created_at": time.time()})
    be = best["evaluation"]
    ok = be["overall"] >= gate and "degeneration" not in be["failure_codes"] and \
        not (cfg.require_clean_positive and be["failure_codes"])
    for o in samples:
        db.update("trials", "id", o["trial"]["id"],
                  {"training_status": "pending" if (o is best and ok) else "not_selected"})
    if ok:
        weight = min(1.0, quality_weight(be["overall"], cfg.min_fidelity_for_positive) + example_weight_boost)
        db.insert("training_examples", {
            "source": source, "feedback_id": best["feedback_id"], "trial_id": best["trial"]["id"],
            "persona": persona, "scenario": best["trial"]["scenario_snapshot"], "response": best["trial"]["response"],
            "weight": max(0.3, weight), "example_type": "response", "created_at": time.time()})
        worst = samples[-1]
        prev = previous or ({"response": worst["trial"]["response"], "overall": worst["evaluation"]["overall"],
                             "axes": worst["evaluation"]["axes"], "failures": worst["evaluation"]["failure_codes"]}
                            if worst["evaluation"]["overall"] < be["overall"] - 5 else None)
        if prev and cfg.use_revision_examples:  # stored only when they will actually be trained on
            sc = dict(best["trial"]["scenario_snapshot"])
            sc["_revision"] = {"previous": prev["response"],
                               "feedback": {"overall": prev["overall"], "axes": prev["axes"], "failures": prev["failures"]}}
            db.insert("training_examples", {
                "source": source, "feedback_id": worst["feedback_id"], "trial_id": worst["trial"]["id"],
                "persona": persona, "scenario": sc, "response": best["trial"]["response"], "weight": 1.0,
                "example_type": "revision", "created_at": time.time()})
    return best, ok, samples


def harvest(db: Database, ctx, model: dict, cfg: LearningConfig, iteration_seed: int, p0: float, p1: float) -> dict:
    """Best-of-N sampling on TRAINING prompts; the evaluator ranks samples; the best become examples."""
    rng = random.Random(iteration_seed)
    scen = db.query("SELECT * FROM scenarios WHERE split IN ('train','user') AND deleted=0")
    user_personas = [get_persona(db, r["id"]) for r in
                     db.query("SELECT id FROM personas WHERE is_benchmark=0 AND deleted=0")] if cfg.harvest_user_personas else []
    exp_id = new_id("harvest")
    db.insert("experiments", {"id": exp_id, "kind": "harvest", "name": f"Feedback harvest ({model['name']})",
                              "model_version": model["name"], "config": cfg.to_dict(), "status": "running",
                              "created_at": time.time(), "job_id": ctx.id if ctx else None})
    n = cfg.feedback_prompts_per_iteration
    added, scores, best_scores = 0, [], []
    for i in range(n):
        if ctx:
            ctx.check()
        if user_personas and rng.random() < 0.25:
            persona = rng.choice(user_personas)
        else:
            persona, _ = sample_persona(rng)
            persona["id"] = None
        s = rng.choice(scen)
        best, ok, samples = harvest_one(db, model, persona, s, cfg, exp_id, ("harvest", iteration_seed, i))
        scores.extend(o["evaluation"]["overall"] for o in samples)
        best_scores.append(best["evaluation"]["overall"])
        added += int(ok)
        if ctx:
            ctx.stage("feedback harvest", p0 + (p1 - p0) * (i + 1) / n,
                      f"prompt {i + 1}/{n}: best of {cfg.samples_per_prompt} = {best['evaluation']['overall']:.1f}"
                      + (" -> example" if ok else " (below threshold)"),
                      {"done": i + 1, "total": n, "best": best["evaluation"]["overall"], "accepted": ok}, force=False)
    summary = {"prompts": n, "samples": len(scores), "examples_added": added,
               "mean_sample_fidelity": round(float(np.mean(scores)), 3) if scores else None,
               "mean_best_fidelity": round(float(np.mean(best_scores)), 3) if best_scores else None}
    db.update("experiments", "id", exp_id, {"status": "completed", "finished_at": time.time(), "summary": summary})
    db.log_event("feedback_harvested", f"Harvested {added} examples from {len(scores)} samples "
                 f"(mean {summary['mean_sample_fidelity']}, best-of-N mean {summary['mean_best_fidelity']})",
                 "experiment", exp_id, summary)
    return summary


def paired_difference(ref: dict, new: dict) -> dict | None:
    """Mean per-item fidelity change on identical benchmark items (same persona, scenario, seeds)
    with its standard error, tells a real improvement from benchmark noise."""
    keys = [k for k in new if k in ref]
    if len(keys) < 3:
        return None
    d = np.array([new[k] - ref[k] for k in keys])
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"n": len(keys), "mean": round(float(d.mean()), 3), "se": round(se, 3),
            "z": round(float(d.mean()) / se, 2) if se > 0 else 0.0}


# =========================================================================== feedback training
def feedback_update(db: Database, ctx, kind: str = "feedback", harvest_first: bool = True,
                    base_name: str | None = None, steps: int | None = None, seed: int | None = None,
                    p0: float = 0.0, p1: float = 1.0, cfg_overrides: dict | None = None) -> dict:
    """One self-improvement iteration: [harvest] -> replay mixture -> fine-tune -> checkpoint
    -> benchmark -> accept/reject. With new+replay ratios of 0 it becomes a *control* update
    (same budget, baseline corpus only, no feedback)."""
    cfg = learning_config(db)
    if cfg_overrides:
        cfg = LearningConfig.from_dict({**cfg.to_dict(), **cfg_overrides})
    base = get_model(db, base_name) if base_name else active_model(db)
    if base is None:
        raise RuntimeError("model not trained: run bootstrap first")
    version = next_version(db)
    name = model_name(version)
    seed = seed if seed is not None else 1000 + version
    span = p1 - p0
    harvest_info = None
    if harvest_first:
        harvest_info = harvest(db, ctx, base, cfg, seed, p0, p0 + 0.35 * span)
    retire_examples_below_criteria(db, cfg)
    rb = ReplayBuffer(db)
    control = cfg.new_feedback_ratio == 0 and cfg.replay_feedback_ratio == 0
    if rb.stats()["new"] == 0 and not control:
        raise RuntimeError("insufficient training data: no new feedback examples above the fidelity threshold")
    if control:
        kind = "control"

    ck = load_checkpoint(checkpoint_dir(base), with_optimizer=True)
    runner, tok = ck.runner, ck.tokenizer
    fmt = SerializationFormat.from_dict(ck.training_state.get("serialization_format"))
    bootstrap_ds = db.one("SELECT * FROM datasets WHERE kind='bootstrap' ORDER BY version LIMIT 1")
    anchor_rows = load_jsonl(_dataset_file(bootstrap_ds, "persona_train"))
    val_rows = load_jsonl(_dataset_file(bootstrap_ds, "persona_val"))
    mix, mix_info = rb.build_mixture(tok, fmt, runner.config.context_length, anchor_rows,
                                     cfg.new_feedback_ratio, cfg.replay_feedback_ratio, seed=seed,
                                     include_revisions=cfg.use_revision_examples)
    val_enc, _ = encode_rows(val_rows, tok, fmt, runner.config.context_length)
    if ctx:
        ctx.stage("replay buffer", p0 + 0.35 * span, f"mixture {mix_info['effective_ratios']} from "
                  f"{mix_info['counts']['new']} new, {mix_info['counts']['replay']} replayed, "
                  f"{mix_info['counts']['anchor']} baseline examples", {"mixture": mix_info["effective_ratios"],
                                                                        "counts": mix_info["counts"]})
    n_steps = steps or (cfg.micro_update_steps if kind == "micro" else cfg.finetune_steps)
    if not steps and mix_info["counts"]["new"] and mix_info["effective_ratios"].get("new"):
        # each new example should be seen at most ~max_epochs_per_new_example times
        t_bs = ck.training_state.get("training_config", {}).get("batch_size", 16)
        cap = int(np.ceil(cfg.max_epochs_per_new_example * mix_info["counts"]["new"] /
                          (t_bs * mix_info["effective_ratios"]["new"])))
        n_steps = min(n_steps, max(10, cap))
    tcfg = TrainingConfig(**{**ck.training_state.get("training_config", {}), "max_steps": n_steps,
                             "learning_rate": cfg.finetune_lr, "warmup_steps": max(1, n_steps // 10),
                             "eval_interval": max(10, n_steps // 3), "early_stopping_patience": 0,
                             "seed": seed, "restore_best": False})
    dataset_id = register_dataset(db, f"{kind}-mix-{name}", "feedback_mix", None,
                                  sum(mix_info["counts"].values()),
                                  {**{k: v for k, v in mix_info.items() if k != "new_example_ids"},
                                   "new_example_ids": mix_info["new_example_ids"][:200],
                                   "anchor_dataset": bootstrap_ds["id"], "base_model": base["name"]},
                                  f"Replay mixture used to train {name} from {base['name']}")
    run_id = _start_run(db, kind, base["name"], name, dataset_id,
                        {**tcfg.to_dict(), "mixture": mix_info["effective_ratios"]}, ctx, runner)
    log: list = []
    db.log_event("training_started", f"{kind} update {base['name']} -> {name}: {n_steps} steps, mixture "
                 f"{mix_info['effective_ratios']} ({mix_info['counts']})", "training_run", run_id, mix_info["counts"])
    tr = Trainer(runner, tcfg, mix, ExampleSource(val_enc),
                 on_event=_on_train_event(ctx, run_id, db, log, f"training {name}", p0 + 0.35 * span, p0 + 0.65 * span))
    try:
        res = tr.run(start_step=0)
    except Exception as e:
        db.update("training_runs", "id", run_id, {"status": "failed", "error": str(e), "finished_at": time.time()})
        raise
    _finish_run(db, run_id, res, log)
    rb.mark_used(mix_info["new_example_ids"], run_id)

    path = CHECKPOINT_DIR / name
    save_checkpoint(path, runner, tok, meta={"name": name, "version": version, "parent": base["name"],
                                              "lineage": "adaptive", "dataset_id": dataset_id},
                    training_state={**ck.training_state, "step": int(ck.training_state.get("step", 0)) + res.steps_completed,
                                    "training_config": tcfg.to_dict(), "seed": seed,
                                    "scheduler": {"type": "warmup_cosine", **tcfg.to_dict()}},
                    metrics={"final_train_loss": res.final_train_loss, "final_val_loss": res.final_val_loss},
                    dataset={"dataset_id": dataset_id, "mixture": mix_info["effective_ratios"]})
    register_model(db, name=name, version=version, parent=base["name"], lineage="adaptive", status="candidate",
                   path=path, dataset_id=dataset_id, runner=runner, tok=tok, training_cfg=tcfg.to_dict(),
                   run_id=run_id, notes=f"{kind} update of {base['name']}")

    # ---- benchmark + acceptance
    try:
        bench = run_benchmark(db, ctx, name, repetitions=cfg.benchmark_repetitions, reference_model=base["name"],
                              regression_threshold=cfg.regression_threshold,
                              progress_base=p0 + 0.65 * span, progress_span=0.35 * span)
    except BaseException as e:  # a candidate that could not be validated is never promoted; the active model stays
        db.update("model_versions", "name", name, {"status": "rejected", "decision": {
            "accepted": False, "reasons": [f"validation did not complete: {type(e).__name__}: {e}"]}})
        raise
    prev = latest_metrics(db, base["name"]) or {}
    base_run = db.one("SELECT final_val_loss FROM training_runs WHERE output_model=? ORDER BY id DESC LIMIT 1",
                      (base["name"],))
    prev_val = base_run["final_val_loss"] if base_run else None
    paired = paired_difference(reference_item_means(db, base["name"]), bench["items"])
    decision = decide({"fidelity": prev.get("fidelity"), "val_loss": prev_val, "failure_rate": prev.get("failure_rate")},
                      {"fidelity": bench["fidelity"], "val_loss": res.final_val_loss, "paired": paired,
                       "regression_count": bench["regression_count"], "failure_rate": bench["failure_rate"]}, cfg)
    # version conflict: someone activated / rolled back another model while this candidate was training
    now_active = active_model(db)
    if decision["accepted"] and now_active and now_active["name"] != base["name"]:
        decision["accepted"] = False
        decision["reasons"].append(f"active model changed from {base['name']} to {now_active['name']} while "
                                   f"training; candidate was built on a stale base")
    decision["paired"] = paired
    if decision["paired"]:
        p = decision["paired"]
        decision["reasons"].append(f"paired over {p['n']} identical benchmark items: {p['mean']:+.2f} ± {p['se']:.2f} "
                                   f"(z = {p['z']:.2f}{', within noise' if abs(p['z']) < 2 else ''})")
    if decision["accepted"]:
        # the checkpoint is complete on disk; flipping the `active_model` pointer is the single atomic switch
        db.update("model_versions", "name", name, {"status": "active", "decision": decision})
        db.set_setting("active_model", name)
        if base["lineage"] != "baseline":
            db.update("model_versions", "name", base["name"], {"status": "archived"})
        db.log_event("model_promoted", f"{name} promoted: " + "; ".join(decision["reasons"]), "model", name, decision)
    else:
        db.update("model_versions", "name", name, {"status": "rejected", "decision": decision})
        db.log_event("model_rejected", f"{name} rejected: " + "; ".join(decision["reasons"]) +
                     f". Active model remains {base['name']}", "model", name, decision)
    engine.evict(name)
    return {"model": name, "base": base["name"], "kind": kind, "accepted": decision["accepted"], "decision": decision,
            "harvest": harvest_info, "mixture": {k: mix_info[k] for k in ("effective_ratios", "counts")},
            "training": {"steps": res.steps_completed, "train_loss": res.final_train_loss,
                         "val_loss": res.final_val_loss},
            "benchmark": {k: bench[k] for k in ("fidelity", "stability", "consistency", "failure_rate",
                                                 "regression_count")}}


# =========================================================================== learning experiment
def learning_experiment(db: Database, ctx, iterations: int = 3) -> dict:
    """Section 30: baseline benchmark -> [harvest -> train -> benchmark -> compare] x iterations."""
    base = baseline_model(db)
    if base is None:
        raise RuntimeError("model not trained: run bootstrap first")
    exp_id = new_id("learn")
    db.insert("experiments", {"id": exp_id, "kind": "learning", "name": f"Learning experiment ({iterations} iterations)",
                              "model_version": base["name"], "config": {"iterations": iterations,
                                                                        "learning": learning_config(db).to_dict()},
                              "status": "running", "created_at": time.time(), "job_id": ctx.id if ctx else None})
    if ctx:
        ctx.stage("baseline", 0.01, f"baseline {base['name']} (frozen)")
    base_metrics = latest_metrics(db, base["name"])
    if base_metrics is None:
        run_benchmark(db, ctx, base["name"], repetitions=learning_config(db).benchmark_repetitions,
                      progress_base=0.0, progress_span=0.1)
        base_metrics = latest_metrics(db, base["name"])
    curve = [{"iteration": 0, "model": base["name"], "accepted": True, "fidelity": base_metrics["fidelity"],
              "stability": base_metrics["stability"], "failure_rate": base_metrics["failure_rate"],
              "val_loss": base_metrics["val_loss"]}]
    start = active_model(db)
    for it in range(1, iterations + 1):
        p0 = 0.1 + 0.9 * (it - 1) / iterations
        p1 = 0.1 + 0.9 * it / iterations
        if ctx:
            ctx.stage(f"iteration {it}/{iterations}", p0, f"self-improvement iteration {it}")
        out = feedback_update(db, ctx, kind="feedback", harvest_first=True, p0=p0, p1=p1)
        b = out["benchmark"]
        curve.append({"iteration": it, "model": out["model"], "accepted": out["accepted"], "fidelity": b["fidelity"],
                      "stability": b["stability"], "failure_rate": b["failure_rate"],
                      "val_loss": out["training"]["val_loss"], "examples_added": (out["harvest"] or {}).get("examples_added"),
                      "reasons": out["decision"]["reasons"]})
        db.update("experiments", "id", exp_id, {"summary": {"curve": curve}})
    final = active_model(db)
    fm = latest_metrics(db, final["name"])
    summary = {
        "curve": curve, "baseline_model": base["name"], "start_model": start["name"], "final_model": final["name"],
        "initial_fidelity": base_metrics["fidelity"], "final_fidelity": fm["fidelity"],
        "improvement": round(fm["fidelity"] - base_metrics["fidelity"], 3),
        "initial_stability": base_metrics["stability"], "final_stability": fm["stability"],
        "initial_failure_rate": base_metrics["failure_rate"], "final_failure_rate": fm["failure_rate"],
        "accepted": sum(1 for c in curve[1:] if c["accepted"]), "rejected": sum(1 for c in curve[1:] if not c["accepted"]),
    }
    db.update("experiments", "id", exp_id, {"status": "completed", "finished_at": time.time(), "summary": summary})
    db.log_event("learning_experiment", f"Learning experiment: fidelity {summary['initial_fidelity']:.2f} -> "
                 f"{summary['final_fidelity']:.2f} ({summary['improvement']:+.2f}); {summary['accepted']} accepted, "
                 f"{summary['rejected']} rejected", "experiment", exp_id, summary)
    return {"experiment_id": exp_id, **summary}
