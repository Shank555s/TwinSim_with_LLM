"""Automatic, background, continual learning. There is no "train" action anywhere: experiments in, a better
model out, with the existing acceptance gate deciding whether "better" is real.

    experiment finished
        -> learning signal stored (learning/signals.py, called from experiments/runner.run_trial)
        -> `maybe_schedule`: enough signals pending and the cooldown has passed -> ONE background job
    learning cycle (job type "learning_cycle", runs on the single worker thread, so cycles never overlap)
        1. claim pending signals
        2. corrective harvest: for experiments that failed / were weak, sample N answers for the SAME persona and
           scenario with the active model, rank them with the SimFidelity evaluator and keep the best failure-free
           one as a (heavily weighted) training example. Failures therefore become targeted corrections.
           High-fidelity experiments were already stored as positive examples when they ran.
        3. feedback_update(): replay mixture (new + older feedback + baseline anchor) -> fine-tune a COPY of the
           active checkpoint -> new candidate checkpoint -> fixed benchmark -> promotion rule
           (fidelity gain, validation loss, failure rate, regressions) -> promote or reject
        4. consume the signals (payload dropped), record the cycle, run retention (bounded storage)

The active model is never touched by training: a failed, cancelled, crashed or rejected cycle leaves it as it was.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from database.db import Database, new_id
from database.repo import active_model, latest_metrics, mark_interrupted, other_worker_alive
from experiments.runner import learning_config
from llm.config import LearningConfig

from .replay_buffer import ReplayBuffer
from .retention import run_retention, storage_summary
from .signals import failure_frequency, pending_count

log = logging.getLogger("twinsim.learning")

MAX_SIGNALS_PER_CYCLE = 200
RECENT_SECONDS = 180  # how long "Model updated" / "Learning cycle completed" stays visible in the status

_sched_lock = threading.Lock()
_timer: threading.Timer | None = None


class _NullCtx:
    id = None

    def check(self): ...
    def cancelled(self): return False
    def stage(self, *a, **k): ...


# ================================================================== scheduling
def trigger_threshold(cfg: LearningConfig) -> int:
    return 1 if cfg.learning_mode == "immediate" else max(1, cfg.batch_trigger_trials)


def maybe_schedule(db: Database) -> dict:
    """Cheap and idempotent; called after every experiment. Never blocks and never raises."""
    try:
        from api.deps import PUBLIC_MODE, jobs
        cfg = learning_config(db)
        pending = pending_count(db)
        info = {"auto": cfg.auto_learning, "pending_signals": pending, "threshold": trigger_threshold(cfg),
                "scheduled": False}
        if PUBLIC_MODE or not cfg.auto_learning or pending < info["threshold"] or active_model(db) is None:
            return info
        global _timer
        with _sched_lock:
            if db.scalar("SELECT COUNT(*) FROM jobs WHERE type='learning_cycle' AND status IN ('queued','running')"):
                return info  # one cycle at a time; it will pick up whatever is pending when it starts
            last = db.scalar("SELECT MAX(finished_at) FROM learning_cycles") or 0.0
            wait = last + cfg.auto_cooldown_s - time.time()
            if wait > 0:
                if _timer is None or not _timer.is_alive():  # make sure the cooled-down cycle still happens
                    _timer = threading.Timer(wait + 1.0, maybe_schedule, args=(db,))
                    _timer.daemon = True
                    _timer.start()
                return info
            jid = jobs().submit("learning_cycle", lambda ctx: run_cycle(db, ctx), {"auto": True, "pending": pending})
            info.update(scheduled=True, job_id=jid)
        return info
    except Exception:  # noqa: BLE001 - learning must never break an experiment
        log.exception("could not schedule an automatic learning cycle")
        return {"auto": True, "scheduled": False, "error": "scheduling failed (see server log)"}


def after_experiment(db: Database, learn: bool = True) -> dict:
    """Everything that follows a finished experiment, all cheap: keep only the latest N experiments raw
    (older ones are compacted first), then start a background learning cycle if one is due."""
    try:
        run_retention(db, learning_config(db), heavy=False)
    except Exception:  # noqa: BLE001
        log.exception("experiment-window compaction failed")
    if not learn:
        return {"scheduled": False, "reason": "learning disabled for this run"}
    return maybe_schedule(db)


def recover(db: Database) -> dict:
    """After a restart: work claimed by a dead process goes back to the queue, half-finished cycles and
    candidate checkpoints are marked as such. Skipped while a live CLI process owns the worker lock."""
    if other_worker_alive():
        return {"recovered": 0}
    n = db.execute("UPDATE learning_signals SET state='pending', cycle_id=NULL WHERE state='claimed'").rowcount
    db.execute("UPDATE learning_cycles SET status='interrupted', finished_at=? WHERE status='running'", (time.time(),))
    db.execute("UPDATE model_versions SET status='rejected', decision=? WHERE status='candidate'",
               (json.dumps({"accepted": False, "reasons": ["interrupted before validation finished"]}),))
    mark_interrupted(db)
    if n:
        log.info("recovered %d claimed learning signals", n)
    return {"recovered": n}


# ================================================================== the cycle
def _claim(db: Database, cycle_id: int, limit: int) -> list[dict]:
    db.execute("UPDATE learning_signals SET state='claimed', cycle_id=?, attempts=attempts+1 WHERE id IN "
               "(SELECT id FROM learning_signals WHERE state='pending' ORDER BY training_weight DESC, id LIMIT ?)",
               (cycle_id, limit))
    return db.query("SELECT * FROM learning_signals WHERE cycle_id=? AND state='claimed'", (cycle_id,))


def _release(db: Database, cycle_id: int, cfg: LearningConfig) -> None:
    """The cycle failed: signals go back to the queue, or are skipped once they have failed too often."""
    db.execute("UPDATE learning_signals SET state=CASE WHEN attempts>=? THEN 'skipped' ELSE 'pending' END, "
               "cycle_id=NULL WHERE cycle_id=? AND state='claimed'", (cfg.max_signal_attempts, cycle_id))
    db.execute("UPDATE learning_signals SET persona=NULL, scenario=NULL, response=NULL, compacted=1 WHERE state='skipped'")


def _consume(db: Database, cycle_id: int) -> int:
    """The learning of these experiments is now in the replay buffer / checkpoint: drop their payload."""
    return db.execute("UPDATE learning_signals SET state='consumed', persona=NULL, scenario=NULL, response=NULL, "
                      "compacted=1 WHERE cycle_id=? AND state='claimed'", (cycle_id,)).rowcount


def corrective_harvest(db: Database, ctx, base: dict, signals: list[dict], cfg: LearningConfig, cycle_id: int,
                       p0: float, p1: float) -> dict:
    """Targeted best-of-N on the exact persona + scenario of each failed / weak experiment."""
    from .trainer import harvest_one
    freq = failure_frequency(db)

    def priority(s):  # failures that keep recurring are corrected first
        return (s["training_weight"] or 0) * (1 + sum(freq.get(f["code"], 0) for f in s["failures"] or []))

    todo = [s for s in signals if not s["harvested"] and s["persona"] and s["scenario"] and
            (s["kind"] == "correct" or (s["kind"] == "monitor" and len(s["weak_axes"] or []) >= 2))]
    todo = sorted(todo, key=priority, reverse=True)[:cfg.max_corrective_per_cycle]
    exp_id = new_id("harvest")
    db.insert("experiments", {"id": exp_id, "kind": "harvest", "name": f"Corrective harvest (cycle {cycle_id})",
                              "model_version": base["name"], "config": {"cycle": cycle_id, "targets": len(todo)},
                              "status": "running", "created_at": time.time(), "job_id": getattr(ctx, "id", None)})
    fixed, by_code = 0, {}
    try:
        for i, s in enumerate(todo):
            ctx.check()
            prev = {"response": s["response"], "overall": s["fidelity"], "axes": s["axes"],
                    "failures": [f["code"] for f in s["failures"] or []]}
            best, ok, _ = harvest_one(db, base, s["persona"], s["scenario"], cfg, exp_id, ("corrective", cycle_id, s["id"]),
                                      previous=prev, min_fidelity=cfg.corrective_min_fidelity,
                                      example_weight_boost=0.25 * (s["failure_score"] or 0), source="corrective")
            db.update("learning_signals", "id", s["id"], {"harvested": 1})
            if ok:
                fixed += 1
                for f in s["failures"] or []:
                    by_code[f["code"]] = by_code.get(f["code"], 0) + 1
            ctx.stage("corrective learning", p0 + (p1 - p0) * (i + 1) / max(1, len(todo)),
                      f"experiment {i + 1}/{len(todo)}: best of {cfg.samples_per_prompt} = "
                      f"{best['evaluation']['overall']:.1f}" + (" -> correction learned" if ok else " (no clean answer found)"),
                      {"done": i + 1, "total": len(todo), "best": best["evaluation"]["overall"], "accepted": ok}, force=False)
    finally:
        db.update("experiments", "id", exp_id, {"status": "completed", "finished_at": time.time(),
                                                "summary": {"targets": len(todo), "corrections": fixed}})
    return {"targets": len(todo), "corrections": fixed, "by_failure": by_code}


def run_cycle(db: Database, ctx=None) -> dict:
    """One learning cycle. Safe to call from the job worker or the CLI."""
    from .trainer import feedback_update
    ctx = ctx or _NullCtx()
    cfg = learning_config(db)
    base = active_model(db)
    if base is None:
        return {"status": "skipped", "reason": "no trained model yet"}
    before = latest_metrics(db, base["name"]) or {}
    cycle_id = db.insert("learning_cycles", {"status": "running", "started_at": time.time(), "model_before": base["name"],
                                             "fidelity_before": before.get("fidelity"),
                                             "failure_rate_before": before.get("failure_rate"),
                                             "job_id": getattr(ctx, "id", None)})
    signals = _claim(db, cycle_id, MAX_SIGNALS_PER_CYCLE)
    row = {"n_signals": len(signals)}
    try:
        if not signals and ReplayBuffer(db).stats()["new"] == 0:
            db.update("learning_cycles", "id", cycle_id, {"status": "skipped", "finished_at": time.time(), "n_signals": 0})
            return {"status": "skipped", "reason": "nothing to learn from"}
        ctx.stage("learning signals", 0.02, f"{len(signals)} experiments to learn from")
        harvest = corrective_harvest(db, ctx, base, signals, cfg, cycle_id, 0.02, 0.30)
        row["n_corrective"] = harvest["corrections"]
        new_examples = ReplayBuffer(db).stats()["new"]
        row["n_examples"] = new_examples
        if new_examples == 0:
            # failures that could not be corrected still count in the learning memory; nothing to train on yet
            _consume(db, cycle_id)
            db.update("learning_cycles", "id", cycle_id, {**row, "status": "completed", "finished_at": time.time(),
                                                          "model_after": base["name"], "accepted": 0,
                                                          "decision": {"reasons": ["no training examples were produced"]}})
            db.log_event("learning_cycle", f"Learning cycle {cycle_id}: {len(signals)} experiments analysed, no training "
                         f"examples produced; active model {base['name']} unchanged", "learning_cycle", cycle_id, row)
            run_retention(db, cfg)
            return {"status": "completed", "accepted": False, "model": base["name"], **row}
        upd = feedback_update(db, ctx, kind="auto", harvest_first=False, p0=0.30, p1=1.0)
        _consume(db, cycle_id)
        dec = upd["decision"]
        total = int(db.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='consumed' AND kind!='held_out'") or 0)
        b = upd["benchmark"]
        db.update("learning_cycles", "id", cycle_id, {
            **row, "status": "completed", "finished_at": time.time(), "model_after": upd["model"],
            "accepted": int(upd["accepted"]), "fidelity_after": b["fidelity"], "failure_rate_after": b["failure_rate"],
            "experiments_incorporated": total, "decision": dec})
        db.update("model_versions", "name", upd["model"], {
            "notes": f"automatic learning cycle {cycle_id}: {len(signals)} experiments ({harvest['corrections']} corrected), "
                     f"{new_examples} new examples; {total} experiments incorporated in total"})
        ff = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
        fr = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
        db.log_event("learning_cycle", (
            f"Learning cycle {cycle_id}: {len(signals)} experiments, {new_examples} new examples -> {upd['model']} "
            f"{'promoted' if upd['accepted'] else 'rejected'} (fidelity {ff(before.get('fidelity'))} -> {ff(b['fidelity'])}, "
            f"failure rate {fr(before.get('failure_rate'))} -> {fr(b['failure_rate'])}). Active model: "
            f"{(active_model(db) or {}).get('name')}"), "learning_cycle", cycle_id, {**row, "decision": dec})
        run_retention(db, cfg)
        return {"status": "completed", "cycle_id": cycle_id, "model": upd["model"], "accepted": upd["accepted"],
                "benchmark": b, "decision": dec, **row}
    except BaseException as e:
        _release(db, cycle_id, cfg)
        db.update("learning_cycles", "id", cycle_id, {**row, "status": "failed" if isinstance(e, Exception) else "cancelled",
                                                      "finished_at": time.time(), "error": f"{type(e).__name__}: {e}"})
        log.warning("learning cycle %s did not finish (%s); active model %s unchanged", cycle_id, e, base["name"])
        raise


# ================================================================== read side
def status(db: Database) -> dict:
    """Small, honest status for the UI. Every number is read from stored measurements."""
    from api.deps import PUBLIC_MODE, jobs
    cfg = learning_config(db)
    am = active_model(db)
    pending = pending_count(db)
    job = db.one("SELECT id, status, stage, progress, message FROM jobs WHERE type='learning_cycle' AND status IN "
                 "('queued','running') ORDER BY created_at DESC LIMIT 1")
    last = db.one("SELECT * FROM learning_cycles WHERE status IN ('completed','failed') ORDER BY id DESC LIMIT 1")
    if PUBLIC_MODE or not cfg.auto_learning:
        state = "disabled"
    elif job:
        state = "learning"
    elif last and time.time() - (last["finished_at"] or 0) < RECENT_SECONDS and last["status"] == "completed":
        state = "updated" if last["accepted"] else "completed"
    else:
        state = "collecting" if pending else "idle"
    from database.repo import baseline_model
    bm = baseline_model(db)
    base_m = latest_metrics(db, bm["name"]) if bm else None
    act_m = latest_metrics(db, am["name"]) if am else None

    def snap(m):
        if not m:
            return None
        d = m.get("details") or {}
        return {"fidelity": m["fidelity"], "failure_rate": m["failure_rate"], "stability": m["stability"],
                "axes": d.get("axes"), "failure_rates": d.get("failure_rates")}
    b, a = snap(base_m), snap(act_m)
    return {
        "state": state, "auto_learning": cfg.auto_learning and not PUBLIC_MODE,
        "message": {"learning": "Learning update in progress…", "updated": "Model updated",
                    "completed": "Learning cycle completed", "collecting": f"{pending} experiment{'s' if pending != 1 else ''} queued for learning",
                    "idle": "Learning is up to date", "disabled": "Automatic learning is off"}[state],
        "active_model": am["name"] if am else None, "version": am["version"] if am else None,
        "pending_signals": pending, "trigger_at": trigger_threshold(cfg),
        "current": job and {"stage": job["stage"], "progress": job["progress"], "message": job["message"]},
        "last_cycle": last and {k: last[k] for k in ("id", "status", "finished_at", "n_signals", "n_corrective", "n_examples",
                                                      "model_before", "model_after", "accepted", "fidelity_before",
                                                      "fidelity_after", "failure_rate_before", "failure_rate_after",
                                                      "experiments_incorporated", "error")},
        "improvement": {"baseline": b, "active": a,
                        "fidelity_delta": None if not (a and b and a["fidelity"] is not None and b["fidelity"] is not None)
                        else round(a["fidelity"] - b["fidelity"], 3),
                        "failure_rate_delta": None if not (a and b and a["failure_rate"] is not None and b["failure_rate"] is not None)
                        else round(a["failure_rate"] - b["failure_rate"], 4)},
        "storage": storage_summary(db, cfg),
        "experiments_incorporated": int(db.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='consumed' "
                                                  "AND kind!='held_out'") or 0),
        "held_out_experiments": int(db.scalar("SELECT COUNT(*) FROM learning_signals WHERE kind='held_out'") or 0),
    }


def cycles(db: Database, limit: int = 30) -> list[dict]:
    return db.query("SELECT * FROM learning_cycles ORDER BY id DESC LIMIT ?", (limit,))
