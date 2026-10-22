"""Bounded storage: RAW HISTORY != LEARNING MEMORY.

    latest N experiments (N = max_recent_experiments)  -> full raw payload (trials, evaluations, feedback)
    older experiments                                  -> compact learning signal (+ a replay example when the
                                                          experiment produced one), then the raw rows are deleted
    replay buffer (training_examples)                  -> at most max_replay_examples, diversity-aware eviction
    learning signals                                   -> at most max_memory_signals, older ones rolled up into counters
    model checkpoints                                  -> at most max_active_checkpoints keep their weights on disk
    logs                                               -> history events, jobs, training logs are capped

The order inside `compact_experiments` is the safety rule of this module: the compact representation is
written and read back FIRST, and only then are the raw rows removed, all in one transaction, so a crash
can never leave an experiment that is neither raw nor compacted.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from database.db import Database
from database.repo import active_model, baseline_model, checkpoint_dir
from llm.config import LearningConfig

from .signals import build_signal

log = logging.getLogger("twinsim.learning")

# purposes that are "experiments" in the UI. benchmark / harvest trials are machinery, handled separately.
RAW_PURPOSES = ("user", "study", "demo", "replay")
_IN = ",".join("?" for _ in RAW_PURPOSES)


class RetentionError(RuntimeError):
    pass


# ============================================================ 1. raw experiments -> compact signals
def _experiment_groups(db: Database) -> list[tuple[float, list[str]]]:
    """Experiments newest first. A study (many trials, one experiment id) counts as ONE experiment."""
    groups: dict[str, list] = defaultdict(lambda: [0.0, []])
    for r in db.query(f"SELECT id, COALESCE(experiment_id, id) AS grp, created_at FROM trials WHERE purpose IN ({_IN})",
                      RAW_PURPOSES):
        g = groups[r["grp"]]
        g[0] = max(g[0], r["created_at"] or 0.0)
        g[1].append(r["id"])
    return sorted(((t, ids) for t, ids in groups.values()), key=lambda x: -x[0])


def _report_from_rows(ev: dict) -> dict:
    return {"overall": ev["overall"], "axes": ev["axes"], "failures": ev["failures"] or [], "drift": ev["drift"] or {},
            "intended": ev["intended"] or {},
            "valid": not any(f["code"] == "degeneration" for f in ev["failures"] or [])}


def compact_trial(db: Database, trial_id: str, cfg: LearningConfig) -> str:
    """Replace one raw trial by its compact learning representation. Returns 'compacted', 'deferred' or 'missing'."""
    t = db.one("SELECT * FROM trials WHERE id=?", (trial_id,))
    if t is None:
        return "missing"
    sig = db.one("SELECT id, state FROM learning_signals WHERE trial_id=?", (trial_id,))
    if sig and sig["state"] == "claimed":
        return "deferred"  # being trained on right now; compact it after the cycle
    ev = db.one("SELECT * FROM evaluations WHERE trial_id=? ORDER BY id DESC LIMIT 1", (trial_id,))
    with db.transaction() as c:
        # STEP 1-4: make sure the learning signal exists in compact form
        if sig is None:
            if ev is None:
                raise RetentionError(f"trial {trial_id} has no evaluation to extract a learning signal from")
            row = build_signal(t["persona_snapshot"], t["scenario_snapshot"], t["response"], _report_from_rows(ev), cfg,
                               t["model_version"], t["created_at"])
            # a historical experiment: whatever replay value it had already lives in training_examples
            row.update(state="consumed", compacted=1, persona=None, scenario=None, response=None, trial_id=trial_id)
            cols = list(row)
            c.execute(f"INSERT OR IGNORE INTO learning_signals ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                      [Database._enc(row[k]) for k in cols])
        # STEP 5: verify it is really there (and that any unused replay example is self-contained) before deleting
        if not c.execute("SELECT 1 FROM learning_signals WHERE trial_id=?", (trial_id,)).fetchone():
            raise RetentionError(f"learning signal for {trial_id} was not stored; raw trial kept")
        c.execute("UPDATE learning_signals SET compacted=1, persona=CASE WHEN state='pending' THEN persona END, "
                  "scenario=CASE WHEN state='pending' THEN scenario END, "
                  "response=CASE WHEN state='pending' THEN response END WHERE trial_id=?", (trial_id,))
        # STEP 6: only now drop the bulky raw payload; training examples keep their own persona/scenario/response
        c.execute("UPDATE training_examples SET trial_id=NULL WHERE trial_id=?", (trial_id,))
        c.execute("UPDATE training_examples SET feedback_id=NULL WHERE feedback_id IN "
                  "(SELECT id FROM feedback WHERE trial_id=?)", (trial_id,))
        for table in ("human_evaluations", "feedback", "evaluations"):
            c.execute(f"DELETE FROM {table} WHERE trial_id=?", (trial_id,))
        c.execute("DELETE FROM trials WHERE id=?", (trial_id,))
    return "compacted"


def compact_experiments(db: Database, cfg: LearningConfig) -> dict:
    groups = _experiment_groups(db)
    old = groups[cfg.max_recent_experiments:]
    done = deferred = 0
    for _, ids in old:
        for tid in ids:
            try:
                r = compact_trial(db, tid, cfg)
            except Exception as e:  # noqa: BLE001 - raw rows are kept when anything goes wrong
                log.warning("compaction of %s skipped: %s", tid, e)
                continue
            done += r == "compacted"
            deferred += r == "deferred"
    # experiment headers of studies whose trials are all gone (their summary stays as the compact record)
    return {"experiments_kept": min(len(groups), cfg.max_recent_experiments), "trials_compacted": done,
            "deferred": deferred}


# ============================================================ 2. bounded replay buffer
def _bucket(ex: dict) -> tuple:
    p, sc = ex["persona"] or {}, ex["scenario"] or {}
    pol = p.get("political", 50)
    lean = "L" if pol < 40 else "R" if pol > 60 else "C"
    traits = {k: p.get(k, 50) for k in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")}
    top = max(traits, key=lambda k: abs(traits[k] - 50))
    return sc.get("domain"), lean, top, "hi" if traits[top] >= 50 else "lo"


def _replay_score(ex: dict) -> float:
    """Higher = more worth keeping. Humans > corrective (fixed a known failure) > rest; heavier weights win."""
    base = {"human_correction": 1.0, "human_rating": 0.6, "corrective": 0.5}.get(ex["source"], 0.0)
    return float(ex["weight"] or 0.0) + base + (0.2 if ex["example_type"] == "revision" else 0.0) \
        + 0.1 * min(3, ex.get("used_count") or 0)  # trained on repeatedly and still around: proven stable anchor


def enforce_replay_cap(db: Database, cfg: LearningConfig) -> dict:
    retired = db.execute("DELETE FROM training_examples WHERE example_type='retired'").rowcount
    rows = db.query("SELECT id, source, weight, example_type, used_count, persona, scenario FROM training_examples")
    over = len(rows) - cfg.max_replay_examples
    if over <= 0:
        return {"replay_examples": len(rows), "evicted": 0, "retired_removed": retired}
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[_bucket(r)].append(r)
    for b in buckets.values():
        b.sort(key=_replay_score, reverse=True)  # weakest last
    evictable = {k: [r for r in b if r["used_count"] > 0 and r["source"] != "human_correction"]
                 for k, b in buckets.items()}  # untrained and human-written examples are never evicted
    size = {k: len(b) for k, b in buckets.items()}
    evict: list[int] = []
    while over > 0:
        # thin out the most redundant bucket first so every scenario type / persona profile stays represented
        cands = [k for k, e in evictable.items() if e]
        if not cands:
            break
        k = max(cands, key=lambda k: (size[k], -_replay_score(evictable[k][-1])))
        evict.append(evictable[k].pop()["id"])
        size[k] -= 1
        over -= 1
    for i in range(0, len(evict), 500):
        chunk = evict[i:i + 500]
        db.execute(f"DELETE FROM training_examples WHERE id IN ({','.join('?' for _ in chunk)})", chunk)
    return {"replay_examples": len(rows) - len(evict), "evicted": len(evict), "retired_removed": retired}


# ============================================================ 3. bounded long-term memory (signals)
def enforce_signal_cap(db: Database, cfg: LearningConfig) -> dict:
    """Keep at most `max_memory_signals` finished signals. Evicted ones are folded into running counters first."""
    rows = db.query("SELECT id, kind, domain, fidelity, training_weight, failures, created_at FROM learning_signals "
                    "WHERE state IN ('consumed','skipped') AND compacted=1")
    over = len(rows) - cfg.max_memory_signals
    if over <= 0:
        return {"signals": len(rows), "rolled_up": 0}
    victims = sorted(rows, key=lambda r: (r["training_weight"] or 0.0, r["created_at"] or 0.0))[:over]
    roll = db.get_setting("learning_rollup") or {"n": 0, "fidelity_sum": 0.0, "by_kind": {}, "by_domain": {}, "failures": {}}
    for r in victims:
        roll["n"] += 1
        roll["fidelity_sum"] += r["fidelity"] or 0.0
        roll["by_kind"][r["kind"]] = roll["by_kind"].get(r["kind"], 0) + 1
        roll["by_domain"][r["domain"]] = roll["by_domain"].get(r["domain"], 0) + 1
        for f in r["failures"] or []:
            roll["failures"][f["code"]] = roll["failures"].get(f["code"], 0) + 1
    ids = [r["id"] for r in victims]
    with db.transaction() as c:  # counters and deletion together, so nothing is counted twice or lost
        c.execute("INSERT INTO settings(key, value) VALUES('learning_rollup', ?) ON CONFLICT(key) DO UPDATE SET "
                  "value=excluded.value", (json.dumps(roll),))
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            c.execute(f"DELETE FROM learning_signals WHERE id IN ({','.join('?' for _ in chunk)})", chunk)
    return {"signals": len(rows) - len(ids), "rolled_up": len(ids)}


# ============================================================ 4. machinery trials (harvest / benchmark)
def _delete_trials_where(db: Database, where: str, args: tuple = ()) -> int:
    with db.transaction() as c:
        sub = f"SELECT id FROM trials WHERE {where}"
        c.execute(f"UPDATE training_examples SET trial_id=NULL WHERE trial_id IN ({sub})", args)
        c.execute(f"UPDATE training_examples SET feedback_id=NULL WHERE feedback_id IN "
                  f"(SELECT id FROM feedback WHERE trial_id IN ({sub}))", args)
        for table in ("human_evaluations", "feedback", "evaluations"):
            c.execute(f"DELETE FROM {table} WHERE trial_id IN ({sub})", args)
        n = c.execute(f"DELETE FROM trials WHERE {where}", args).rowcount
    return n


def prune_machinery_trials(db: Database, keep_models: set[str]) -> dict:
    """Harvest samples were only needed to create replay examples (those hold their own copy), and benchmark
    trials only matter for model versions that still have weights (the per-item scores stay in the
    benchmark experiment summary, so paired comparisons keep working)."""
    running = [r["id"] for r in db.query("SELECT id FROM experiments WHERE kind='harvest' AND status='running'")]
    guard = f" AND experiment_id NOT IN ({','.join('?' for _ in running)})" if running else ""
    h = _delete_trials_where(db, "purpose='harvest'" + guard, tuple(running))
    marks = ",".join("?" for _ in keep_models)
    b = _delete_trials_where(db, f"purpose='benchmark' AND model_version NOT IN ({marks})", tuple(keep_models)) \
        if keep_models else 0
    db.execute("DELETE FROM experiments WHERE kind='harvest' AND status!='running' AND id NOT IN "
               "(SELECT id FROM experiments WHERE kind='harvest' ORDER BY created_at DESC LIMIT 20)")
    return {"harvest_trials_removed": h, "benchmark_trials_removed": b}


# ============================================================ 5. checkpoints
def checkpoints_to_keep(db: Database, cfg: LearningConfig) -> set[str]:
    """Always: baseline, active model, and the active model's parent (the rollback target).
    Remaining slots go to the newest other versions, accepted ones before rejected candidates."""
    models = db.query("SELECT name, version, parent, status, weights_pruned FROM model_versions ORDER BY version")
    by_name = {m["name"]: m for m in models}
    keep: set[str] = set()
    b, a = baseline_model(db), active_model(db)
    if b:
        keep.add(b["name"])
    if a:
        keep.add(a["name"])
        if a.get("parent"):
            keep.add(a["parent"])
    keep |= {m["name"] for m in models if m["status"] == "candidate"}  # in flight
    rest = sorted((m for m in models if m["name"] not in keep and not m["weights_pruned"]),
                  key=lambda m: (m["status"] == "rejected", -m["version"]))
    for m in rest:
        if len(keep) >= cfg.max_active_checkpoints:
            break
        keep.add(m["name"])
    return keep & set(by_name)


def prune_checkpoints(db: Database, cfg: LearningConfig) -> dict:
    keep = checkpoints_to_keep(db, cfg)
    pruned, freed = [], 0
    for m in db.query("SELECT * FROM model_versions WHERE weights_pruned=0"):
        if m["name"] in keep:
            continue
        d = checkpoint_dir(m)
        db.update("model_versions", "name", m["name"], {"weights_pruned": 1})  # flag first: it never lies the other way
        for fname in ("weights.npz", "optimizer.npz"):
            f = Path(d) / fname
            try:
                if f.exists():
                    freed += f.stat().st_size
                    f.unlink()
            except OSError as e:
                log.warning("could not delete %s: %s", f, e)
        db.execute("UPDATE checkpoints SET size_bytes=? WHERE model_name=?",
                   (sum(x.stat().st_size for x in Path(d).iterdir()) if Path(d).exists() else 0, m["name"]))
        try:
            from twin.twin_engine import engine
            engine.evict(m["name"])
        except Exception:  # noqa: BLE001
            pass
        pruned.append(m["name"])
    if pruned:
        db.log_event("checkpoints_pruned", f"Retention: weights of {', '.join(pruned)} removed ({freed / 1e6:.1f} MB); "
                     f"kept {', '.join(sorted(keep))}", "model", None, {"pruned": pruned, "kept": sorted(keep)})
    return {"pruned": pruned, "kept": sorted(keep), "freed_bytes": freed}


# ============================================================ 6. logs
def prune_logs(db: Database, cfg: LearningConfig) -> dict:
    n = db.execute("DELETE FROM history_events WHERE id <= (SELECT MAX(id) FROM history_events) - ?",
                   (cfg.max_history_events,)).rowcount
    db.execute("UPDATE training_runs SET log='[]' WHERE id NOT IN (SELECT id FROM training_runs ORDER BY id DESC LIMIT 10)")
    db.execute("UPDATE jobs SET events='[]' WHERE id NOT IN (SELECT id FROM jobs ORDER BY created_at DESC LIMIT 20)")
    db.execute("DELETE FROM jobs WHERE status NOT IN ('queued','running') AND id NOT IN "
               "(SELECT id FROM jobs ORDER BY created_at DESC LIMIT 200)")
    db.execute("DELETE FROM learning_cycles WHERE id NOT IN (SELECT id FROM learning_cycles ORDER BY id DESC LIMIT 100)")
    return {"history_events_removed": n}


# ============================================================ orchestration
def run_retention(db: Database, cfg: LearningConfig, heavy: bool = True) -> dict:
    """`heavy=False` (after every experiment) only compacts the experiment window; the full pass runs after
    each learning cycle. Every step is isolated: a failure in one is logged and never blocks the others."""
    out: dict = {}
    steps = [("experiments", lambda: compact_experiments(db, cfg))]
    if heavy:
        steps += [("replay", lambda: enforce_replay_cap(db, cfg)),
                  ("memory", lambda: enforce_signal_cap(db, cfg)),
                  ("checkpoints", lambda: prune_checkpoints(db, cfg)),
                  ("machinery", lambda: prune_machinery_trials(
                      db, {m["name"] for m in db.query("SELECT name FROM model_versions WHERE weights_pruned=0")})),
                  ("logs", lambda: prune_logs(db, cfg))]
    for name, fn in steps:
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            log.exception("retention step %s failed", name)
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def storage_summary(db: Database, cfg: LearningConfig) -> dict:
    """What is stored right now vs. its limit (shown in the API status, read-only)."""
    groups = len(_experiment_groups(db))
    return {
        "recent_experiments": {"n": groups, "limit": cfg.max_recent_experiments},
        "replay_examples": {"n": int(db.scalar("SELECT COUNT(*) FROM training_examples") or 0),
                            "limit": cfg.max_replay_examples},
        "memory_signals": {"n": int(db.scalar("SELECT COUNT(*) FROM learning_signals WHERE state IN ('consumed','skipped')") or 0),
                           "limit": cfg.max_memory_signals},
        "checkpoints": {"n": int(db.scalar("SELECT COUNT(*) FROM model_versions WHERE weights_pruned=0") or 0),
                        "limit": cfg.max_active_checkpoints},
    }
