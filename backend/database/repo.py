"""Domain-level data access: personas, scenarios, model registry rows, trials."""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from twin.personas import RESEARCH_PERSONAS, validate_persona
from twin.scenarios import scenario_library, validate_scenario

from .db import Database, new_id


# --------------------------------------------------------------------------- seeding
def seed(db: Database) -> None:
    now = time.time()
    for p in RESEARCH_PERSONAS:
        if not db.one("SELECT id FROM personas WHERE id=?", (p["id"],)):
            db.insert("personas", {"id": p["id"], "label": p["label"], "data": validate_persona(p),
                                   "source": "preset", "is_benchmark": 1, "created_at": now, "updated_at": now})
    for s in scenario_library():
        if not db.one("SELECT id FROM scenarios WHERE id=?", (s["id"],)):
            db.insert("scenarios", {"id": s["id"], "domain": s["domain"], "issue": s["issue"], "text": s["text"],
                                    "split": s["split"], "source": "library", "created_at": now})


# --------------------------------------------------------------------------- personas
def _persona_out(r: dict) -> dict:
    d = dict(r["data"])
    d.update(id=r["id"], label=r["label"], source=r["source"], is_benchmark=bool(r["is_benchmark"]))
    return d


def list_personas(db: Database) -> list[dict]:
    return [_persona_out(r) for r in db.query(
        "SELECT * FROM personas WHERE deleted=0 ORDER BY is_benchmark DESC, created_at")]


def get_persona(db: Database, pid: str) -> dict | None:
    r = db.one("SELECT * FROM personas WHERE id=?", (pid,))
    return _persona_out(r) if r else None


def create_persona(db: Database, data: dict) -> dict:
    p = validate_persona(data)
    pid = new_id("persona")
    now = time.time()
    db.insert("personas", {"id": pid, "label": p["label"], "data": p, "source": "user", "is_benchmark": 0,
                           "created_at": now, "updated_at": now})
    db.log_event("persona_created", f"Persona '{p['label']}' created", "persona", pid, {"persona": p})
    return get_persona(db, pid)


def update_persona(db: Database, pid: str, data: dict) -> dict:
    cur = get_persona(db, pid)
    if not cur:
        raise KeyError(pid)
    if cur["is_benchmark"]:
        raise PermissionError("benchmark personas are fixed so model versions stay comparable; duplicate it instead")
    p = validate_persona(data)
    db.update("personas", "id", pid, {"label": p["label"], "data": p, "updated_at": time.time()})
    changed = {k: [cur.get(k), p.get(k)] for k in p if cur.get(k) != p.get(k)}
    db.log_event("persona_edited", f"Persona '{p['label']}' edited ({len(changed)} fields)", "persona", pid,
                 {"changes": changed})
    return get_persona(db, pid)


def delete_persona(db: Database, pid: str) -> None:
    cur = get_persona(db, pid)
    if not cur:
        raise KeyError(pid)
    if cur["is_benchmark"]:
        raise PermissionError("benchmark personas cannot be deleted")
    db.update("personas", "id", pid, {"deleted": 1})
    db.log_event("persona_deleted", f"Persona '{cur['label']}' archived", "persona", pid)


# --------------------------------------------------------------------------- scenarios
def list_scenarios(db: Database, split: str | None = None) -> list[dict]:
    if split:
        return db.query("SELECT * FROM scenarios WHERE deleted=0 AND split=? ORDER BY domain, id", (split,))
    return db.query("SELECT * FROM scenarios WHERE deleted=0 ORDER BY CASE split WHEN 'benchmark' THEN 0 "
                    "WHEN 'user' THEN 1 ELSE 2 END, domain, id")


def get_scenario(db: Database, sid: str) -> dict | None:
    return db.one("SELECT * FROM scenarios WHERE id=?", (sid,))


def create_scenario(db: Database, data: dict) -> dict:
    s = validate_scenario(data)
    sid = new_id("scenario")
    db.insert("scenarios", {"id": sid, "domain": s["domain"], "issue": s["issue"], "text": s["text"],
                            "split": "user", "source": "user", "created_at": time.time()})
    db.log_event("scenario_created", f"Scenario created ({s['domain']})", "scenario", sid, s)
    return get_scenario(db, sid)


def delete_scenario(db: Database, sid: str) -> None:
    s = get_scenario(db, sid)
    if not s:
        raise KeyError(sid)
    if s["split"] in ("benchmark", "validation"):
        raise PermissionError("benchmark/validation scenarios are fixed")
    db.update("scenarios", "id", sid, {"deleted": 1})
    db.log_event("scenario_deleted", "Scenario archived", "scenario", sid)


def benchmark_keys(db: Database) -> tuple[set[str], set[str]]:
    """IDs and texts that must never enter training data."""
    ps = {r["id"] for r in db.query("SELECT id FROM personas WHERE is_benchmark=1")}
    ss = {r["text"] for r in db.query("SELECT text FROM scenarios WHERE split IN ('benchmark','validation')")}
    return ps, ss


# --------------------------------------------------------------------------- models
def list_models(db: Database) -> list[dict]:
    return db.query("SELECT * FROM model_versions ORDER BY version")


def get_model(db: Database, name: str) -> dict | None:
    return db.one("SELECT * FROM model_versions WHERE name=?", (name,))


def active_model(db: Database) -> dict | None:
    name = db.get_setting("active_model")
    if name:
        m = get_model(db, name)
        if m:
            return m
    return db.one("SELECT * FROM model_versions WHERE status='active' ORDER BY version DESC LIMIT 1")


def baseline_model(db: Database) -> dict | None:
    return db.one("SELECT * FROM model_versions WHERE lineage='baseline' ORDER BY version LIMIT 1")


def next_version(db: Database) -> int:
    v = db.scalar("SELECT MAX(version) FROM model_versions")
    return 0 if v is None else int(v) + 1


def latest_metrics(db: Database, model_name: str) -> dict | None:
    return db.one("SELECT * FROM model_metrics WHERE model_name=? ORDER BY id DESC LIMIT 1", (model_name,))


def checkpoint_dir(model: dict) -> Path:
    """Checkpoint folder for a model row. Falls back to <TWINSIM_HOME>/checkpoints/<name> so a
    runtime/ folder copied to another machine keeps working."""
    from .db import HOME
    p = Path(model["checkpoint_path"])
    return p if p.exists() else Path(HOME) / "checkpoints" / model["name"]


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if sys.platform.startswith("win"):
        return True  # os.kill(pid, 0) would terminate the process on Windows; assume alive
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _lock_dir() -> Path:
    from .db import HOME
    d = Path(HOME) / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def worker_lock():
    """Held while a process trains, harvests or benchmarks. Other processes (the server and the CLI can run
    at the same time) check it so they never mark a live run as interrupted."""
    f = _lock_dir() / str(os.getpid())
    f.write_text(str(time.time()))
    try:
        yield
    finally:
        try:
            f.unlink()
        except OSError:
            pass


def other_worker_alive() -> bool:
    for f in _lock_dir().iterdir():
        if not f.name.isdigit() or int(f.name) == os.getpid():
            continue
        if _pid_alive(int(f.name)):
            return True
        try:
            f.unlink()  # stale lock from a process that died
        except OSError:
            pass
    return False


def mark_interrupted(db: Database) -> int:
    """Rows left 'running' by a process that died (server restart, killed CLI) are marked as
    interrupted instead of silently looking like they are still in progress. Skipped while another
    live process (server job or CLI) holds a worker lock, because its rows are genuinely running."""
    if other_worker_alive():
        return 0
    n = 0
    for table in ("training_runs", "experiments"):
        for r in db.query(f"SELECT id FROM {table} WHERE status='running'"):
            db.update(table, "id", r["id"], {"status": "interrupted", "finished_at": time.time()})
            n += 1
    if n:
        db.log_event("interrupted", f"{n} unfinished training runs / experiments from a previous process marked interrupted")
    return n
