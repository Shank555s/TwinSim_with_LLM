"""Model registry: list, inspect, activate / roll back, compare, benchmark."""
from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import BaseModel

from database import repo
from experiments.metrics import compare_models
from experiments.runner import learning_config
from learning.benchmark import run_benchmark

from .deps import db, guarded, jobs

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models():
    d = db()
    am = repo.active_model(d)
    out = []
    for m in repo.list_models(d):
        m["metrics"] = repo.latest_metrics(d, m["name"])
        m["is_active"] = bool(am and am["name"] == m["name"])
        m["kind"] = "bootstrap" if m["lineage"] == "baseline" else (
            d.scalar("SELECT kind FROM training_runs WHERE id=?", (m["training_run_id"],)) or "?")
        out.append(m)
    return {"models": out, "active": am["name"] if am else None}


@router.get("/compare/{a}/{b}")
def compare(a: str, b: str):
    def f():
        d = db()
        for n in (a, b):
            if not repo.get_model(d, n):
                raise KeyError(n)
        return compare_models(d, a, b)
    return guarded(f)


@router.get("/{name}")
def get_model(name: str):
    def f():
        d = db()
        m = repo.get_model(d, name)
        if not m:
            raise KeyError(name)
        p = repo.checkpoint_dir(m)
        m["checkpoint"] = {
            "path": str(p), "exists": p.exists(),
            "files": sorted(x.name for x in p.iterdir()) if p.exists() else [],
            "meta": json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else None,
            "training_state": json.loads((p / "training_state.json").read_text()) if (p / "training_state.json").exists() else None,
        }
        m["metrics_history"] = d.query("SELECT * FROM model_metrics WHERE model_name=? ORDER BY id", (name,))
        m["training_run"] = d.one("SELECT id, kind, steps, tokens_seen, final_train_loss, final_val_loss, best_val_loss, "
                                  "started_at, finished_at FROM training_runs WHERE id=?",
                                  (m["training_run_id"],)) if m["training_run_id"] else None
        m["dataset"] = d.one("SELECT * FROM datasets WHERE id=?", (m["dataset_id"],)) if m["dataset_id"] else None
        return m
    return guarded(f)


class ActivateRequest(BaseModel):
    force: bool = False  # required to activate a candidate that failed acceptance


@router.post("/{name}/activate")
def activate(name: str, req: ActivateRequest = ActivateRequest()):
    def f():
        d = db()
        m = repo.get_model(d, name)
        if not m:
            raise KeyError(name)
        if m.get("weights_pruned"):
            raise ValueError(f"{name}: weights were removed by the checkpoint retention policy and cannot be activated")
        if m["status"] == "rejected" and not req.force:
            raise PermissionError(f"{name} failed the acceptance criteria; pass force=true to activate it anyway")
        cur = repo.active_model(d)
        if cur and cur["name"] == name:
            return {"active": name, "changed": False}
        if cur and cur["lineage"] != "baseline":
            d.update("model_versions", "name", cur["name"], {"status": "archived"})
        if m["lineage"] != "baseline":
            d.update("model_versions", "name", name, {"status": "active"})
        d.set_setting("active_model", name)
        rollback = cur is not None and m["version"] < cur["version"]
        d.log_event("model_rollback" if rollback else "model_activated",
                    f"{'Rolled back' if rollback else 'Activated'} {name} (was {cur['name'] if cur else 'none'})"
                    + (" [forced despite rejection]" if m["status"] == "rejected" else ""), "model", name)
        return {"active": name, "changed": True, "rollback": rollback}
    return guarded(f)


@router.post("/{name}/benchmark")
def benchmark(name: str):
    def f():
        d = db()
        m = repo.get_model(d, name)
        if not m:
            raise KeyError(name)
        if m.get("weights_pruned"):
            raise ValueError(f"{name}: weights were removed by the checkpoint retention policy")
        reps = learning_config(d).benchmark_repetitions
        jid = jobs().submit("benchmark", lambda ctx: run_benchmark(d, ctx, name, repetitions=reps), {"model": name})
        return {"job_id": jid}
    return guarded(f)
