"""TwinSim FastAPI application.

Run (from backend/):   uvicorn main:app --port 8000
The React dev server (npm run dev) proxies /api to this process; a production build in ../dist
is served directly by this app.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api import experiments as experiments_api
from api import history as history_api
from api import models as models_api
from api import personas as personas_api
from api import scenarios as scenarios_api
from api import training as training_api
from api.deps import PUBLIC_MODE, db, jobs
from data.ingestion import ingestion_status
from database.repo import active_model, checkpoint_dir, other_worker_alive, seed
from evaluation.failure_modes import FAILURE_LABELS
from evaluation.fidelity import EVALUATOR_VERSION, evaluate
from experiments.calibration import calibration
from llm.backend import format_parameter_count
from llm.config import EvaluatorConfig
from llm.hardware import detect_hardware, recommend_profile

_HW: dict | None = None


def startup() -> None:
    global _HW
    d = db()
    seed(d)
    jobs()  # start the worker
    from learning.auto import maybe_schedule, recover
    recover(d)  # work claimed by a process that died goes back to the queue
    maybe_schedule(d)
    _HW = detect_hardware()
    m = active_model(d)
    lines = ["", "=" * 46, "TwinSim LLM"]
    if m:
        mc = m["model_config"]
        lines += [f"Active model: {m['name']} ({m['status']})",
                  f"Parameters: {format_parameter_count(m['parameters'])} ({m['parameters']:,})",
                  f"Layers: {mc['num_layers']}", f"Heads: {mc['num_heads']}", f"Embedding: {mc['embedding_dim']}",
                  f"Context: {mc['context_length']}", f"Vocabulary: {mc['vocab_size']}"]
    else:
        lines.append("No model trained yet - run bootstrap from the Learning page or scripts/train.py")
    lines += [f"Backend: {_HW['backend']}", f"Device: {'CUDA (' + str(_HW['gpu']) + ')' if _HW['cuda_available'] else 'CPU'}",
              "=" * 46]
    print("\n".join(lines), flush=True)


@asynccontextmanager
async def lifespan(_app):
    startup()
    yield


app = FastAPI(lifespan=lifespan, title="TwinSim API", version="2.1",
              description="From-scratch Transformer digital-twin research platform")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
for r in (personas_api, scenarios_api, experiments_api, training_api, models_api, history_api):
    app.include_router(r.router)

# Public mode (TWINSIM_PUBLIC=1): for a site on the open internet. Visitors can still build twins, run
# scenarios and rate answers, but cannot start training, benchmarks, studies, change settings or switch
# the active model, which are CPU-heavy or change shared state. Run those from the CLI on the server.
_BLOCKED_PREFIXES = ("/api/training/", "/api/learning/", "/api/demo/", "/api/experiments/study", "/api/models/",
                     "/api/jobs/")


@app.middleware("http")
async def public_mode_guard(request: Request, call_next):
    if PUBLIC_MODE and request.method not in ("GET", "HEAD", "OPTIONS") and \
            (request.method in ("PUT", "DELETE") or request.url.path.startswith(_BLOCKED_PREFIXES)):
        return JSONResponse(status_code=403, content={"detail": {
            "code": "public_mode", "message": "This action is disabled on the public site. "
            "Run it on the server with backend/scripts/train.py."}})
    return await call_next(request)


def _component_status(d, m) -> dict:
    """Live status of each subsystem for the UI's command center (checked, not assumed)."""
    out = {}
    t0 = time.perf_counter()
    try:
        d.scalar("SELECT 1")
        out["database"] = {"state": "connected", "detail": f"SQLite · {(time.perf_counter() - t0) * 1000:.1f} ms"}
    except Exception as e:  # noqa: BLE001
        out["database"] = {"state": "error", "detail": str(e)}
    if not m:
        out["llm"] = {"state": "untrained", "detail": "no checkpoint yet"}
    else:
        ok = (checkpoint_dir(m) / "weights.npz").exists()
        out["llm"] = {"state": "ready" if ok else "error", "detail": m["name"] if ok else "checkpoint missing"}
    try:
        evaluate({"political": 50}, "x", "I would think about it.")
        out["evaluator"] = {"state": "ready", "detail": EVALUATOR_VERSION}
    except Exception as e:  # noqa: BLE001
        out["evaluator"] = {"state": "error", "detail": str(e)}
    hw = _HW or detect_hardware()
    out["compute"] = {"state": "cuda" if hw["cuda_available"] else "cpu",
                      "detail": f"{hw['gpu']} · {hw['vram_gb']} GB" if hw["cuda_available"] else
                      f"{hw['cpu_cores']} cores · {hw['backend']} backend"}
    jm = jobs()
    cur = d.one("SELECT type, stage, progress FROM jobs WHERE id=?", (jm.current,)) if jm.current else None
    if cur:
        out["training"] = {"state": "busy", "detail": f"{cur['type'].replace('_', ' ')} · {cur['stage']}",
                           "progress": cur["progress"]}
    elif other_worker_alive():
        run = d.one("SELECT kind, output_model, steps FROM training_runs WHERE status='running' ORDER BY id DESC LIMIT 1")
        out["training"] = {"state": "busy", "progress": None,
                           "detail": f"{run['kind']} -> {run['output_model']} (command line)" if run else "command-line job"}
    else:
        out["training"] = {"state": "idle", "detail": "worker idle", "progress": None}
    return out


@app.get("/api/system")
def system():
    d = db()
    m = active_model(d)
    hw = _HW or detect_hardware()
    return {
        "hardware": hw,
        "recommended": recommend_profile(hw),
        "active_model": m,
        "model_summary": None if not m else {
            "name": m["name"], "parameters": m["parameters"], "parameters_h": format_parameter_count(m["parameters"]),
            "layers": m["model_config"]["num_layers"], "heads": m["model_config"]["num_heads"],
            "embedding_dim": m["model_config"]["embedding_dim"], "context_length": m["model_config"]["context_length"],
            "vocab_size": m["model_config"]["vocab_size"], "backend": m["backend"], "tokenizer": m["tokenizer_version"]},
        "evaluator": {"version": EVALUATOR_VERSION, "kind": "rubric / lexicon (no language model)",
                      "weights": (d.get_setting("evaluator_weights") or EvaluatorConfig().weights),
                      "failure_labels": FAILURE_LABELS, "calibration": calibration(d)},
        "ingestion": ingestion_status(),
        "worker": {"busy": jobs().busy(), "current_job": jobs().current},
        "external_llm": False,
        "public_mode": PUBLIC_MODE,
    }


@app.get("/api/status")
def status():
    """Lightweight poll target for the persistent command-center panel."""
    d = db()
    m = active_model(d)
    return {"components": _component_status(d, m), "active_model": m["name"] if m else None,
            "public_mode": PUBLIC_MODE,
            "parameters_h": format_parameter_count(m["parameters"]) if m else None}


@app.get("/api/health")
def health():
    return {"ok": True}


DIST = Path(__file__).resolve().parent.parent / "dist"
if DIST.exists():
    if (DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
