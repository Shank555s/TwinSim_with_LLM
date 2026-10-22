"""Twin generation, trials (plain and live-streamed), studies, replay, human evaluation."""
from __future__ import annotations

import json
import queue
import random
import threading

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from database import repo
from evaluation.fidelity import evaluate
from experiments.runner import evaluator_config, replay_trial, resolve_model, run_study, run_trial
from learning.auto import after_experiment
from learning.feedback import record_human_evaluation
from llm.config import GenerationConfig
from twin.personas import validate_persona
from twin.scenarios import validate_scenario

from .deps import db, guarded, http_error, jobs

router = APIRouter(prefix="/api", tags=["experiments"])


class GenParams(BaseModel):
    max_new_tokens: int = Field(96, ge=1, le=512)
    min_new_tokens: int = Field(4, ge=0, le=64)
    temperature: float = Field(0.8, ge=0.0, le=3.0)
    top_k: int = Field(40, ge=0, le=100000)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0, le=3.0)
    deterministic: bool = False
    seed: int | None = 0


class TwinRequest(BaseModel):
    persona_id: str | None = None
    persona: dict | None = None
    scenario_id: str | None = None
    scenario: dict | None = None
    model_version: str | None = "active"
    generation: GenParams = GenParams()
    learn: bool = True  # store feedback / training example for the adaptive twin


def _resolve(req: TwinRequest):
    d = db()
    if req.persona_id:
        persona = repo.get_persona(d, req.persona_id)
        if persona is None:
            raise KeyError(f"persona {req.persona_id}")
    elif req.persona:
        persona = validate_persona(req.persona)
        persona["id"] = None
    else:
        raise ValueError("persona_id or persona is required")
    if req.scenario_id:
        scenario = repo.get_scenario(d, req.scenario_id)
        if scenario is None:
            raise KeyError(f"scenario {req.scenario_id}")
    elif req.scenario:
        scenario = validate_scenario(req.scenario)
        scenario.setdefault("split", "user")
    else:
        raise ValueError("scenario_id or scenario is required")
    return persona, scenario


def _gen(req: TwinRequest) -> GenerationConfig:
    gen = GenerationConfig(**req.generation.model_dump())
    if gen.seed is None:
        gen.seed = random.randint(0, 2**31 - 1)  # recorded, so the trial stays reproducible
    return gen


@router.post("/twin/generate")
def twin_generate(req: TwinRequest):
    """Generation only (no evaluation / storage). Response comes from our Transformer checkpoint."""
    def f():
        from twin.twin_engine import engine
        persona, scenario = _resolve(req)
        model = resolve_model(db(), req.model_version)
        res, info = engine.generate(model, persona, scenario, _gen(req))
        return {"response": res.text, "model_version": model["name"], "tokens_generated": res.tokens_generated,
                "latency_ms": round(res.latency_ms, 1), "stop_reason": res.stop_reason, **info}
    return guarded(f)


@router.post("/evaluate")
def evaluate_text(body: dict = Body(...)):
    """Score any response text against a persona with the rubric evaluator."""
    def f():
        persona = validate_persona(body.get("persona") or {})
        return evaluate(persona, str(body.get("scenario", "")), str(body.get("response", "")),
                        evaluator_config(db())).to_dict()
    return guarded(f)


def _finish_trial(d, out, persona, scenario, model, learn):
    d.log_event("experiment_run", f"Trial on {model['name']}: fidelity {out['evaluation']['overall']:.1f}",
                "trial", out["trial"]["id"], {"persona": persona.get("label"), "domain": scenario["domain"]})
    out["learning"] = after_experiment(d, learn)  # compaction of old experiments + background learning, never blocking
    return out


@router.post("/experiments/trial")
def trial(req: TwinRequest):
    """generate -> evaluate -> store experiment + feedback (+ training example) -> maybe trigger learning."""
    def f():
        d = db()
        persona, scenario = _resolve(req)
        model = resolve_model(d, req.model_version)
        out = run_trial(d, persona=persona, scenario=scenario, model=model, gen=_gen(req), purpose="user",
                        store_feedback=req.learn)
        return _finish_trial(d, out, persona, scenario, model, req.learn)
    return guarded(f)


@router.post("/experiments/trial/stream")
def trial_stream(req: TwinRequest):
    """Same as /experiments/trial, but streams the real pipeline as server-sent events:
    preparing -> generating (+ every token) -> evaluating -> feedback -> completed | failed."""
    q: "queue.Queue[dict | None]" = queue.Queue()

    def work():
        try:
            d = db()
            q.put({"stage": "preparing"})
            persona, scenario = _resolve(req)
            model = resolve_model(d, req.model_version)
            q.put({"stage": "preparing", "model": model["name"], "persona": persona.get("label")})
            out = run_trial(d, persona=persona, scenario=scenario, model=model, gen=_gen(req), purpose="user",
                            store_feedback=req.learn,
                            on_stage=lambda s, data=None: q.put({"stage": s, **(data or {})}),
                            on_text=lambda text, n: q.put({"stage": "generating", "text": text, "tokens": n}))
            out = _finish_trial(d, out, persona, scenario, model, req.learn)
            q.put({"stage": "completed", "result": out})
        except Exception as e:  # noqa: BLE001 - reported to the client, never swallowed
            err = http_error(e)
            q.put({"stage": "failed", "error": err.detail})
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def events():
        while True:
            ev = q.get()
            if ev is None:
                return
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class StudyRequest(BaseModel):
    persona_ids: list[str] = []
    scenario_ids: list[str] = []
    repetitions: int = Field(1, ge=1, le=20)
    model_version: str | None = "active"
    generation: GenParams = GenParams()
    name: str = "Study"


@router.post("/experiments/study")
def study(req: StudyRequest):
    def f():
        d = db()
        pids = req.persona_ids or [p["id"] for p in repo.list_personas(d) if p["is_benchmark"]]
        sids = req.scenario_ids or [s["id"] for s in repo.list_scenarios(d, "benchmark")][:4]
        total = len(pids) * len(sids) * req.repetitions
        if total > 2000:
            raise ValueError(f"study too large ({total} trials, max 2000)")
        resolve_model(d, req.model_version)
        gen = GenerationConfig(**req.generation.model_dump())

        def job(ctx):
            out = run_study(d, ctx, persona_ids=pids, scenario_ids=sids, repetitions=req.repetitions,
                            model_name=req.model_version, gen=gen, name=req.name)
            after_experiment(d)
            return out
        jid = jobs().submit("study", job,
                            {"trials": total, "personas": len(pids), "scenarios": len(sids), "reps": req.repetitions})
        return {"job_id": jid, "trials": total}
    return guarded(f)


@router.get("/experiments")
def list_experiments(kind: str | None = None, limit: int = 50):
    q = "SELECT id, kind, name, model_version, status, created_at, finished_at, summary, job_id FROM experiments"
    args = []
    if kind:
        q += " WHERE kind=?"
        args.append(kind)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    return {"experiments": db().query(q, args)}


@router.get("/experiments/{exp_id}")
def get_experiment(exp_id: str):
    def f():
        d = db()
        e = d.one("SELECT * FROM experiments WHERE id=?", (exp_id,))
        if not e:
            raise KeyError(exp_id)
        e["trials"] = d.query("SELECT t.id, t.persona_id, t.persona_snapshot, t.scenario_snapshot, t.repetition, t.seed, "
                              "t.response, t.model_version, e.overall, e.axes, e.failures FROM trials t "
                              "JOIN evaluations e ON e.trial_id=t.id WHERE t.experiment_id=? ORDER BY t.created_at",
                              (exp_id,))
        return e
    return guarded(f)


@router.get("/trials")
def list_trials(purpose: str | None = None, model: str | None = None, limit: int = 200, offset: int = 0):
    q = ("SELECT t.*, e.overall, e.axes, e.failures, e.evaluator FROM trials t JOIN evaluations e ON e.trial_id=t.id "
         "WHERE 1=1")
    args: list = []
    if purpose:
        q += " AND t.purpose=?"
        args.append(purpose)
    else:
        q += " AND t.purpose IN ('user','study','demo','replay')"
    if model:
        q += " AND t.model_version=?"
        args.append(model)
    q += " ORDER BY t.created_at DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    return {"trials": db().query(q, args)}


@router.get("/trials/{tid}")
def get_trial(tid: str):
    def f():
        d = db()
        t = d.one("SELECT * FROM trials WHERE id=?", (tid,))
        if not t:
            raise KeyError(tid)
        t["evaluation"] = d.one("SELECT * FROM evaluations WHERE trial_id=? ORDER BY id DESC LIMIT 1", (tid,))
        t["feedback"] = d.query("SELECT * FROM feedback WHERE trial_id=?", (tid,))
        t["human_evaluations"] = d.query("SELECT * FROM human_evaluations WHERE trial_id=?", (tid,))
        return t
    return guarded(f)


@router.post("/trials/{tid}/replay")
def replay(tid: str):
    return guarded(lambda: replay_trial(db(), tid))


# ---------------------------------------------------------------- human evaluation
@router.get("/human-eval/next")
def human_eval_next(blind: bool = True, model: str | None = None):
    """A random evaluated trial not yet rated by a human. Model version hidden when blind."""
    d = db()
    q = ("SELECT t.id, t.persona_snapshot, t.scenario_snapshot, t.response, t.model_version FROM trials t "
         "WHERE t.purpose IN ('user','study','benchmark','demo') AND t.id NOT IN (SELECT trial_id FROM human_evaluations)")
    args = []
    if model:
        q += " AND t.model_version=?"
        args.append(model)
    q += " ORDER BY RANDOM() LIMIT 1"
    t = d.one(q, args)
    if not t:
        return {"trial": None}
    if blind:
        t.pop("model_version")
    return {"trial": t, "blind": blind,
            "remaining": d.scalar("SELECT COUNT(*) FROM trials WHERE purpose IN ('user','study','benchmark','demo') "
                                  "AND id NOT IN (SELECT trial_id FROM human_evaluations)")}


class HumanEval(BaseModel):
    trial_id: str
    score: int = Field(..., ge=1, le=10)
    blind: bool = True
    rater: str = "researcher"
    corrected_response: str | None = None
    notes: str = ""


@router.post("/human-eval")
def human_eval(req: HumanEval):
    def f():
        d = db()
        t = d.one("SELECT * FROM trials WHERE id=?", (req.trial_id,))
        if not t:
            raise KeyError(req.trial_id)
        return record_human_evaluation(d, t, req.score, req.blind, req.rater, req.corrected_response, req.notes)
    return guarded(f)


@router.get("/human-eval/summary")
def human_eval_summary():
    d = db()
    rows = d.query("SELECT t.model_version, COUNT(*) AS n, AVG(h.score) AS human_mean, AVG(e.overall) AS evaluator_mean "
                   "FROM human_evaluations h JOIN trials t ON t.id=h.trial_id "
                   "JOIN evaluations e ON e.trial_id=t.id GROUP BY t.model_version ORDER BY t.model_version")
    pairs = d.query("SELECT h.score, e.overall FROM human_evaluations h JOIN evaluations e ON e.trial_id=h.trial_id")
    corr = None
    if len(pairs) >= 3:
        import numpy as np
        a = np.array([p["score"] for p in pairs], float)
        b = np.array([p["overall"] for p in pairs], float)
        if a.std() > 0 and b.std() > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
    return {"by_model": rows, "n": len(pairs), "human_vs_evaluator_correlation": corr,
            "note": "Human scores (1-10) are stored separately from evaluator scores (0-100) and never mixed."}
