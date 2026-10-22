"""Training jobs, learning configuration, training runs, job status, demo pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from database import repo
from experiments.runner import learning_config, resolve_model, run_trial, stable_seed
from learning.feedback import new_feedback_count
from learning.replay_buffer import ReplayBuffer
from llm.config import MODEL_PROFILES, GenerationConfig, LearningConfig

from .deps import db, guarded, jobs

router = APIRouter(prefix="/api", tags=["training"])


class BootstrapRequest(BaseModel):
    profile: str = "dev"
    pretrain_steps: int | None = Field(None, ge=1)
    persona_steps: int | None = Field(None, ge=1)
    persona_examples: int = Field(8000, ge=50)
    pretrain_docs: int = Field(6000, ge=50)
    seed: int = 1337


@router.post("/training/bootstrap")
def start_bootstrap(req: BootstrapRequest):
    def f():
        from learning.trainer import bootstrap
        if req.profile not in MODEL_PROFILES:
            raise ValueError(f"unknown profile {req.profile}")
        if db().scalar("SELECT COUNT(*) FROM model_versions"):
            raise ValueError("TwinSim-v0 already exists; bootstrap runs once per research run")
        d = db()
        jid = jobs().submit("bootstrap", lambda ctx: bootstrap(
            d, ctx, profile=req.profile, pretrain_steps=req.pretrain_steps, persona_steps=req.persona_steps,
            n_persona_examples=req.persona_examples, n_pretrain_docs=req.pretrain_docs, seed=req.seed),
            req.model_dump())
        return {"job_id": jid}
    return guarded(f)


class FeedbackRequest(BaseModel):
    harvest: bool = True
    steps: int | None = Field(None, ge=1, le=100000)
    control: bool = False  # ablation: same training budget on the baseline corpus only, no feedback
    prompts: int | None = Field(None, ge=1, le=2000)  # override harvest size for this run


@router.post("/training/feedback")
def start_feedback(req: FeedbackRequest):
    def f():
        from learning.trainer import feedback_update
        d = db()
        resolve_model(d, "active")
        if req.control:
            ov = {"new_feedback_ratio": 0.0, "replay_feedback_ratio": 0.0}
            jid = jobs().submit("control_update", lambda ctx: feedback_update(
                d, ctx, kind="control", harvest_first=False, steps=req.steps or learning_config(d).finetune_steps,
                cfg_overrides=ov), req.model_dump())
            return {"job_id": jid}
        if not req.harvest and new_feedback_count(d) == 0:
            raise ValueError("insufficient training data: no new feedback examples (enable harvesting or run trials)")
        ov = {"feedback_prompts_per_iteration": req.prompts} if req.prompts else None
        jid = jobs().submit("feedback_update", lambda ctx: feedback_update(d, ctx, kind="feedback",
                                                                         harvest_first=req.harvest, steps=req.steps,
                                                                         cfg_overrides=ov),
                            req.model_dump())
        return {"job_id": jid}
    return guarded(f)


class LearnRequest(BaseModel):
    iterations: int = Field(3, ge=1, le=20)


@router.post("/learning/experiment")
def start_learning(req: LearnRequest):
    def f():
        from learning.trainer import learning_experiment
        d = db()
        resolve_model(d, "baseline")
        jid = jobs().submit("learning_experiment", lambda ctx: learning_experiment(d, ctx, req.iterations),
                            req.model_dump())
        return {"job_id": jid}
    return guarded(f)


@router.get("/learning/status")
def learning_status():
    """Small read-only status of the automatic learner (there is nothing to start or stop from here)."""
    from learning.auto import status
    return guarded(lambda: status(db()))


@router.get("/learning/cycles")
def learning_cycles(limit: int = 30):
    from learning.auto import cycles
    return {"cycles": cycles(db(), min(max(limit, 1), 100))}


@router.get("/learning/config")
def get_learning_config():
    d = db()
    cfg = learning_config(d)
    return {"config": cfg.to_dict(), "defaults": LearningConfig().to_dict(),
            "replay_buffer": ReplayBuffer(d).stats(), "pending_examples": new_feedback_count(d)}


@router.put("/learning/config")
def put_learning_config(body: dict = Body(...)):
    def f():
        d = db()
        cur = learning_config(d).to_dict()
        unknown = set(body) - set(cur)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        cur.update(body)
        cfg = LearningConfig.from_dict(cur)
        if cfg.learning_mode not in ("batch", "immediate"):
            raise ValueError("learning_mode must be 'batch' or 'immediate'")
        if not 0 <= cfg.new_feedback_ratio <= 1 or not 0 <= cfg.replay_feedback_ratio <= 1 or \
                cfg.new_feedback_ratio + cfg.replay_feedback_ratio > 1:
            raise ValueError("ratios must be in [0,1] and new + replay must not exceed 1")
        for k, lo in (("max_recent_experiments", 1), ("max_replay_examples", 10), ("max_memory_signals", 10),
                      ("max_active_checkpoints", 3), ("max_history_events", 100), ("batch_trigger_trials", 1)):
            if getattr(cfg, k) < lo:
                raise ValueError(f"{k} must be at least {lo} (the baseline, the active model and a rollback target are always kept)")
        d.set_setting("learning_config", cfg.to_dict())
        d.log_event("settings_changed", "Learning configuration updated", "settings", "learning_config", body)
        return {"config": cfg.to_dict()}
    return guarded(f)


@router.get("/training/runs")
def list_runs(limit: int = 50):
    return {"runs": db().query("SELECT id, kind, status, base_model, output_model, dataset_id, config, started_at, "
                               "finished_at, steps, tokens_seen, final_train_loss, final_val_loss, best_val_loss, "
                               "error, job_id, device, backend FROM training_runs ORDER BY id DESC LIMIT ?", (limit,))}


@router.get("/training/runs/{run_id}")
def get_run(run_id: int):
    def f():
        r = db().one("SELECT * FROM training_runs WHERE id=?", (run_id,))
        if not r:
            raise KeyError(run_id)
        return r
    return guarded(f)


@router.get("/jobs")
def list_jobs(active: bool = False, limit: int = 30):
    q = "SELECT id, type, status, stage, progress, message, payload, created_at, started_at, finished_at FROM jobs"
    if active:
        q += " WHERE status IN ('queued','running')"
    q += " ORDER BY created_at DESC LIMIT ?"
    return {"jobs": db().query(q, (limit,))}


@router.get("/jobs/{jid}")
def get_job(jid: str):
    def f():
        j = db().one("SELECT * FROM jobs WHERE id=?", (jid,))
        if not j:
            raise KeyError(jid)
        return j
    return guarded(f)


@router.post("/jobs/{jid}/cancel")
def cancel_job(jid: str):
    return {"cancel_requested": jobs().cancel(jid)}


# ---------------------------------------------------------------- demo (section 49)
class DemoRequest(BaseModel):
    persona_id: str
    scenario_id: str
    seed: int = 7
    prompts: int = Field(64, ge=4, le=512)


@router.post("/demo/run")
def demo(req: DemoRequest):
    """Persona -> scenario -> twin response -> fidelity -> feedback -> train -> new version -> same scenario again."""
    def f():
        d = db()
        before_model = resolve_model(d, "active")
        persona = repo.get_persona(d, req.persona_id)
        scenario = repo.get_scenario(d, req.scenario_id)
        if not persona or not scenario:
            raise KeyError("persona or scenario")

        def run(ctx):
            from learning.trainer import feedback_update
            gen = GenerationConfig(temperature=0.7, top_k=40, top_p=0.95, repetition_penalty=1.15,
                                   seed=stable_seed("demo", req.seed, persona["id"], scenario["id"]))
            ctx.stage("generate", 0.02, f"{before_model['name']} answering as {persona['label']}")
            first = run_trial(d, persona=persona, scenario=scenario, model=before_model, gen=gen, purpose="demo")
            ctx.stage("evaluate", 0.05, f"fidelity {first['evaluation']['overall']:.1f}",
                      {"first": {"response": first["trial"]["response"], "evaluation": first["evaluation"],
                                 "training_status": first["trial"]["training_status"]}})
            ctx.stage("feedback", 0.06, f"feedback stored ({first['trial']['training_status']})")
            upd = feedback_update(d, ctx, kind="feedback", harvest_first=True, p0=0.07, p1=0.93,
                                  cfg_overrides={"feedback_prompts_per_iteration": req.prompts})
            new_model = repo.get_model(d, upd["model"])
            ctx.stage("rerun", 0.95, f"{new_model['name']} answering the same scenario with the same seed")
            second = run_trial(d, persona=persona, scenario=scenario, model=new_model, gen=gen, purpose="demo")
            return {"before_model": before_model["name"], "after_model": new_model["name"],
                    "accepted": upd["accepted"], "decision": upd["decision"],
                    "first": {"response": first["trial"]["response"], "evaluation": first["evaluation"],
                              "trial_id": first["trial"]["id"], "training_status": first["trial"]["training_status"]},
                    "second": {"response": second["trial"]["response"], "evaluation": second["evaluation"],
                               "trial_id": second["trial"]["id"]},
                    "improvement": round(second["evaluation"]["overall"] - first["evaluation"]["overall"], 2),
                    "benchmark": upd["benchmark"], "harvest": upd["harvest"], "training": upd["training"]}
        jid = jobs().submit("demo", run, req.model_dump())
        return {"job_id": jid}
    return guarded(f)
