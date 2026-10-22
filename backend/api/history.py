"""History, research metrics, report and exports."""
from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response

from experiments import metrics
from experiments.report import build_report, report_markdown

from .deps import db, guarded

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history")
def history(type: str | None = None, limit: int = 300, offset: int = 0):
    q = "SELECT * FROM history_events"
    args: list = []
    if type:
        q += " WHERE type=?"
        args.append(type)
    q += " ORDER BY ts DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    d = db()
    types = [r["type"] for r in d.query("SELECT DISTINCT type FROM history_events ORDER BY type")]
    return {"events": d.query(q, args), "types": types,
            "total": d.scalar("SELECT COUNT(*) FROM history_events" + (" WHERE type=?" if type else ""),
                              [type] if type else [])}


@router.get("/metrics/dashboard")
def dashboard():
    return metrics.dashboard(db())


@router.get("/metrics/learning-curve")
def learning_curve():
    return {"curve": metrics.learning_curve(db())}


@router.get("/metrics/failures")
def failures():
    return metrics.failure_analysis(db())


@router.get("/metrics/dna")
def dna(model: str, persona_id: str):
    return guarded(lambda: metrics.twin_dna(db(), model, persona_id))


@router.get("/metrics/drift")
def drift(model: str):
    return {"model": model, "drift": metrics.persona_drift(db(), model)}


@router.get("/report")
def report():
    return build_report(db())


@router.get("/export/report.md", response_class=PlainTextResponse)
def report_md():
    return PlainTextResponse(report_markdown(build_report(db())), headers={
        "Content-Disposition": "attachment; filename=twinsim_report.md"})


def _csv(rows: list[list], header: list[str], filename: str) -> Response:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


AXES = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "political", "style"]


@router.get("/export/trials.csv")
def export_trials(purpose: str | None = None):
    """Original TwinSim CSV columns preserved (first 16), extended with reproducibility columns."""
    d = db()
    q = ("SELECT t.*, e.overall, e.axes, e.failures, e.explanation, e.evaluator FROM trials t "
         "JOIN evaluations e ON e.trial_id=t.id")
    args = []
    if purpose:
        q += " WHERE t.purpose=?"
        args.append(purpose)
    q += " ORDER BY t.created_at"
    rows = []
    for t in d.query(q, args):
        p, s = t["persona_snapshot"], t["scenario_snapshot"]
        rows.append([t["id"], t["created_at"], s.get("domain"), s.get("text"), t["response"],
                     *[t["axes"].get(a, "") for a in AXES], "|".join(f["code"] for f in t["failures"]),
                     (t["explanation"] or [""])[0] if isinstance(t["explanation"], list) else t["explanation"],
                     p.get("age"), p.get("gender"), p.get("occupation"), p.get("political"),
                     t["overall"], t["model_version"], t["purpose"], t["seed"], json.dumps(t["gen_params"]),
                     t["persona_id"], t["scenario_id"], t["evaluator"], t["tokenizer_version"], t["training_status"]])
    header = ["id", "timestamp", "scenario_type", "scenario", "response", *[f"fidelity_{a}" for a in AXES],
              "failures", "comment", "persona_age", "persona_gender", "persona_occupation", "persona_political",
              "overall_fidelity", "model_version", "purpose", "seed", "generation_params", "persona_id", "scenario_id",
              "evaluator_version", "tokenizer_version", "training_status"]
    return _csv(rows, header, "twinsim_trials.csv")


@router.get("/export/benchmark.csv")
def export_benchmark():
    d = db()
    rows = []
    for t in d.query("SELECT t.*, e.overall, e.axes, e.failures FROM trials t JOIN evaluations e ON e.trial_id=t.id "
                     "WHERE t.purpose='benchmark' ORDER BY t.model_version, t.persona_id, t.scenario_id, t.repetition"):
        rows.append([t["model_version"], t["experiment_id"], t["persona_id"], t["scenario_id"], t["repetition"],
                     t["seed"], t["overall"], *[t["axes"].get(a, "") for a in AXES],
                     "|".join(f["code"] for f in t["failures"]), t["response"]])
    return _csv(rows, ["model_version", "benchmark_id", "persona_id", "scenario_id", "repetition", "seed",
                       "overall_fidelity", *[f"fidelity_{a}" for a in AXES], "failures", "response"],
                "twinsim_benchmark.csv")


@router.get("/export/models.json")
def export_models():
    d = db()
    data = {"models": metrics.learning_curve(d), "training_runs": d.query(
        "SELECT id, kind, status, base_model, output_model, dataset_id, config, steps, tokens_seen, final_train_loss, "
        "final_val_loss, best_val_loss, backend, device FROM training_runs ORDER BY id"),
        "datasets": d.query("SELECT * FROM datasets ORDER BY version")}
    return Response(json.dumps(data, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=twinsim_models.json"})


@router.get("/export/experiments.json")
def export_experiments():
    d = db()
    exps = d.query("SELECT * FROM experiments ORDER BY created_at")
    for e in exps:
        e["trials"] = d.query("SELECT t.id, t.persona_id, t.scenario_id, t.model_version, t.seed, t.gen_params, "
                              "t.response, t.repetition, t.purpose, t.tokenizer_version, t.backend, e.overall, e.axes, "
                              "e.failures, e.evaluator FROM trials t JOIN evaluations e ON e.trial_id=t.id "
                              "WHERE t.experiment_id=?", (e["id"],))
    return Response(json.dumps({"experiments": exps}, default=str), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=twinsim_experiments.json"})
