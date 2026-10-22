"""End-to-end: bootstrap a tiny model -> benchmark -> trial -> feedback -> fine-tune -> accept/reject,
plus the HTTP API (including the live SSE trial stream). Uses a throw-away TWINSIM_HOME."""
import json

import pytest
from fastapi.testclient import TestClient

from database import repo
from database.db import get_db
from experiments.runner import run_trial
from learning.feedback import record_human_evaluation
from learning.promotion import decide
from learning.trainer import bootstrap, feedback_update
from llm.config import GenerationConfig, LearningConfig


class Ctx:
    id = "test"

    def __init__(self):
        self.stages = []

    def stage(self, stage, progress=None, message="", data=None, force=True):
        self.stages.append(stage)

    def check(self):
        pass


@pytest.fixture(scope="module")
def trained():
    d = get_db()
    repo.seed(d)
    out = bootstrap(d, Ctx(), profile="tiny", pretrain_steps=20, persona_steps=20, n_pretrain_docs=150,
                    n_persona_examples=150, n_val_examples=40,
                    model_overrides={"context_length": 384, "vocab_size": 1024})
    return d, out


def test_bootstrap_registers_baseline(trained):
    d, out = trained
    v0 = repo.get_model(d, out["model"])
    assert v0["name"].endswith("v0") and v0["status"] == "baseline" and v0["lineage"] == "baseline"
    assert repo.active_model(d)["name"] == v0["name"]
    assert out["parameters"] > 0 and 0 <= out["benchmark"]["fidelity"] <= 100
    # fixed benchmark: 6 personas x 8 scenarios x 2 repetitions
    assert d.scalar("SELECT COUNT(*) FROM trials WHERE purpose='benchmark' AND model_version=?", (v0["name"],)) == 96
    with pytest.raises(RuntimeError):
        bootstrap(d, Ctx(), profile="tiny")


def test_trial_feedback_and_update(trained):
    d, _ = trained
    model = repo.active_model(d)
    # benchmark personas (p1-p6) are protected from training, so use a fresh custom persona
    persona = {k: v for k, v in repo.get_persona(d, "p1").items() if k not in ("id", "label")}
    persona.update(political=35, occupation="librarian")
    scen = next(s for s in repo.list_scenarios(d) if s["split"] == "train")
    stages = []
    r = run_trial(d, persona=persona, scenario=scen, model=model, gen=GenerationConfig(seed=3, max_new_tokens=24),
                  on_stage=lambda s, data=None: stages.append(s))
    assert stages == ["generating", "evaluating", "feedback"]
    assert r["trial"]["model_version"] == model["name"]
    # the baseline is frozen: its trial never trains anything by itself; a human correction does
    trial = d.one("SELECT * FROM trials WHERE id=?", (r["trial"]["id"],))
    h = record_human_evaluation(d, trial, 3, blind=True,
                                corrected_response="I would support it, because it helps families like mine.")
    assert h["examples_added"] == ["human_correction"]
    upd = feedback_update(d, Ctx(), harvest_first=False, steps=5,
                          cfg_overrides={"new_feedback_ratio": 0.5, "replay_feedback_ratio": 0.0})
    cand = repo.get_model(d, upd["model"])
    assert cand["parent"] == model["name"] and cand["version"] == 1
    assert cand["status"] == ("active" if upd["accepted"] else "rejected")
    if not upd["accepted"]:
        assert repo.active_model(d)["name"] == model["name"]  # rejection leaves the active model untouched
    assert upd["decision"]["reasons"]


def test_promotion_is_strict():
    cfg = LearningConfig(min_improvement=1.0, max_val_loss_increase=0.5, max_regressions=-1)
    assert not decide({"fidelity": 70.0}, {"fidelity": 70.9}, cfg)["accepted"]
    assert not decide({"fidelity": 70.0}, {"fidelity": 71.0}, cfg)["accepted"]  # equal to the margin
    assert decide({"fidelity": 70.0, "val_loss": 1.0}, {"fidelity": 71.5, "val_loss": 1.2}, cfg)["accepted"]
    assert not decide({"fidelity": 70.0, "val_loss": 1.0}, {"fidelity": 80.0, "val_loss": 1.6}, cfg)["accepted"]
    assert not decide({"fidelity": None}, {"fidelity": 80.0}, cfg)["accepted"]


def test_api(trained):
    from main import app
    with TestClient(app) as c:
        for url in ["/api/system", "/api/status", "/api/personas", "/api/scenarios", "/api/models",
                    "/api/metrics/dashboard", "/api/metrics/learning-curve", "/api/metrics/failures", "/api/history",
                    "/api/report", "/api/learning/config", "/api/training/runs", "/api/jobs",
                    "/api/export/trials.csv", "/api/export/report.md"]:
            assert c.get(url).status_code == 200, url
        st = c.get("/api/status").json()["components"]
        assert set(st) == {"database", "llm", "evaluator", "compute", "training"}
        assert st["llm"]["state"] == "ready"
        assert c.get("/api/models/nope").status_code == 404
        assert c.put("/api/learning/config", json={"bogus": 1}).status_code in (400, 422)
        assert c.post("/api/training/bootstrap", json={"profile": "tiny"}).status_code in (400, 409, 422)

        scen = next(s for s in c.get("/api/scenarios").json()["scenarios"] if s["split"] == "train")
        body = {"persona_id": "p2", "scenario_id": scen["id"], "generation": {"seed": 4, "max_new_tokens": 16}}
        events = []
        with c.stream("POST", "/api/experiments/trial/stream", json=body) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:]))
        order = [e["stage"] for e in events]
        assert order[0] == "preparing" and order[-1] == "completed"
        assert order.index("generating") < order.index("evaluating") < order.index("feedback")
        assert 0 <= events[-1]["result"]["evaluation"]["overall"] <= 100
