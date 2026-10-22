"""Automatic continual learning + bounded storage.

File name starts with `test_z` so it runs after test_pipeline.py, which bootstraps the shared tiny model
(the end-to-end tests here reuse it and bootstrap one only when this file is run alone).
The unit tests use their own throw-away database.
"""
import json
import time

import pytest
from fastapi.testclient import TestClient

from database import repo
from database.db import Database, get_db, new_id
from evaluation.fidelity import evaluate
from experiments.runner import run_trial
from learning import auto, retention
from learning.promotion import decide
from learning.signals import build_signal, record_signal
from llm.config import GenerationConfig, LearningConfig

GOOD = ("I would think it through carefully and plan ahead. I support a balanced approach, because families like mine "
        "need stability, and I would rather discuss it calmly with everyone involved.")
LOOP = "yes yes yes yes yes yes yes yes"


class Ctx:
    id = "test"

    def __init__(self):
        self.stages = []

    def stage(self, stage, progress=None, message="", data=None, force=True):
        self.stages.append(stage)

    def check(self):
        pass

    def cancelled(self):
        return False


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    repo.seed(d)
    return d


def user_persona(db, **over):
    p = {k: v for k, v in repo.get_persona(db, "p1").items() if k not in ("id", "label", "source", "is_benchmark")}
    p.update(label="Custom twin", **over)
    return p


def train_scenario(db):
    return next(s for s in repo.list_scenarios(db) if s["split"] == "train")


def store_trial(db, text=GOOD, purpose="user", ts=None, experiment_id=None, persona=None, persona_id=None, model="M"):
    """Insert a finished experiment exactly like run_trial does, without needing a model."""
    persona = persona or user_persona(db)
    sc = train_scenario(db)
    rep = evaluate(persona, sc["text"], text).to_dict()
    tid = new_id("trial")
    t = {"id": tid, "experiment_id": experiment_id, "persona_id": persona_id, "persona_snapshot": persona,
         "scenario_id": sc["id"], "scenario_snapshot": {k: sc.get(k) for k in ("id", "domain", "issue", "text", "split")},
         "model_version": model, "seed": 1, "gen_params": {}, "response": text, "tokens_generated": 10, "latency_ms": 1.0,
         "repetition": 0, "purpose": purpose, "created_at": ts or time.time(), "training_status": "not_eligible",
         "tokenizer_version": "x", "backend": "numpy"}
    db.insert("trials", t)
    db.insert("evaluations", {"trial_id": tid, "evaluator": "e", "evaluator_kind": "k", "overall": rep["overall"],
                              "axes": rep["axes"], "observed": rep["observed"], "intended": rep["intended"],
                              "drift": rep["drift"], "failures": rep["failures"], "explanation": rep["explanation"],
                              "style_detail": rep["style_detail"], "created_at": t["created_at"]})
    return t, rep


# ------------------------------------------------------------------ learning signals
def test_signal_extracts_fidelity_and_failure_information(db):
    cfg = LearningConfig()
    t, rep = store_trial(db, LOOP)
    s = build_signal(t["persona_snapshot"], t["scenario_snapshot"], t["response"], rep, cfg)
    assert s["kind"] == "correct" and rep["overall"] == 0.0
    assert [f["code"] for f in s["failures"]] == ["degeneration"] and s["failure_score"] == 1.0
    assert s["training_weight"] == 1.0 and s["reduce"] and "degenerate" in s["reduce"][0].lower()
    assert s["persona"] and s["response"] == LOOP  # payload is kept until the signal is consumed

    t2, rep2 = store_trial(db, GOOD)
    s2 = build_signal(t2["persona_snapshot"], t2["scenario_snapshot"], t2["response"], rep2, cfg)
    assert s2["kind"] == ("correct" if rep2["failures"] else "reinforce" if rep2["overall"] >= cfg.min_fidelity_for_positive else "monitor")
    assert set(s2["axes"]) == set(rep2["axes"]) and s2["fidelity"] == rep2["overall"]
    assert all(a in s2["axes"] for a in s2["weak_axes"]) and 0 <= s2["failure_score"] <= 1


def test_signal_is_recorded_once_and_benchmark_is_held_out(db):
    cfg = LearningConfig()
    t, rep = store_trial(db)
    assert record_signal(db, t, rep, cfg) is not None
    assert record_signal(db, t, rep, cfg) is None  # same experiment never becomes a second signal
    assert db.scalar("SELECT COUNT(*) FROM learning_signals WHERE trial_id=?", (t["id"],)) == 1
    assert db.one("SELECT state FROM learning_signals WHERE trial_id=?", (t["id"],))["state"] == "pending"
    # benchmark persona: kept out of training, never queued
    bp = repo.get_persona(db, "p1")
    tb, repb = store_trial(db, persona={k: v for k, v in bp.items() if k not in ("id", "source", "is_benchmark")}, persona_id="p1")
    record_signal(db, tb, repb, cfg)
    r = db.one("SELECT state, kind, persona FROM learning_signals WHERE trial_id=?", (tb["id"],))
    assert (r["state"], r["kind"], r["persona"]) == ("consumed", "held_out", None)


# ------------------------------------------------------------------ latest-N window + compaction
def test_only_latest_n_experiments_stay_raw_and_old_ones_become_signals(db):
    cfg = LearningConfig(max_recent_experiments=5)
    now = time.time()
    trials = [store_trial(db, GOOD if i % 2 else LOOP, ts=now + i)[0] for i in range(9)]
    # experiment 0 produced a replay example, which must survive the removal of its raw trial
    db.insert("training_examples", {"source": "evaluator_feedback", "trial_id": trials[0]["id"], "persona": trials[0]["persona_snapshot"],
                                    "scenario": trials[0]["scenario_snapshot"], "response": GOOD, "weight": 0.9,
                                    "example_type": "response", "created_at": now, "used_count": 1})
    out = retention.compact_experiments(db, cfg)
    assert out["trials_compacted"] == 4
    left = [r["id"] for r in db.query("SELECT id FROM trials WHERE purpose='user' ORDER BY created_at")]
    assert left == [t["id"] for t in trials[4:]]  # the 5 newest, full payload
    for t in trials[:4]:
        sig = db.one("SELECT * FROM learning_signals WHERE trial_id=?", (t["id"],))
        assert sig and sig["compacted"] == 1 and sig["fidelity"] is not None and sig["training_weight"] is not None
        assert sig["persona"] is None and sig["response"] is None  # bulky payload gone, learning signal kept
        assert db.scalar("SELECT COUNT(*) FROM evaluations WHERE trial_id=?", (t["id"],)) == 0
    ex = db.one("SELECT * FROM training_examples")
    assert ex["trial_id"] is None and ex["response"] == GOOD and ex["persona"]["label"] == "Custom twin"
    assert retention.compact_experiments(db, cfg)["trials_compacted"] == 0  # idempotent


def test_window_counts_a_study_as_one_experiment(db):
    cfg = LearningConfig(max_recent_experiments=2)
    db.insert("experiments", {"id": "study-1", "kind": "study", "name": "s", "status": "completed", "created_at": 1.0})
    now = time.time()
    for i in range(4):
        store_trial(db, ts=now + i, purpose="study", experiment_id="study-1")
    store_trial(db, ts=now + 10)
    store_trial(db, ts=now + 11)
    assert retention.compact_experiments(db, cfg)["trials_compacted"] == 4  # the whole study left the window
    assert db.scalar("SELECT COUNT(*) FROM trials") == 2


def test_pending_signal_keeps_its_payload_and_claimed_work_is_deferred(db):
    cfg = LearningConfig(max_recent_experiments=1)
    now = time.time()
    t_pending, rep = store_trial(db, LOOP, ts=now)
    record_signal(db, t_pending, rep, cfg)
    t_claimed, rep2 = store_trial(db, LOOP, ts=now + 1)
    record_signal(db, t_claimed, rep2, cfg)
    db.execute("UPDATE learning_signals SET state='claimed' WHERE trial_id=?", (t_claimed["id"],))
    store_trial(db, ts=now + 5)
    out = retention.compact_experiments(db, cfg)
    assert out["trials_compacted"] == 1 and out["deferred"] == 1
    assert db.one("SELECT id FROM trials WHERE id=?", (t_claimed["id"],))  # still raw: it is being trained on
    p = db.one("SELECT * FROM learning_signals WHERE trial_id=?", (t_pending["id"],))
    assert p["state"] == "pending" and p["persona"] and p["response"] == LOOP  # can still be learned from later


def test_raw_data_is_kept_when_the_signal_cannot_be_made(db):
    cfg = LearningConfig(max_recent_experiments=0)
    t, _ = store_trial(db)
    db.execute("DELETE FROM evaluations WHERE trial_id=?", (t["id"],))  # nothing to extract a signal from
    assert retention.compact_experiments(db, cfg)["trials_compacted"] == 0
    assert db.one("SELECT id FROM trials WHERE id=?", (t["id"],))
    assert db.scalar("SELECT COUNT(*) FROM learning_signals") == 0


# ------------------------------------------------------------------ bounded replay / memory / checkpoints
def add_example(db, domain, lean, source="auto_best_of_n", weight=0.7, used=1, trait_val=80):
    p = {"political": {"L": 20, "C": 50, "R": 80}[lean], "openness": trait_val, "conscientiousness": 50,
         "extraversion": 50, "agreeableness": 50, "neuroticism": 50}
    return db.insert("training_examples", {"source": source, "persona": p, "scenario": {"domain": domain, "text": "x"},
                                           "response": "r", "weight": weight, "example_type": "response",
                                           "created_at": time.time(), "used_count": used})


def test_replay_cap_is_diverse_and_never_evicts_untrained_or_human_examples(db):
    cfg = LearningConfig(max_replay_examples=20)
    for _ in range(22):
        add_example(db, "economic", "L")  # one very redundant bucket
    for lean in ("C", "R"):
        for _ in range(2):
            add_example(db, "health", lean)  # rare buckets
    keep_new = add_example(db, "economic", "L", used=0, weight=0.1)
    keep_human = add_example(db, "economic", "L", source="human_correction", weight=0.1)
    db.insert("training_examples", {"source": "x", "persona": {}, "scenario": {}, "response": "r", "weight": 1,
                                    "example_type": "retired", "created_at": 1.0})
    out = retention.enforce_replay_cap(db, cfg)
    assert db.scalar("SELECT COUNT(*) FROM training_examples") == 20 and out["evicted"] == 8 and out["retired_removed"] == 1
    assert db.scalar("SELECT COUNT(*) FROM training_examples WHERE scenario LIKE '%health%'") == 4  # rare ones all survive
    assert db.one("SELECT id FROM training_examples WHERE id=?", (keep_new,))
    assert db.one("SELECT id FROM training_examples WHERE id=?", (keep_human,))


def test_signal_memory_is_bounded_and_rolled_up(db):
    cfg = LearningConfig(max_memory_signals=10)
    for i in range(16):
        db.insert("learning_signals", {"trial_id": f"t{i}", "state": "consumed", "kind": "correct" if i < 6 else "reinforce",
                                       "compacted": 1, "domain": "health", "fidelity": 60.0, "training_weight": 0.3 + i / 100,
                                       "failures": [{"code": "exaggeration", "severity": 0.5}], "created_at": float(i)})
    out = retention.enforce_signal_cap(db, cfg)
    assert out["rolled_up"] == 6 and db.scalar("SELECT COUNT(*) FROM learning_signals") == 10
    roll = db.get_setting("learning_rollup")
    assert roll["n"] == 6 and roll["failures"]["exaggeration"] == 6 and roll["by_domain"]["health"] == 6  # statistics survive


def make_models(db, tmp_path, statuses, active):
    for i, st in enumerate(statuses):
        d = tmp_path / f"ck{i}"
        d.mkdir()
        for f in ("weights.npz", "optimizer.npz", "meta.json"):
            (d / f).write_bytes(b"x" * 100)
        db.insert("model_versions", {"name": f"M{i}", "version": i, "parent": f"M{i - 1}" if i else None,
                                     "lineage": "baseline" if i == 0 else "adaptive", "status": st, "created_at": float(i),
                                     "checkpoint_path": str(d), "weights_pruned": 0})
        db.insert("checkpoints", {"model_name": f"M{i}", "path": str(d), "size_bytes": 300, "created_at": 0.0})
    db.set_setting("active_model", active)


def test_checkpoint_pruning_keeps_baseline_active_and_rollback(db, tmp_path):
    make_models(db, tmp_path, ["baseline", "archived", "archived", "active", "rejected", "rejected", "rejected"], "M3")
    out = retention.prune_checkpoints(db, LearningConfig(max_active_checkpoints=4))
    assert {"M0", "M3", "M2"} <= set(out["kept"]) and len(out["kept"]) == 4  # baseline, active, its parent + newest other
    for name in out["kept"]:
        assert (tmp_path / f"ck{name[1:]}" / "weights.npz").exists()
    for name in out["pruned"]:
        i = name[1:]
        assert not (tmp_path / f"ck{i}" / "weights.npz").exists() and not (tmp_path / f"ck{i}" / "optimizer.npz").exists()
        assert (tmp_path / f"ck{i}" / "meta.json").exists()  # provenance stays
        assert repo.get_model(db, name)["weights_pruned"] == 1
    assert repo.active_model(db)["name"] == "M3"
    assert retention.prune_checkpoints(db, LearningConfig(max_active_checkpoints=4))["pruned"] == []  # idempotent


def test_checkpoint_pruning_never_touches_an_in_flight_candidate(db, tmp_path):
    make_models(db, tmp_path, ["baseline", "active", "candidate"], "M1")
    out = retention.prune_checkpoints(db, LearningConfig(max_active_checkpoints=3))
    assert out["pruned"] == [] and set(out["kept"]) == {"M0", "M1", "M2"}


# ------------------------------------------------------------------ promotion rule
def test_promotion_rejects_a_candidate_that_raises_the_failure_rate():
    cfg = LearningConfig(min_improvement=1.0, max_failure_rate_increase=0.05)
    ok = decide({"fidelity": 70, "failure_rate": 0.20}, {"fidelity": 73, "failure_rate": 0.15}, cfg)
    assert ok["accepted"] and ok["failure_rate"] == 0.15
    bad = decide({"fidelity": 70, "failure_rate": 0.20}, {"fidelity": 73, "failure_rate": 0.30}, cfg)
    assert not bad["accepted"] and any("failure rate rose" in r for r in bad["reasons"])
    assert decide({"fidelity": 70}, {"fidelity": 73}, cfg)["accepted"]  # unknown failure rates do not block


def test_promotion_requires_the_gain_to_beat_benchmark_noise():
    cfg = LearningConfig(min_improvement=1.0, min_paired_z=1.0)
    noisy = decide({"fidelity": 70}, {"fidelity": 71.03, "paired": {"n": 48, "mean": 1.03, "se": 3.52, "z": 0.29}}, cfg)
    assert not noisy["accepted"] and any("within benchmark noise" in r for r in noisy["reasons"])
    real = decide({"fidelity": 70}, {"fidelity": 72.5, "paired": {"n": 48, "mean": 2.5, "se": 1.0, "z": 2.5}}, cfg)
    assert real["accepted"] and any("significant" in r for r in real["reasons"])
    off = decide({"fidelity": 70}, {"fidelity": 71.03, "paired": {"n": 48, "mean": 1.03, "se": 3.52, "z": 0.29}},
                 LearningConfig(min_paired_z=0.0))
    assert off["accepted"]


# ------------------------------------------------------------------ end to end (tiny real model)
@pytest.fixture(scope="module")
def trained():
    from learning.trainer import bootstrap
    d = get_db()
    repo.seed(d)
    if not repo.baseline_model(d):
        bootstrap(d, Ctx(), profile="tiny", pretrain_steps=20, persona_steps=20, n_pretrain_docs=150,
                  n_persona_examples=150, n_val_examples=40, model_overrides={"context_length": 384, "vocab_size": 1024})
    d.set_setting("learning_config", LearningConfig(
        samples_per_prompt=3, max_corrective_per_cycle=2, finetune_steps=8, micro_update_steps=8, benchmark_repetitions=1,
        batch_trigger_trials=2, auto_cooldown_s=0.0).to_dict())
    return d


def run_user_trials(d, n, **persona_over):
    model = repo.active_model(d)
    persona = user_persona(d, **persona_over)
    sc = train_scenario(d)
    return [run_trial(d, persona=persona, scenario=sc, model=model, gen=GenerationConfig(seed=100 + i, max_new_tokens=24),
                      purpose="user") for i in range(n)]


def test_experiments_create_signals_automatically(trained):
    d = trained
    before = d.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='pending'")
    outs = run_user_trials(d, 3, political=30)
    after = d.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='pending'")
    assert after - before == 3
    row = d.one("SELECT * FROM learning_signals WHERE trial_id=?", (outs[0]["trial"]["id"],))
    assert row["fidelity"] == outs[0]["evaluation"]["overall"] and row["expected_behavior"]


def test_failed_cycle_keeps_active_model_and_returns_signals_to_the_queue(trained, monkeypatch):
    d = trained
    active = repo.active_model(d)["name"]
    pending_before = d.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='pending'")
    assert pending_before >= 1
    import learning.trainer as tr

    def boom(*a, **k):
        raise RuntimeError("training crashed")
    monkeypatch.setattr(tr, "feedback_update", boom)
    # force at least one trainable example so the cycle reaches the training step
    d.insert("training_examples", {"source": "human_correction", "persona": user_persona(d), "scenario": train_scenario(d),
                                   "response": GOOD, "weight": 1.0, "example_type": "response", "created_at": time.time()})
    with pytest.raises(RuntimeError):
        auto.run_cycle(d, Ctx())
    assert repo.active_model(d)["name"] == active
    assert d.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='claimed'") == 0
    assert d.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='pending'") == pending_before  # nothing lost
    assert d.one("SELECT status, error FROM learning_cycles ORDER BY id DESC LIMIT 1")["status"] == "failed"
    d.execute("DELETE FROM training_examples WHERE source='human_correction' AND response=?", (GOOD,))


def test_cycle_trains_a_candidate_and_only_promotes_a_measured_improvement(trained):
    d = trained
    run_user_trials(d, 3, political=30)
    base = repo.active_model(d)
    n_models = d.scalar("SELECT COUNT(*) FROM model_versions")
    ctx = Ctx()
    out = auto.run_cycle(d, ctx)
    assert out["status"] == "completed"
    if out["model"] == base["name"]:
        pytest.skip("no training examples were produced in this cycle")  # honest: nothing was trained
    assert d.scalar("SELECT COUNT(*) FROM model_versions") == n_models + 1
    cand = repo.get_model(d, out["model"])
    assert cand["parent"] == base["name"] and "automatic learning cycle" in cand["notes"]
    dec = cand["decision"]
    print("CYCLE", out["accepted"], out["n_signals"], out["n_corrective"], out["n_examples"], dec["reasons"])
    after = repo.active_model(d)["name"]
    if out["accepted"]:  # promoted only because the benchmark said so
        assert after == cand["name"] and dec["improvement"] > 1.0 and dec["new_score"] > dec["previous_score"]
    else:
        assert after == base["name"] and cand["status"] == "rejected" and dec["reasons"]  # rejection keeps the active model
    assert d.scalar("SELECT COUNT(*) FROM learning_signals WHERE state='claimed'") == 0
    sig = d.one("SELECT * FROM learning_signals WHERE state='consumed' AND kind!='held_out' ORDER BY id DESC LIMIT 1")
    assert sig["persona"] is None and sig["compacted"] == 1 and sig["fidelity"] is not None
    cyc = d.one("SELECT * FROM learning_cycles ORDER BY id DESC LIMIT 1")
    assert cyc["status"] == "completed" and cyc["model_before"] == base["name"] and cyc["fidelity_after"] == out["benchmark"]["fidelity"]
    st = auto.status(d)
    assert st["last_cycle"]["id"] == cyc["id"] and st["improvement"]["baseline"]["fidelity"] is not None
    assert st["storage"]["replay_examples"]["n"] <= st["storage"]["replay_examples"]["limit"]


def test_recover_returns_orphaned_work_to_the_queue(trained):
    d = trained
    t, rep = store_trial(d, LOOP)
    sid = record_signal(d, t, rep, LearningConfig())
    cid = d.insert("learning_cycles", {"status": "running", "started_at": time.time()})
    d.execute("UPDATE learning_signals SET state='claimed', cycle_id=? WHERE id=?", (cid, sid))
    auto.recover(d)
    assert d.one("SELECT state FROM learning_signals WHERE id=?", (sid,))["state"] == "pending"
    assert d.one("SELECT status FROM learning_cycles WHERE id=?", (cid,))["status"] == "interrupted"


def test_background_learning_runs_without_any_user_action(trained):
    """Run an experiment through the HTTP API and wait: a learning cycle starts by itself on the worker."""
    from api.deps import jobs
    from main import app
    d = trained
    d.execute("UPDATE learning_signals SET state='consumed' WHERE state='pending'")  # clean queue
    d.set_setting("learning_config", LearningConfig(
        samples_per_prompt=3, max_corrective_per_cycle=2, finetune_steps=8, benchmark_repetitions=1,
        batch_trigger_trials=1, auto_cooldown_s=0.0).to_dict())
    scen = train_scenario(d)
    persona = {k: v for k, v in user_persona(d, political=70).items() if k != "label"} | {"label": "API twin"}
    with TestClient(app) as c:
        r = c.post("/api/experiments/trial", json={"persona": persona, "scenario_id": scen["id"],
                                                    "generation": {"seed": 5, "max_new_tokens": 24}})
        assert r.status_code == 200, r.text
        info = r.json()["learning"]
        assert info["scheduled"] is True and info["job_id"]
        job = jobs().wait(info["job_id"], timeout=600)
        assert job["status"] == "completed", job.get("error")
        st = c.get("/api/learning/status").json()
        assert st["state"] in ("updated", "completed") and st["last_cycle"]["status"] == "completed"
        assert c.get("/api/learning/cycles").json()["cycles"]
    d.set_setting("learning_config", LearningConfig().to_dict())
