"""Persona/scenario handling and the rubric evaluator (no language model involved)."""
import random

import pytest

from database.db import Database
from database.repo import seed
from data.synthetic import build_persona_examples, generate_response
from evaluation.fidelity import AXES, evaluate
from evaluation.stability import stability_metrics
from experiments.calibration import calibration
from twin.persona_serializer import serialize_prompt
from twin.personas import PersonaError, validate_persona
from twin.scenarios import scenario_library, scenarios_by_split, validate_scenario

LEFT = {"political": 10, "openness": 80, "agreeableness": 75, "commStyle": "warm and empathetic"}
RIGHT = {"political": 90, "openness": 30, "agreeableness": 40, "commStyle": "direct and blunt"}


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    d = Database(tmp_path_factory.mktemp("ev") / "t.db")
    seed(d)
    return d


def test_persona_validation():
    p = validate_persona({"political": 70, "openness": "40"})
    assert p["political"] == 70 and p["openness"] == 40
    with pytest.raises(PersonaError):
        validate_persona({"political": 150})
    with pytest.raises(PersonaError):
        validate_persona("nope")


def test_scenario_validation_and_splits():
    with pytest.raises(ValueError):
        validate_scenario({"domain": "nonsense", "text": "a long enough text here"})
    with pytest.raises(ValueError):
        validate_scenario({"domain": next(iter(scenario_library()))["domain"], "text": "short"})
    texts = {s: {x["text"] for x in scenarios_by_split(s)} for s in ("train", "validation", "benchmark")}
    assert texts["benchmark"] and texts["train"]
    assert not texts["benchmark"] & texts["train"]
    assert not texts["validation"] & texts["train"]


def test_training_data_never_uses_benchmark_scenarios():
    bench = {x["text"] for x in scenarios_by_split("benchmark")}
    rows = build_persona_examples(200, "train", 5)
    assert not {r["scenario"]["text"] for r in rows} & bench


def test_serializer_contains_persona():
    s = serialize_prompt({**validate_persona(LEFT), "occupation": "nurse"},
                         {"domain": "political", "text": "Should the city expand bus service?"})
    assert "nurse" in s and "bus service" in s


def test_axes_and_range():
    r = evaluate(LEFT, "Should taxes on the wealthy go up?", "I care about everyone involved and I support it.")
    assert set(r.axes) == set(AXES)
    assert all(0 <= v <= 10 for v in r.axes.values())
    assert 0 <= r.overall <= 100
    assert r.evaluator_kind.startswith("rubric")


def test_empty_and_degenerate_responses_score_zero():
    assert evaluate(LEFT, "x", "").overall == 0
    bad = evaluate(LEFT, "x", "the the the the the the the the the the the the the the the the")
    assert bad.overall == 0 and not bad.valid
    assert "degeneration" in [f["code"] for f in bad.failures]


def test_evaluator_prefers_matching_persona(db):
    cal = calibration(db, refresh=True)
    assert cal["oracle"] > cal["shuffled"] + 5
    assert cal["oracle"] > cal["bland"]


def test_political_direction_detected():
    scen = next(s for s in scenarios_by_split("train") if s["domain"] == "political")
    rng = random.Random(1)
    left_resp = generate_response(validate_persona(LEFT), scen, "warm", rng)
    right_resp = generate_response(validate_persona(RIGHT), scen, "direct", rng)
    assert evaluate(LEFT, scen["text"], left_resp).observed["political"] < \
        evaluate(LEFT, scen["text"], right_resp).observed["political"]


def test_stability():
    same = ["I support this policy because it helps families."] * 4
    assert stability_metrics(same)["stability"] == 100.0
    assert stability_metrics(same[:1])["stability"] is None


def test_ingestion_validates_and_holds_out_benchmark(tmp_path):
    import json
    from data.ingestion import load_persona_jsonl_dir
    bench = scenarios_by_split("benchmark")[0]
    rows = [{"persona": LEFT, "scenario": {"domain": "political", "text": "Should the town build a new library?"},
             "response": "Yes, I would."},
            {"persona": LEFT, "scenario": {"domain": bench["domain"], "text": bench["text"]}, "response": "leak"},
            {"persona": {"political": 999}, "scenario": {"domain": "political", "text": "x" * 20}, "response": "a"}]
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    ok, rep = load_persona_jsonl_dir(tmp_path)
    assert len(ok) == 1 and rep["skipped"] == 3
