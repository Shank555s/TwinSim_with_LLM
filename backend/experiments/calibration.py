"""Evaluator calibration anchors on the benchmark items, how to read a fidelity number.

  oracle    responses written by the corpus generator for the *correct* persona (upper reference)
  bland     a fixed non-committal answer that expresses no traits ("no-information" reference)
  shuffled  oracle responses scored against the *wrong* benchmark persona (wrong-person reference)
  empty     unreadable text (validity gate -> 0)

A model is only simulating the persona if it beats `bland`; it approaches the evaluator's
ceiling as it nears `oracle`. These anchors are computed, not assumed.
"""
from __future__ import annotations

import random

import numpy as np

from data.synthetic import generate_response
from database.db import Database
from evaluation.fidelity import evaluate
from evaluation.style import FAMILIES, expected_style

BLAND = "I would think about it for a while and then decide what makes sense."


def calibration(db: Database, refresh: bool = False) -> dict:
    cached = db.get_setting("evaluator_calibration")
    if cached and not refresh:
        return cached
    from learning.benchmark import benchmark_items
    personas, scenarios = benchmark_items(db)
    if not personas:
        return {}
    rng = random.Random(0)
    oracle, shuffled, bland = [], [], []
    for i, p in enumerate(personas):
        exp = expected_style(p.get("commStyle", ""))
        fam = max(FAMILIES, key=lambda f: exp[f]) if any(exp.values()) else "warm"
        other = personas[(i + 3) % len(personas)]
        for s in scenarios:
            for _ in range(2):
                r = generate_response(p, s, fam, rng)
                oracle.append(evaluate(p, s["text"], r).overall)
                shuffled.append(evaluate(other, s["text"], r).overall)
            bland.append(evaluate(p, s["text"], BLAND).overall)
    out = {"oracle": round(float(np.mean(oracle)), 2), "bland": round(float(np.mean(bland)), 2),
           "shuffled": round(float(np.mean(shuffled)), 2), "empty": 0.0,
           "note": "benchmark personas x benchmark scenarios; oracle = template generator (development corpus style)"}
    db.set_setting("evaluator_calibration", out)
    return out
