"""Replay buffer (section 15). Each feedback-training batch mixes three pools:

    new      feedback examples never trained on before            (ratio: new_feedback_ratio)
    replay   feedback examples already used in an earlier update  (ratio: replay_feedback_ratio)
    anchor   the baseline persona corpus (dataset the model was bootstrapped on)  (the remainder)

Mixing old and anchor data into every update guards against catastrophic forgetting and
persona collapse. When a pool is empty its share is redistributed and the *effective* ratios
are recorded in the training run.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from database.db import Database
from llm.tokenizer import TwinSimTokenizer
from llm.training import ExampleSource, MixtureSource
from twin.persona_serializer import SerializationFormat, serialize_example, serialize_feedback_example


def encode_example(tok: TwinSimTokenizer, fmt: SerializationFormat, persona: dict, scenario: dict, response: str,
                   weight: float, ctx: int):
    prompt, completion = serialize_example(persona, scenario, response, fmt)
    p_ids = tok.encode(prompt, add_bos=True)
    c_ids = tok.encode(completion) + [tok.eos_id]
    ids = p_ids + c_ids
    if len(ids) > ctx + 1:
        return None
    return ids, len(p_ids), float(weight)


def encode_revision(tok, fmt, r: dict, ctx: int):
    sc = dict(r["scenario"])
    rev = sc.pop("_revision")
    prompt, completion = serialize_feedback_example(r["persona"], sc, rev["previous"], rev["feedback"], r["response"], fmt)
    p_ids = tok.encode(prompt, add_bos=True)
    ids = p_ids + tok.encode(completion) + [tok.eos_id]
    return None if len(ids) > ctx + 1 else (ids, len(p_ids), float(r.get("weight", 1.0)))


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_rows(rows: list[dict], tok, fmt, ctx, weight_key: str | None = None) -> tuple[list, int]:
    out, skipped = [], 0
    for r in rows:
        if r.get("example_type") == "revision":
            e = encode_revision(tok, fmt, r, ctx)
        else:
            e = encode_example(tok, fmt, r["persona"], r["scenario"], r["response"],
                               r.get(weight_key, 1.0) if weight_key else 1.0, ctx)
        if e is None:
            skipped += 1
        else:
            out.append(e)
    return out, skipped


def _count_by(rows, key):
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


class ReplayBuffer:
    def __init__(self, db: Database):
        self.db = db

    def stats(self) -> dict:
        rows = self.db.query("SELECT source, used_count>0 AS used, COUNT(*) AS n FROM training_examples "
                             "WHERE example_type='response' GROUP BY source, used_count>0")
        out = {"new": 0, "replay": 0, "by_source": {}}
        for r in rows:
            out["replay" if r["used"] else "new"] += r["n"]
            out["by_source"][r["source"]] = out["by_source"].get(r["source"], 0) + r["n"]
        return out

    def fetch(self, used: bool, limit: int = 5000, include_revisions: bool = False) -> list[dict]:
        cond = "used_count>0" if used else "used_count=0"
        types = "('response','revision')" if include_revisions else "('response')"
        return self.db.query(f"SELECT * FROM training_examples WHERE {cond} AND example_type IN {types} "
                             f"ORDER BY id DESC LIMIT ?", (limit,))

    def build_mixture(self, tok, fmt, ctx: int, anchor_rows: list[dict], new_ratio: float, replay_ratio: float,
                      seed: int = 0, anchor_limit: int = 3000, include_revisions: bool = False):
        new_rows = self.fetch(used=False, include_revisions=include_revisions)
        old_rows = self.fetch(used=True, include_revisions=include_revisions)
        rng = random.Random(seed)
        anchor = anchor_rows if len(anchor_rows) <= anchor_limit else rng.sample(anchor_rows, anchor_limit)
        new_enc, s1 = encode_rows(new_rows, tok, fmt, ctx, "weight")
        old_enc, s2 = encode_rows(old_rows, tok, fmt, ctx, "weight")
        anc_enc, s3 = encode_rows(anchor, tok, fmt, ctx)
        anchor_ratio = max(0.0, 1.0 - new_ratio - replay_ratio)
        mix = MixtureSource([("new", ExampleSource(new_enc), new_ratio),
                             ("replay", ExampleSource(old_enc), replay_ratio),
                             ("anchor", ExampleSource(anc_enc), anchor_ratio)])
        info = {
            "requested_ratios": {"new": new_ratio, "replay": replay_ratio, "anchor": round(anchor_ratio, 4)},
            "effective_ratios": {k: round(v, 4) for k, v in mix.effective_ratios.items()},
            "counts": {"new": len(new_enc), "replay": len(old_enc), "anchor": len(anc_enc)},
            "skipped_too_long": s1 + s2 + s3,
            "new_example_ids": [r["id"] for r in new_rows],
            "replay_example_ids_count": len(old_rows),
            "sources_new": _count_by(new_rows, "source"),
        }
        return mix, info

    def mark_used(self, example_ids: list[int], run_id: int) -> None:
        for i in range(0, len(example_ids), 500):
            chunk = example_ids[i:i + 500]
            self.db.execute(f"UPDATE training_examples SET used_count=used_count+1, last_used_run=? "
                            f"WHERE id IN ({','.join('?' for _ in chunk)})", [run_id, *chunk])
        self.db.execute("UPDATE trials SET training_status='used' WHERE id IN "
                        "(SELECT trial_id FROM training_examples WHERE last_used_run=?)", (run_id,))
