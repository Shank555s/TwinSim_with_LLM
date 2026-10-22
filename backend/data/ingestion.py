"""Dataset ingestion (sections 8 & 60). Drop files into these folders before bootstrapping:

  data/corpus/*.txt            plain text, one document per blank-line-separated block
  data/corpus/*.jsonl          {"text": "..."} per line                 -> stage-1 pre-training
  data/persona_data/*.jsonl    {"persona": {...}, "scenario": {"domain": "...", "text": "..."},
                                "response": "...", "source": "..."}       -> stage-2 persona training

Persona objects use the TwinSim schema (see twin/personas.py). Every record is validated;
invalid records are skipped and counted, never silently "fixed".

Nothing here downloads data. Suitable public corpora (check each licence yourself):
  * TinyStories (Eldan & Li, 2023): simple English for stage-1 pre-training
  * Twin-2K-500 (Columbia DAPLab, CC BY 4.0), real persona survey answers; requires a
    conversion script mapping its question blocks to the persona schema above.
The paths can be overridden with TWINSIM_CORPUS_DIR / TWINSIM_PERSONA_DATA_DIR.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from twin.personas import PersonaError, validate_persona
from twin.scenarios import scenarios_by_split, validate_scenario

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = Path(os.environ.get("TWINSIM_CORPUS_DIR", DATA_DIR / "corpus"))
PERSONA_DIR = Path(os.environ.get("TWINSIM_PERSONA_DATA_DIR", DATA_DIR / "persona_data"))
MAX_DOC_CHARS = 20_000


def load_extra_pretraining_texts(directory: Path | None = None, max_docs: int = 200_000) -> list[str]:
    d = directory or CORPUS_DIR
    docs: list[str] = []
    if not d.exists():
        return docs
    for f in sorted(d.glob("*.txt")):
        for block in f.read_text(errors="replace").split("\n\n"):
            block = block.strip()
            if len(block) > 20:
                docs.append(block[:MAX_DOC_CHARS])
    for f in sorted(d.glob("*.jsonl")):
        with open(f, errors="replace") as fh:
            for line in fh:
                try:
                    t = json.loads(line).get("text", "")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if isinstance(t, str) and len(t.strip()) > 20:
                    docs.append(t.strip()[:MAX_DOC_CHARS])
    return docs[:max_docs]


def load_extra_persona_examples(directory: Path | None = None) -> list[dict]:
    rows, _ = load_persona_jsonl_dir(directory or PERSONA_DIR)
    return rows


def load_persona_jsonl_dir(directory: Path) -> tuple[list[dict], dict]:
    rows, report = [], {"files": 0, "ok": 0, "skipped": 0, "errors": []}
    held_out = {s["text"] for s in scenarios_by_split("benchmark")}
    if not directory.exists():
        return rows, report
    for f in sorted(directory.glob("*.jsonl")):
        report["files"] += 1
        with open(f, errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                try:
                    r = json.loads(line)
                    p = validate_persona(r["persona"])
                    s = validate_scenario(r["scenario"])
                    resp = str(r["response"]).strip()
                    if not resp:
                        raise ValueError("empty response")
                    if s["text"] in held_out:
                        raise ValueError("uses a held-out benchmark scenario")
                    rows.append({"persona": p, "scenario": s, "response": resp,
                                 "source": str(r.get("source", f.name))})
                    report["ok"] += 1
                except (json.JSONDecodeError, KeyError, ValueError, PersonaError) as e:
                    report["skipped"] += 1
                    if len(report["errors"]) < 20:
                        report["errors"].append(f"{f.name}:{n}: {e}")
    return rows, report


def ingestion_status() -> dict:
    docs = load_extra_pretraining_texts()
    _, rep = load_persona_jsonl_dir(PERSONA_DIR)
    return {"corpus_dir": str(CORPUS_DIR), "corpus_documents": len(docs), "persona_dir": str(PERSONA_DIR),
            "persona_examples": rep}
