"""Communication-style matching.

expected profile : the persona's free-text `commStyle` -> weights over five style families
observed profile : measurable surface features of the response -> affinity (0-1) per family
"""
from __future__ import annotations

import re

from .lexicons import ANALYTICAL_MARKERS, FORMAL_MARKERS, INFORMAL_MARKERS, STYLE_KEYWORDS, WARMTH_MARKERS

FAMILIES = ["terse", "analytical", "formal", "expressive", "warm"]
_CONTRACTION = re.compile(r"\b\w+'(d|m|s|t|re|ve|ll)\b")


def expected_style(comm_style: str) -> dict[str, float]:
    text = (comm_style or "").lower()
    raw = {f: sum(1 for kw in STYLE_KEYWORDS[f] if kw in text) for f in FAMILIES}
    total = sum(raw.values())
    if total == 0:
        return {f: 0.0 for f in FAMILIES}  # unknown style -> style axis is neutral (see score_style)
    return {f: raw[f] / total for f in FAMILIES}


def _count(patterns, text):
    return sum(len(re.findall(p, text)) for p in patterns)


def observed_style(text: str) -> dict[str, float]:
    low = text.lower()
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()] or [text]
    words = re.findall(r"[A-Za-z0-9']+", text)
    avg_len = len(words) / max(1, len(sents))
    contractions = len(_CONTRACTION.findall(low))
    exclaims = text.count("!")

    def clip(x):
        return max(0.0, min(1.0, x))

    return {
        "terse": clip((13.0 - avg_len) / 7.0) * (1.0 if len(words) <= 45 else 0.4),
        "analytical": clip(_count(ANALYTICAL_MARKERS, low) / max(1.5, 0.5 * len(sents))),
        "formal": 0.5 * clip(1.0 - contractions / 3.0) + 0.5 * clip(_count(FORMAL_MARKERS, low) / 2.0),
        "expressive": clip((exclaims + _count(INFORMAL_MARKERS, low)) / 2.5),
        "warm": clip(_count(WARMTH_MARKERS, low) / 2.0),
    }


def score_style(comm_style: str, text: str) -> tuple[float, dict]:
    """Style fidelity 0-10.

    score = 10 * (1 - 0.7 * sum_f e_f * (1 - a_f) - 0.3 * mean_{f not expected} a_f)
    where e_f are expected family weights and a_f observed affinities.
    """
    e = expected_style(comm_style)
    a = observed_style(text)
    if sum(e.values()) == 0:
        return 5.0, {"expected": e, "observed": a, "note": "communication style not recognised; neutral score"}
    miss = sum(e[f] * (1 - a[f]) for f in FAMILIES)
    others = [a[f] for f in FAMILIES if e[f] == 0]
    spill = sum(others) / len(others) if others else 0.0
    score = 10.0 * max(0.0, min(1.0, 1 - 0.7 * miss - 0.3 * spill))
    return round(score, 3), {"expected": e, "observed": {k: round(v, 3) for k, v in a.items()}}
