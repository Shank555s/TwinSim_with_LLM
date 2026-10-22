"""Observed-behaviour extraction: text -> estimated Big-5 levels, political lean, stance, intensity.

Scale: every observed value is on 0-10 (5 = no evidence either way), matching the persona's
0-100 sliders divided by 10. For political lean 0 = left, 10 = right (same direction as the UI).
Evidence is counted once per sentence, so repeating a cue word inside one sentence does not
inflate a score.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .lexicons import INTENSIFIERS, NEGATIONS, POLITICAL_CUES, STANCE_CUES, TRAIT_CUES

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
_WORD = re.compile(r"[a-z0-9']+")


def _compile(patterns):
    return [re.compile(rf"(?<![a-z]){p}(?![a-z])") for p in patterns]


_TRAIT_RE = {t: {pole: _compile(ps) for pole, ps in poles.items()} for t, poles in TRAIT_CUES.items()}
_POL_RE = {k: _compile(v) for k, v in POLITICAL_CUES.items()}
_STANCE_RE = {k: _compile(v) for k, v in STANCE_CUES.items()}
_INT_RE = [re.compile(p) for p in INTENSIFIERS]


def _matches(text: str, groups: dict[str, list[re.Pattern]]):
    """All non-overlapping cue matches (longest wins) -> list of (start, end, group, phrase)."""
    found = []
    for g, pats in groups.items():
        for pat in pats:
            for m in pat.finditer(text):
                if m.end() > m.start():
                    found.append((m.start(), m.end(), g, m.group(0)))
    found.sort(key=lambda x: -(x[1] - x[0]))
    kept, taken = [], []
    for s, e, g, ph in found:
        if any(s < te and ts < e for ts, te in taken):
            continue
        kept.append((s, e, g, ph))
        taken.append((s, e))
    return sorted(kept)


def _negated(text: str, start: int, phrase: str) -> bool:
    if phrase.split()[0] in NEGATIONS or phrase.startswith(("not ", "don", "doesn", "no ")):
        return False  # the cue itself already encodes the negation
    before = _WORD.findall(text[max(0, start - 30):start])[-3:]
    return any(w in NEGATIONS or w.endswith("n't") for w in before)


def squash(evidence: float, k: float) -> float:
    return 5.0 + 5.0 * math.tanh(k * evidence)


@dataclass
class Observation:
    traits: dict[str, float]
    political: float
    evidence: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    political_evidence: dict[str, list[str]] = field(default_factory=dict)
    stance: str = "neutral"
    stance_counts: dict[str, int] = field(default_factory=dict)
    intensifiers: int = 0
    words: int = 0
    sentences: int = 0

    def vector(self) -> list[float]:
        return [self.traits[t] for t in TRAITS] + [self.political]


def _sentence_index(text: str):
    """Map a character offset to its sentence number (evidence is counted once per sentence)."""
    bounds = [m.end() for m in re.finditer(r"[.!?]+(\s|$)", text)]
    return lambda pos: sum(1 for b in bounds if b <= pos)


def _per_sentence(items):
    """items: list of (sentence_idx, group, phrase) -> {group: count of sentences containing it}"""
    seen = {(i, g) for i, g, _ in items}
    out: dict[str, int] = {}
    for _, g in seen:
        out[g] = out.get(g, 0) + 1
    return out


def observe(text: str) -> Observation:
    low = text.lower()
    sent_of = _sentence_index(low)
    traits, evidence = {}, {}
    for t in TRAITS:
        ev = {"high": [], "low": []}
        items = []
        for s, e, pole, ph in _matches(low, _TRAIT_RE[t]):
            if _negated(low, s, ph):
                pole = "low" if pole == "high" else "high"
                ph = "not " + ph
            ev[pole].append(ph)
            items.append((sent_of(s), pole, ph))
        evidence[t] = ev
        n = _per_sentence(items)
        traits[t] = round(squash(n.get("high", 0) - n.get("low", 0), 0.55), 3)
    pol = {"left": [], "right": [], "center": []}
    items = []
    for s, e, g, ph in _matches(low, _POL_RE):
        if not _negated(low, s, ph):
            pol[g].append(ph)
            items.append((sent_of(s), g, ph))
    n = _per_sentence(items)
    raw = squash(n.get("right", 0) - n.get("left", 0), 0.4)
    political = 5.0 + (raw - 5.0) / (1.0 + 0.6 * n.get("center", 0))
    st = {k: len(_matches(low, {k: v})) for k, v in _STANCE_RE.items()}
    stance = "support" if st["support"] > st["oppose"] else "oppose" if st["oppose"] > st["support"] else "neutral"
    sentences = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))
    return Observation(
        traits=traits, political=round(political, 3), evidence=evidence, political_evidence=pol,
        stance=stance, stance_counts=st, intensifiers=sum(len(p.findall(low)) for p in _INT_RE),
        words=len(_WORD.findall(low)), sentences=sentences,
    )
