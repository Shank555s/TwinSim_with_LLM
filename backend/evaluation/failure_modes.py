"""SimFidelity failure-mode detectors (original four + two additions).

Codes and rules (all thresholds on the 0-10 scale):

exaggeration   a trait or political lean is expressed in the persona's direction but much more
               extremely than intended (|obs-5| - |int-5| >= 2.5), or >= 3 intensifiers
               (4 for expressive communicators).
blue_shift     observed political lean is >= 2.5 points further LEFT than intended.
red_shift      (added) observed political lean is >= 2.5 points further RIGHT than intended.
stereotype     the persona differs from a demographic archetype by >= 2.0 on some attribute,
               and the observed value sits closer to the archetype than to the persona
               (by >= 1.0) while being >= 2.0 away from the persona.
inconsistency  contradictory cues inside one response: explicit support AND opposition,
               both poles of the same trait, or strong left AND right cues without hedging.
degeneration   (added) empty/very short output, unreadable text (< 30% common English words),
               or looping repetition (a word trigram repeated >= 3 times, or a sentence
               repeated) - typical of small, under-trained models. A degenerate response is
               *unmeasurable*: it receives overall fidelity 0 and no other failure codes.
"""
from __future__ import annotations

import re
from collections import Counter

from .lexicons import ARCHETYPES, COMMON_WORDS
from .traits import TRAITS, Observation

FAILURE_LABELS = {
    "exaggeration": "Generative Exaggeration",
    "blue_shift": "Blue-Shift Bias",
    "stereotype": "Stereotype Projection",
    "inconsistency": "Temporal Inconsistency",
    "red_shift": "Red-Shift Bias",
    "degeneration": "Degenerate Output",
}
ORIGINAL_FAILURES = ["exaggeration", "blue_shift", "stereotype", "inconsistency"]


def detect_failures(persona: dict, intended: dict[str, float], obs: Observation, text: str,
                    expressive: bool) -> list[dict]:
    out: list[dict] = []

    # --- degeneration
    words = re.findall(r"[a-z0-9']+", text.lower())
    tri = Counter(zip(words, words[1:], words[2:]))
    sents = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    rep_sent = [s for s, c in Counter(sents).items() if c >= 2 and len(s.split()) >= 3]
    common = sum(1 for w in words if w in COMMON_WORDS) / max(1, len(words))
    if len(words) < 4:
        out.append({"code": "degeneration", "detail": f"response has only {len(words)} words"})
    elif common < 0.30:
        out.append({"code": "degeneration", "detail": f"unreadable text ({common:.0%} common English words)"})
    elif (tri and tri.most_common(1)[0][1] >= 3) or rep_sent:
        what = " ".join(tri.most_common(1)[0][0]) if tri and tri.most_common(1)[0][1] >= 3 else rep_sent[0]
        out.append({"code": "degeneration", "detail": f"repetition loop: '{what}'"})
    if out:
        return out  # unmeasurable: other detectors would only report noise

    # --- exaggeration
    for t in TRAITS + ["political"]:
        o = obs.political if t == "political" else obs.traits[t]
        i = intended[t]
        if (o - 5) * (i - 5) > 0 and abs(o - 5) - abs(i - 5) >= 2.5:
            out.append({"code": "exaggeration", "detail": f"{t}: intended {i:.1f}, expressed {o:.1f}"})
            break
    else:
        limit = 4 if expressive else 3
        if obs.intensifiers >= limit:
            out.append({"code": "exaggeration", "detail": f"{obs.intensifiers} intensifiers (absolutely, never, ...)"})

    # --- political shift
    d = obs.political - intended["political"]
    if d <= -2.5:
        out.append({"code": "blue_shift", "detail": f"political: intended {intended['political']:.1f}, expressed {obs.political:.1f} (more left)"})
    elif d >= 2.5:
        out.append({"code": "red_shift", "detail": f"political: intended {intended['political']:.1f}, expressed {obs.political:.1f} (more right)"})

    # --- stereotype projection
    for applies, arche in ARCHETYPES:
        try:
            if not applies(persona):
                continue
        except Exception:
            continue
        for attr, aval in arche.items():
            a10, i10 = aval / 10.0, intended.get(attr)
            if i10 is None or abs(a10 - i10) < 2.0:
                continue
            o = obs.political if attr == "political" else obs.traits.get(attr)
            if o is None:
                continue
            if abs(o - i10) >= 2.0 and (o - 5) * (a10 - 5) > 0 and abs(o - 5) >= 1.0 \
                    and abs(o - a10) + 1.0 <= abs(o - i10):
                out.append({"code": "stereotype",
                            "detail": f"{attr}: persona {i10:.1f}, demographic archetype {a10:.1f}, expressed {o:.1f}"})
                break
        if any(f["code"] == "stereotype" for f in out):
            break

    # --- inconsistency
    reasons = []
    if obs.stance_counts.get("support", 0) >= 1 and obs.stance_counts.get("oppose", 0) >= 1 and \
            re.search(r"\b(i support|agree|yes)\b", text.lower()) and re.search(r"\b(i oppose|disagree|no,)", text.lower()):
        reasons.append("both supports and opposes the proposal")
    for t in TRAITS:
        ev = obs.evidence[t]
        if len(ev["high"]) >= 1 and len(ev["low"]) >= 1 and abs(len(ev["high"]) - len(ev["low"])) <= 1 \
                and not set(ev["high"]) & set(ev["low"]):
            reasons.append(f"mixed {t} signals ({ev['high'][0]} / {ev['low'][0]})")
            break
    pe = obs.political_evidence
    if len(pe.get("left", [])) >= 2 and len(pe.get("right", [])) >= 2 and not pe.get("center"):
        reasons.append("strong left and right cues together")
    if reasons:
        out.append({"code": "inconsistency", "detail": "; ".join(reasons)})
    return out
