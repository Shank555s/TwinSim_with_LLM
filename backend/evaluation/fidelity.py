"""Rubric-based fidelity evaluator (no language model involved).

Per-axis fidelity (0-10):
    trait axis t      : 10 - |intended_t - observed_t|          intended_t = persona slider / 10
    political         : 10 - |intended_pol - observed_pol|
    style             : see evaluation/style.py

Overall fidelity (0-100):
    overall = 10 * sum_a (w_a * axis_a) / sum_a w_a
    default weights: each Big-5 trait 1.0, political 2.0, style 1.0  (configurable)

Validity gate: a response flagged `degeneration` (empty, unreadable or looping) cannot express
a persona, so all axes and the overall score are set to 0. Otherwise failure modes are
reported separately and do NOT change the score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from llm.config import EvaluatorConfig

from .failure_modes import FAILURE_LABELS, detect_failures
from .style import expected_style, score_style
from .traits import TRAITS, Observation, observe

EVALUATOR_VERSION = "rubric-lexicon-v1"
AXES = TRAITS + ["political", "style"]


@dataclass
class FidelityReport:
    axes: dict[str, float]  # 0-10 per axis
    overall: float  # 0-100
    intended: dict[str, float]  # 0-10
    observed: dict[str, float]  # 0-10
    drift: dict[str, float]  # observed - intended
    failures: list[dict]
    explanation: list[str]
    style_detail: dict
    evidence: dict
    evaluator: str = EVALUATOR_VERSION
    evaluator_kind: str = "rubric (non-LLM)"
    valid: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failure_codes"] = [f["code"] for f in self.failures]
        return d


def intended_profile(persona: dict) -> dict[str, float]:
    d = {t: float(persona.get(t, 50)) / 10.0 for t in TRAITS}
    d["political"] = float(persona.get("political", 50)) / 10.0
    return d


def _strength(v: float) -> str:
    return "strong" if v >= 8 else "moderate" if v >= 6 else "weak"


def evaluate(persona: dict, scenario_text: str, response: str, cfg: EvaluatorConfig | None = None) -> FidelityReport:
    cfg = cfg or EvaluatorConfig()
    if not isinstance(response, str):
        raise ValueError("response must be text")
    obs: Observation = observe(response)
    intended = intended_profile(persona)
    axes = {t: round(max(0.0, 10.0 - abs(intended[t] - obs.traits[t])), 2) for t in TRAITS}
    axes["political"] = round(max(0.0, 10.0 - abs(intended["political"] - obs.political)), 2)
    style_score, style_detail = score_style(persona.get("commStyle", ""), response)
    axes["style"] = round(style_score, 2)
    w = cfg.weights
    overall = 10.0 * sum(w.get(a, 1.0) * axes[a] for a in AXES) / sum(w.get(a, 1.0) for a in AXES)
    observed = {**{t: obs.traits[t] for t in TRAITS}, "political": obs.political}
    drift = {k: round(observed[k] - intended[k], 2) for k in observed}
    expressive = expected_style(persona.get("commStyle", "")).get("expressive", 0) > 0.2
    failures = detect_failures(persona, intended, obs, response, expressive)
    valid = not any(f["code"] == "degeneration" for f in failures)
    if not valid:
        axes = {a: 0.0 for a in AXES}
        overall = 0.0

    expl = [] if valid else ["validity gate: response is degenerate/unreadable, so fidelity is 0"]
    for a in (AXES if valid else []):
        line = f"{a}: {_strength(axes[a])} ({axes[a]:.1f}/10)"
        if a in TRAITS:
            ev = obs.evidence[a]
            cues = [f"+{c}" for c in ev["high"][:2]] + [f"-{c}" for c in ev["low"][:2]]
            line += f" - intended {intended[a]:.1f}, expressed {obs.traits[a]:.1f}" + (f" [{', '.join(cues)}]" if cues else " [no cues]")
        elif a == "political":
            pe = obs.political_evidence
            cues = [f"L:{c}" for c in pe["left"][:2]] + [f"R:{c}" for c in pe["right"][:2]] + [f"C:{c}" for c in pe["center"][:1]]
            line += f" - intended {intended[a]:.1f}, expressed {obs.political:.1f}" + (f" [{', '.join(cues)}]" if cues else " [no cues]")
        expl.append(line)
    for f in failures:
        expl.append(f"failure: {FAILURE_LABELS[f['code']]} - {f['detail']}")

    return FidelityReport(axes=axes, overall=round(overall, 2), intended=intended, observed=observed, drift=drift,
                          failures=failures, explanation=expl, style_detail=style_detail, valid=valid,
                          evidence={"traits": obs.evidence, "political": obs.political_evidence,
                                    "stance": obs.stance, "words": obs.words})
