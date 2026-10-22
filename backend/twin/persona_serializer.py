"""Persona + scenario -> model input text (section 9). The format is configurable but the same
format object must be used for training and inference (it is stored with each checkpoint)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .personas import TRAITS


def level(v: float) -> str:
    return "low" if v < 40 else "high" if v > 60 else "mid"


def lean(v: float) -> str:
    return "left" if v < 40 else "right" if v > 60 else "center"


@dataclass
class SerializationFormat:
    persona_fields: list = field(default_factory=lambda: [
        "age", "gender", "occupation", "location", "traits", "politics", "environment",
        "economic", "communication", "notes"])
    trait_format: str = "both"  # "level" (low/mid/high) | "decimal" (0.72) | "both" ("high 0.72")
    include_domain: bool = True
    separator: str = "\n"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SerializationFormat":
        return cls(**(d or {}))


def _fmt(v: float, kind: str, fmt: SerializationFormat) -> str:
    word = lean(v) if kind == "politics" else level(v)
    dec = f"{v / 100:.2f}"
    return {"level": word, "decimal": dec}.get(fmt.trait_format, f"{word} {dec}")


def serialize_persona(p: dict, fmt: SerializationFormat | None = None) -> str:
    fmt = fmt or SerializationFormat()
    lines = []
    for f in fmt.persona_fields:
        if f == "traits":
            lines.extend(f"{t}={_fmt(float(p.get(t, 50)), 'trait', fmt)}" for t in TRAITS)
        elif f == "politics":
            lines.append(f"politics={_fmt(float(p.get('political', 50)), 'politics', fmt)}")
        elif f == "environment":
            lines.append(f"environment={_fmt(float(p.get('environmental', 50)), 'trait', fmt)}")
        elif f == "communication":
            if p.get("commStyle"):
                lines.append(f"communication={p['commStyle']}")
        else:
            v = p.get(f)
            if v not in (None, ""):
                lines.append(f"{f}={v}")
    return "<persona>" + fmt.separator + fmt.separator.join(lines) + fmt.separator + "</persona>"


def serialize_prompt(p: dict, scenario: dict, fmt: SerializationFormat | None = None) -> str:
    """Everything up to and including the <response> tag; the model writes the rest."""
    fmt = fmt or SerializationFormat()
    sc = []
    if fmt.include_domain and scenario.get("domain"):
        sc.append(f"domain={scenario['domain']}")
    sc.append(scenario["text"])
    return (serialize_persona(p, fmt) + fmt.separator + "<scenario>" + fmt.separator +
            fmt.separator.join(sc) + fmt.separator + "</scenario>" + fmt.separator + "<response>" + fmt.separator)


def serialize_example(p: dict, scenario: dict, response: str, fmt: SerializationFormat | None = None) -> tuple[str, str]:
    """Returns (prompt, completion) where completion ends with the closing tag."""
    fmt = fmt or SerializationFormat()
    return serialize_prompt(p, scenario, fmt), response.strip() + fmt.separator + "</response>"


def serialize_feedback_example(p: dict, scenario: dict, previous: str, feedback: dict, target: str,
                               fmt: SerializationFormat | None = None) -> tuple[str, str]:
    """Critique -> revision format (section 13). Used only when `use_revision_examples` is enabled."""
    fmt = fmt or SerializationFormat()
    fb_lines = [f"overall={feedback.get('overall', 0):.1f}"]
    for k, v in (feedback.get("axes") or {}).items():
        fb_lines.append(f"{k}={v:.1f}")
    if feedback.get("failures"):
        fb_lines.append("failures=" + ",".join(feedback["failures"]))
    prompt = (serialize_persona(p, fmt) + fmt.separator + "<scenario>" + fmt.separator + scenario["text"] +
              fmt.separator + "</scenario>" + fmt.separator + "<response>" + fmt.separator + previous.strip() +
              fmt.separator + "</response>" + fmt.separator + "<feedback>" + fmt.separator +
              fmt.separator.join(fb_lines) + fmt.separator + "</feedback>" + fmt.separator + "<response>" + fmt.separator)
    return prompt, target.strip() + fmt.separator + "</response>"
