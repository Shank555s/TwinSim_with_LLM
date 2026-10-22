"""Twin Stability and Consistency over repeated runs of the same persona x scenario (section 20).

stability   = 100 * (1 - mean_axis_std / 5)
              axes: 5 observed traits + observed political lean + 5 style affinities (scaled 0-10).
              5 is the largest standard deviation possible on a 0-10 scale, so 100 = identical
              behaviour every run, 0 = maximally erratic.
consistency = 100 * mean(stance agreement, political-bucket agreement) * (1 - inconsistency rate)
              agreement = share of runs equal to the most common value; inconsistency rate =
              share of runs flagged with the `inconsistency` failure mode.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .style import FAMILIES, observed_style
from .traits import observe


def _bucket(pol: float) -> str:
    return "left" if pol < 4 else "right" if pol > 6 else "center"


def stability_metrics(responses: list[str], inconsistency_flags: list[bool] | None = None) -> dict:
    if len(responses) < 2:
        return {"stability": None, "consistency": None, "n": len(responses)}
    vecs, stances, buckets = [], [], []
    for r in responses:
        o = observe(r)
        st = observed_style(r)
        vecs.append(o.vector() + [10 * st[f] for f in FAMILIES])
        stances.append(o.stance)
        buckets.append(_bucket(o.political))
    std = np.asarray(vecs).std(axis=0).mean()
    stability = 100.0 * max(0.0, 1.0 - std / 5.0)
    agree = lambda xs: Counter(xs).most_common(1)[0][1] / len(xs)  # noqa: E731
    inc = (sum(inconsistency_flags) / len(inconsistency_flags)) if inconsistency_flags else 0.0
    consistency = 100.0 * (agree(stances) + agree(buckets)) / 2 * (1 - inc)
    return {"stability": round(float(stability), 2), "consistency": round(float(consistency), 2), "n": len(responses),
            "mean_axis_std": round(float(std), 3)}
