"""Deterministic entropy selective-consumption baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping


REGIMES = ("normal", "supplier_delay", "demand_surge")


def _validate(belief: Mapping[str, float]) -> tuple[float, float, float]:
    values = tuple(float(belief[r]) for r in REGIMES)
    if any(v < 0 or v > 1 for v in values):
        raise ValueError("belief probabilities must be in [0,1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
        raise ValueError("belief probabilities must sum to one")
    return values


def normalized_entropy(belief: Mapping[str, float]) -> float:
    values = _validate(belief)
    entropy = -sum(v * math.log(v) for v in values if v > 0)
    return entropy / math.log(3.0)


def entropy_weight(belief: Mapping[str, float]) -> float:
    return 1.0 - normalized_entropy(belief)


def shrink_to_prior(
    belief: Mapping[str, float],
    prior: Mapping[str, float],
) -> dict[str, float]:
    _validate(belief)
    _validate(prior)
    c = entropy_weight(belief)
    out = {r: c * float(belief[r]) + (1.0 - c) * float(prior[r]) for r in REGIMES}
    total = sum(out.values())
    return {r: out[r] / total for r in REGIMES}
