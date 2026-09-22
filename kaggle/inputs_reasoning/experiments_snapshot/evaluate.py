"""Semantic + operational metrics (reuse frozen formulas; new code lives in-track)."""

from __future__ import annotations

import math


def brier(probs: dict, true_regime: str) -> float:
    onehot = {k: 1.0 if k == true_regime else 0.0 for k in probs}
    return sum((probs[k] - onehot[k]) ** 2 for k in probs) / len(probs)


def logloss(probs: dict, true_regime: str, eps: float = 1e-9) -> float:
    return -math.log(max(probs[true_regime], eps))


def accuracy(probs: dict, true_regime: str) -> float:
    return 1.0 if max(probs, key=probs.get) == true_regime else 0.0


def entropy(probs: dict) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in probs.values())


def ece_10bin(confidences: list[float], corrects: list[float]) -> float:
    n = len(confidences)
    if not n:
        return 0.0
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        idx = [i for i, c in enumerate(confidences) if (c > lo or (b == 0 and c == 0.0)) and c <= hi]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(conf - acc)
    return ece
