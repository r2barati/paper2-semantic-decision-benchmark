"""Paired contrasts + crossed warning x seed bootstrap + mechanism cells."""

from __future__ import annotations

import numpy as np

HEADLINE = [("A1", "A0"), ("A2", "A1"), ("R0", "A2"), ("A3", "R0")]
SECONDARY = [("A2", "A0"), ("A3", "A0"), ("A3", "A2")]


def paired_stats(x: np.ndarray, seed: int = 42, B: int = 5000) -> dict:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(x, dtype=float)
    boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(B)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    sd = diffs.std(ddof=1) if len(diffs) > 1 else 0.0
    return {"mean": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "win_frac": float((diffs > 0).mean()), "cohens_d": float(diffs.mean() / sd) if sd else 0.0,
            "n": int(len(diffs))}


def crossed_bootstrap(delta: np.ndarray, seed: int = 42, B: int = 5000) -> dict:
    """delta: (n_warnings x n_seeds) matrix; resample warnings and seeds independently."""
    rng = np.random.default_rng(seed)
    W, S = delta.shape
    boots = np.empty(B)
    for b in range(B):
        wi = rng.integers(0, W, W)
        si = rng.integers(0, S, S)
        boots[b] = delta[np.ix_(wi, si)].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(delta.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def mechanism_cell(d_brier: float, action_changed: bool, d_profit: float) -> str:
    if d_brier < 0 and action_changed and d_profit > 0:
        return "sem_up_act_change_util_up"
    if d_brier < 0 and not action_changed:
        return "sem_up_no_act_change"
    if d_brier < 0 and action_changed and d_profit <= 0:
        return "sem_up_harmful_act"
    if d_brier >= 0 and not action_changed:
        return "sem_down_unchanged"
    return "sem_down_harmful"
