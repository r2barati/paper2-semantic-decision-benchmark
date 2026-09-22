"""Frozen controller+simulator replay (read-only reuse of src/env.py + src/experiment_phase5.py).

Belief (per warning x arm) is replayed across all paired seeds. Records the FULL
action sequence per episode for the pathwise dead-zone test (Correction 3):
same seed + equal actions at every t => identical trajectory/profit by induction
(demand draws are seed-fixed and belief-independent).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.env import CausalOptimizer, InventoryEnv  # noqa: E402
from src.events import P5_DEMAND_MEAN, P5_HORIZON, P5_WARNING_TIME, Regime  # noqa: E402
from src.interpreter import RegimeInterpretation, no_info_regime_belief  # noqa: E402
from src.metrics import compute_episode_metrics  # noqa: E402


def belief_to_interp(b: dict) -> RegimeInterpretation:
    return RegimeInterpretation(
        regime_probabilities={"normal": b["normal"], "supplier_delay": b["supplier_delay"],
                              "demand_surge": b["demand_surge"]})


def run_episode(belief: dict, regime: Regime, seed: int) -> dict:
    env = InventoryEnv(seed=seed, regime=regime)
    env.reset()
    interp = belief_to_interp(belief).normalized()
    probs = interp.regime_probabilities
    prior = no_info_regime_belief().normalized().regime_probabilities
    opt = CausalOptimizer(seed=seed, regime_probabilities=dict(prior),
                          horizon=P5_HORIZON, initial_inventory=30,
                          demand_mean=P5_DEMAND_MEAN)
    released = P5_WARNING_TIME <= 0
    if released:
        opt.set_regime_probabilities(probs)
    actions: list[float] = []
    for t in range(P5_HORIZON):
        if not released and t >= P5_WARNING_TIME:
            released = True
            opt.set_regime_probabilities(probs)
        a = float(opt.decide(env._state))
        actions.append(a)
        env.step(a)
    # NOTE: env keeps internal history; recompute metrics from a fresh replay is
    # unnecessary — run via experiment_phase5 helper in full runner. Here we
    # re-run identically to capture history + actions together.
    return {"seed": seed, "actions": actions}


def run_episode_with_history(belief: dict, regime: Regime, seed: int) -> tuple[dict, list]:
    from src.experiment_phase5 import _run_p5_episode

    interp = belief_to_interp(belief)
    result, history = _run_p5_episode(
        regime=regime, regime_interp=interp, seed=seed, sensor="reasoning",
        controller="optimizer", template_id="", ambiguity_level="",
        warning_time=P5_WARNING_TIME)
    # Recover per-period actions from history (order_quantity field).
    actions = [float(h.order_quantity) for h in history]
    row = {"seed": seed, "total_profit": result.total_profit, "fill_rate": result.fill_rate,
           "actions": actions}
    return row, history


def pathwise_equal(a: list[float], b: list[float], tol: float = 1e-9) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))
