"""Second-controller robustness (LABELED EXTENSION, Round-C escalation).

A belief-adaptive base-stock policy for the CONTROLLED single-item
environment — a different fixed policy class from CausalOptimizer's
receding-horizon LP, taking the SAME frozen beliefs as input. Answers the
top repeated reviewer demand: does C1 harm survive a different controller,
or is it controller-specific? Frozen code untouched (new file); same paired
seeds; same episodes protocol; extension-labeled, outside the Holm family.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.events import (P5_DEMAND_MEAN, P5_NORMAL_LEAD_TIME, REGIME_PARAMS,
                        Regime)


class BeliefBaseStock:
    """Order-up-to policy with belief-weighted demand/lead time.

    effective_mu = sum_z b(z) * base_mu * mult(z)
    effective_lt = sum_z b(z) * (normal_lt + lt_increase(z))
    target = effective_mu * effective_lt
             + 1.65 * sqrt(effective_lt * effective_mu)
    order = max(0, target - (on_hand + pipeline))

    Same interface as CausalOptimizer.decide(state). No learning, no tuning:
    the 1.65 safety factor mirrors the heuristic family used elsewhere.
    """

    def __init__(self, regime_probabilities, demand_mean=P5_DEMAND_MEAN,
                 normal_lt=P5_NORMAL_LEAD_TIME, safety=1.65):
        self.probs = dict(regime_probabilities)
        total = sum(self.probs.values()) or 1.0
        self.probs = {k: v / total for k, v in self.probs.items()}
        self.demand_mean = demand_mean
        self.normal_lt = normal_lt
        self.safety = safety

    def _effective(self):
        mu = sum(self.probs.get(r.value if isinstance(r, Regime) else r, 0.0)
                 * self.demand_mean * REGIME_PARAMS[r]["demand_multiplier"]
                 for r in (Regime.NORMAL, Regime.SUPPLIER_DELAY,
                           Regime.DEMAND_SURGE))
        lt = sum(self.probs.get(r.value, 0.0)
                 * (self.normal_lt + REGIME_PARAMS[r]["lead_time_increase"])
                 for r in (Regime.NORMAL, Regime.SUPPLIER_DELAY,
                           Regime.DEMAND_SURGE))
        return mu, max(lt, 1.0)

    def decide(self, state) -> float:
        mu, lt = self._effective()
        target = mu * lt + self.safety * math.sqrt(max(lt, 1) * max(mu, 1e-9))
        inv_pos = float(state.on_hand) + float(state.pipeline)
        return max(0.0, target - inv_pos)
