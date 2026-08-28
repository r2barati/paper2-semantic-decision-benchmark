"""Decision policies for the inventory environment.

Policies receive the current inventory state and return an order quantity.
Two policies are provided:

    Expert A — Base-stock heuristic (conservative, normal-lead-time tuned)
    Expert B — Disruption-aware (uses interpreted parameters to set target)

The disruption-aware policy is parameterized so that different semantic
interpretations produce different operational actions. This is essential
for the experiment: interpretation quality must be capable of affecting
downstream decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.env import InventoryState, DEMAND_MEAN, HOLDING_COST_PER_UNIT, ORDERING_VARIABLE_COST


def _safety_stock(lead_time: int, z: float = 1.65) -> float:
    """Approximate safety stock using normal approximation to Poisson.

    safety_stock = z * sqrt(L * lambda)
    """
    return z * math.sqrt(lead_time * DEMAND_MEAN)


@dataclass
class BaseStockPolicy:
    """Expert A: conservative base-stock heuristic.

    Orders up to a target inventory position = (mean demand over lead time)
    + safety stock.  Tuned for normal lead time.
    """

    lead_time: int = 2
    z: float = 1.65  # ~95% service level

    @property
    def target(self) -> float:
        return DEMAND_MEAN * self.lead_time + _safety_stock(self.lead_time, self.z)

    def __call__(self, state: InventoryState) -> float:
        """Order up to target from current inventory position."""
        inv_pos = state.on_hand + state.pipeline
        order = max(0.0, self.target - inv_pos)
        return math.ceil(order)


@dataclass
class DisruptionAwarePolicy:
    """Expert B: parameterized disruption-aware policy.

    Uses the *interpreted* lead time increase to set the inventory target.
    Different semantic interpretations produce different target levels,
    allowing us to measure how interpretation quality affects decisions.

    estimated_lt_increase: how much longer the lead time is expected to be.
    effective_duration: how many periods the adapted policy remains active.
    """

    normal_lead_time: int = 2
    z: float = 1.65

    def target(self, estimated_lt_increase: int) -> float:
        """Compute target inventory position given estimated LT increase."""
        effective_lt = self.normal_lead_time + estimated_lt_increase
        return DEMAND_MEAN * effective_lt + _safety_stock(effective_lt, self.z)

    def __call__(
        self,
        state: InventoryState,
        estimated_lt_increase: int = 3,
    ) -> float:
        """Order up to disruption-adjusted target using interpreted parameters."""
        inv_pos = state.on_hand + state.pipeline
        order = max(0.0, self.target(estimated_lt_increase) - inv_pos)
        return math.ceil(order)
