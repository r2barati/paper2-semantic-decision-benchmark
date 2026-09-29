"""v5 controller extensions.

WorstCaseBaseStock is deliberately evidence-insensitive. It is a safety
control, not a third belief-dependent controller.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.events import P5_DEMAND_MEAN, P5_NORMAL_LEAD_TIME, REGIME_PARAMS, Regime


class WorstCaseBaseStock:
    """Evidence-insensitive maximum regime target."""

    def __init__(self, safety: float = 1.65):
        self.safety = float(safety)

    def _target(self, regime: Regime) -> float:
        params = REGIME_PARAMS[regime]
        mu = P5_DEMAND_MEAN * params["demand_multiplier"]
        lead = P5_NORMAL_LEAD_TIME + params["lead_time_increase"]
        return mu * lead + self.safety * math.sqrt(max(mu * lead, 1.0))

    @property
    def target(self) -> float:
        return max(self._target(r) for r in Regime)

    def decide(self, state) -> float:
        inventory_position = float(state.on_hand) + float(state.pipeline)
        return max(0.0, self.target - inventory_position)
