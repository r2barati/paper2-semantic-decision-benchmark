"""No-text and degraded-text control conditions.

The headline result of this benchmark is that reading a warning improves
downstream operational value.  That claim is only supported if reading text
beats *competitive text-free operation* -- not merely the one particular
NoInfo belief that happens to be the benchmark's prior.

The September 2026 audit showed the distinction matters: a controller that
simply assumes the event is certain, without reading anything, outperformed
three of four interpreters in the multi-echelon confirmation and all four in
the capacity-drop transfer.  A large part of what looked like "semantic value"
was really the base-stock controller being poorly positioned under the prior.

This module supplies the missing arms:

``constant``
    A fixed event probability, reading no text.  The endpoints (0, 1), the
    benchmark prior, and the uninformative 0.5 midpoint are the diagnostic
    controls.
``tuned_notext``
    A constant belief whose probability is *selected on separate development
    seeds* and then evaluated on the locked confirmation seeds.  This is the
    honest text-free competitor: it is allowed to be tuned, but never on the
    evaluation data.
``shuffled_text``
    The real interpreter applied to a deterministically permuted assignment of
    texts to worlds.  This destroys the text-world relationship while keeping
    the marginal distribution of beliefs identical, isolating how much of the
    effect comes from *reading the right text* rather than from the belief
    distribution's shape.
``argmax``
    The real interpreter's hard label as a degenerate belief.  Comparing it
    with the same interpreter's soft probabilities isolates probability
    quality from label quality.

All of these reuse the existing controllers and simulators unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from src.events import Regime

# Development seeds are disjoint from every confirmation seed range used by
# the frozen experiments (8B: 3100-3129, 9A: 4000-4029, 9B: 4100-4109).
DEV_SEEDS_8B = list(range(3200, 3230))
DEV_SEEDS_9A = list(range(4200, 4230))

# The probability grid searched by the tuned no-text control.
TUNING_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

CONTROL_CONSTANT = "constant"
CONTROL_TUNED = "tuned_notext"
CONTROL_SHUFFLED = "shuffled_text"
CONTROL_ARGMAX = "argmax"


def constant_belief(p_event: float, event_regime: Regime) -> dict[str, float]:
    """A fixed two-class belief that reads no text at all."""
    p = float(min(1.0, max(0.0, p_event)))
    return {"normal": 1.0 - p, event_regime.value: p}


def argmax_belief(probs: Mapping[str, float], event_regime: Regime) -> dict[str, float]:
    """Collapse a soft belief onto its most likely class."""
    p_event = float(probs.get(event_regime.value, 0.0))
    return constant_belief(1.0 if p_event >= 0.5 else 0.0, event_regime)


def shuffled_text_assignment(
    template_ids: Sequence[str], seed: int = 20260907
) -> dict[str, str]:
    """Deterministic permutation mapping each template to another's text slot.

    A derangement is attempted so no template keeps its own text; with fewer
    than two templates the identity is returned.
    """
    ids = list(template_ids)
    if len(ids) < 2:
        return {t: t for t in ids}
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = rng.permutation(len(ids))
        if all(perm[i] != i for i in range(len(ids))):
            break
    return {ids[i]: ids[int(perm[i])] for i in range(len(ids))}


@dataclass
class TunedControlResult:
    """Outcome of selecting a no-text operating point on development seeds."""

    p_event: float
    dev_grid: dict
    dev_seeds: list
    event_regime: str
    balanced: bool = True

    def belief(self) -> dict[str, float]:
        return constant_belief(self.p_event, Regime(self.event_regime))

    def as_dict(self) -> dict:
        return {
            "control": CONTROL_TUNED,
            "p_event": self.p_event,
            "dev_seeds": list(self.dev_seeds),
            "dev_grid": {str(k): v for k, v in self.dev_grid.items()},
            "event_regime": self.event_regime,
            "selection_estimand": "balanced" if self.balanced else "event_only",
            "selection_data": "development seeds, disjoint from confirmation seeds",
        }


def tune_notext_control(
    episode_runner: Callable,
    event_regime: Regime,
    dev_seeds: Sequence[int],
    grid: Sequence[float] = TUNING_GRID,
    balanced: bool = True,
    runner_kwargs: Optional[dict] = None,
) -> TunedControlResult:
    """Pick the best constant event probability on development seeds only.

    ``episode_runner`` must accept ``seed``, ``regime``, ``sensor`` and
    ``regime_probabilities`` and return ``(result, history)`` where ``result``
    has ``.total_reward``.  The evaluation regimes are the balanced pair
    (normal and the event) unless ``balanced`` is False.
    """
    runner_kwargs = dict(runner_kwargs or {})
    regimes = [Regime.NORMAL, event_regime] if balanced else [event_regime]

    dev_grid = {}
    for p in grid:
        belief = constant_belief(p, event_regime)
        regime_means = []
        for regime in regimes:
            rewards = [
                episode_runner(
                    seed=int(seed), regime=regime, sensor=f"{CONTROL_TUNED}_dev",
                    regime_probabilities=belief, **runner_kwargs,
                )[0].total_reward
                for seed in dev_seeds
            ]
            regime_means.append(float(np.mean(rewards)))
        dev_grid[float(p)] = float(np.mean(regime_means))

    best_p = max(dev_grid, key=lambda k: dev_grid[k])
    return TunedControlResult(
        p_event=float(best_p),
        dev_grid=dev_grid,
        dev_seeds=list(dev_seeds),
        event_regime=event_regime.value,
        balanced=balanced,
    )


def constant_control_suite(
    prior_p_event: float, event_regime: Regime
) -> dict[str, dict[str, float]]:
    """The standard constant-belief diagnostic endpoints."""
    return {
        "constant_p0.0": constant_belief(0.0, event_regime),
        "constant_prior": constant_belief(prior_p_event, event_regime),
        "constant_p0.5": constant_belief(0.5, event_regime),
        "constant_p1.0": constant_belief(1.0, event_regime),
    }
