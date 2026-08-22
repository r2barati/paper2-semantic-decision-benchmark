"""Metrics for evaluating inventory performance across conditions."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.env import InventoryState, DEMAND_MEAN, HORIZON


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for one complete episode (one seed, one condition, one template)."""

    seed: int
    condition: str
    template_id: str = ""
    ambiguity_level: str = ""
    total_profit: float = 0.0
    total_demand: float = 0.0
    total_sales: float = 0.0
    total_lost_sales: float = 0.0
    total_holding: float = 0.0
    periods_stockout: int = 0
    periods: int = 0
    avg_inventory: float = 0.0
    fill_rate: float = 0.0

    # Semantic extraction errors (populated if applicable)
    predicted_event_type: str = ""
    predicted_probability: float = 0.0
    predicted_lt_increase: int = 0
    predicted_duration: int = 0
    event_type_correct: Optional[bool] = None
    lead_time_error: Optional[int] = None
    duration_error: Optional[int] = None


def compute_episode_metrics(
    history: list[InventoryState],
    seed: int,
    condition: str,
    template_id: str = "",
    ambiguity_level: str = "",
    predicted_event_type: str = "",
    predicted_probability: float = 0.0,
    predicted_lt_increase: int = 0,
    predicted_duration: int = 0,
    event_type_correct: Optional[bool] = None,
    lead_time_error: Optional[int] = None,
    duration_error: Optional[int] = None,
) -> EpisodeMetrics:
    """Compute metrics from a list of per-period states."""
    m = EpisodeMetrics(
        seed=seed,
        condition=condition,
        template_id=template_id,
        ambiguity_level=ambiguity_level,
    )
    inventory_levels = []

    for s in history:
        m.total_profit += s.period_profit
        m.total_demand += s.demand
        m.total_sales += s.sales
        m.total_lost_sales += s.lost_sales
        m.total_holding += s.holding_cost
        if s.lost_sales > 0:
            m.periods_stockout += 1
        inventory_levels.append(s.on_hand)
        m.periods += 1

    m.avg_inventory = float(np.mean(inventory_levels)) if inventory_levels else 0.0
    m.fill_rate = m.total_sales / m.total_demand if m.total_demand > 0 else 0.0

    m.predicted_event_type = predicted_event_type
    m.predicted_probability = predicted_probability
    m.predicted_lt_increase = predicted_lt_increase
    m.predicted_duration = predicted_duration
    m.event_type_correct = event_type_correct
    m.lead_time_error = lead_time_error
    m.duration_error = duration_error

    return m


def information_value(
    j_condition: float,
    j_no_info: float,
    j_perfect_semantic: float,
) -> float:
    """Compute Semantic Information Value Recovery (SIVR).

    SIVR = (J_condition - J_no_info) / (J_PerfectSemantic - J_no_info)

    Denominator is PerfectSemantic (perfect semantic interpretation + heuristic policy),
    NOT HindsightOracle. This isolates the value of semantic interpretation quality.
    Returns 0.0 if PerfectSemantic provides no gain (denominator near zero).
    """
    denom = j_perfect_semantic - j_no_info
    if abs(denom) < 1e-9:
        return 0.0
    return (j_condition - j_no_info) / denom


def regret_to_perfect_semantic(j_condition: float, j_perfect_semantic: float) -> float:
    """Compute regret relative to PerfectSemantic.

    Regret = J_PerfectSemantic - J_condition.
    Positive means the condition underperformed PerfectSemantic.
    """
    return j_perfect_semantic - j_condition


def semantic_regret(j_llm: float, j_perfect_semantic: float) -> float:
    """Semantic regret: cost of imperfect interpretation quality.

    SemanticRegret = J_PerfectSemantic - J_LLM.
    Measures performance loss attributable to semantic extraction errors.
    """
    return j_perfect_semantic - j_llm


def controller_gap(
    j_causal_optimizer: float, j_perfect_semantic: float
) -> float:
    """Controller gap: gain from stronger causal policy with same semantic info.

    ControllerGap = J_CausalOptimizer - J_PerfectSemantic.
    Measures how much performance improves by replacing the heuristic policy
    with a receding-horizon optimizer, while keeping semantic information identical.
    """
    return j_causal_optimizer - j_perfect_semantic


def hindsight_advantage(
    j_hindsight: float, j_causal_optimizer: float
) -> float:
    """Hindsight advantage: additional gain from knowing realized future demand.

    HindsightAdvantage = J_HindsightOracle - J_CausalOptimizer.
    Measures the value of perfect future knowledge beyond what a strong
    causal optimizer can achieve.
    """
    return j_hindsight - j_causal_optimizer


def total_gap(j_hindsight: float, j_llm: float) -> float:
    """Total gap: overall distance from LLM to privileged optimum.

    TotalGap = J_HindsightOracle - J_LLM.
    Decomposes as: SemanticRegret + ControllerGap + HindsightAdvantage.
    """
    return j_hindsight - j_llm


def paired_difference_ci(
    values_a: list[float],
    values_b: list[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute mean difference and confidence interval for paired samples.

    Uses t-based CI for paired differences.

    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    a = np.array(values_a)
    b = np.array(values_b)
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    if n < 2:
        return mean_diff, mean_diff, mean_diff
    se = float(np.std(diff, ddof=1)) / np.sqrt(n)
    from scipy import stats
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    return mean_diff, ci_lower, ci_upper
