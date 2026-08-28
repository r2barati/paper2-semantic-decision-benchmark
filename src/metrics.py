"""Metrics for evaluating inventory performance across conditions."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

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


# ---------------------------------------------------------------------------
# Authoritative semantic-value metrics
# ---------------------------------------------------------------------------

SIVR_VALID = "VALID"
SIVR_ZERO_OR_NEAR_ZERO_REFERENCE_VALUE = "ZERO_OR_NEAR_ZERO_REFERENCE_VALUE"
SIVR_NEGATIVE_ORACLE_REFERENCE_VALUE = "NEGATIVE_ORACLE_REFERENCE_VALUE"


@dataclass(frozen=True)
class SIVRResult:
    """Signed SIVR together with the reference-value status.

    OracleSemantic is a fixed-controller semantic reference, not an upper
    bound on reward.  Consequently the denominator is signed and a negative
    oracle information value is explicitly marked rather than normalized
    with ``abs``.
    """

    value: float
    oiv: float
    status: str

    def as_dict(self) -> Dict[str, Any]:
        return {"sivr": self.value, "oiv": self.oiv, "status": self.status}


def signed_sivr(
    j_condition: float,
    j_no_info: float,
    j_oracle_semantic: float,
    epsilon: float = 1e-9,
) -> SIVRResult:
    """Compute the canonical signed SIVR.

    If the oracle reference value is zero or numerically negligible, SIVR is
    undefined and returned as NaN.  If the oracle reference is negative, the
    signed diagnostic is returned but marked as unsuitable for an
    information-value-recovery interpretation.
    """

    oiv = float(j_oracle_semantic - j_no_info)
    if abs(oiv) <= epsilon:
        return SIVRResult(float("nan"), oiv, SIVR_ZERO_OR_NEAR_ZERO_REFERENCE_VALUE)
    status = SIVR_VALID if oiv > 0 else SIVR_NEGATIVE_ORACLE_REFERENCE_VALUE
    return SIVRResult(float((j_condition - j_no_info) / oiv), oiv, status)


def standard_brier_score(
    probabilities: Mapping[str, float],
    true_label: str,
    class_order: Optional[list] = None,
) -> float:
    """Return the conventional multiclass Brier score.

    This is ``sum_k (p_k - y_k)^2`` over the supplied class order.  Missing
    classes receive probability zero, and the result is not the older
    true-class-only squared error used by some historical runners.
    """

    classes = list(class_order) if class_order is not None else sorted(probabilities)
    return float(sum((float(probabilities.get(c, 0.0)) - (1.0 if c == true_label else 0.0)) ** 2
                     for c in classes))


def family_seed_bootstrap_ci(
    diffs_by_regime_family_variant: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    regime_weights: Optional[Mapping[str, float]] = None,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap paired effects at the regime/family/variant/seed hierarchy.

    Within each regime, template families are sampled with replacement.  For
    every sampled family, its observed variant count is retained while
    variants are sampled with replacement; paired seeds are then sampled with
    replacement within each selected variant.  Regime means are combined with
    explicit weights, giving separate balanced and deployment-prior
    estimands without changing the frozen episodes.
    """

    regimes = sorted(diffs_by_regime_family_variant)
    if not regimes:
        raise ValueError("At least one regime is required")
    if regime_weights is None:
        weights = {r: 1.0 / len(regimes) for r in regimes}
    else:
        raw = {r: float(regime_weights.get(r, 0.0)) for r in regimes}
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("Regime weights must have positive mass")
        weights = {r: raw[r] / total for r in regimes}

    def point_regime_mean(regime: str) -> float:
        family_means = []
        for family in sorted(diffs_by_regime_family_variant[regime]):
            variant_means = [float(np.mean(np.asarray(values, dtype=float)))
                             for values in diffs_by_regime_family_variant[regime][family].values()]
            if variant_means:
                family_means.append(float(np.mean(variant_means)))
        if not family_means:
            raise ValueError("Each regime must contain at least one family")
        return float(np.mean(family_means))

    point = float(sum(weights[r] * point_regime_mean(r) for r in regimes))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        regime_means = {}
        for regime in regimes:
            families = sorted(diffs_by_regime_family_variant[regime])
            sampled_family_ids = rng.choice(families, size=len(families), replace=True)
            sampled_family_means = []
            for family in sampled_family_ids:
                variants = sorted(diffs_by_regime_family_variant[regime][family])
                sampled_variants = rng.choice(variants, size=len(variants), replace=True)
                variant_means = []
                for variant in sampled_variants:
                    values = np.asarray(diffs_by_regime_family_variant[regime][family][variant], dtype=float)
                    if values.size == 0:
                        continue
                    sampled_values = values[rng.integers(0, values.size, size=values.size)]
                    variant_means.append(float(np.mean(sampled_values)))
                sampled_family_means.append(float(np.mean(variant_means)))
            regime_means[regime] = float(np.mean(sampled_family_means))
        boot[b] = sum(weights[r] * regime_means[r] for r in regimes)

    return {
        "mean": point,
        "ci_lower": float(np.percentile(boot, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "bootstrap_samples": boot,
        "regime_weights": weights,
        "bootstrap_seed": seed,
        "bootstrap_hierarchy": "regime -> template_family -> variant -> paired_seed",
    }


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
    return signed_sivr(j_condition, j_no_info, j_perfect_semantic).value


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
