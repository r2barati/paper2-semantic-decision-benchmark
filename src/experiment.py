"""Experiment runner — Phase 3.

Runs a full factorial experiment across:
  - all warning templates (clear / moderate / vague)
  - 6 conditions (NoInfo, PerfectSemantic, LLM, ActiveWrong, ActiveWrongOver, HindsightOracle)
  - paired demand seeds

Phase 3 key additions:
  - PerfectSemantic replaces Oracle: perfect semantic interpretation + heuristic policy.
  - HindsightOracle: true decision upper bound using full future demand knowledge.
  - Regret decomposition: SemanticRegret, DecisionRegret, TotalRegret.
  - SIVR (Semantic Information Value Recovery) replaces IVR.
  - Cost-asymmetry sensitivity analysis.
  - Trajectory-level diagnostics (order qty, inventory, costs per period).
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.env import InventoryEnv, HORIZON, HindsightOracle, CausalOptimizer
from src.events import (
    DEFAULT_DISRUPTION,
    WARNING_TEMPLATES,
    P4_WARNING_TIME,
    TRUE_LT_INCREASE,
    TRUE_DURATION,
    TRUE_EVENT_TYPE,
    SupplierDisruption,
)
from src.interpreter import (
    interpret,
    mock_interpret,
    active_wrong_interpret,
    active_wrong_overestimate,
    Interpretation,
)
from src.policies import BaseStockPolicy, DisruptionAwarePolicy
from src.metrics import (
    EpisodeMetrics,
    compute_episode_metrics,
    information_value,
    regret_to_perfect_semantic,
    paired_difference_ci,
    semantic_regret,
    controller_gap,
    hindsight_advantage,
    total_gap,
)

# Configuration
NUM_SEEDS = 30
SEED_BASE = 1000
WARNING_TIME = P4_WARNING_TIME
PROBABILITY_THRESHOLD = 0.70
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CONDITIONS = ["NoInfo", "PerfectSemantic", "LLM", "ActiveWrong", "ActiveWrongOver", "CausalOptimizer", "HindsightOracle"]


def _run_episode(
    seed: int,
    condition: str,
    disruption: SupplierDisruption,
    template_id: str = "",
    ambiguity_level: str = "",
    interpretation: Optional[Interpretation] = None,
    cost_overrides: Optional[dict] = None,
) -> EpisodeMetrics:
    """Run one episode under one condition.

    The critical design:
      - NoInfo: never switches policy
      - PerfectSemantic: switches using TRUE parameters (perfect semantic + heuristic)
      - LLM: switches using INTERPRETED parameters (if probability >= threshold)
      - ActiveWrong: switches using INCORRECT parameters (probability always >= threshold)
      - ActiveWrongOver: switches using OVERESTIMATED parameters (always >= threshold)
      - HindsightOracle: pre-computes optimal order sequence using full future demand knowledge
    """
    # Special case: HindsightOracle uses pre-computed optimal orders
    if condition == "HindsightOracle":
        oracle = HindsightOracle(
            seed=seed,
            disruption=disruption,
        )
        history, optimal_orders = oracle.compute_optimal_trajectory()

        return compute_episode_metrics(
            history,
            seed=seed,
            condition=condition,
            template_id=template_id,
            ambiguity_level=ambiguity_level,
            predicted_event_type="",
            predicted_probability=1.0,
            predicted_lt_increase=0,
            predicted_duration=0,
            event_type_correct=None,
            lead_time_error=None,
            duration_error=None,
        )

    env = InventoryEnv(seed=seed, disruption=disruption)
    state = env.reset()
    history = []

    base_policy = BaseStockPolicy(lead_time=disruption.normal_lead_time)
    disrupt_policy = DisruptionAwarePolicy(
        normal_lead_time=disruption.normal_lead_time,
    )

    # CausalOptimizer: receding-horizon with perfect semantic knowledge.
    # This condition is *defined* as an informed reference, so the true
    # disruption is disclosed deliberately and explicitly.
    causal_optimizer = None
    if condition == "CausalOptimizer":
        causal_optimizer = CausalOptimizer(
            seed=seed,
            disruption=disruption,
            knows_true_disruption=True,
        )

    # Determine whether and how to adapt
    switch = False
    estimated_lt_increase = 0
    estimated_duration = HORIZON  # default: stay adapted forever

    if condition == "PerfectSemantic":
        switch = True
        estimated_lt_increase = disruption.disrupted_lead_time - disruption.normal_lead_time
        estimated_duration = disruption.duration

    elif condition == "LLM" and interpretation is not None:
        if interpretation.probability >= PROBABILITY_THRESHOLD:
            switch = True
            estimated_lt_increase = interpretation.estimated_lead_time_increase
            estimated_duration = interpretation.estimated_duration

    elif condition == "ActiveWrong" and interpretation is not None:
        # Always switches (high confidence) but uses wrong parameters
        switch = True
        estimated_lt_increase = interpretation.estimated_lead_time_increase
        estimated_duration = interpretation.estimated_duration

    elif condition == "ActiveWrongOver" and interpretation is not None:
        # Always switches with overestimated parameters
        switch = True
        estimated_lt_increase = interpretation.estimated_lead_time_increase
        estimated_duration = interpretation.estimated_duration

    # Track adaptation window
    adaptation_end = WARNING_TIME + estimated_duration if switch else 0

    for t in range(HORIZON):
        # Decide order quantity
        if causal_optimizer is not None:
            order_qty = causal_optimizer.decide(state)
        else:
            use_disruption = switch and (WARNING_TIME <= t < adaptation_end)
            if use_disruption:
                order_qty = disrupt_policy(state, estimated_lt_increase=estimated_lt_increase)
            else:
                order_qty = base_policy(state)

        state = env.step(order_qty)
        history.append(state)

    # Compute semantic extraction errors
    predicted_event_type = ""
    predicted_probability = 0.0
    predicted_lt_increase = 0
    predicted_duration_val = 0
    event_type_correct = None
    lt_error = None
    dur_error = None

    if condition == "PerfectSemantic":
        event_type_correct = True
        lt_error = 0
        dur_error = 0
    elif interpretation is not None:
        predicted_event_type = interpretation.event_type
        predicted_probability = interpretation.probability
        predicted_lt_increase = interpretation.estimated_lead_time_increase
        predicted_duration_val = interpretation.estimated_duration
        event_type_correct = interpretation.event_type == TRUE_EVENT_TYPE
        lt_error = abs(TRUE_LT_INCREASE - interpretation.estimated_lead_time_increase)
        dur_error = abs(TRUE_DURATION - interpretation.estimated_duration)

    return compute_episode_metrics(
        history,
        seed=seed,
        condition=condition,
        template_id=template_id,
        ambiguity_level=ambiguity_level,
        predicted_event_type=predicted_event_type,
        predicted_probability=predicted_probability,
        predicted_lt_increase=predicted_lt_increase,
        predicted_duration=predicted_duration_val,
        event_type_correct=event_type_correct,
        lead_time_error=lt_error,
        duration_error=dur_error,
    )


def run_experiment(
    num_seeds: int = NUM_SEEDS,
    seed_base: int = SEED_BASE,
    disruption: SupplierDisruption = DEFAULT_DISRUPTION,
    warning_time: int = WARNING_TIME,
    output_dir: Optional[Path] = None,
    use_llm: bool = False,
    templates: Optional[list[dict]] = None,
    conditions: Optional[list[str]] = None,
    cost_overrides: Optional[dict] = None,
) -> dict:
    """Run the complete multi-template, multi-condition experiment.

    Returns dictionary with all episode metrics, summaries, and comparisons.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if templates is None:
        templates = WARNING_TEMPLATES
    if conditions is None:
        conditions = CONDITIONS

    interpreter_mode = "REAL_LLM" if use_llm else "MOCK_LLM"
    print(f"Interpreter mode: {interpreter_mode}")
    print(f"Templates: {len(templates)} | Seeds: {num_seeds} | Conditions: {len(conditions)}")
    print(f"Total episodes: {len(templates) * num_seeds * len(conditions)}")

    # --- Phase 1: Get interpretations for all templates ---
    print("\n--- Semantic Interpretations ---")
    template_interpretations: dict[str, dict] = {}  # template_id -> {llm, activewrong, activewrong_over}

    for tmpl in templates:
        tid = tmpl["template_id"]
        text = tmpl["text"]

        if use_llm:
            llm_interp = interpret(text)
        else:
            llm_interp = mock_interpret(text)

        aw_interp = active_wrong_interpret(text)
        awo_interp = active_wrong_overestimate(text)

        template_interpretations[tid] = {
            "llm": llm_interp,
            "activewrong": aw_interp,
            "activewrong_over": awo_interp,
        }

        print(f"  {tid:15s} [{tmpl['ambiguity_level']:8s}] "
              f"LLM: prob={llm_interp.probability:.2f} lt+={llm_interp.estimated_lead_time_increase} dur={llm_interp.estimated_duration} | "
              f"AW: prob={aw_interp.probability:.2f} lt+={aw_interp.estimated_lead_time_increase} dur={aw_interp.estimated_duration} | "
              f"AWO: prob={awo_interp.probability:.2f} lt+={awo_interp.estimated_lead_time_increase} dur={awo_interp.estimated_duration}")

    # --- Phase 2: Run all episodes ---
    all_episodes: list[EpisodeMetrics] = []
    total = len(templates) * num_seeds * len(conditions)
    done = 0

    for tmpl in templates:
        tid = tmpl["template_id"]
        alevel = tmpl["ambiguity_level"]
        interps = template_interpretations[tid]

        for seed_i in range(num_seeds):
            seed = seed_base + seed_i

            for condition in conditions:
                interp = None
                if condition == "LLM":
                    interp = interps["llm"]
                elif condition == "ActiveWrong":
                    interp = interps["activewrong"]
                elif condition == "ActiveWrongOver":
                    interp = interps["activewrong_over"]

                metrics = _run_episode(
                    seed=seed,
                    condition=condition,
                    disruption=disruption,
                    template_id=tid,
                    ambiguity_level=alevel,
                    interpretation=interp,
                    cost_overrides=cost_overrides,
                )
                all_episodes.append(metrics)
                done += 1

            if done % (len(conditions) * 5) == 0 or done == total:
                print(f"  Progress: {done}/{total} episodes")

    # --- Phase 3: Save semantic interpretations ---
    interp_path = output_dir / "semantic_interpretations.csv"
    with open(interp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "template_id", "ambiguity_level", "interpreter_mode",
            "predicted_event_type", "true_event_type",
            "predicted_probability", "predicted_lt_increase", "predicted_duration",
            "true_lt_increase", "true_duration",
            "event_type_correct", "lead_time_error", "duration_error",
        ])
        for tmpl in templates:
            tid = tmpl["template_id"]
            interps = template_interpretations[tid]
            for label, interp_key in [("LLM", "llm"), ("ActiveWrong", "activewrong"), ("ActiveWrongOver", "activewrong_over")]:
                interp = interps[interp_key]
                lt_err = abs(TRUE_LT_INCREASE - interp.estimated_lead_time_increase)
                dur_err = abs(TRUE_DURATION - interp.estimated_duration)
                etc = interp.event_type == TRUE_EVENT_TYPE
                writer.writerow([
                    tid, tmpl["ambiguity_level"], f"{interpreter_mode}_{label}",
                    interp.event_type, TRUE_EVENT_TYPE,
                    f"{interp.probability:.4f}", interp.estimated_lead_time_increase, interp.estimated_duration,
                    TRUE_LT_INCREASE, TRUE_DURATION,
                    etc, lt_err, dur_err,
                ])

    # --- Phase 4: Save episode results ---
    episode_path = output_dir / "episode_results.csv"
    with open(episode_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "template_id", "ambiguity_level", "seed", "condition",
            "total_profit", "fill_rate", "total_lost_sales", "avg_inventory", "periods_stockout",
            "predicted_event_type", "predicted_probability", "predicted_lt_increase", "predicted_duration",
            "event_type_correct", "lead_time_error", "duration_error",
        ])
        for m in all_episodes:
            writer.writerow([
                m.template_id, m.ambiguity_level, m.seed, m.condition,
                f"{m.total_profit:.2f}", f"{m.fill_rate:.4f}", f"{m.total_lost_sales:.0f}",
                f"{m.avg_inventory:.2f}", m.periods_stockout,
                m.predicted_event_type, f"{m.predicted_probability:.4f}",
                m.predicted_lt_increase, m.predicted_duration,
                m.event_type_correct, m.lead_time_error, m.duration_error,
            ])

    # --- Phase 5: Compute summaries ---

    # 5a: Overall summary by condition
    overall_summary = _compute_condition_summary(all_episodes, conditions)

    overall_path = output_dir / "overall_summary.csv"
    _write_summary_csv(overall_summary, overall_path)

    # 5b: Template-level summary
    template_summary = _compute_template_summary(all_episodes, templates, conditions)

    template_path = output_dir / "template_summary.csv"
    _write_template_csv(template_summary, template_path)

    # 5c: Ambiguity-level summary
    ambiguity_summary = _compute_ambiguity_summary(all_episodes, templates, conditions)

    ambiguity_path = output_dir / "ambiguity_summary.csv"
    _write_ambiguity_csv(ambiguity_summary, ambiguity_path)

    # 5d: Paired comparisons
    paired = _compute_paired_comparisons(all_episodes, templates, conditions)

    paired_path = output_dir / "paired_comparisons.csv"
    _write_paired_csv(paired, paired_path)

    # 5e: Semantic-error ↔ operational-value analysis
    analysis = _compute_semantic_operational_analysis(all_episodes, templates)

    analysis_path = output_dir / "semantic_operational_analysis.csv"
    with open(analysis_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["template_id", "ambiguity_level", "lt_error", "duration_error", "sivr_llm", "regret_llm"])
        for row in analysis:
            writer.writerow([row["template_id"], row["ambiguity_level"],
                           row["lt_error"], row["duration_error"],
                           f"{row['sivr_llm']:.4f}", f"{row['regret_llm']:.2f}"])

    # 5f: Regret decomposition (Semantic, Decision, Total)
    regret_decomp = _compute_regret_decomposition(all_episodes, templates, conditions)
    regret_path = output_dir / "regret_decomposition.csv"
    _write_regret_decomposition_csv(regret_decomp, regret_path)

    # 5g: Trajectory diagnostics (PerfectSemantic vs ActiveWrongOver)
    trajectory_diag = _compute_trajectory_diagnostics(all_episodes, templates, conditions)
    if trajectory_diag:
        traj_path = output_dir / "trajectory_diagnostics.csv"
        _write_trajectory_diagnostics_csv(trajectory_diag, traj_path)

    # --- Phase 6: Print summary ---
    _print_summary(overall_summary, template_summary, ambiguity_summary, paired, analysis,
                   regret_decomp, trajectory_diag,
                   interpreter_mode, disruption, warning_time, num_seeds, len(templates))

    # --- Phase 7: Generate plots ---
    _generate_all_plots(
        overall_summary, template_summary, ambiguity_summary, paired, analysis,
        regret_decomp, trajectory_diag,
        output_dir, interpreter_mode,
    )

    return {
        "episodes": all_episodes,
        "overall_summary": overall_summary,
        "template_summary": template_summary,
        "ambiguity_summary": ambiguity_summary,
        "paired_comparisons": paired,
        "semantic_analysis": analysis,
        "regret_decomposition": regret_decomp,
        "trajectory_diagnostics": trajectory_diag,
        "interpreter_mode": interpreter_mode,
    }


# --- Summary computation helpers ---

def _compute_condition_summary(
    episodes: list[EpisodeMetrics],
    conditions: list[str],
) -> list[dict]:
    """Compute summary statistics grouped by condition (pooled across templates)."""
    by_cond = defaultdict(list)
    for m in episodes:
        by_cond[m.condition].append(m)

    rows = []
    for cond in conditions:
        ms = by_cond[cond]
        profits = [m.total_profit for m in ms]
        n = len(profits)
        mean_p = float(np.mean(profits))
        std_p = float(np.std(profits, ddof=1)) if n > 1 else 0.0
        median_p = float(np.median(profits))
        se_p = std_p / np.sqrt(n) if n > 1 else 0.0
        from scipy import stats as sp_stats
        t_crit = sp_stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
        ci_lo = mean_p - t_crit * se_p
        ci_hi = mean_p + t_crit * se_p

        # IVR and regret
        ni_profits = [m.total_profit for m in by_cond["NoInfo"]]
        ps_profits = [m.total_profit for m in by_cond.get("PerfectSemantic", [])]
        mean_ni = float(np.mean(ni_profits))
        mean_ps = float(np.mean(ps_profits)) if ps_profits else mean_p

        sivr = information_value(mean_p, mean_ni, mean_ps)
        regret = regret_to_perfect_semantic(mean_p, mean_ps)

        # Semantic errors for this condition
        lt_errors = [m.lead_time_error for m in ms if m.lead_time_error is not None]
        dur_errors = [m.duration_error for m in ms if m.duration_error is not None]
        mean_lt_err = float(np.mean(lt_errors)) if lt_errors else None
        mean_dur_err = float(np.mean(dur_errors)) if dur_errors else None

        rows.append({
            "condition": cond,
            "mean_profit": mean_p,
            "std_profit": std_p,
            "median_profit": median_p,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "mean_fill_rate": float(np.mean([m.fill_rate for m in ms])),
            "mean_lost_sales": float(np.mean([m.total_lost_sales for m in ms])),
            "mean_avg_inventory": float(np.mean([m.avg_inventory for m in ms])),
            "mean_stockout_periods": float(np.mean([m.periods_stockout for m in ms])),
            "sivr": sivr,
            "regret": regret,
            "mean_lt_error": mean_lt_err,
            "mean_duration_error": mean_dur_err,
            "n": n,
        })
    return rows


def _compute_template_summary(
    episodes: list[EpisodeMetrics],
    templates: list[dict],
    conditions: list[str],
) -> list[dict]:
    """Compute summary for each template × condition."""
    by_tmpl_cond = defaultdict(list)
    for m in episodes:
        by_tmpl_cond[(m.template_id, m.condition)].append(m)

    # Get PerfectSemantic/noinfo per template for SIVR
    ps_by_tmpl = defaultdict(list)
    ninfo_by_tmpl = defaultdict(list)
    for m in episodes:
        if m.condition == "PerfectSemantic":
            ps_by_tmpl[m.template_id].append(m.total_profit)
        elif m.condition == "NoInfo":
            ninfo_by_tmpl[m.template_id].append(m.total_profit)

    rows = []
    for tmpl in templates:
        tid = tmpl["template_id"]
        alevel = tmpl["ambiguity_level"]
        mean_ps = float(np.mean(ps_by_tmpl[tid])) if ps_by_tmpl[tid] else 0.0
        mean_ninfo = float(np.mean(ninfo_by_tmpl[tid])) if ninfo_by_tmpl[tid] else 0.0

        for cond in conditions:
            ms = by_tmpl_cond.get((tid, cond), [])
            if not ms:
                continue
            profits = [m.total_profit for m in ms]
            mean_p = float(np.mean(profits))
            std_p = float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0
            n = len(profits)

            sivr = information_value(mean_p, mean_ninfo, mean_ps)
            regret = regret_to_perfect_semantic(mean_p, mean_ps)

            lt_errors = [m.lead_time_error for m in ms if m.lead_time_error is not None]
            dur_errors = [m.duration_error for m in ms if m.duration_error is not None]

            rows.append({
                "template_id": tid,
                "ambiguity_level": alevel,
                "condition": cond,
                "mean_profit": mean_p,
                "std_profit": std_p,
                "n": n,
                "sivr": sivr,
                "regret": regret,
                "mean_lt_error": float(np.mean(lt_errors)) if lt_errors else None,
                "mean_duration_error": float(np.mean(dur_errors)) if dur_errors else None,
            })
    return rows


def _compute_ambiguity_summary(
    episodes: list[EpisodeMetrics],
    templates: list[dict],
    conditions: list[str],
) -> list[dict]:
    """Compute summary grouped by ambiguity level × condition."""
    by_ambig_cond = defaultdict(list)
    for m in episodes:
        by_ambig_cond[(m.ambiguity_level, m.condition)].append(m)

    ps_by_ambig = defaultdict(list)
    ninfo_by_ambig = defaultdict(list)
    for m in episodes:
        if m.condition == "PerfectSemantic":
            ps_by_ambig[m.ambiguity_level].append(m.total_profit)
        elif m.condition == "NoInfo":
            ninfo_by_ambig[m.ambiguity_level].append(m.total_profit)

    rows = []
    for alevel in ["clear", "moderate", "vague"]:
        mean_ps = float(np.mean(ps_by_ambig[alevel])) if ps_by_ambig[alevel] else 0.0
        mean_ninfo = float(np.mean(ninfo_by_ambig[alevel])) if ninfo_by_ambig[alevel] else 0.0

        for cond in conditions:
            ms = by_ambig_cond.get((alevel, cond), [])
            if not ms:
                continue
            profits = [m.total_profit for m in ms]
            mean_p = float(np.mean(profits))
            std_p = float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0
            n = len(profits)

            sivr = information_value(mean_p, mean_ninfo, mean_ps)
            regret = regret_to_perfect_semantic(mean_p, mean_ps)

            lt_errors = [m.lead_time_error for m in ms if m.lead_time_error is not None]
            dur_errors = [m.duration_error for m in ms if m.duration_error is not None]

            rows.append({
                "ambiguity_level": alevel,
                "condition": cond,
                "mean_profit": mean_p,
                "std_profit": std_p,
                "n": n,
                "sivr": sivr,
                "regret": regret,
                "mean_lt_error": float(np.mean(lt_errors)) if lt_errors else None,
                "mean_duration_error": float(np.mean(dur_errors)) if dur_errors else None,
            })
    return rows


def _compute_paired_comparisons(
    episodes: list[EpisodeMetrics],
    templates: list[dict],
    conditions: list[str],
) -> list[dict]:
    """Compute paired comparisons at template level and overall."""
    # Group by (template, seed) for paired analysis
    by_tmpl_seed = defaultdict(dict)
    for m in episodes:
        by_tmpl_seed[(m.template_id, m.seed)][m.condition] = m.total_profit

    comparisons = []
    # Overall paired comparisons (flatten all template×seed pairs)
    pairs_ni_ps = []
    pairs_ni_llm = []
    pairs_ni_aw = []
    pairs_ni_awo = []
    pairs_llm_ps = []
    pairs_aw_ps = []
    pairs_awo_ps = []
    pairs_llm_ho = []
    pairs_ps_ho = []

    for key, cond_profits in by_tmpl_seed.items():
        ni = cond_profits.get("NoInfo")
        ps = cond_profits.get("PerfectSemantic")
        llm = cond_profits.get("LLM")
        aw = cond_profits.get("ActiveWrong")
        awo = cond_profits.get("ActiveWrongOver")
        ho = cond_profits.get("HindsightOracle")

        if ni is not None and ps is not None:
            pairs_ni_ps.append((ps, ni))
        if ni is not None and llm is not None:
            pairs_ni_llm.append((llm, ni))
        if ni is not None and aw is not None:
            pairs_ni_aw.append((aw, ni))
        if ni is not None and awo is not None:
            pairs_ni_awo.append((awo, ni))
        if llm is not None and ps is not None:
            pairs_llm_ps.append((llm, ps))
        if aw is not None and ps is not None:
            pairs_aw_ps.append((aw, ps))
        if awo is not None and ps is not None:
            pairs_awo_ps.append((awo, ps))
        if llm is not None and ho is not None:
            pairs_llm_ho.append((llm, ho))
        if ps is not None and ho is not None:
            pairs_ps_ho.append((ps, ho))

    for label, pairs in [
        ("PerfectSemantic - NoInfo", pairs_ni_ps),
        ("LLM - NoInfo", pairs_ni_llm),
        ("ActiveWrong - NoInfo", pairs_ni_aw),
        ("ActiveWrongOver - NoInfo", pairs_ni_awo),
        ("LLM - PerfectSemantic", pairs_llm_ps),
        ("ActiveWrong - PerfectSemantic", pairs_aw_ps),
        ("ActiveWrongOver - PerfectSemantic", pairs_awo_ps),
        ("LLM - HindsightOracle", pairs_llm_ho),
        ("PerfectSemantic - HindsightOracle", pairs_ps_ho),
    ]:
        if pairs:
            a_vals = [p[0] for p in pairs]
            b_vals = [p[1] for p in pairs]
            mean_diff, ci_lo, ci_hi = paired_difference_ci(a_vals, b_vals)
        else:
            mean_diff, ci_lo, ci_hi = 0.0, 0.0, 0.0

        comparisons.append({
            "comparison": label,
            "mean_diff": float(mean_diff),
            "ci_95_lower": float(ci_lo),
            "ci_95_upper": float(ci_hi),
            "n_pairs": len(pairs),
        })

    # Per-ambiguity-level paired comparisons for LLM - NoInfo
    for alevel in ["clear", "moderate", "vague"]:
        pairs = []
        for key, cond_profits in by_tmpl_seed.items():
            tmpl_id = key[0]
            tmpl_meta = next((t for t in templates if t["template_id"] == tmpl_id), None)
            if tmpl_meta and tmpl_meta["ambiguity_level"] != alevel:
                continue
            ni = cond_profits.get("NoInfo")
            llm = cond_profits.get("LLM")
            if ni is not None and llm is not None:
                pairs.append((llm, ni))

        if pairs:
            a_vals = [p[0] for p in pairs]
            b_vals = [p[1] for p in pairs]
            mean_diff, ci_lo, ci_hi = paired_difference_ci(a_vals, b_vals)
        else:
            mean_diff, ci_lo, ci_hi = 0.0, 0.0, 0.0

        comparisons.append({
            "comparison": f"LLM - NoInfo ({alevel})",
            "mean_diff": float(mean_diff),
            "ci_95_lower": float(ci_lo),
            "ci_95_upper": float(ci_hi),
            "n_pairs": len(pairs),
        })

    return comparisons


def _compute_semantic_operational_analysis(
    episodes: list[EpisodeMetrics],
    templates: list[dict],
) -> list[dict]:
    """Template-level analysis relating semantic error to operational outcome."""
    # For LLM condition: group by template
    llm_by_tmpl = defaultdict(list)
    for m in episodes:
        if m.condition == "LLM":
            llm_by_tmpl[m.template_id].append(m)

    results = []
    for tmpl in templates:
        tid = tmpl["template_id"]
        alevel = tmpl["ambiguity_level"]
        ms = llm_by_tmpl.get(tid, [])
        if not ms:
            continue

        # Compute SIVR for this template (need PerfectSemantic/noinfo means)
        ps_profits = [m.total_profit for m in episodes
                         if m.template_id == tid and m.condition == "PerfectSemantic"]
        ninfo_profits = [m.total_profit for m in episodes
                        if m.template_id == tid and m.condition == "NoInfo"]

        mean_llm = float(np.mean([m.total_profit for m in ms]))
        mean_ps = float(np.mean(ps_profits)) if ps_profits else mean_llm
        mean_ninfo = float(np.mean(ninfo_profits)) if ninfo_profits else mean_llm

        sivr = information_value(mean_llm, mean_ninfo, mean_ps)
        regret = regret_to_perfect_semantic(mean_llm, mean_ps)

        lt_errors = [m.lead_time_error for m in ms if m.lead_time_error is not None]
        dur_errors = [m.duration_error for m in ms if m.duration_error is not None]

        results.append({
            "template_id": tid,
            "ambiguity_level": alevel,
            "lt_error": float(np.mean(lt_errors)) if lt_errors else 0.0,
            "duration_error": float(np.mean(dur_errors)) if dur_errors else 0.0,
            "sivr_llm": sivr,
            "regret_llm": regret,
        })
    return results


# --- CSV writers ---

def _write_summary_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "mean_profit", "std_profit", "median_profit",
            "ci_95_lower", "ci_95_upper", "mean_fill_rate",
            "mean_lost_sales", "mean_avg_inventory", "mean_stockout_periods",
            "sivr", "regret", "mean_lt_error", "mean_duration_error", "n",
        ])
        for r in rows:
            writer.writerow([r[k] for k in [
                "condition", "mean_profit", "std_profit", "median_profit",
                "ci_95_lower", "ci_95_upper", "mean_fill_rate",
                "mean_lost_sales", "mean_avg_inventory", "mean_stockout_periods",
                "sivr", "regret", "mean_lt_error", "mean_duration_error", "n",
            ]])


def _write_template_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "template_id", "ambiguity_level", "condition",
            "mean_profit", "std_profit", "n", "sivr", "regret",
            "mean_lt_error", "mean_duration_error",
        ])
        for r in rows:
            writer.writerow([
                r["template_id"], r["ambiguity_level"], r["condition"],
                f"{r['mean_profit']:.2f}", f"{r['std_profit']:.2f}", r["n"],
                f"{r['sivr']:.4f}", f"{r['regret']:.2f}",
                r["mean_lt_error"] if r["mean_lt_error"] is not None else "",
                r["mean_duration_error"] if r["mean_duration_error"] is not None else "",
            ])


def _write_ambiguity_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ambiguity_level", "condition",
            "mean_profit", "std_profit", "n", "sivr", "regret",
            "mean_lt_error", "mean_duration_error",
        ])
        for r in rows:
            writer.writerow([
                r["ambiguity_level"], r["condition"],
                f"{r['mean_profit']:.2f}", f"{r['std_profit']:.2f}", r["n"],
                f"{r['sivr']:.4f}", f"{r['regret']:.2f}",
                r["mean_lt_error"] if r["mean_lt_error"] is not None else "",
                r["mean_duration_error"] if r["mean_duration_error"] is not None else "",
            ])


def _write_paired_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "mean_diff", "ci_95_lower", "ci_95_upper", "n_pairs"])
        for r in rows:
            writer.writerow([
                r["comparison"],
                f"{r['mean_diff']:.2f}", f"{r['ci_95_lower']:.2f}", f"{r['ci_95_upper']:.2f}",
                r["n_pairs"],
            ])


# --- Regret decomposition ---

def _compute_regret_decomposition(
    episodes: list[EpisodeMetrics],
    templates: list[dict],
    conditions: list[str],
) -> list[dict]:
    """Compute clean regret decomposition.

    SemanticRegret = J_PerfectSemantic - J_LLM
    ControllerGap = J_CausalOptimizer - J_PerfectSemantic
    HindsightAdvantage = J_HindsightOracle - J_CausalOptimizer
    TotalGap = J_HindsightOracle - J_LLM = SemanticRegret + ControllerGap + HindsightAdvantage
    """
    by_tmpl_cond = defaultdict(list)
    for m in episodes:
        by_tmpl_cond[(m.template_id, m.condition)].append(m)

    results = []
    for tmpl in templates:
        tid = tmpl["template_id"]
        alevel = tmpl["ambiguity_level"]

        def mean_profit(condition: str) -> float:
            ms = by_tmpl_cond.get((tid, condition), [])
            return float(np.mean([m.total_profit for m in ms])) if ms else 0.0

        j_noinfo = mean_profit("NoInfo")
        j_ps = mean_profit("PerfectSemantic")
        j_llm = mean_profit("LLM")
        j_aw = mean_profit("ActiveWrong")
        j_awo = mean_profit("ActiveWrongOver")
        j_co = mean_profit("CausalOptimizer")
        j_ho = mean_profit("HindsightOracle")

        # Decomposition for each condition vs HindsightOracle
        for cond, j_cond in [("LLM", j_llm), ("ActiveWrong", j_aw),
                              ("ActiveWrongOver", j_awo), ("PerfectSemantic", j_ps),
                              ("CausalOptimizer", j_co)]:
            sr = semantic_regret(j_cond, j_ps)
            cg = controller_gap(j_co, j_ps)
            ha = hindsight_advantage(j_ho, j_co)
            tg = total_gap(j_ho, j_cond)

            results.append({
                "template_id": tid,
                "ambiguity_level": alevel,
                "condition": cond,
                "j_no_info": j_noinfo,
                "j_perfect_semantic": j_ps,
                "j_causal_optimizer": j_co,
                "j_hindsight_oracle": j_ho,
                "j_condition": j_cond,
                "semantic_regret": sr,
                "controller_gap": cg,
                "hindsight_advantage": ha,
                "total_gap": tg,
                "sivr": information_value(j_cond, j_noinfo, j_ps),
            })

    return results


def _write_regret_decomposition_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "template_id", "ambiguity_level", "condition",
            "j_no_info", "j_perfect_semantic", "j_causal_optimizer",
            "j_hindsight_oracle", "j_condition",
            "semantic_regret", "controller_gap", "hindsight_advantage",
            "total_gap", "sivr",
        ])
        for r in rows:
            writer.writerow([
                r["template_id"], r["ambiguity_level"], r["condition"],
                f"{r['j_no_info']:.2f}", f"{r['j_perfect_semantic']:.2f}",
                f"{r['j_causal_optimizer']:.2f}", f"{r['j_hindsight_oracle']:.2f}",
                f"{r['j_condition']:.2f}",
                f"{r['semantic_regret']:.2f}", f"{r['controller_gap']:.2f}",
                f"{r['hindsight_advantage']:.2f}", f"{r['total_gap']:.2f}",
                f"{r['sivr']:.4f}",
            ])


# --- Trajectory diagnostics ---

def _compute_trajectory_diagnostics(
    episodes: list[EpisodeMetrics],
    templates: list[dict],
    conditions: list[str],
) -> list[dict]:
    """Compare trajectory-level behavior of PerfectSemantic vs ActiveWrongOver.

    Since episode metrics don't store per-period data, this summarizes the
    observable differences: fill rate, lost sales, avg inventory, stockout periods.
    """
    if "PerfectSemantic" not in conditions or "ActiveWrongOver" not in conditions:
        return []

    by_tmpl_cond = defaultdict(list)
    for m in episodes:
        by_tmpl_cond[(m.template_id, m.condition)].append(m)

    results = []
    for tmpl in templates:
        tid = tmpl["template_id"]
        alevel = tmpl["ambiguity_level"]

        ps_ms = by_tmpl_cond.get((tid, "PerfectSemantic"), [])
        awo_ms = by_tmpl_cond.get((tid, "ActiveWrongOver"), [])

        if not ps_ms or not awo_ms:
            continue

        def safe_mean(ms, attr):
            vals = [getattr(m, attr) for m in ms]
            return float(np.mean(vals)) if vals else 0.0

        results.append({
            "template_id": tid,
            "ambiguity_level": alevel,
            "ps_mean_profit": safe_mean(ps_ms, "total_profit"),
            "awo_mean_profit": safe_mean(awo_ms, "total_profit"),
            "ps_mean_fill_rate": safe_mean(ps_ms, "fill_rate"),
            "awo_mean_fill_rate": safe_mean(awo_ms, "fill_rate"),
            "ps_mean_lost_sales": safe_mean(ps_ms, "total_lost_sales"),
            "awo_mean_lost_sales": safe_mean(awo_ms, "total_lost_sales"),
            "ps_mean_avg_inventory": safe_mean(ps_ms, "avg_inventory"),
            "awo_mean_avg_inventory": safe_mean(awo_ms, "avg_inventory"),
            "ps_mean_stockout_periods": safe_mean(ps_ms, "periods_stockout"),
            "awo_mean_stockout_periods": safe_mean(awo_ms, "periods_stockout"),
        })

    return results


def _write_trajectory_diagnostics_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "template_id", "ambiguity_level",
            "ps_mean_profit", "awo_mean_profit",
            "ps_mean_fill_rate", "awo_mean_fill_rate",
            "ps_mean_lost_sales", "awo_mean_lost_sales",
            "ps_mean_avg_inventory", "awo_mean_avg_inventory",
            "ps_mean_stockout_periods", "awo_mean_stockout_periods",
        ])
        for r in rows:
            writer.writerow([
                r["template_id"], r["ambiguity_level"],
                f"{r['ps_mean_profit']:.2f}", f"{r['awo_mean_profit']:.2f}",
                f"{r['ps_mean_fill_rate']:.4f}", f"{r['awo_mean_fill_rate']:.4f}",
                f"{r['ps_mean_lost_sales']:.1f}", f"{r['awo_mean_lost_sales']:.1f}",
                f"{r['ps_mean_avg_inventory']:.2f}", f"{r['awo_mean_avg_inventory']:.2f}",
                f"{r['ps_mean_stockout_periods']:.1f}", f"{r['awo_mean_stockout_periods']:.1f}",
            ])


# --- Cost sensitivity ---

def run_cost_sensitivity(
    num_seeds: int = NUM_SEEDS,
    seed_base: int = SEED_BASE,
    output_dir: Optional[Path] = None,
    templates: Optional[list[dict]] = None,
    conditions: Optional[list[str]] = None,
) -> list[dict]:
    """Run experiment across different cost ratios (stockout/holding).

    Tests robustness of ActiveWrongOver advantage to cost parameter changes.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "cost_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    if templates is None:
        templates = WARNING_TEMPLATES
    if conditions is None:
        conditions = CONDITIONS

    # Cost ratio configurations: (stockout_cost, holding_cost, label)
    cost_configs = [
        (8.0, 1.0, "baseline_8:1"),
        (4.0, 2.0, "reduced_2:1"),
        (12.0, 1.0, "high_stockout_12:1"),
        (20.0, 1.0, "extreme_stockout_20:1"),
        (2.0, 1.0, "low_stockout_2:1"),
    ]

    all_results = []
    for stockout_cost, holding_cost, label in cost_configs:
        print(f"\n--- Cost sensitivity: {label} (stockout=${stockout_cost}, holding=${holding_cost}) ---")
        cost_overrides = {
            "stockout_cost_per_unit": stockout_cost,
            "holding_cost_per_unit": holding_cost,
        }
        results = run_experiment(
            num_seeds=num_seeds,
            seed_base=seed_base,
            output_dir=output_dir / label,
            use_llm=False,
            templates=templates,
            conditions=conditions,
            cost_overrides=cost_overrides,
        )
        # Extract summary for this cost config
        for row in results["overall_summary"]:
            row["cost_config"] = label
            row["stockout_cost"] = stockout_cost
            row["holding_cost"] = holding_cost
            row["cost_ratio"] = stockout_cost / holding_cost
            all_results.append(row)

    # Write cost sensitivity summary
    sens_path = output_dir / "cost_sensitivity_summary.csv"
    with open(sens_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cost_config", "stockout_cost", "holding_cost", "cost_ratio",
            "condition", "mean_profit", "std_profit", "sivr", "regret", "n",
        ])
        for r in all_results:
            writer.writerow([
                r["cost_config"], r["stockout_cost"], r["holding_cost"],
                f"{r['cost_ratio']:.1f}", r["condition"],
                f"{r['mean_profit']:.2f}", f"{r['std_profit']:.2f}",
                f"{r['sivr']:.4f}" if r['sivr'] is not None else "",
                f"{r['regret']:.2f}" if r['regret'] is not None else "",
                r["n"],
            ])

    return all_results


# --- Printing ---

def _print_summary(
    overall, template_summary, ambiguity_summary, paired, analysis,
    regret_decomp, trajectory_diag,
    interpreter_mode, disruption, warning_time, num_seeds, num_templates,
):
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"Interpreter: {interpreter_mode}")
    print(f"Templates: {num_templates} | Seeds: {num_seeds} | Conditions: {len(CONDITIONS)}")
    print(f"Disruption: start={disruption.start_time}, duration={disruption.duration}, "
          f"LT {disruption.normal_lead_time} -> {disruption.disrupted_lead_time}")
    print(f"Warning time: t={warning_time} | Policy switch threshold: {PROBABILITY_THRESHOLD}")

    print("\n--- Overall Mean Cumulative Profit by Condition ---")
    for r in overall:
        sivr_str = f"SIVR={r['sivr']:.3f}" if r['sivr'] is not None else ""
        regret_str = f"Regret={r['regret']:.1f}" if r['regret'] is not None else ""
        print(
            f"  {r['condition']:22s}: {r['mean_profit']:8.1f} "
            f"(sd={r['std_profit']:.1f}, 95% CI=[{r['ci_95_lower']:.1f}, {r['ci_95_upper']:.1f}]) "
            f"{sivr_str} {regret_str}"
        )

    print("\n--- Paired Comparisons (mean diff, 95% CI) ---")
    for r in paired:
        print(
            f"  {r['comparison']:35s}: {r['mean_diff']:+8.1f} "
            f"[{r['ci_95_lower']:+.1f}, {r['ci_95_upper']:+.1f}] (n={r['n_pairs']})"
        )

    print("\n--- SIVR by Ambiguity Level (LLM condition) ---")
    for alevel in ["clear", "moderate", "vague"]:
        rows = [r for r in ambiguity_summary if r["ambiguity_level"] == alevel and r["condition"] == "LLM"]
        if rows:
            r = rows[0]
            lt_err = f"LT_err={r['mean_lt_error']:.1f}" if r['mean_lt_error'] is not None else ""
            dur_err = f"Dur_err={r['mean_duration_error']:.1f}" if r['mean_duration_error'] is not None else ""
            print(f"  {alevel:10s}: SIVR={r['sivr']:.3f}  Profit={r['mean_profit']:.1f}  {lt_err}  {dur_err}")

    # Regret decomposition summary
    if regret_decomp:
        print("\n--- Regret Decomposition (LLM condition) ---")
        print(f"  {'Template':15s} {'Ambig':8s} {'SemReg':8s} {'CtrlGap':8s} {'HindAdv':8s} {'TotGap':8s} {'SIVR':6s}")
        llm_decomp = [r for r in regret_decomp if r["condition"] == "LLM"]
        for r in llm_decomp:
            print(f"  {r['template_id']:15s} {r['ambiguity_level']:8s} "
                  f"{r['semantic_regret']:+8.1f} {r['controller_gap']:+8.1f} "
                  f"{r['hindsight_advantage']:+8.1f} {r['total_gap']:+8.1f} {r['sivr']:6.3f}")

    # Trajectory diagnostics summary
    if trajectory_diag:
        print("\n--- Trajectory Diagnostics: PerfectSemantic vs ActiveWrongOver ---")
        for r in trajectory_diag:
            print(f"  {r['template_id']:15s}: "
                  f"PS_fill={r['ps_mean_fill_rate']:.3f} AWO_fill={r['awo_mean_fill_rate']:.3f} | "
                  f"PS_inv={r['ps_mean_avg_inventory']:.1f} AWO_inv={r['awo_mean_avg_inventory']:.1f} | "
                  f"PS_stockout={r['ps_mean_stockout_periods']:.1f} AWO_stockout={r['awo_mean_stockout_periods']:.1f}")

    print("\n--- Semantic Error ↔ Operational Value (per template, LLM) ---")
    print(f"  {'Template':15s} {'Ambig':8s} {'LT_err':6s} {'Dur_err':7s} {'SIVR':6s} {'Regret':8s}")
    for r in analysis:
        print(f"  {r['template_id']:15s} {r['ambiguity_level']:8s} "
              f"{r['lt_error']:6.1f} {r['duration_error']:7.1f} "
              f"{r['sivr_llm']:6.3f} {r['regret_llm']:+8.1f}")

    print("=" * 70)


# --- Plot generation ---

def _generate_all_plots(
    overall_summary, template_summary, ambiguity_summary, paired, analysis,
    regret_decomp, trajectory_diag,
    output_dir, interpreter_mode,
):
    """Generate all publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    condition_colors = {
        "NoInfo": "#d62728",
        "PerfectSemantic": "#2ca02c",
        "LLM": "#1f77b4",
        "ActiveWrong": "#ff7f0e",
        "ActiveWrongOver": "#9467bd",
        "CausalOptimizer": "#e377c2",
        "HindsightOracle": "#8c564b",
    }
    condition_labels = {
        "NoInfo": "No Info",
        "PerfectSemantic": "Perfect\nSemantic",
        "LLM": "LLM",
        "ActiveWrong": "Active\nWrong",
        "ActiveWrongOver": "ActiveWrong\nOver",
        "CausalOptimizer": "Causal\nOptimizer",
        "HindsightOracle": "Hindsight\nOracle",
    }

    # Plot 1: Mean profit by condition (overall)
    fig, ax = plt.subplots(figsize=(12, 5))
    conds = [r["condition"] for r in overall_summary]
    means = [r["mean_profit"] for r in overall_summary]
    colors = [condition_colors.get(c, "#333") for c in conds]
    labels = [condition_labels.get(c, c) for c in conds]
    bars = ax.bar(labels, means, color=colors, alpha=0.8, edgecolor="black")
    cis_lo = [r["ci_95_lower"] for r in overall_summary]
    cis_hi = [r["ci_95_upper"] for r in overall_summary]
    errs_lower = [m - lo for m, lo in zip(means, cis_lo)]
    errs_upper = [hi - m for m, hi in zip(means, cis_hi)]
    ax.errorbar(labels, means, yerr=[errs_lower, errs_upper],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel("Mean Cumulative Profit")
    ax.set_title(f"Inventory Performance by Condition ({interpreter_mode})")
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "profit_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 2: SIVR by template
    llm_tmpl_rows = [r for r in template_summary if r["condition"] == "LLM"]
    if llm_tmpl_rows:
        fig, ax = plt.subplots(figsize=(12, 5))
        tids = [r["template_id"] for r in llm_tmpl_rows]
        sivrs = [r["sivr"] for r in llm_tmpl_rows]
        ambig_colors = {"clear": "#2ca02c", "moderate": "#ff7f0e", "vague": "#d62728"}
        bar_colors = [ambig_colors[r["ambiguity_level"]] for r in llm_tmpl_rows]
        bars = ax.bar(range(len(tids)), sivrs, color=bar_colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(tids)))
        ax.set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("SIVR")
        ax.set_title("Semantic Information Value Recovery by Warning Template (LLM)")
        ax.set_ylim(-0.2, 1.3)
        ax.grid(axis="y", alpha=0.3)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, edgecolor="black", label=l)
                          for l, c in ambig_colors.items()]
        ax.legend(handles=legend_elements, title="Ambiguity")
        plt.tight_layout()
        fig.savefig(output_dir / "sivr_by_template.png", dpi=150)
        plt.close(fig)

    # Plot 3: SIVR grouped by ambiguity level
    llm_ambig_rows = [r for r in ambiguity_summary if r["condition"] == "LLM"]
    aw_ambig_rows = [r for r in ambiguity_summary if r["condition"] == "ActiveWrong"]
    awo_ambig_rows = [r for r in ambiguity_summary if r["condition"] == "ActiveWrongOver"]

    if llm_ambig_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        alevels = ["clear", "moderate", "vague"]
        x = np.arange(len(alevels))
        width = 0.25

        llm_sivrs = []
        aw_sivrs = []
        awo_sivrs = []
        for al in alevels:
            r_llm = next((r for r in llm_ambig_rows if r["ambiguity_level"] == al), None)
            r_aw = next((r for r in aw_ambig_rows if r["ambiguity_level"] == al), None)
            r_awo = next((r for r in awo_ambig_rows if r["ambiguity_level"] == al), None)
            llm_sivrs.append(r_llm["sivr"] if r_llm else 0)
            aw_sivrs.append(r_aw["sivr"] if r_aw else 0)
            awo_sivrs.append(r_awo["sivr"] if r_awo else 0)

        ax.bar(x - width, llm_sivrs, width, label="LLM", color="#1f77b4", alpha=0.8, edgecolor="black")
        ax.bar(x, aw_sivrs, width, label="ActiveWrong", color="#ff7f0e", alpha=0.8, edgecolor="black")
        ax.bar(x + width, awo_sivrs, width, label="ActiveWrongOver", color="#9467bd", alpha=0.8, edgecolor="black")
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(alevels)
        ax.set_ylabel("SIVR")
        ax.set_title("SIVR by Ambiguity Level and Condition")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "sivr_by_ambiguity.png", dpi=150)
        plt.close(fig)

    # Plot 4: Semantic error vs SIVR
    if analysis:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ambig_markers = {"clear": "o", "moderate": "s", "vague": "^"}
        ambig_colors = {"clear": "#2ca02c", "moderate": "#ff7f0e", "vague": "#d62728"}

        for al in ["clear", "moderate", "vague"]:
            pts = [(r["lt_error"], r["sivr_llm"]) for r in analysis if r["ambiguity_level"] == al]
            if pts:
                axes[0].scatter([p[0] for p in pts], [p[1] for p in pts],
                               marker=ambig_markers[al], color=ambig_colors[al],
                               s=80, alpha=0.8, edgecolor="black", label=al)
        axes[0].set_xlabel("Lead-Time Estimation Error")
        axes[0].set_ylabel("SIVR")
        axes[0].set_title("LT Error vs SIVR by Template")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        for al in ["clear", "moderate", "vague"]:
            pts = [(r["duration_error"], r["sivr_llm"]) for r in analysis if r["ambiguity_level"] == al]
            if pts:
                axes[1].scatter([p[0] for p in pts], [p[1] for p in pts],
                               marker=ambig_markers[al], color=ambig_colors[al],
                               s=80, alpha=0.8, edgecolor="black", label=al)
        axes[1].set_xlabel("Duration Estimation Error")
        axes[1].set_ylabel("SIVR")
        axes[1].set_title("Duration Error vs SIVR by Template")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / "semantic_error_vs_sivr.png", dpi=150)
        plt.close(fig)

    # Plot 5: Profit by condition for each ambiguity level
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, alevel in enumerate(["clear", "moderate", "vague"]):
        ax = axes[idx]
        ambig_rows = [r for r in ambiguity_summary if r["ambiguity_level"] == alevel]
        conds = [r["condition"] for r in ambig_rows]
        means = [r["mean_profit"] for r in ambig_rows]
        colors = [condition_colors.get(c, "#333") for c in conds]
        labels = [condition_labels.get(c, c) for c in conds]
        bars = ax.bar(labels, means, color=colors, alpha=0.8, edgecolor="black")
        ax.set_title(f"{alevel.upper()} warnings")
        ax.grid(axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Mean Cumulative Profit")
    plt.suptitle(f"Profit by Condition and Ambiguity Level ({interpreter_mode})", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "profit_by_ambiguity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 6: Regret decomposition (LLM condition)
    if regret_decomp:
        llm_decomp = [r for r in regret_decomp if r["condition"] == "LLM"]
        if llm_decomp:
            fig, ax = plt.subplots(figsize=(12, 5))
            tids = [r["template_id"] for r in llm_decomp]
            sem_regrets = [r["semantic_regret"] for r in llm_decomp]
            ctrl_gaps = [r["controller_gap"] for r in llm_decomp]
            hind_advs = [r["hindsight_advantage"] for r in llm_decomp]
            x = np.arange(len(tids))
            width = 0.25
            ax.bar(x - width, sem_regrets, width, label="Semantic Regret", color="#1f77b4", alpha=0.8, edgecolor="black")
            ax.bar(x, ctrl_gaps, width, label="Controller Gap", color="#ff7f0e", alpha=0.8, edgecolor="black")
            ax.bar(x + width, hind_advs, width, label="Hindsight Advantage", color="#9467bd", alpha=0.8, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Gap ($)")
            ax.set_title("Gap Decomposition by Template (LLM condition)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
            plt.tight_layout()
            fig.savefig(output_dir / "regret_decomposition.png", dpi=150)
            plt.close(fig)

    # Plot 7: Trajectory diagnostics — PerfectSemantic vs ActiveWrongOver
    if trajectory_diag:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        tids = [r["template_id"] for r in trajectory_diag]
        x = np.arange(len(tids))
        width = 0.35

        # Fill rate
        ps_fills = [r["ps_mean_fill_rate"] for r in trajectory_diag]
        awo_fills = [r["awo_mean_fill_rate"] for r in trajectory_diag]
        axes[0].bar(x - width/2, ps_fills, width, label="PerfectSemantic", color="#2ca02c", alpha=0.8, edgecolor="black")
        axes[0].bar(x + width/2, awo_fills, width, label="ActiveWrongOver", color="#9467bd", alpha=0.8, edgecolor="black")
        axes[0].set_ylabel("Fill Rate")
        axes[0].set_title("Fill Rate Comparison")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # Average inventory
        ps_inv = [r["ps_mean_avg_inventory"] for r in trajectory_diag]
        awo_inv = [r["awo_mean_avg_inventory"] for r in trajectory_diag]
        axes[1].bar(x - width/2, ps_inv, width, label="PerfectSemantic", color="#2ca02c", alpha=0.8, edgecolor="black")
        axes[1].bar(x + width/2, awo_inv, width, label="ActiveWrongOver", color="#9467bd", alpha=0.8, edgecolor="black")
        axes[1].set_ylabel("Average Inventory")
        axes[1].set_title("Average Inventory Comparison")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        # Stockout periods
        ps_so = [r["ps_mean_stockout_periods"] for r in trajectory_diag]
        awo_so = [r["awo_mean_stockout_periods"] for r in trajectory_diag]
        axes[2].bar(x - width/2, ps_so, width, label="PerfectSemantic", color="#2ca02c", alpha=0.8, edgecolor="black")
        axes[2].bar(x + width/2, awo_so, width, label="ActiveWrongOver", color="#9467bd", alpha=0.8, edgecolor="black")
        axes[2].set_ylabel("Stockout Periods")
        axes[2].set_title("Stockout Periods Comparison")
        axes[2].legend()
        axes[2].grid(axis="y", alpha=0.3)

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)

        plt.suptitle("Trajectory Diagnostics: PerfectSemantic vs ActiveWrongOver", y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / "trajectory_diagnostics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    use_llm = bool(os.environ.get("LLM_API_KEY"))
    run_experiment(use_llm=use_llm)


# =====================================================================
# Phase 4: Semantic-Sensor x Controller Matrix Experiment
# =====================================================================

from src.interpreter import (
    rule_based_extract,
    llm_interpret_with_result,
    LLMResult,
    load_dotenv,
    EXTRACTION_PROMPT,
)

# Semantic sensor types
SENSOR_NOINFO = "NoInfo"
SENSOR_RULEBASED = "RuleBased"
SENSOR_LLM = "LLM"
SENSOR_PERFECT = "PerfectSemantic"

# Controller types
CONTROLLER_HEURISTIC = "Heuristic"
CONTROLLER_OPTIMIZER = "CausalOptimizer"

# Privileged baseline (outside matrix)
CONDITION_HINDSIGHT = "HindsightOracle"


def _run_episode_matrix(
    seed: int,
    sensor: str,
    controller: str,
    disruption: SupplierDisruption,
    template_id: str = "",
    ambiguity_level: str = "",
    interpretation: Optional[Interpretation] = None,
    cost_overrides: Optional[dict] = None,
    threshold: float = PROBABILITY_THRESHOLD,
) -> EpisodeMetrics:
    """Run one episode under an explicit sensor x controller pair.

    CRITICAL: CausalOptimizer receives the INTERPRETED disruption parameters
    (from the sensor), NOT the true parameters.
    """
    if sensor == CONDITION_HINDSIGHT:
        oracle = HindsightOracle(seed=seed, disruption=disruption)
        history, _ = oracle.compute_optimal_trajectory()
        return compute_episode_metrics(
            history, seed=seed, condition="HindsightOracle",
            template_id=template_id, ambiguity_level=ambiguity_level,
        )

    env = InventoryEnv(seed=seed, disruption=disruption)
    state = env.reset()
    history = []

    base_policy = BaseStockPolicy(lead_time=disruption.normal_lead_time)
    disrupt_policy = DisruptionAwarePolicy(normal_lead_time=disruption.normal_lead_time)

    switch = False
    estimated_lt_increase = 0
    estimated_duration = HORIZON

    if sensor == SENSOR_PERFECT:
        switch = True
        estimated_lt_increase = disruption.disrupted_lead_time - disruption.normal_lead_time
        estimated_duration = disruption.duration
    elif (sensor == SENSOR_RULEBASED or sensor.startswith("LLM:")) and interpretation is not None:
        if interpretation.probability >= threshold:
            switch = True
            estimated_lt_increase = interpretation.estimated_lead_time_increase
            estimated_duration = interpretation.estimated_duration

    causal_optimizer = None
    if controller == CONTROLLER_OPTIMIZER:
        if sensor == SENSOR_PERFECT:
            # The perfect-semantic reference is the ONLY controller entitled to
            # plan against the environment's true disruption.
            causal_optimizer = CausalOptimizer(
                seed=seed, disruption=disruption, knows_true_disruption=True,
            )
        elif (sensor == SENSOR_RULEBASED or sensor.startswith("LLM:")) and interpretation is not None:
            if interpretation.probability >= threshold:
                causal_optimizer = CausalOptimizer(
                    seed=seed, disruption=disruption,
                    assumed_lt_increase=interpretation.estimated_lead_time_increase,
                    assumed_duration=interpretation.estimated_duration,
                )
            else:
                # Interpretation below threshold: fall back to the nominal
                # no-disruption plan, NOT to the true disruption.
                causal_optimizer = CausalOptimizer(seed=seed, disruption=disruption)
        else:
            # NoInfo and every other uninformed sensor plan under the nominal
            # lead time; `disruption` supplies only its public normal_lead_time.
            causal_optimizer = CausalOptimizer(seed=seed, disruption=disruption)

    adaptation_end = WARNING_TIME + estimated_duration if switch else 0

    for t in range(HORIZON):
        if causal_optimizer is not None:
            order_qty = causal_optimizer.decide(state)
        else:
            use_disruption = switch and (WARNING_TIME <= t < adaptation_end)
            if use_disruption:
                order_qty = disrupt_policy(state, estimated_lt_increase=estimated_lt_increase)
            else:
                order_qty = base_policy(state)
        state = env.step(order_qty)
        history.append(state)

    predicted_event_type = ""
    predicted_probability = 0.0
    predicted_lt_increase = 0
    predicted_duration_val = 0
    event_type_correct = None
    lt_error = None
    dur_error = None

    if sensor == SENSOR_PERFECT:
        event_type_correct = True
        lt_error = 0
        dur_error = 0
    elif interpretation is not None:
        predicted_event_type = interpretation.event_type
        predicted_probability = interpretation.probability
        predicted_lt_increase = interpretation.estimated_lead_time_increase
        predicted_duration_val = interpretation.estimated_duration
        event_type_correct = interpretation.event_type == TRUE_EVENT_TYPE
        lt_error = abs(TRUE_LT_INCREASE - interpretation.estimated_lead_time_increase)
        dur_error = abs(TRUE_DURATION - interpretation.estimated_duration)

    condition_label = f"{sensor}+{controller}"

    return compute_episode_metrics(
        history, seed=seed, condition=condition_label,
        template_id=template_id, ambiguity_level=ambiguity_level,
        predicted_event_type=predicted_event_type,
        predicted_probability=predicted_probability,
        predicted_lt_increase=predicted_lt_increase,
        predicted_duration=predicted_duration_val,
        event_type_correct=event_type_correct,
        lead_time_error=lt_error,
        duration_error=dur_error,
    )


def _get_sensor_interpretation(
    sensor: str, text: str, llm_model: Optional[str] = None
) -> tuple:
    """Get interpretation from a semantic sensor. Returns (interpretation, llm_result)."""
    if sensor == SENSOR_NOINFO:
        return Interpretation("none", 0.0, 0, 0), None
    elif sensor == SENSOR_RULEBASED:
        return rule_based_extract(text), None
    elif sensor == SENSOR_LLM:
        result = llm_interpret_with_result(text, model=llm_model)
        return result.interpretation, result
    elif sensor == SENSOR_PERFECT:
        return Interpretation("supply_disruption", 1.0, TRUE_LT_INCREASE, TRUE_DURATION), None
    else:
        raise ValueError(f"Unknown sensor: {sensor}")


def run_matrix_experiment(
    sensors=None,
    controllers=None,
    num_seeds=15,
    seed_base=SEED_BASE,
    disruption=DEFAULT_DISRUPTION,
    output_dir=None,
    llm_models=None,
    threshold=PROBABILITY_THRESHOLD,
):
    """Run the Phase 4 semantic-sensor x controller matrix experiment."""
    load_dotenv()

    if sensors is None:
        sensors = [SENSOR_NOINFO, SENSOR_RULEBASED, SENSOR_PERFECT]
    if controllers is None:
        controllers = [CONTROLLER_HEURISTIC, CONTROLLER_OPTIMIZER]
    if llm_models is None:
        llm_models = []
    if output_dir is None:
        output_dir = RESULTS_DIR / "phase4_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sensors = list(dict.fromkeys(sensors))
    for model in llm_models:
        label = f"LLM:{model}"
        if label not in all_sensors:
            all_sensors.append(label)

    # Save extraction prompt
    prompt_dir = output_dir.parent / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    (prompt_dir / "semantic_extraction_v1.txt").write_text(EXTRACTION_PROMPT)

    templates = WARNING_TEMPLATES
    total_episodes = (len(all_sensors) * len(controllers) * num_seeds * len(templates)
                      + len(templates) * num_seeds)

    print(f"Phase 4 Matrix Experiment")
    print(f"Sensors: {all_sensors}")
    print(f"Controllers: {controllers}")
    print(f"Templates: {len(templates)} | Seeds: {num_seeds}")
    print(f"Total episodes: {total_episodes}")

    # Phase 1: Get all interpretations
    print("\n--- Semantic Interpretations ---")
    template_interps = {}

    for tmpl in templates:
        tid = tmpl["template_id"]
        text = tmpl["text"]
        template_interps[tid] = {}
        for sensor in all_sensors:
            if sensor.startswith("LLM:"):
                model = sensor.split(":", 1)[1]
                interp, llm_result = _get_sensor_interpretation(SENSOR_LLM, text, llm_model=model)
            else:
                interp, llm_result = _get_sensor_interpretation(sensor, text)
            template_interps[tid][sensor] = (interp, llm_result)

        parts = []
        for sensor in all_sensors:
            interp, _ = template_interps[tid][sensor]
            parts.append(f"{sensor[:12]}: prob={interp.probability:.2f} lt+={interp.estimated_lead_time_increase} dur={interp.estimated_duration}")
        print(f"  {tid:15s} " + " | ".join(parts))

    # Phase 2: Run all episodes
    all_episodes = []
    all_llm_results = []
    done = 0

    for tmpl in templates:
        tid = tmpl["template_id"]
        alevel = tmpl["ambiguity_level"]

        for seed_i in range(num_seeds):
            seed = seed_base + seed_i

            metrics = _run_episode_matrix(
                seed=seed, sensor=CONDITION_HINDSIGHT, controller=CONTROLLER_HEURISTIC,
                disruption=disruption, template_id=tid, ambiguity_level=alevel,
            )
            all_episodes.append(metrics)
            done += 1

            for sensor in all_sensors:
                interp, llm_result = template_interps[tid][sensor]

                if llm_result is not None:
                    all_llm_results.append({
                        "model": llm_result.model,
                        "template_id": tid,
                        "ambiguity_level": alevel,
                        "raw_response": llm_result.raw_response,
                        "parsed_json": json.dumps(interp.to_dict()),
                        "latency_ms": f"{llm_result.latency_ms:.1f}",
                        "from_cache": llm_result.from_cache,
                        "retries": llm_result.retries,
                        "malformed": llm_result.malformed,
                        "event_type_correct": interp.event_type == TRUE_EVENT_TYPE,
                        "lt_error": abs(TRUE_LT_INCREASE - interp.estimated_lead_time_increase),
                        "duration_error": abs(TRUE_DURATION - interp.estimated_duration),
                    })

                for controller in controllers:
                    metrics = _run_episode_matrix(
                        seed=seed, sensor=sensor, controller=controller,
                        disruption=disruption, template_id=tid, ambiguity_level=alevel,
                        interpretation=interp, threshold=threshold,
                    )
                    all_episodes.append(metrics)
                    done += 1

            if done % 50 == 0 or done == total_episodes:
                print(f"  Progress: {done}/{total_episodes} episodes")

    # Phase 3: Save outputs
    _save_matrix_outputs(all_episodes, all_llm_results, all_sensors, controllers,
                         llm_models, templates, output_dir, threshold)

    # Phase 4: Compute analysis
    results = _compute_matrix_analysis(all_episodes, all_sensors, controllers,
                                       llm_models, templates, output_dir)
    results["episodes"] = all_episodes
    results["llm_results"] = all_llm_results
    results["all_sensors"] = all_sensors
    results["controllers"] = controllers
    results["llm_models"] = llm_models
    return results


def _save_matrix_outputs(episodes, llm_results, sensors, controllers,
                         llm_models, templates, output_dir, threshold):
    """Save all CSV outputs for the matrix experiment."""

    if llm_results:
        with open(output_dir / "real_llm_interpretations.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "template_id", "ambiguity_level",
                "raw_response", "parsed_json", "latency_ms",
                "from_cache", "retries", "malformed",
                "event_type_correct", "lt_error", "duration_error",
            ])
            writer.writeheader()
            writer.writerows(llm_results)

    with open(output_dir / "episode_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "template_id", "ambiguity_level", "seed", "condition",
            "total_profit", "fill_rate", "total_lost_sales", "avg_inventory", "periods_stockout",
            "predicted_event_type", "predicted_probability", "predicted_lt_increase", "predicted_duration",
            "event_type_correct", "lead_time_error", "duration_error",
        ])
        for m in episodes:
            writer.writerow([
                m.template_id, m.ambiguity_level, m.seed, m.condition,
                f"{m.total_profit:.2f}", f"{m.fill_rate:.4f}", f"{m.total_lost_sales:.0f}",
                f"{m.avg_inventory:.2f}", m.periods_stockout,
                m.predicted_event_type, f"{m.predicted_probability:.4f}",
                m.predicted_lt_increase, m.predicted_duration,
                m.event_type_correct, m.lead_time_error, m.duration_error,
            ])

    with open(output_dir / "information_sets.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "semantic_info", "policy", "optimization", "known_future_demand"])
        for sensor in sensors:
            for controller in controllers:
                writer.writerow([
                    f"{sensor}+{controller}", sensor, controller,
                    "LP optimizer" if controller == "CausalOptimizer" else "Heuristic", "No",
                ])
        writer.writerow(["HindsightOracle", "Perfect", "MILP global", "Full MILP", "Yes"])

    if llm_results:
        with open(output_dir / "model_semantic_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "template_id", "ambiguity_level",
                "event_type_correct", "lt_error", "duration_error",
                "probability", "malformed",
            ])
            for r in llm_results:
                interp = Interpretation.from_dict(json.loads(r["parsed_json"]))
                writer.writerow([
                    r["model"], r["template_id"], r["ambiguity_level"],
                    r["event_type_correct"], r["lt_error"], r["duration_error"],
                    f"{interp.probability:.4f}", r["malformed"],
                ])


def _compute_matrix_analysis(episodes, sensors, controllers, llm_models, templates, output_dir):
    """Compute the full matrix analysis."""
    def mean_profit(condition, template_id=""):
        if template_id:
            ms = [m for m in episodes if m.condition == condition and m.template_id == template_id]
        else:
            ms = [m for m in episodes if m.condition == condition]
        return float(np.mean([m.total_profit for m in ms])) if ms else 0.0

    all_sensor_labels = list(dict.fromkeys(sensors))
    for model in llm_models:
        label = f"LLM:{model}"
        if label not in all_sensor_labels:
            all_sensor_labels.append(label)

    # Controller comparison matrix
    matrix_rows = []
    for sensor in all_sensor_labels:
        for controller in controllers:
            cond = f"{sensor}+{controller}"
            ms = [m for m in episodes if m.condition == cond]
            if not ms:
                continue
            profits = [m.total_profit for m in ms]
            matrix_rows.append({
                "sensor": sensor, "controller": controller, "condition": cond,
                "mean_profit": float(np.mean(profits)),
                "std_profit": float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0,
                "mean_fill_rate": float(np.mean([m.fill_rate for m in ms])),
                "mean_lost_sales": float(np.mean([m.total_lost_sales for m in ms])),
                "n": len(ms),
            })

    with open(output_dir / "controller_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sensor", "controller", "condition", "mean_profit", "std_profit",
            "mean_fill_rate", "mean_lost_sales", "n",
        ])
        writer.writeheader()
        writer.writerows(matrix_rows)

    # SIVR by sensor under each controller
    sivr_rows = []
    for controller in controllers:
        j_noinfo = mean_profit(f"{SENSOR_NOINFO}+{controller}")
        j_ps = mean_profit(f"{SENSOR_PERFECT}+{controller}")
        for sensor in all_sensor_labels:
            j_cond = mean_profit(f"{sensor}+{controller}")
            sivr_rows.append({
                "controller": controller, "sensor": sensor,
                "j_no_info": j_noinfo, "j_perfect_semantic": j_ps,
                "j_condition": j_cond,
                "sivr": information_value(j_cond, j_noinfo, j_ps),
            })

    with open(output_dir / "sivr_by_model_controller.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "controller", "sensor", "j_no_info", "j_perfect_semantic", "j_condition", "sivr",
        ])
        writer.writeheader()
        writer.writerows(sivr_rows)

    # Semantic regret by controller
    regret_rows = []
    for controller in controllers:
        j_ps = mean_profit(f"{SENSOR_PERFECT}+{controller}")
        for sensor in all_sensor_labels:
            if sensor in (SENSOR_NOINFO, SENSOR_PERFECT):
                continue
            j_sensor = mean_profit(f"{sensor}+{controller}")
            regret_rows.append({
                "controller": controller, "sensor": sensor,
                "j_perfect_semantic": j_ps, "j_sensor": j_sensor,
                "semantic_regret": j_ps - j_sensor,
            })

    with open(output_dir / "semantic_regret_by_controller.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "controller", "sensor", "j_perfect_semantic", "j_sensor", "semantic_regret",
        ])
        writer.writeheader()
        writer.writerows(regret_rows)

    # Rule-based comparison
    rb_rows = []
    for controller in controllers:
        j_noinfo = mean_profit(f"{SENSOR_NOINFO}+{controller}")
        j_rb = mean_profit(f"{SENSOR_RULEBASED}+{controller}")
        j_ps = mean_profit(f"{SENSOR_PERFECT}+{controller}")
        rb_rows.append({
            "controller": controller,
            "j_no_info": j_noinfo, "j_rule_based": j_rb, "j_perfect_semantic": j_ps,
            "rb_value_above_noinfo": j_rb - j_noinfo,
            "rb_sivr": (j_rb - j_noinfo) / (j_ps - j_noinfo) if abs(j_ps - j_noinfo) > 1e-9 else 0.0,
        })

    with open(output_dir / "rule_based_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "controller", "j_no_info", "j_rule_based", "j_perfect_semantic",
            "rb_value_above_noinfo", "rb_sivr",
        ])
        writer.writeheader()
        writer.writerows(rb_rows)

    # Model ambiguity summary
    if llm_models:
        ambig_rows = []
        for model in llm_models:
            sensor = f"LLM:{model}"
            for alevel in ["clear", "moderate", "vague"]:
                for controller in controllers:
                    cond = f"{sensor}+{controller}"
                    ms = [m for m in episodes if m.condition == cond and m.ambiguity_level == alevel]
                    if not ms:
                        continue
                    j_ps = mean_profit(f"{SENSOR_PERFECT}+{controller}")
                    j_noinfo = mean_profit(f"{SENSOR_NOINFO}+{controller}")
                    j_cond = float(np.mean([m.total_profit for m in ms]))
                    lt_errors = [m.lead_time_error for m in ms if m.lead_time_error is not None]
                    dur_errors = [m.duration_error for m in ms if m.duration_error is not None]
                    ambig_rows.append({
                        "model": model, "ambiguity_level": alevel, "controller": controller,
                        "mean_profit": j_cond,
                        "sivr": information_value(j_cond, j_noinfo, j_ps),
                        "mean_lt_error": float(np.mean(lt_errors)) if lt_errors else None,
                        "mean_duration_error": float(np.mean(dur_errors)) if dur_errors else None,
                        "n": len(ms),
                    })

        with open(output_dir / "model_ambiguity_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "ambiguity_level", "controller", "mean_profit",
                "sivr", "mean_lt_error", "mean_duration_error", "n",
            ])
            writer.writeheader()
            writer.writerows(ambig_rows)

    # Threshold sensitivity
    _run_threshold_sensitivity(episodes, sensors, controllers, llm_models,
                               templates, output_dir.parent / "threshold_sensitivity")

    # Matrix plots
    _generate_matrix_plots(matrix_rows, sivr_rows, regret_rows, rb_rows,
                           llm_models, controllers, output_dir)

    # Print summary
    _print_matrix_summary(episodes, all_sensor_labels, controllers, llm_models,
                          sivr_rows, regret_rows)

    return {
        "matrix_rows": matrix_rows,
        "sivr_rows": sivr_rows,
        "regret_rows": regret_rows,
        "rb_rows": rb_rows,
    }


def _print_matrix_summary(episodes, all_sensors, controllers, llm_models, sivr_rows, regret_rows):
    """Print the Phase 4 matrix results."""
    print("\n" + "=" * 70)
    print("PHASE 4 MATRIX RESULTS")
    print("=" * 70)

    print(f"\n--- Semantic Sensor x Controller Matrix (Mean Profit) ---")
    header = f"  {'Sensor':20s}"
    for c in controllers:
        header += f" {c:>18s}"
    print(header)
    print("  " + "-" * (20 + 19 * len(controllers)))

    for sensor in all_sensors:
        row = f"  {sensor:20s}"
        for controller in controllers:
            cond = f"{sensor}+{controller}"
            ms = [m for m in episodes if m.condition == cond]
            if ms:
                row += f" {np.mean([m.total_profit for m in ms]):18.1f}"
            else:
                row += f" {'N/A':>18s}"
        print(row)

    ho_ms = [m for m in episodes if m.condition == "HindsightOracle"]
    if ho_ms:
        print(f"  {'HindsightOracle':20s} {np.mean([m.total_profit for m in ho_ms]):18.1f}")

    print(f"\n--- SIVR by Controller ---")
    for controller in controllers:
        print(f"\n  {controller}:")
        for row in sivr_rows:
            if row["controller"] == controller:
                print(f"    {row['sensor']:20s} SIVR={row['sivr']:.3f}  (J={row['j_condition']:.1f})")

    print(f"\n--- Semantic Regret by Controller ---")
    for controller in controllers:
        print(f"\n  {controller}:")
        for row in regret_rows:
            if row["controller"] == controller:
                print(f"    {row['sensor']:20s} Regret={row['semantic_regret']:+.1f}")

    print("=" * 70)


def _run_threshold_sensitivity(episodes, sensors, controllers, llm_models,
                                templates, output_dir):
    """Evaluate results under multiple threshold settings (offline analysis)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []

    all_sensor_labels = list(dict.fromkeys(sensors))
    for model in llm_models:
        label = f"LLM:{model}"
        if label not in all_sensor_labels:
            all_sensor_labels.append(label)

    # Group episodes by (condition, template, seed) for re-evaluation
    by_cond_tmpl_seed = defaultdict(dict)
    for m in episodes:
        by_cond_tmpl_seed[(m.condition, m.template_id, m.seed)] = m

    # For each threshold, we need to re-evaluate which episodes switch policy
    # We use the semantic interpretations to determine switching
    for threshold in thresholds:
        for sensor in all_sensor_labels:
            if sensor in (SENSOR_NOINFO, SENSOR_PERFECT):
                # NoInfo and PerfectSemantic are threshold-independent
                for controller in controllers:
                    cond = f"{sensor}+{controller}"
                    ms = [m for m in episodes if m.condition == cond]
                    if ms:
                        rows.append({
                            "threshold": threshold,
                            "sensor": sensor,
                            "controller": controller,
                            "mean_profit": float(np.mean([m.total_profit for m in ms])),
                            "n": len(ms),
                        })
            else:
                # For LLM/RuleBased: re-classify which episodes switch at this threshold
                for controller in controllers:
                    cond = f"{sensor}+{controller}"
                    switching = []
                    for (c, tid, seed), m in by_cond_tmpl_seed.items():
                        if c != cond:
                            continue
                        if sensor == SENSOR_RULEBASED:
                            # RuleBased: switches if probability > 0 (always for RuleBased)
                            switching.append(m)
                        else:
                            # LLM: switches if probability >= threshold
                            if m.predicted_probability >= threshold:
                                switching.append(m)

                    if switching:
                        rows.append({
                            "threshold": threshold,
                            "sensor": sensor,
                            "controller": controller,
                            "mean_profit": float(np.mean([m.total_profit for m in switching])),
                            "n": len(switching),
                        })

    if rows:
        with open(output_dir / "threshold_sensitivity.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "threshold", "sensor", "controller", "mean_profit", "n",
            ])
            writer.writeheader()
            writer.writerows(rows)

    # Print summary
    print(f"\n--- Threshold Sensitivity ---")
    print(f"  {'Threshold':>10s} {'Sensor':15s} {'Controller':15s} {'Profit':>10s}")
    for row in rows:
        print(f"  {row['threshold']:10.2f} {row['sensor']:15s} {row['controller']:15s} {row['mean_profit']:10.1f}")


def _generate_matrix_plots(matrix_rows, sivr_rows, regret_rows, rb_rows,
                           llm_models, controllers, output_dir):
    """Generate Phase 4 matrix plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sensor_colors = {
        "NoInfo": "#d62728",
        "RuleBased": "#ff7f0e",
        "PerfectSemantic": "#2ca02c",
    }
    for i, model in enumerate(llm_models):
        sensor_colors[f"LLM:{model}"] = f"C{i}"

    # Plot 1: Profit matrix (sensor x controller)
    fig, axes = plt.subplots(1, len(controllers), figsize=(7 * len(controllers), 5), sharey=True)
    if len(controllers) == 1:
        axes = [axes]

    for idx, controller in enumerate(controllers):
        ax = axes[idx]
        ctrl_rows = [r for r in matrix_rows if r["controller"] == controller]
        labels = [r["sensor"] for r in ctrl_rows]
        means = [r["mean_profit"] for r in ctrl_rows]
        colors = [sensor_colors.get(l, "#333") for l in labels]
        bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Controller: {controller}")
        ax.grid(axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Mean Cumulative Profit")
    plt.suptitle("Profit by Semantic Sensor and Controller", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "profit_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: SIVR comparison
    fig, axes = plt.subplots(1, len(controllers), figsize=(7 * len(controllers), 5), sharey=True)
    if len(controllers) == 1:
        axes = [axes]

    for idx, controller in enumerate(controllers):
        ax = axes[idx]
        ctrl_sivrs = [r for r in sivr_rows if r["controller"] == controller]
        labels = [r["sensor"] for r in ctrl_sivrs]
        values = [r["sivr"] for r in ctrl_sivrs]
        colors = [sensor_colors.get(l, "#333") for l in labels]
        ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Controller: {controller}")
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-0.2, 1.3)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("SIVR")
    plt.suptitle("Semantic Information Value Recovery by Sensor and Controller", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "sivr_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Semantic regret comparison
    if regret_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(controllers))
        width = 0.25
        sensors_for_plot = list(set(r["sensor"] for r in regret_rows))

        for i, sensor in enumerate(sensors_for_plot):
            vals = []
            for controller in controllers:
                row = next((r for r in regret_rows if r["sensor"] == sensor and r["controller"] == controller), None)
                vals.append(row["semantic_regret"] if row else 0)
            ax.bar(x + i * width, vals, width, label=sensor,
                   color=sensor_colors.get(sensor, f"C{i}"), alpha=0.8, edgecolor="black")

        ax.set_xticks(x + width * (len(sensors_for_plot) - 1) / 2)
        ax.set_xticklabels(controllers)
        ax.set_ylabel("Semantic Regret ($)")
        ax.set_title("Semantic Regret by Controller")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(output_dir / "semantic_regret_comparison.png", dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to {output_dir}")
