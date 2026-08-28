"""Phase 6: Confirmation experiment.

Publication-grade robustness, replication, and statistical scrutiny
of the Phase-5.5 finding that semantic belief quality causally affects
downstream operational value.

DO NOT tune the frozen benchmark. This module confirms, it does not modify.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import warnings
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.events import (
    Regime, REGIME_PRIOR, REGIME_PARAMS, REGIME_WARNING_TEMPLATES,
    P5_HORIZON, P5_NORMAL_LEAD_TIME, P5_DEMAND_MEAN,
    P5_EVENT_START, P5_EVENT_DURATION, P5_WARNING_TIME,
)
from src.env import (
    InventoryEnv, InventoryState, CausalOptimizer, HindsightOracle,
    REVENUE_PER_UNIT, ORDERING_FIXED_COST, ORDERING_VARIABLE_COST,
    HOLDING_COST_PER_UNIT, STOCKOUT_COST_PER_UNIT, INITIAL_INVENTORY,
)
from src.interpreter import (
    RegimeInterpretation, no_info_regime_belief, perfect_semantic_regime_belief,
    rule_based_regime_extract, llm_regime_interpret_with_result, load_dotenv,
    EXTRACTION_PROMPT, REGIME_EXTRACTION_PROMPT,
)
from src.experiment_phase5 import _run_p5_episode
from src.confirmation_templates import CONFIRMATION_TEMPLATES
from src.metrics import signed_sivr, standard_brier_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase6"

NOINFO = "NoInfo"
RULEBASED = "RuleBased"
PERFECT = "PerfectSemantic"
CONTROLLER = "CausalOptimizer"


def compute_episode_count(num_regimes, num_seeds, num_templates_per_regime, num_sensors):
    return num_regimes * num_seeds * num_templates_per_regime * num_sensors

def compute_semantic_call_count(num_text_sensors, num_templates):
    return num_text_sensors * num_templates

def compute_paired_comparison_count(num_comparison_sensors, num_seeds, num_regimes):
    return num_comparison_sensors * num_seeds * num_regimes

def brier_score(predicted, true_regime):
    return standard_brier_score(
        predicted, true_regime,
        class_order=["normal", "supplier_delay", "demand_surge"],
    )

def log_loss_safe(predicted, true_regime, eps=1e-10):
    prob = max(predicted.get(true_regime, eps), eps)
    return -math.log(prob)

def aggregate_sivr(sensor_profits, ni_profits, ps_profits):
    num = np.sum(sensor_profits - ni_profits)
    return signed_sivr(float(np.mean(sensor_profits)), float(np.mean(ni_profits)),
                       float(np.mean(ps_profits))).value

def macro_sivr(sivr_values):
    if not sivr_values:
        return 0.0
    return float(np.mean(sivr_values))


def paired_bootstrap_ci(data_a, data_b, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    diffs = data_a - data_b
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diffs[idx])
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(ci_low), float(ci_high)


def hierarchical_bootstrap_ci(template_profits, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    template_ids = list(template_profits.keys())
    n_templates = len(template_ids)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        t_idx = rng.integers(0, n_templates, size=n_templates)
        vals = []
        for ti in t_idx:
            tid = template_ids[ti]
            arr = template_profits[tid]
            s_idx = rng.integers(0, len(arr), size=len(arr))
            vals.extend(arr[s_idx])
        boot_means[i] = np.mean(vals)
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(boot_means)), float(ci_low), float(ci_high)


def calibration_analysis(beliefs, n_bins=5):
    if not beliefs:
        return {"ece": 0.0, "bins": []}
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [b for b in beliefs if lo <= b["confidence"] < hi]
        if not in_bin:
            continue
        mean_conf = float(np.mean([b["confidence"] for b in in_bin]))
        mean_acc = float(np.mean([1.0 if b["correct"] else 0.0 for b in in_bin]))
        bin_data.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "mean_confidence": mean_conf, "mean_accuracy": mean_acc,
            "count": len(in_bin),
        })
    total = sum(b["count"] for b in bin_data)
    if total == 0:
        return {"ece": 0.0, "bins": []}
    ece = sum(b["count"] / total * abs(b["mean_confidence"] - b["mean_accuracy"])
              for b in bin_data)
    return {"ece": float(ece), "bins": bin_data}


def _save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_confirmation_experiment(
    num_seeds=50, seed_base=2000, llm_models=None,
    use_confirmation_templates=True, run_stochasticity=True, stochasticity_reps=5,
):
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
        os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"

    if llm_models is None:
        llm_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

    if use_confirmation_templates:
        templates = CONFIRMATION_TEMPLATES
        template_source = "confirmation"
    else:
        templates = REGIME_WARNING_TEMPLATES
        template_source = "original"

    regimes = [Regime.NORMAL, Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]
    seeds = list(range(seed_base, seed_base + num_seeds))

    sensors = [NOINFO, RULEBASED, PERFECT]
    text_sensors = [RULEBASED]
    for model in llm_models:
        sensors.append("LLM:" + model)
        text_sensors.append("LLM:" + model)

    templates_per_regime = len([t for t in templates if t["regime"] == regimes[0]])
    total_templates = len(templates)
    num_text_sensors = len(text_sensors)
    num_comparison_sensors = len(sensors) - 1

    ep_count = compute_episode_count(len(regimes), len(seeds), templates_per_regime, len(sensors))
    sem_calls = compute_semantic_call_count(num_text_sensors, total_templates)
    paired_count = compute_paired_comparison_count(num_comparison_sensors, len(seeds), len(regimes))

    accounting = {
        "num_sensors": len(sensors), "num_text_sensors": num_text_sensors,
        "num_llm_models": len(llm_models), "llm_models": json.dumps(llm_models),
        "num_templates": total_templates, "templates_per_regime": templates_per_regime,
        "num_regimes": len(regimes), "num_seeds": len(seeds), "seed_base": seed_base,
        "controller": CONTROLLER, "template_source": template_source,
        "baselines_repeated_per_template": True,
        "total_operational_episodes": ep_count,
        "unique_semantic_calls": sem_calls,
        "unique_llm_api_calls": len(llm_models) * total_templates,
        "paired_comparisons": paired_count,
    }
    _save_csv([accounting], output_dir / "experiment_accounting.csv")

    print("=" * 70)
    print("PHASE 6: CONFIRMATION EXPERIMENT")
    print("=" * 70)
    print("Template source: {} ({} templates)".format(template_source, total_templates))
    print("Seeds: {} ({}-{})".format(len(seeds), seeds[0], seeds[-1]))
    print("Sensors: {}".format(len(sensors)))
    print("Episodes: {} | Semantic calls: {} | LLM API: {}".format(
        ep_count, sem_calls, accounting["unique_llm_api_calls"]))
    print()

    print("Pre-caching LLM responses...")
    for model in llm_models:
        for tmpl in templates:
            ri, result = llm_regime_interpret_with_result(tmpl["text"], model=model)
            status = "cached" if result.from_cache else "fetched"
            print("  {} | {} | {}".format(model, tmpl["template_id"], status))
    print()

    print("Running operational episodes...")
    all_episodes = []
    all_beliefs = []
    all_raw_responses = []
    api_stats = {"total_calls": 0, "cache_hits": 0, "malformed": 0, "api_errors": 0}

    regime_templates_map = {r: [t for t in templates if t["regime"] == r] for r in regimes}

    for regime in regimes:
        r_templates = regime_templates_map[regime]
        for seed in seeds:
            for tmpl in r_templates:
                text = tmpl["text"]
                tid = tmpl["template_id"]
                alevel = tmpl["ambiguity_level"]

                for sensor in sensors:
                    if sensor == NOINFO:
                        ri = no_info_regime_belief()
                        raw_resp, from_cache, latency, malformed = "", False, 0.0, False
                    elif sensor == PERFECT:
                        ri = perfect_semantic_regime_belief(regime)
                        raw_resp, from_cache, latency, malformed = "", False, 0.0, False
                    elif sensor == RULEBASED:
                        ri = rule_based_regime_extract(text)
                        raw_resp = json.dumps(ri.to_dict())
                        from_cache, latency, malformed = False, 0.0, False
                    elif sensor.startswith("LLM:"):
                        model = sensor.split(":", 1)[1]
                        api_stats["total_calls"] += 1
                        try:
                            ri, llm_result = llm_regime_interpret_with_result(text, model=model)
                            raw_resp = llm_result.raw_response
                            from_cache = llm_result.from_cache
                            latency = llm_result.latency_ms
                            malformed = llm_result.malformed
                            if from_cache:
                                api_stats["cache_hits"] += 1
                            if malformed:
                                api_stats["malformed"] += 1
                        except Exception as e:
                            ri = no_info_regime_belief()
                            raw_resp = str(e)
                            from_cache, latency, malformed = False, 0.0, True
                            api_stats["api_errors"] += 1
                    else:
                        ri = no_info_regime_belief()
                        raw_resp, from_cache, latency, malformed = "", False, 0.0, False

                    belief = ri.normalized()

                    result, history = _run_p5_episode(
                        seed=seed, regime=regime, sensor=sensor,
                        controller=CONTROLLER, template_id=tid,
                        ambiguity_level=alevel, regime_interp=ri,
                    )

                    all_episodes.append({
                        "seed": seed, "regime": regime.value,
                        "template_id": tid, "ambiguity_level": alevel,
                        "sensor": sensor, "controller": CONTROLLER,
                        "total_profit": result.total_profit,
                        "fill_rate": result.fill_rate,
                        "total_lost_sales": result.total_lost_sales,
                        "avg_inventory": result.avg_inventory,
                        "periods_stockout": result.periods_stockout,
                        "belief_normal": belief.regime_probabilities.get("normal", 0.0),
                        "belief_delay": belief.regime_probabilities.get("supplier_delay", 0.0),
                        "belief_surge": belief.regime_probabilities.get("demand_surge", 0.0),
                        "most_likely_regime": belief.most_likely_regime,
                        "belief_correct": belief.most_likely_regime == regime.value,
                        "est_lt_increase": belief.estimated_lt_increase,
                        "est_duration": belief.estimated_duration,
                        "est_demand_multiplier": belief.estimated_demand_multiplier,
                        "template_source": template_source,
                    })

                    confidence = max(belief.regime_probabilities.values())
                    all_beliefs.append({
                        "sensor": sensor, "regime": regime.value,
                        "template_id": tid, "ambiguity_level": alevel,
                        "seed": seed, "template_source": template_source,
                        "belief_normal": belief.regime_probabilities.get("normal", 0.0),
                        "belief_delay": belief.regime_probabilities.get("supplier_delay", 0.0),
                        "belief_surge": belief.regime_probabilities.get("demand_surge", 0.0),
                        "most_likely": belief.most_likely_regime,
                        "correct": belief.most_likely_regime == regime.value,
                        "true_regime": regime.value,
                        "confidence": float(confidence),
                        "brier": brier_score(belief.regime_probabilities, regime.value),
                        "log_loss": log_loss_safe(belief.regime_probabilities, regime.value),
                        "est_lt_increase": belief.estimated_lt_increase,
                        "est_duration": belief.estimated_duration,
                        "est_demand_multiplier": belief.estimated_demand_multiplier,
                    })

                    if sensor.startswith("LLM:") or sensor == RULEBASED:
                        all_raw_responses.append({
                            "model": sensor, "template_id": tid,
                            "regime": regime.value, "ambiguity_level": alevel,
                            "template_source": template_source,
                            "raw_response": raw_resp,
                            "parsed_json": json.dumps(belief.to_dict()),
                            "latency_ms": latency, "from_cache": from_cache,
                            "malformed": malformed,
                        })

    _save_csv(all_episodes, output_dir / "operational_results.csv")
    _save_csv(all_beliefs, output_dir / "belief_metrics.csv")
    _save_csv(all_raw_responses, output_dir / "raw_llm_responses.csv")
    _save_csv([api_stats], output_dir / "api_reliability.csv")

    # =========================================================================
    # Compute baselines per regime
    # =========================================================================
    ni_by_regime = defaultdict(list)
    ps_by_regime = defaultdict(list)
    for ep in all_episodes:
        if ep["sensor"] == NOINFO:
            ni_by_regime[ep["regime"]].append(ep["total_profit"])
        elif ep["sensor"] == PERFECT:
            ps_by_regime[ep["regime"]].append(ep["total_profit"])

    ni_means = {r: float(np.mean(v)) for r, v in ni_by_regime.items()}
    ps_means = {r: float(np.mean(v)) for r, v in ps_by_regime.items()}

    # =========================================================================
    # SIVR by sensor x regime (template-level then aggregated)
    # =========================================================================
    sivr_rows = []
    template_sivr_data = defaultdict(lambda: defaultdict(list))

    for sensor in sensors:
        for regime in regimes:
            sensor_profits_by_template = defaultdict(list)
            for ep in all_episodes:
                if ep["sensor"] == sensor and ep["regime"] == regime.value:
                    sensor_profits_by_template[ep["template_id"]].append(ep["total_profit"])

            ni_arr = np.array(ni_by_regime[regime.value])
            ps_arr = np.array(ps_by_regime[regime.value])

            template_sivrs = []
            for tid, profits in sensor_profits_by_template.items():
                sp = np.array(profits)
                ts = aggregate_sivr(sp, ni_arr[:len(sp)], ps_arr[:len(sp)])
                template_sivrs.append(ts)
                template_sivr_data[sensor][regime.value].append(ts)

            all_sensor_profits = []
            for profits in sensor_profits_by_template.values():
                all_sensor_profits.extend(profits)
            sensor_mean = float(np.mean(all_sensor_profits)) if all_sensor_profits else 0.0
            ni_mean = ni_means[regime.value]
            ps_mean = ps_means[regime.value]
            overall_sivr = signed_sivr(sensor_mean, ni_mean, ps_mean).value

            sivr_rows.append({
                "sensor": sensor, "regime": regime.value,
                "template_source": template_source,
                "sensor_profit": sensor_mean,
                "noinfo_profit": ni_mean, "perfect_profit": ps_mean,
                "oiv": ps_mean - ni_mean,
                "rov": sensor_mean - ni_mean,
                "semantic_regret": ps_mean - sensor_mean,
                "sivr": overall_sivr,
                "macro_sivr": macro_sivr(template_sivrs),
                "n": len(all_sensor_profits),
                "std": float(np.std(all_sensor_profits)) if all_sensor_profits else 0.0,
            })

    _save_csv(sivr_rows, output_dir / "sivr_by_regime.csv")
    _save_csv(sivr_rows, output_dir / "aggregate_sivr.csv")

    # =========================================================================
    # SIVR by sensor x ambiguity
    # =========================================================================
    sivr_ambig_rows = []
    for sensor in sensors:
        for alevel in ["clear", "moderate", "vague"]:
            sensor_profits = [ep["total_profit"] for ep in all_episodes
                              if ep["sensor"] == sensor and ep["ambiguity_level"] == alevel]
            all_ni = [ep["total_profit"] for ep in all_episodes if ep["sensor"] == NOINFO]
            all_ps = [ep["total_profit"] for ep in all_episodes if ep["sensor"] == PERFECT]
            if not sensor_profits:
                continue
            sensor_mean = float(np.mean(sensor_profits))
            ni_mean = float(np.mean(all_ni))
            ps_mean = float(np.mean(all_ps))
            rov = sensor_mean - ni_mean
            sr = ps_mean - sensor_mean
            sivr = signed_sivr(sensor_mean, ni_mean, ps_mean).value
            oiv = ps_mean - ni_mean

            sivr_ambig_rows.append({
                "sensor": sensor, "ambiguity_level": alevel,
                "template_source": template_source,
                "sensor_profit": sensor_mean, "noinfo_profit": ni_mean,
                "perfect_profit": ps_mean, "oiv": oiv, "rov": rov,
                "semantic_regret": sr, "sivr": sivr,
                "n": len(sensor_profits),
            })
    _save_csv(sivr_ambig_rows, output_dir / "ambiguity_summary.csv")

    # =========================================================================
    # Belief quality by sensor
    # =========================================================================
    belief_metrics = []
    for sensor in sensors:
        sensor_beliefs = [b for b in all_beliefs if b["sensor"] == sensor]
        if not sensor_beliefs:
            continue
        acc = float(np.mean([b["correct"] for b in sensor_beliefs]))
        brier = float(np.mean([b["brier"] for b in sensor_beliefs]))
        logloss = float(np.mean([b["log_loss"] for b in sensor_beliefs]))
        true_probs = []
        for b in sensor_beliefs:
            tr = b["true_regime"]
            if tr == "normal":
                true_probs.append(b["belief_normal"])
            elif tr == "supplier_delay":
                true_probs.append(b["belief_delay"])
            elif tr == "demand_surge":
                true_probs.append(b["belief_surge"])
        true_prob = float(np.mean(true_probs)) if true_probs else 0.0
        mean_conf = float(np.mean([b["confidence"] for b in sensor_beliefs]))

        cal = calibration_analysis([
            {"confidence": b["confidence"], "correct": b["correct"]}
            for b in sensor_beliefs
        ])

        belief_metrics.append({
            "sensor": sensor, "template_source": template_source,
            "accuracy": acc, "brier_score": brier, "log_loss": logloss,
            "true_regime_probability": true_prob, "mean_confidence": mean_conf,
            "ece": cal["ece"], "n": len(sensor_beliefs),
        })
    _save_csv(belief_metrics, output_dir / "belief_metrics_summary.csv")

    # =========================================================================
    # Paired bootstrap CIs
    # =========================================================================
    ci_rows = []
    for sensor in sensors:
        if sensor == NOINFO:
            continue
        for regime in regimes:
            sensor_profits = np.array([
                ep["total_profit"] for ep in all_episodes
                if ep["sensor"] == sensor and ep["regime"] == regime.value
            ])
            ni_profits = np.array([
                ep["total_profit"] for ep in all_episodes
                if ep["sensor"] == NOINFO and ep["regime"] == regime.value
            ])
            ps_profits = np.array([
                ep["total_profit"] for ep in all_episodes
                if ep["sensor"] == PERFECT and ep["regime"] == regime.value
            ])
            min_len = min(len(sensor_profits), len(ni_profits), len(ps_profits))
            sensor_profits = sensor_profits[:min_len]
            ni_profits = ni_profits[:min_len]
            ps_profits = ps_profits[:min_len]

            mean_s_ni, lo_s_ni, hi_s_ni = paired_bootstrap_ci(sensor_profits, ni_profits)
            mean_s_ps, lo_s_ps, hi_s_ps = paired_bootstrap_ci(sensor_profits, ps_profits)

            ci_rows.append({
                "sensor": sensor, "regime": regime.value,
                "template_source": template_source,
                "comparison": "sensor_vs_noinfo",
                "mean_diff": mean_s_ni, "ci_low": lo_s_ni, "ci_high": hi_s_ni,
                "n": min_len,
            })
            ci_rows.append({
                "sensor": sensor, "regime": regime.value,
                "template_source": template_source,
                "comparison": "sensor_vs_perfect",
                "mean_diff": mean_s_ps, "ci_low": lo_s_ps, "ci_high": hi_s_ps,
                "n": min_len,
            })
    _save_csv(ci_rows, output_dir / "paired_confirmatory_tests.csv")

    # =========================================================================
    # Regime summary
    # =========================================================================
    regime_summary = []
    for regime in regimes:
        ni_p = ni_means[regime.value]
        ps_p = ps_means[regime.value]
        oiv = ps_p - ni_p
        for sensor in sensors:
            s_profits = [ep["total_profit"] for ep in all_episodes
                         if ep["sensor"] == sensor and ep["regime"] == regime.value]
            s_mean = float(np.mean(s_profits)) if s_profits else 0.0
            rov = s_mean - ni_p
            sr = ps_p - s_mean
            sivr = rov / oiv if abs(oiv) > 1e-9 else 0.0
            regime_summary.append({
                "regime": regime.value, "sensor": sensor,
                "template_source": template_source,
                "noinfo_profit": ni_p, "perfect_profit": ps_p,
                "sensor_profit": s_mean,
                "oiv": oiv, "rov": rov, "semantic_regret": sr,
                "sivr": sivr,
            })
    _save_csv(regime_summary, output_dir / "regime_summary.csv")

    # =========================================================================
    # Semantic regret
    # =========================================================================
    sr_rows = []
    for sensor in sensors:
        for regime in regimes:
            s_profits = [ep["total_profit"] for ep in all_episodes
                         if ep["sensor"] == sensor and ep["regime"] == regime.value]
            ps_p = ps_means[regime.value]
            s_mean = float(np.mean(s_profits)) if s_profits else 0.0
            sr_rows.append({
                "sensor": sensor, "regime": regime.value,
                "template_source": template_source,
                "semantic_regret": ps_p - s_mean,
                "n": len(s_profits),
            })
    _save_csv(sr_rows, output_dir / "semantic_regret.csv")

    # =========================================================================
    # Hierarchical bootstrap for overall SIVR per sensor
    # =========================================================================
    hier_rows = []
    for sensor in sensors:
        if sensor == NOINFO:
            continue
        sensor_by_template = defaultdict(list)
        ni_by_template = defaultdict(list)
        ps_by_template = defaultdict(list)
        for ep in all_episodes:
            if ep["sensor"] == sensor:
                sensor_by_template[ep["template_id"]].append(ep["total_profit"])
            elif ep["sensor"] == NOINFO:
                ni_by_template[ep["template_id"]].append(ep["total_profit"])
            elif ep["sensor"] == PERFECT:
                ps_by_template[ep["template_id"]].append(ep["total_profit"])

        template_aggregate_diffs = {}
        for tid in sensor_by_template:
            s_arr = np.array(sensor_by_template[tid])
            ni_arr = np.array(ni_by_template.get(tid, s_arr))
            ps_arr = np.array(ps_by_template.get(tid, s_arr))
            min_l = min(len(s_arr), len(ni_arr), len(ps_arr))
            template_aggregate_diffs[tid] = s_arr[:min_l] - ni_arr[:min_l]

        if template_aggregate_diffs:
            mean_val, ci_lo, ci_hi = hierarchical_bootstrap_ci(template_aggregate_diffs)
            hier_rows.append({
                "sensor": sensor, "template_source": template_source,
                "mean_profit_diff_vs_noinfo": mean_val,
                "ci_low": ci_lo, "ci_high": ci_hi,
                "n_templates": len(template_aggregate_diffs),
            })
    _save_csv(hier_rows, output_dir / "hierarchical_bootstrap_results.csv")

    # =========================================================================
    # Calibration by model
    # =========================================================================
    cal_summary = []
    cal_bins = []
    for sensor in sensors:
        sensor_beliefs = [b for b in all_beliefs if b["sensor"] == sensor]
        if not sensor_beliefs:
            continue
        cal = calibration_analysis(sensor_beliefs)
        cal_summary.append({
            "sensor": sensor, "template_source": template_source,
            "ece": cal["ece"], "num_bins": len(cal["bins"]),
        })
        for b in cal["bins"]:
            cal_bins.append({
                "sensor": sensor, "template_source": template_source,
                "bin_lo": b["bin_lo"], "bin_hi": b["bin_hi"],
                "mean_confidence": b["mean_confidence"],
                "mean_accuracy": b["mean_accuracy"],
                "count": b["count"],
            })
    _save_csv(cal_summary, output_dir / "calibration_metrics.csv")
    _save_csv(cal_bins, output_dir / "calibration_bins.csv")

    # =========================================================================
    # RuleBased vs LLM comparison
    # =========================================================================
    rb_vs_llm = []
    for regime in regimes:
        for alevel in ["clear", "moderate", "vague"]:
            rb_eps = [ep for ep in all_episodes if ep["sensor"] == RULEBASED
                      and ep["regime"] == regime.value and ep["ambiguity_level"] == alevel]
            for model in llm_models:
                llm_eps = [ep for ep in all_episodes if ep["sensor"] == "LLM:" + model
                           and ep["regime"] == regime.value and ep["ambiguity_level"] == alevel]
                if rb_eps and llm_eps:
                    rb_profit = float(np.mean([e["total_profit"] for e in rb_eps]))
                    llm_profit = float(np.mean([e["total_profit"] for e in llm_eps]))
                    rb_profits = np.array([e["total_profit"] for e in rb_eps])
                    llm_profits = np.array([e["total_profit"] for e in llm_eps])
                    min_l = min(len(rb_profits), len(llm_profits))
                    _, ci_lo, ci_hi = paired_bootstrap_ci(llm_profits[:min_l], rb_profits[:min_l])
                    rb_vs_llm.append({
                        "regime": regime.value, "ambiguity_level": alevel,
                        "model": model, "template_source": template_source,
                        "rulebased_profit": rb_profit, "llm_profit": llm_profit,
                        "profit_diff": llm_profit - rb_profit,
                        "ci_low": ci_lo, "ci_high": ci_hi,
                    })
    _save_csv(rb_vs_llm, output_dir / "rulebased_vs_llm.csv")

    # =========================================================================
    # Semantic stochasticity experiment
    # =========================================================================
    stochasticity_rows = []
    if run_stochasticity:
        print("Running semantic stochasticity experiment...")
        from src.interpreter import _get_client
        api_key = os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        if api_key:
            client = _get_client(api_key, base_url)
            for model in llm_models:
                for tmpl in templates:
                    text = tmpl["text"]
                    regime_probs_list = []
                    for rep in range(stochasticity_reps):
                        try:
                            response = client.chat.completions.create(
                                model=model, temperature=0.3,
                                messages=[
                                    {"role": "system", "content": REGIME_EXTRACTION_PROMPT},
                                    {"role": "user", "content": text},
                                ],
                            )
                            raw = response.choices[0].message.content.strip()
                            if raw.startswith("```"):
                                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                            parsed = json.loads(raw)
                            ri = RegimeInterpretation.from_dict(parsed).normalized()
                            probs = ri.regime_probabilities
                            stochasticity_rows.append({
                                "model": model, "template_id": tmpl["template_id"],
                                "regime": tmpl["regime"].value,
                                "ambiguity_level": tmpl["ambiguity_level"],
                                "rep": rep, "template_source": template_source,
                                "belief_normal": probs.get("normal", 0.0),
                                "belief_delay": probs.get("supplier_delay", 0.0),
                                "belief_surge": probs.get("demand_surge", 0.0),
                                "confidence": float(max(probs.values())),
                            })
                        except Exception as e:
                            stochasticity_rows.append({
                                "model": model, "template_id": tmpl["template_id"],
                                "regime": tmpl["regime"].value,
                                "ambiguity_level": tmpl["ambiguity_level"],
                                "rep": rep, "template_source": template_source,
                                "error": str(e),
                            })
            print("Stochasticity: {} rows collected".format(len(stochasticity_rows)))
        else:
            print("No API key available, skipping stochasticity experiment")
    _save_csv(stochasticity_rows, output_dir / "semantic_stochasticity.csv")

    # =========================================================================
    # Trajectory mechanisms
    # =========================================================================
    traj_rows = []
    for regime in regimes:
        for seed in seeds[:3]:
            for sensor in [NOINFO, RULEBASED, PERFECT]:
                r_templates = [t for t in templates if t["regime"] == regime]
                if not r_templates:
                    continue
                tmpl = r_templates[0]
                text = tmpl["text"]
                if sensor == NOINFO:
                    ri = no_info_regime_belief()
                elif sensor == PERFECT:
                    ri = perfect_semantic_regime_belief(regime)
                else:
                    ri = rule_based_regime_extract(text)

                result, history = _run_p5_episode(
                    seed=seed, regime=regime, sensor=sensor,
                    controller=CONTROLLER, template_id=tmpl["template_id"],
                    ambiguity_level=tmpl["ambiguity_level"], regime_interp=ri,
                )

                for state in history[:25]:
                    traj_rows.append({
                        "regime": regime.value, "seed": seed,
                        "sensor": sensor, "template_source": template_source,
                        "time": state.time, "on_hand": state.on_hand,
                        "pipeline": state.pipeline, "order_quantity": state.order_quantity,
                        "lost_sales": state.lost_sales,
                        "cumulative_profit": state.cumulative_profit,
                    })

                # Add best LLM if available
                for model in llm_models:
                    llm_sensor = "LLM:" + model
                    llm_ri, _ = llm_regime_interpret_with_result(text, model=model)
                    llm_result, llm_history = _run_p5_episode(
                        seed=seed, regime=regime, sensor=llm_sensor,
                        controller=CONTROLLER, template_id=tmpl["template_id"],
                        ambiguity_level=tmpl["ambiguity_level"], regime_interp=llm_ri,
                    )
                    for state in llm_history[:25]:
                        traj_rows.append({
                            "regime": regime.value, "seed": seed,
                            "sensor": llm_sensor, "template_source": template_source,
                            "time": state.time, "on_hand": state.on_hand,
                            "pipeline": state.pipeline, "order_quantity": state.order_quantity,
                            "lost_sales": state.lost_sales,
                            "cumulative_profit": state.cumulative_profit,
                        })
    _save_csv(traj_rows, output_dir / "trajectory_mechanisms.csv")

    # =========================================================================
    # Print summary
    # =========================================================================
    print()
    print("SIVR by Sensor and Regime:")
    for row in sivr_rows:
        print("  {:25s} {:15s} SIVR={:.3f} (OIV={:.1f} ROV={:.1f} SR={:.1f})".format(
            row["sensor"], row["regime"], row["sivr"],
            row["oiv"], row["rov"], row["semantic_regret"]))

    print()
    print("Aggregate SIVR (value-weighted):")
    for sensor in sensors:
        sensor_eps = [ep for ep in all_episodes if ep["sensor"] == sensor]
        ni_eps = [ep for ep in all_episodes if ep["sensor"] == NOINFO]
        ps_eps = [ep for ep in all_episodes if ep["sensor"] == PERFECT]
        agg = aggregate_sivr(
            np.array([e["total_profit"] for e in sensor_eps]),
            np.array([e["total_profit"] for e in ni_eps]),
            np.array([e["total_profit"] for e in ps_eps]),
        )
        print("  {:25s} AggregateSIVR={:.3f}".format(sensor, agg))

    print()
    print("Belief Quality:")
    for row in belief_metrics:
        print("  {:25s} acc={:.3f} brier={:.3f} logloss={:.3f} ECE={:.3f}".format(
            row["sensor"], row["accuracy"], row["brier_score"],
            row["log_loss"], row["ece"]))

    print()
    print("API Stats:", api_stats)
    print("Outputs saved to", output_dir)

    return {
        "accounting": accounting,
        "sivr_rows": sivr_rows,
        "belief_metrics": belief_metrics,
        "all_episodes": all_episodes,
        "all_beliefs": all_beliefs,
        "api_stats": api_stats,
    }


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] if len(sys.argv) > 1 else None
    run_confirmation_experiment(llm_models=models)
