"""Phase 5.5: Frozen Real-LLM Semantic Sensing Experiment.

Evaluates real LLMs as semantic sensors on the frozen Phase 5 benchmark.
Measures how much operational decision value LLMs recover from unstructured
warning text under a strong causal optimizer.

Do NOT redesign the environment. This module evaluates, it does not tune.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
from scipy.stats import spearmanr

from src.events import (
    Regime, REGIME_PRIOR, REGIME_PARAMS, REGIME_WARNING_TEMPLATES,
    P5_HORIZON, P5_NORMAL_LEAD_TIME, P5_DEMAND_MEAN,
    P5_EVENT_START, P5_EVENT_DURATION, P5_WARNING_TIME,
)
from src.env import (
    InventoryEnv, InventoryState, CausalOptimizer, HindsightOracle,
    REVENUE_PER_UNIT, ORDERING_FIXED_COST, ORDERING_VARIABLE_COST,
    HOLDING_COST_PER_UNIT, STOCKOUT_COST_PER_UNIT,
    INITIAL_INVENTORY, HORIZON, DEMAND_MEAN,
)
from src.interpreter import (
    RegimeInterpretation, no_info_regime_belief, perfect_semantic_regime_belief,
    rule_based_regime_extract, llm_regime_interpret_with_result, load_dotenv,
    EXTRACTION_PROMPT,
)
from src.metrics import (
    compute_episode_metrics, information_value, signed_sivr, standard_brier_score,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase5_5"


# --- Frozen Benchmark Manifest ---

def compute_frozen_manifest() -> dict:
    """Compute hashes of all frozen benchmark components."""
    components = {}

    # Regime definitions
    regime_data = {r.value: REGIME_PARAMS[r] for r in Regime}
    components["regime_definitions"] = hashlib.sha256(
        json.dumps(regime_data, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Prior
    components["regime_prior"] = hashlib.sha256(
        json.dumps({k: v for k, v in sorted(REGIME_PRIOR.items())}).encode()
    ).hexdigest()[:16]

    # Templates (text only, not metadata)
    template_texts = sorted([t["text"] for t in REGIME_WARNING_TEMPLATES])
    components["warning_templates"] = hashlib.sha256(
        json.dumps(template_texts).encode()
    ).hexdigest()[:16]

    # Economics
    from src.env import (REVENUE_PER_UNIT, ORDERING_FIXED_COST,
                          ORDERING_VARIABLE_COST, HOLDING_COST_PER_UNIT,
                          STOCKOUT_COST_PER_UNIT, INITIAL_INVENTORY)
    economics = {
        "revenue": REVENUE_PER_UNIT, "ordering_fixed": ORDERING_FIXED_COST,
        "ordering_var": ORDERING_VARIABLE_COST, "holding": HOLDING_COST_PER_UNIT,
        "stockout": STOCKOUT_COST_PER_UNIT, "initial_inv": INITIAL_INVENTORY,
    }
    components["economics"] = hashlib.sha256(
        json.dumps(economics, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Timing
    timing = {
        "horizon": P5_HORIZON, "normal_lt": P5_NORMAL_LEAD_TIME,
        "demand_mean": P5_DEMAND_MEAN, "event_start": P5_EVENT_START,
        "event_duration": P5_EVENT_DURATION, "warning_time": P5_WARNING_TIME,
    }
    components["timing"] = hashlib.sha256(
        json.dumps(timing, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Extraction prompt
    components["extraction_prompt"] = hashlib.sha256(
        EXTRACTION_PROMPT.encode()
    ).hexdigest()[:16]

    return {
        "phase": "5.5",
        "frozen": True,
        "components": components,
        "num_seeds": 15,
        "seed_base": 1000,
        "llm_models": [],
    }


# --- Metric computations ---

def brier_score(predicted: dict[str, float], true_regime: str) -> float:
    """Brier score for a single prediction (lower is better)."""
    return standard_brier_score(
        predicted, true_regime,
        class_order=["normal", "supplier_delay", "demand_surge"],
    )


def log_loss_safe(predicted: dict[str, float], true_regime: str, eps: float = 1e-10) -> float:
    """Log loss with numerical clipping for stability."""
    prob = max(predicted.get(true_regime, eps), eps)
    return -math.log(prob)


def false_positive_cost(
    ni_profit: float, sensor_profit: float,
    true_regime: str, predicted_regime: str,
) -> dict:
    """Economic cost of false positive (predicting disruption when Normal)."""
    is_false_positive = (true_regime == "normal" and predicted_regime != "normal")
    is_false_negative = (true_regime != "normal" and predicted_regime == "normal")
    return {
        "is_false_positive": is_false_positive,
        "is_false_negative": is_false_negative,
        "profit_loss": float(ni_profit - sensor_profit) if is_false_positive else 0.0,
        "additional_holding": float(ni_profit - sensor_profit) if is_false_positive else 0.0,
        "missed_protection": float(ni_profit - sensor_profit) if is_false_negative else 0.0,
    }


# --- Main experiment ---

def run_phase5_5(
    num_seeds: int = 15,
    seed_base: int = 1000,
    llm_models: Optional[list[str]] = None,
) -> dict:
    """Run the complete Phase 5.5 frozen benchmark experiment."""
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
        os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"

    # 1. Freeze benchmark
    manifest = compute_frozen_manifest()
    if llm_models:
        manifest["llm_models"] = llm_models
    with open(output_dir / "frozen_benchmark_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print("PHASE 5.5: FROZEN REAL-LLM SEMANTIC SENSING EXPERIMENT")
    print("=" * 70)
    print(f"Seeds: {num_seeds} | Models: {llm_models or 'none'}")
    print(f"Benchmark hash: {json.dumps(manifest['components'], indent=2)}")
    print()

    regimes = [Regime.NORMAL, Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]
    seeds = [seed_base + i for i in range(num_seeds)]

    sensors = ["NoInfo", "RuleBased", "PerfectSemantic"]
    if llm_models:
        for model in llm_models:
            sensors.append(f"LLM:{model}")

    # Storage
    all_episodes = []
    all_beliefs = []
    all_raw_responses = []
    llm_api_stats = {"total_calls": 0, "cache_hits": 0, "malformed": 0}

    # --- Run all conditions ---
    for regime in regimes:
        regime_templates = [t for t in REGIME_WARNING_TEMPLATES if t["regime"] == regime]

        for seed in seeds:
            for tmpl in regime_templates:
                text = tmpl["text"]
                tid = tmpl["template_id"]
                alevel = tmpl["ambiguity_level"]

                for sensor in sensors:
                    # Get interpretation
                    if sensor == "NoInfo":
                        ri = no_info_regime_belief()
                        raw_resp = ""
                        from_cache = False
                        latency = 0.0
                        malformed = False
                    elif sensor == "PerfectSemantic":
                        ri = perfect_semantic_regime_belief(regime)
                        raw_resp = ""
                        from_cache = False
                        latency = 0.0
                        malformed = False
                    elif sensor == "RuleBased":
                        ri = rule_based_regime_extract(text)
                        raw_resp = json.dumps(ri.to_dict())
                        from_cache = False
                        latency = 0.0
                        malformed = False
                    elif sensor.startswith("LLM:"):
                        model = sensor.split(":", 1)[1]
                        llm_api_stats["total_calls"] += 1
                        try:
                            ri, llm_result = llm_regime_interpret_with_result(
                                text, model=model,
                            )
                            raw_resp = llm_result.raw_response
                            from_cache = llm_result.from_cache
                            latency = llm_result.latency_ms
                            malformed = llm_result.malformed
                            if from_cache:
                                llm_api_stats["cache_hits"] += 1
                            if malformed:
                                llm_api_stats["malformed"] += 1
                        except Exception as e:
                            ri = no_info_regime_belief()
                            raw_resp = str(e)
                            from_cache = False
                            latency = 0.0
                            malformed = True
                            llm_api_stats["malformed"] += 1
                    else:
                        ri = no_info_regime_belief()
                        raw_resp = ""
                        from_cache = False
                        latency = 0.0
                        malformed = False

                    belief = ri.normalized()

                    # Run episode
                    from src.experiment_phase5 import _run_p5_episode
                    result, history = _run_p5_episode(
                        seed=seed, regime=regime,
                        sensor=sensor, controller="CausalOptimizer",
                        template_id=tid, ambiguity_level=alevel,
                        regime_interp=ri,
                    )

                    # Record
                    all_episodes.append({
                        "seed": seed, "regime": regime.value,
                        "template_id": tid, "ambiguity_level": alevel,
                        "sensor": sensor, "controller": "CausalOptimizer",
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
                    })

                    all_beliefs.append({
                        "sensor": sensor, "regime": regime.value,
                        "template_id": tid, "ambiguity_level": alevel,
                        "seed": seed,
                        "belief_normal": belief.regime_probabilities.get("normal", 0.0),
                        "belief_delay": belief.regime_probabilities.get("supplier_delay", 0.0),
                        "belief_surge": belief.regime_probabilities.get("demand_surge", 0.0),
                        "most_likely": belief.most_likely_regime,
                        "correct": belief.most_likely_regime == regime.value,
                        "true_regime": regime.value,
                        "brier": brier_score(belief.regime_probabilities, regime.value),
                        "log_loss": log_loss_safe(belief.regime_probabilities, regime.value),
                        "est_lt_increase": belief.estimated_lt_increase,
                        "est_duration": belief.estimated_duration,
                        "est_demand_multiplier": belief.estimated_demand_multiplier,
                    })

                    if sensor.startswith("LLM:") or sensor == "RuleBased":
                        all_raw_responses.append({
                            "model": sensor, "template_id": tid,
                            "regime": regime.value, "ambiguity_level": alevel,
                            "raw_response": raw_resp,
                            "parsed_json": json.dumps(belief.to_dict()),
                            "latency_ms": latency, "from_cache": from_cache,
                            "malformed": malformed,
                        })

    # --- Compute SIVR_O ---
    # Get NoInfo and PerfectSemantic baselines per regime
    ni_by_regime = {}
    ps_by_regime = {}
    for ep in all_episodes:
        key = (ep["regime"], ep["sensor"])
        if ep["sensor"] == "NoInfo":
            ni_by_regime.setdefault(ep["regime"], []).append(ep["total_profit"])
        elif ep["sensor"] == "PerfectSemantic":
            ps_by_regime.setdefault(ep["regime"], []).append(ep["total_profit"])

    ni_means = {r: np.mean(v) for r, v in ni_by_regime.items()}
    ps_means = {r: np.mean(v) for r, v in ps_by_regime.items()}

    # SIVR_O by sensor x regime
    sivr_rows = []
    for sensor in sensors:
        for regime in regimes:
            sensor_profits = [ep["total_profit"] for ep in all_episodes
                              if ep["sensor"] == sensor and ep["regime"] == regime.value]
            if not sensor_profits:
                continue
            sensor_mean = np.mean(sensor_profits)
            ni_mean = ni_means[regime.value]
            ps_mean = ps_means[regime.value]
            sivr = signed_sivr(sensor_mean, ni_mean, ps_mean).value

            sivr_rows.append({
                "sensor": sensor, "regime": regime.value,
                "sensor_profit": float(sensor_mean),
                "noinfo_profit": float(ni_mean),
                "perfect_profit": float(ps_mean),
                "sivr": float(sivr),
                "n": len(sensor_profits),
                "std": float(np.std(sensor_profits)),
            })

    # SIVR_O by sensor x ambiguity
    sivr_ambig_rows = []
    for sensor in sensors:
        for alevel in ["clear", "moderate", "vague"]:
            sensor_profits = [ep["total_profit"] for ep in all_episodes
                              if ep["sensor"] == sensor and ep["ambiguity_level"] == alevel]
            # Overall NoInfo/PerfectSemantic for denominator
            all_ni = [ep["total_profit"] for ep in all_episodes if ep["sensor"] == "NoInfo"]
            all_ps = [ep["total_profit"] for ep in all_episodes if ep["sensor"] == "PerfectSemantic"]
            if not sensor_profits or not all_ni or not all_ps:
                continue
            sensor_mean = np.mean(sensor_profits)
            sivr = signed_sivr(sensor_mean, float(np.mean(all_ni)),
                               float(np.mean(all_ps))).value

            sivr_ambig_rows.append({
                "sensor": sensor, "ambiguity_level": alevel,
                "sensor_profit": float(sensor_mean),
                "sivr": float(sivr),
                "n": len(sensor_profits),
            })

    # --- Belief metrics by model ---
    belief_metrics = []
    for sensor in sensors:
        sensor_beliefs = [b for b in all_beliefs if b["sensor"] == sensor]
        if not sensor_beliefs:
            continue
        acc = np.mean([b["correct"] for b in sensor_beliefs])
        brier = np.mean([b["brier"] for b in sensor_beliefs])
        logloss = np.mean([b["log_loss"] for b in sensor_beliefs])
        true_prob = np.mean([b[f"belief_{b['true_regime'].split('_')[0] if 'supplier' not in b['true_regime'] else 'delay'}"]
                             for b in sensor_beliefs
                             if b["true_regime"] in b]) if sensor_beliefs else 0.0

        # More robust true-regime probability
        true_probs = []
        for b in sensor_beliefs:
            tr = b["true_regime"]
            if tr == "normal":
                true_probs.append(b["belief_normal"])
            elif tr == "supplier_delay":
                true_probs.append(b["belief_delay"])
            elif tr == "demand_surge":
                true_probs.append(b["belief_surge"])
        true_prob = np.mean(true_probs) if true_probs else 0.0

        belief_metrics.append({
            "sensor": sensor,
            "accuracy": float(acc),
            "brier_score": float(brier),
            "log_loss": float(logloss),
            "true_regime_probability": float(true_prob),
            "n": len(sensor_beliefs),
        })

    # --- False positive / negative costs ---
    fpfn_rows = []
    for sensor in sensors:
        sensor_episodes = [ep for ep in all_episodes if ep["sensor"] == sensor]
        ni_profit_by_regime = ni_means

        for ep in sensor_episodes:
            is_fp = ep["regime"] == "normal" and ep["most_likely_regime"] != "normal"
            is_fn = ep["regime"] != "normal" and ep["most_likely_regime"] == "normal"
            fpfn_rows.append({
                "sensor": sensor,
                "regime": ep["regime"],
                "template_id": ep["template_id"],
                "ambiguity_level": ep["ambiguity_level"],
                "seed": ep["seed"],
                "true_regime": ep["regime"],
                "predicted_regime": ep["most_likely_regime"],
                "is_false_positive": is_fp,
                "is_false_negative": is_fn,
                "sensor_profit": ep["total_profit"],
                "noinfo_profit": float(ni_profit_by_regime.get(ep["regime"], 0.0)),
                "profit_loss_vs_noinfo": float(
                    ni_profit_by_regime.get(ep["regime"], 0.0) - ep["total_profit"]
                ),
            })

    # --- Paired comparisons ---
    paired_rows = []
    for regime in regimes:
        for seed in seeds:
            regime_eps = {ep["sensor"]: ep for ep in all_episodes
                          if ep["regime"] == regime.value and ep["seed"] == seed}
            ni_ep = regime_eps.get("NoInfo")
            for sensor in sensors:
                if sensor == "NoInfo":
                    continue
                s_ep = regime_eps.get(sensor)
                if ni_ep and s_ep:
                    paired_rows.append({
                        "regime": regime.value, "seed": seed,
                        "sensor": sensor,
                        "sensor_profit": s_ep["total_profit"],
                        "noinfo_profit": ni_ep["total_profit"],
                        "diff": s_ep["total_profit"] - ni_ep["total_profit"],
                    })

    # --- Save all outputs ---
    _save_csv(all_episodes, output_dir / "operational_results.csv")
    _save_csv(all_beliefs, output_dir / "belief_metrics.csv")
    _save_csv(all_raw_responses, output_dir / "raw_llm_responses.csv")
    _save_csv(sivr_rows, output_dir / "sivr_by_model.csv")
    _save_csv(sivr_ambig_rows, output_dir / "sivr_by_ambiguity.csv")
    _save_csv(belief_metrics, output_dir / "belief_to_value_analysis.csv")
    _save_csv(fpfn_rows, output_dir / "false_positive_negative_costs.csv")
    _save_csv(paired_rows, output_dir / "paired_comparisons.csv")

    # SIVR by regime
    _save_csv(sivr_rows, output_dir / "sivr_by_regime.csv")

    # RuleBased vs LLM comparison
    rb_vs_llm = []
    for regime in regimes:
        for alevel in ["clear", "moderate", "vague"]:
            rb_eps = [ep for ep in all_episodes if ep["sensor"] == "RuleBased"
                      and ep["regime"] == regime.value and ep["ambiguity_level"] == alevel]
            for model in (llm_models or []):
                llm_eps = [ep for ep in all_episodes if ep["sensor"] == f"LLM:{model}"
                           and ep["regime"] == regime.value and ep["ambiguity_level"] == alevel]
                if rb_eps and llm_eps:
                    rb_profit = np.mean([e["total_profit"] for e in rb_eps])
                    llm_profit = np.mean([e["total_profit"] for e in llm_eps])
                    rb_vs_llm.append({
                        "regime": regime.value, "ambiguity_level": alevel,
                        "model": model,
                        "rulebased_profit": float(rb_profit),
                        "llm_profit": float(llm_profit),
                        "profit_diff": float(llm_profit - rb_profit),
                    })
    _save_csv(rb_vs_llm, output_dir / "rulebased_vs_llm.csv")

    # --- Print summary ---
    print("\nSIVR_O by Sensor and Regime:")
    for row in sivr_rows:
        print(f"  {row['sensor']:20s} {row['regime']:15s} SIVR={row['sivr']:.3f} "
              f"(profit={row['sensor_profit']:.1f} n={row['n']})")

    print("\nBelief Quality:")
    for row in belief_metrics:
        print(f"  {row['sensor']:20s} acc={row['accuracy']:.3f} brier={row['brier_score']:.3f} "
              f"logloss={row['log_loss']:.3f} true_prob={row['true_regime_probability']:.3f}")

    if llm_models:
        print(f"\nLLM API Stats: {llm_api_stats}")

    print(f"\nOutputs saved to {output_dir}")

    return {
        "manifest": manifest,
        "sivr_rows": sivr_rows,
        "belief_metrics": belief_metrics,
        "all_episodes": all_episodes,
        "all_beliefs": all_beliefs,
        "api_stats": llm_api_stats,
    }


def _save_csv(rows: list[dict], path: Path) -> None:
    """Save list of dicts to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] if len(sys.argv) > 1 else ["gpt-4o-mini"]
    run_phase5_5(llm_models=models)
