"""Phase 8: Replicate Paper-2 semantic decision-value experiment inside
Paper-1 Gymnasium inventory environment (GymInvMgmt/Serial-v0).

Architecture:
  Warning text
    -> Semantic interpreter (one of: NoInfo, RuleBased, TFIDF, LLM, PerfectSemantic)
    -> 2-class regime belief [Normal, DemandSurge]
    -> BeliefAdaptiveController (base-stock policy on Paper-1 env)
    -> Operational outcome (profit, fill rate, inventory)

Outputs go to results/phase8_gym_replication/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()
if os.environ.get("OPENAI_API_KEY") and not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
    os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"

from src.events import Regime, REGIME_WARNING_TEMPLATES
from src.interpreter import (
    no_info_regime_belief,
    perfect_semantic_regime_belief,
    rule_based_regime_extract,
    llm_regime_interpret_with_result,
    RegimeInterpretation,
)
from src.gym_adapter import (
    GymEpisodeResult,
    run_gym_episode,
    get_no_info_probs,
    get_perfect_semantic_probs,
    get_rulebased_probs,
    get_tfidf_probs,
    get_llm_probs,
    DEFAULT_BASE_MU,
    SURGE_MULTIPLIER,
    DEFAULT_NUM_PERIODS,
    DEFAULT_EVENT_START,
    DEFAULT_SCENARIO,
)

from src.experiment_phase6 import (
    aggregate_sivr,
    paired_bootstrap_ci,
    brier_score,
)

RESULTS_DIR = ROOT / "results" / "phase8_gym_replication"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = ["NoInfo", "RuleBased", "TFIDF_LogReg", "gpt-4o", "PerfectSemantic"]

BASE_MU = DEFAULT_BASE_MU
SHOCK_MAG = SURGE_MULTIPLIER
NUM_PERIODS = DEFAULT_NUM_PERIODS
EVENT_START = DEFAULT_EVENT_START
SCENARIO = DEFAULT_SCENARIO

REGIMES_2 = [Regime.NORMAL, Regime.DEMAND_SURGE]


@dataclass
class Phase8Row:
    seed: int
    regime: str
    sensor: str
    template_id: str
    ambiguity_level: str
    total_reward: float
    total_demand: float
    total_sold: float
    fill_rate: float
    avg_inventory: float
    inventory_position_final: float
    belief_normal: float
    belief_surge: float
    most_likely_regime: str
    belief_correct: bool
    oiv: float = 0.0
    sivr: float = 0.0


def _save_csv(rows, path):
    import csv
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name} ({len(rows)} rows)")


def _get_surge_templates():
    """Return templates relevant to our 2-class regime space."""
    templates = []
    for t in REGIME_WARNING_TEMPLATES:
        if t["regime"] in REGIMES_2:
            templates.append(t)
    return templates


def _get_sensor_belief(
    sensor: str,
    text: str,
    true_regime: Regime,
    tfidf_model=None,
    llm_model: str = "gpt-4o",
) -> dict[str, float]:
    """Get regime probabilities for a sensor+text combination."""
    if sensor == "NoInfo":
        return get_no_info_probs()
    elif sensor == "PerfectSemantic":
        return get_perfect_semantic_probs(true_regime)
    elif sensor == "RuleBased":
        return get_rulebased_probs(text)
    elif sensor == "TFIDF_LogReg" and tfidf_model is not None:
        return get_tfidf_probs(text, tfidf_model)
    elif sensor == "gpt-4o":
        return get_llm_probs(text, model=llm_model)
    else:
        return get_no_info_probs()


def run_phase8(
    num_seeds: int = 20,
    seed_base: int = 2000,
    llm_models: list[str] | None = None,
    base_mu: float = BASE_MU,
    shock_mag: float = SHOCK_MAG,
):
    if llm_models is None:
        llm_models = ["gpt-4o"]

    seeds = list(range(seed_base, seed_base + num_seeds))
    templates = _get_surge_templates()
    sensors = SENSORS

    print(f"Phase 8: {num_seeds} seeds x {len(templates)} templates x {len(sensors)} sensors")
    print(f"  base_mu={base_mu}, shock_mag={shock_mag}, scenario={SCENARIO}")
    print(f"  Results dir: {RESULTS_DIR}")

    tfidf_model = None
    pkl_path = ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_calibrated_model.pkl"
    if pkl_path.exists():
        from src.classical_baseline import TFIDFLogReg
        tfidf_model = TFIDFLogReg.load(str(pkl_path))
        print(f"  Loaded TFIDF model from {pkl_path.name}")
    else:
        print(f"  WARNING: TFIDF model not found at {pkl_path}")
        if "TFIDF_LogReg" in sensors:
            sensors.remove("TFIDF_LogReg")

    for model in llm_models:
        print(f"  Pre-caching LLM responses for {model}...")
        for tmpl in templates:
            ri, result = llm_regime_interpret_with_result(tmpl["text"], model=model)
            if result.from_cache:
                print(f"    Cached: {tmpl['template_id']}")
            else:
                print(f"    Fetched: {tmpl['template_id']} ({result.latency_ms:.0f}ms)")

    rows = []
    total_calls = len(seeds) * len(templates) * len(sensors)
    call_count = 0

    for regime in REGIMES_2:
        r_templates = [t for t in templates if t["regime"] == regime]
        if not r_templates:
            r_templates = templates

        for seed in seeds:
            for tmpl in r_templates:
                text = tmpl["text"]
                tid = tmpl["template_id"]
                alevel = tmpl["ambiguity_level"]

                for sensor in sensors:
                    call_count += 1
                    probs = _get_sensor_belief(
                        sensor, text, regime, tfidf_model=tfidf_model,
                        llm_model=llm_models[0] if llm_models else "gpt-4o",
                    )

                    result, trajectory = run_gym_episode(
                        seed=seed,
                        regime=regime,
                        sensor=sensor,
                        regime_probabilities=probs,
                        scenario=SCENARIO,
                        num_periods=NUM_PERIODS,
                        base_mu=base_mu,
                        event_start=EVENT_START,
                        shock_mag=shock_mag,
                    )

                    row = Phase8Row(
                        seed=seed,
                        regime=regime.value,
                        sensor=sensor,
                        template_id=tid,
                        ambiguity_level=alevel,
                        total_reward=result.total_reward,
                        total_demand=result.total_demand,
                        total_sold=result.total_sold,
                        fill_rate=result.fill_rate,
                        avg_inventory=result.avg_inventory,
                        inventory_position_final=result.inventory_position_final,
                        belief_normal=result.belief_normal,
                        belief_surge=result.belief_surge,
                        most_likely_regime=result.most_likely_regime,
                        belief_correct=result.belief_correct,
                    )
                    rows.append(row)

                    if call_count % 50 == 0 or call_count == total_calls:
                        print(f"  [{call_count}/{total_calls}] seed={seed} regime={regime.value} "
                              f"sensor={sensor} OIV=? reward={result.total_reward:.1f}")

    print(f"\nAll {call_count} episodes complete. Computing metrics...")
    return _compute_and_save(rows, seeds, templates, base_mu, shock_mag)


def _compute_and_save(rows, seeds, templates, base_mu, shock_mag):
    operational_rows = [asdict(r) for r in rows]
    _save_csv(operational_rows, RESULTS_DIR / "operational_results.csv")

    ni_rewards = {}
    ps_rewards = {}
    for r in rows:
        key = (r.seed, r.regime, r.template_id)
        if r.sensor == "NoInfo":
            ni_rewards[key] = r.total_reward
        elif r.sensor == "PerfectSemantic":
            ps_rewards[key] = r.total_reward

    for r in rows:
        key = (r.seed, r.regime, r.template_id)
        ni = ni_rewards.get(key, 0.0)
        ps = ps_rewards.get(key, 0.0)
        denom = ps - ni if abs(ps - ni) > 1e-9 else 1.0
        r.oiv = ps - ni
        r.sivr = (r.total_reward - ni) / abs(denom) if abs(denom) > 1e-9 else 0.0

    operational_rows = [asdict(r) for r in rows]
    _save_csv(operational_rows, RESULTS_DIR / "operational_results.csv")

    sensor_agg = {}
    for sensor in SENSORS:
        sensor_rows = [r for r in rows if r.sensor == sensor]
        rewards = [r.total_reward for r in sensor_rows]
        ni_list = [ni_rewards.get((r.seed, r.regime, r.template_id), 0.0) for r in sensor_rows]
        ps_list = [ps_rewards.get((r.seed, r.regime, r.template_id), 0.0) for r in sensor_rows]
        sivr_vals = [r.sivr for r in sensor_rows]
        belief_corrects = [r.belief_correct for r in sensor_rows]

        s_arr = np.array(rewards)
        ni_arr = np.array(ni_list)
        ps_arr = np.array(ps_list)
        agg_sivr = aggregate_sivr(s_arr, ni_arr, ps_arr)

        sensor_agg[sensor] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_fill_rate": float(np.mean([r.fill_rate for r in sensor_rows])),
            "mean_sivr": float(np.mean(sivr_vals)),
            "std_sivr": float(np.std(sivr_vals)),
            "aggregate_sivr": agg_sivr,
            "belief_accuracy": float(np.mean(belief_corrects)),
            "n": len(sensor_rows),
        }

    agg_df = []
    for sensor, metrics in sensor_agg.items():
        row = {"sensor": sensor}
        row.update(metrics)
        agg_df.append(row)
    _save_csv(agg_df, RESULTS_DIR / "aggregate_sivr.csv")

    print("\n=== Aggregate SIVR Summary ===")
    for row in agg_df:
        print(f"  {row['sensor']:20s}  AggSIVR={row['aggregate_sivr']:.3f}  "
              f"MeanRew={row['mean_reward']:.1f}  FillRate={row['mean_fill_rate']:.3f}  "
              f"BelAcc={row['belief_accuracy']:.3f}")

    brier_scores = {}
    for sensor in SENSORS:
        sensor_rows = [r for r in rows if r.sensor == sensor]
        beliefs = []
        for r in sensor_rows:
            true = r.regime
            b = r.belief_normal if true == "normal" else r.belief_surge
            beliefs.append(b)
        brier_scores[sensor] = float(np.mean([(1 - b) ** 2 for b in beliefs])) if beliefs else 0.0

    _save_csv(
        [{"sensor": s, "brier_score": b} for s, b in brier_scores.items()],
        RESULTS_DIR / "belief_metrics.csv",
    )

    ci_results = []
    for sensor in SENSORS:
        if sensor in ("NoInfo", "PerfectSemantic"):
            continue
        sensor_rewards = [r.total_reward for r in rows if r.sensor == sensor]
        ni_all = [r.total_reward for r in rows if r.sensor == "NoInfo"]
        if len(sensor_rewards) == len(ni_all) and len(sensor_rewards) > 0:
            ci = paired_bootstrap_ci(
                np.array(sensor_rewards), np.array(ni_all), n_boot=5000
            )
            ci_results.append({
                "sensor": sensor,
                "vs_sensor": "NoInfo",
                "mean_diff": float(ci[0]),
                "ci_lower": float(ci[1]),
                "ci_upper": float(ci[2]),
                "significant": bool(ci[1] > 0 or ci[2] < 0),
            })
            sig = "YES" if ci_results[-1]["significant"] else "NO"
            print(f"  CI {sensor} vs NoInfo: diff={ci[0]:.1f} [{ci[1]:.1f}, {ci[2]:.1f}] sig={sig}")

    _save_csv(ci_results, RESULTS_DIR / "paired_comparisons.csv")

    regime_summary = []
    for regime in REGIMES_2:
        for sensor in SENSORS:
            sensor_regime_rows = [r for r in rows if r.sensor == sensor and r.regime == regime.value]
            if sensor_regime_rows:
                regime_summary.append({
                    "regime": regime.value,
                    "sensor": sensor,
                    "mean_reward": float(np.mean([r.total_reward for r in sensor_regime_rows])),
                    "mean_fill_rate": float(np.mean([r.fill_rate for r in sensor_regime_rows])),
                    "mean_sivr": float(np.mean([r.sivr for r in sensor_regime_rows])),
                    "n": len(sensor_regime_rows),
                })
    _save_csv(regime_summary, RESULTS_DIR / "regime_summary.csv")

    ambiguity_summary = []
    for alevel in ["clear", "moderate", "vague"]:
        for sensor in SENSORS:
            sensor_amb_rows = [r for r in rows if r.sensor == sensor and r.ambiguity_level == alevel]
            if sensor_amb_rows:
                ambiguity_summary.append({
                    "ambiguity_level": alevel,
                    "sensor": sensor,
                    "mean_reward": float(np.mean([r.total_reward for r in sensor_amb_rows])),
                    "mean_fill_rate": float(np.mean([r.fill_rate for r in sensor_amb_rows])),
                    "mean_sivr": float(np.mean([r.sivr for r in sensor_amb_rows])),
                    "n": len(sensor_amb_rows),
                })
    _save_csv(ambiguity_summary, RESULTS_DIR / "ambiguity_summary.csv")

    win_rates = {}
    for sensor in SENSORS:
        if sensor == "NoInfo":
            continue
        s_dict = {(r.seed, r.regime, r.template_id): r.total_reward for r in rows if r.sensor == sensor}
        ni_dict = {(r.seed, r.regime, r.template_id): r.total_reward for r in rows if r.sensor == "NoInfo"}
        common_keys = set(s_dict.keys()) & set(ni_dict.keys())
        if common_keys:
            wins = sum(1 for k in common_keys if s_dict[k] > ni_dict[k])
            ties = sum(1 for k in common_keys if abs(s_dict[k] - ni_dict[k]) < 1e-9)
            win_rates[sensor] = {
                "win_rate_vs_NoInfo": wins / len(common_keys) if common_keys else 0,
                "tie_rate": ties / len(common_keys) if common_keys else 0,
                "n_pairs": len(common_keys),
            }
    _save_csv(
        [{"sensor": s, **v} for s, v in win_rates.items()],
        RESULTS_DIR / "win_rates.csv",
    )

    manifest = {
        "phase": 8,
        "description": "Paper-1 Gymnasium replication of semantic decision-value experiment",
        "num_seeds": len(seeds),
        "seed_base": seeds[0],
        "seeds": seeds,
        "templates_used": len(templates),
        "sensors": SENSORS,
        "base_mu": base_mu,
        "shock_mag": shock_mag,
        "scenario": SCENARIO,
        "num_periods": NUM_PERIODS,
        "event_start": EVENT_START,
        "results_dir": str(RESULTS_DIR),
        "output_files": [f.name for f in RESULTS_DIR.iterdir() if f.suffix == ".csv"],
        "aggregate_sivr": {s: sensor_agg[s]["aggregate_sivr"] for s in SENSORS},
    }
    (RESULTS_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(f"\nPhase 8 complete. Results in {RESULTS_DIR}")
    return rows, sensor_agg, ci_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8: Paper-1 Gym replication")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--llm-models", nargs="*", default=["gpt-4o"])
    parser.add_argument("--base-mu", type=float, default=DEFAULT_BASE_MU)
    parser.add_argument("--shock-mag", type=float, default=SURGE_MULTIPLIER)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    if args.no_llm:
        llm_models = []
        SENSORS.remove("gpt-4o")
    else:
        llm_models = args.llm_models

    run_phase8(
        num_seeds=args.seeds,
        seed_base=args.seed_base,
        llm_models=llm_models,
        base_mu=args.base_mu,
        shock_mag=args.shock_mag,
    )
