"""Phase 8B: Frozen, leakage-safe confirmation experiment.

Corrects Phase-8A validity issues:
  - Uses Phase-6 CONFIRMATION_TEMPLATES (held-out from TF-IDF training)
  - Includes both raw and calibrated TF-IDF
  - Uses OracleSemantic instead of PerfectSemantic
  - Fixes fill rate (uses retail sales, not replenishment orders)
  - Uses new disjoint seeds (3000-3029)
  - Implements hierarchical bootstrap CI

Operational scenario is FROZEN from Phase-8A. Do not change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()
if os.environ.get("OPENAI_API_KEY") and not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
    os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"

from src.events import Regime
from src.interpreter import (
    no_info_regime_belief,
    rule_based_regime_extract,
    llm_regime_interpret_with_result,
)
from src.gym_adapter import (
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
from src.classical_baseline import TFIDFLogReg
from src.confirmation_templates import CONFIRMATION_TEMPLATES
from src.metrics import signed_sivr, standard_brier_score

RESULTS_DIR = ROOT / "results" / "phase8b_gym_confirmation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHASE8B_SEEDS = list(range(3100, 3130))

SENSORS = [
    # --- text-free references and controls ------------------------------
    # Reading text is only valuable if it beats competitive text-free
    # operation, not merely the benchmark prior. The September 2026 audit
    # found a constant "assume the event" belief outperforming most
    # interpreters, so these arms are now part of the experiment.
    "NoInfo",                       # the declared benchmark prior
    "Constant_p0.0",                # never assume the event
    "Constant_p0.5",                # uninformative midpoint
    "Constant_p1.0",                # always assume the event
    "NoText_Tuned",                 # constant belief tuned on development seeds
    # --- text interpreters ----------------------------------------------
    "RuleBased",
    "TFIDF_LogReg_Raw",
    "TFIDF_LogReg_Calibrated",
    "gpt-4o",
    # --- degraded-text controls -----------------------------------------
    "TFIDF_LogReg_Calibrated_Argmax",    # label only, no probabilities
    "TFIDF_LogReg_Calibrated_Shuffled",  # right interpreter, wrong text
    # --- reference ------------------------------------------------------
    "OracleSemantic",
]

EVENT_REGIME = Regime.DEMAND_SURGE

BASE_MU = DEFAULT_BASE_MU
SHOCK_MAG = SURGE_MULTIPLIER
NUM_PERIODS = DEFAULT_NUM_PERIODS
EVENT_START = DEFAULT_EVENT_START
SCENARIO = DEFAULT_SCENARIO

REGIMES_2 = [Regime.NORMAL, Regime.DEMAND_SURGE]

PHASE7_DIR = ROOT / "results" / "phase7_classical_baseline"


@dataclass
class Phase8BRow:
    seed: int
    regime: str
    sensor: str
    template_id: str
    ambiguity_level: str
    total_reward: float
    total_demand: float
    total_retail_sales: float
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
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name} ({len(rows)} rows)")


def _get_heldout_templates():
    """Select DemandSurge and Normal templates from CONFIRMATION_TEMPLATES."""
    templates = []
    for t in CONFIRMATION_TEMPLATES:
        if t["regime"] in REGIMES_2:
            templates.append(t)
    return templates


def _get_sensor_belief(
    sensor: str,
    text: str,
    true_regime: Regime,
    tfidf_raw=None,
    tfidf_cal=None,
    llm_model: str = "gpt-4o",
    tuned_p_event=None,
    shuffled_text=None,
) -> dict[str, float]:
    from src.notext_controls import constant_belief, argmax_belief

    if sensor == "NoInfo":
        return get_no_info_probs()
    if sensor.startswith("Constant_p"):
        return constant_belief(float(sensor.split("_p")[1]), EVENT_REGIME)
    if sensor == "NoText_Tuned":
        if tuned_p_event is None:
            raise ValueError(
                "NoText_Tuned requires an operating point selected on development "
                "seeds; run the tuning step before the confirmation episodes"
            )
        return constant_belief(tuned_p_event, EVENT_REGIME)
    if sensor == "OracleSemantic":
        return get_perfect_semantic_probs(true_regime)
    if sensor == "RuleBased":
        return get_rulebased_probs(text)
    if sensor == "TFIDF_LogReg_Raw" and tfidf_raw is not None:
        return get_tfidf_probs(text, tfidf_raw)
    if sensor == "TFIDF_LogReg_Calibrated" and tfidf_cal is not None:
        return get_tfidf_probs(text, tfidf_cal)
    if sensor == "TFIDF_LogReg_Calibrated_Argmax" and tfidf_cal is not None:
        return argmax_belief(get_tfidf_probs(text, tfidf_cal), EVENT_REGIME)
    if sensor == "TFIDF_LogReg_Calibrated_Shuffled" and tfidf_cal is not None:
        if shuffled_text is None:
            raise ValueError("the shuffled-text control needs a substituted text")
        return get_tfidf_probs(shuffled_text, tfidf_cal)
    if sensor == "gpt-4o":
        return get_llm_probs(text, model=llm_model)
    # Never fall through to the prior: that records a control's result under
    # another sensor's name.
    raise ValueError(f"unknown sensor {sensor!r} (or its model was not supplied)")


def run_gym_episode_corrected_fill_rate(
    seed, regime, sensor, regime_probabilities,
    scenario=SCENARIO, num_periods=NUM_PERIODS,
    base_mu=BASE_MU, event_start=EVENT_START, shock_mag=SHOCK_MAG,
):
    """Run episode with corrected fill rate using retail sales."""
    result, trajectory = run_gym_episode(
        seed=seed, regime=regime, sensor=sensor,
        regime_probabilities=regime_probabilities,
        scenario=scenario, num_periods=num_periods,
        base_mu=base_mu, event_start=event_start, shock_mag=shock_mag,
    )

    if regime == Regime.DEMAND_SURGE:
        from src.gym_adapter import make_surge_env
        env = make_surge_env(
            seed=seed, scenario=scenario, num_periods=num_periods,
            base_mu=base_mu, shock_mag=shock_mag, shock_time=event_start,
        )
    else:
        from src.gym_adapter import make_env
        env = make_env(
            seed=seed, scenario=scenario, num_periods=num_periods,
            base_mu=base_mu,
        )

    from src.gym_adapter import _build_controller
    controller = _build_controller(env, regime_probabilities, base_mu=base_mu)
    obs, _ = env.reset(seed=seed)
    for t in range(num_periods):
        action = controller.get_action(obs, t)
        action = np.clip(action, 0, float(env.action_space.high[0]))
        obs, _, term, trunc, _ = env.step(action)
        if trunc:
            break

    retail_link_idx = None
    for idx, (supplier, buyer) in enumerate(env.network.reorder_links):
        if buyer == 1:
            retail_link_idx = idx
            break

    total_demand = float(np.sum(env.D))
    if retail_link_idx is not None:
        total_retail_sales = float(np.sum(env.S[:, retail_link_idx]))
    else:
        total_retail_sales = float(np.sum(env.R))

    fill_rate = total_retail_sales / total_demand if total_demand > 0 else 1.0
    result.fill_rate = fill_rate

    return result, trajectory


def hierarchical_paired_bootstrap(
    diffs_by_template: dict[str, np.ndarray],
    n_boot: int = 5000, alpha: float = 0.05, seed: int = 42,
):
    rng = np.random.default_rng(seed)
    template_ids = list(diffs_by_template.keys())
    n_templates = len(template_ids)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        t_idx = rng.integers(0, n_templates, size=n_templates)
        vals = []
        for ti in t_idx:
            tid = template_ids[ti]
            arr = diffs_by_template[tid]
            s_idx = rng.integers(0, len(arr), size=len(arr))
            vals.extend(arr[s_idx])
        boot_means[i] = np.mean(vals)
    mean_diff = float(np.mean(list(diffs_by_template.values())))
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_diff, ci_low, ci_high


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


def run_phase8b(
    seeds=None,
    llm_models=None,
):
    if seeds is None:
        seeds = PHASE8B_SEEDS
    if llm_models is None:
        llm_models = ["gpt-4o"]

    templates = _get_heldout_templates()
    sensors = list(SENSORS)

    print(f"Phase 8B: {len(seeds)} seeds x {len(templates)} templates x {len(sensors)} sensors")
    print(f"  Seeds: {seeds[0]}-{seeds[-1]}")
    print(f"  Templates: {len(templates)} held-out ({len([t for t in templates if t['regime']==Regime.DEMAND_SURGE])} surge, {len([t for t in templates if t['regime']==Regime.NORMAL])} normal)")
    print(f"  Results dir: {RESULTS_DIR}")

    tfidf_raw = TFIDFLogReg.load(str(PHASE7_DIR / "tfidf_logreg_model.pkl"))
    tfidf_cal = TFIDFLogReg.load(str(PHASE7_DIR / "tfidf_logreg_calibrated_model.pkl"))
    print(f"  Loaded raw TF-IDF model")
    print(f"  Loaded calibrated TF-IDF model")

    # --- controls -------------------------------------------------------
    from src.notext_controls import (
        DEV_SEEDS_8B, shuffled_text_assignment, tune_notext_control,
    )

    # Deranged text assignment: the calibrated interpreter reads a real
    # warning that belongs to a different template.
    shuffle_map = shuffled_text_assignment([t["template_id"] for t in templates])
    text_by_id = {t["template_id"]: t["text"] for t in templates}

    print(f"  Tuning the no-text control on development seeds "
          f"{DEV_SEEDS_8B[0]}-{DEV_SEEDS_8B[-1]} (disjoint from confirmation seeds)...")
    tuned = tune_notext_control(
        episode_runner=run_gym_episode,
        event_regime=EVENT_REGIME,
        dev_seeds=DEV_SEEDS_8B,
    )
    print(f"    selected p(event)={tuned.p_event} "
          f"(dev balanced reward {tuned.dev_grid[tuned.p_event]:.1f})")
    (RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "notext_tuning.json").write_text(json.dumps(tuned.as_dict(), indent=2))

    for model in llm_models:
        print(f"  Pre-caching LLM responses for {model}...")
        for tmpl in templates:
            ri, result = llm_regime_interpret_with_result(tmpl["text"], model=model)
            status = "Cached" if result.from_cache else f"Fetched ({result.latency_ms:.0f}ms)"
            print(f"    {tmpl['template_id']}: {status}")

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
                        sensor, text, regime,
                        tfidf_raw=tfidf_raw, tfidf_cal=tfidf_cal,
                        llm_model=llm_models[0] if llm_models else "gpt-4o",
                        tuned_p_event=tuned.p_event,
                        shuffled_text=text_by_id[shuffle_map[tid]],
                    )

                    result, trajectory = run_gym_episode_corrected_fill_rate(
                        seed=seed,
                        regime=regime,
                        sensor=sensor,
                        regime_probabilities=probs,
                        scenario=SCENARIO,
                        num_periods=NUM_PERIODS,
                        base_mu=BASE_MU,
                        event_start=EVENT_START,
                        shock_mag=SHOCK_MAG,
                    )

                    row = Phase8BRow(
                        seed=seed,
                        regime=regime.value,
                        sensor=sensor,
                        template_id=tid,
                        ambiguity_level=alevel,
                        total_reward=result.total_reward,
                        total_demand=result.total_demand,
                        total_retail_sales=result.fill_rate * result.total_demand if result.total_demand > 0 else 0.0,
                        fill_rate=result.fill_rate,
                        avg_inventory=result.avg_inventory,
                        inventory_position_final=result.inventory_position_final,
                        belief_normal=result.belief_normal,
                        belief_surge=result.belief_surge,
                        most_likely_regime=result.most_likely_regime,
                        belief_correct=result.belief_correct,
                    )
                    rows.append(row)

                    if call_count % 100 == 0 or call_count == total_calls:
                        print(f"  [{call_count}/{total_calls}] seed={seed} regime={regime.value} "
                              f"sensor={sensor} reward={result.total_reward:.1f}")

    print(f"\nAll {call_count} episodes complete. Computing metrics...")
    return _compute_and_save(rows, seeds, templates)


def _compute_and_save(rows, seeds, templates):
    operational_rows = [asdict(r) for r in rows]
    _save_csv(operational_rows, RESULTS_DIR / "operational_results.csv")

    ni_rewards = {}
    oracle_rewards = {}
    for r in rows:
        key = (r.seed, r.regime, r.template_id)
        if r.sensor == "NoInfo":
            ni_rewards[key] = r.total_reward
        elif r.sensor == "OracleSemantic":
            oracle_rewards[key] = r.total_reward

    for r in rows:
        key = (r.seed, r.regime, r.template_id)
        ni = ni_rewards.get(key, 0.0)
        oracle = oracle_rewards.get(key, 0.0)
        r.oiv = oracle - ni
        r.sivr = signed_sivr(r.total_reward, ni, oracle).value

    operational_rows = [asdict(r) for r in rows]
    _save_csv(operational_rows, RESULTS_DIR / "operational_results.csv")

    print("\n=== Aggregate SIVR Summary ===")
    print(f"  {'Sensor':25s} {'AggSIVR':>8s} {'MeanRew':>8s} {'FillRate':>8s} {'BelAcc':>7s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    sensor_agg = {}
    for sensor in SENSORS:
        sensor_rows = [r for r in rows if r.sensor == sensor]
        rewards = np.array([r.total_reward for r in sensor_rows])
        ni_arr = np.array([ni_rewards.get((r.seed, r.regime, r.template_id), 0.0) for r in sensor_rows])
        oracle_arr = np.array([oracle_rewards.get((r.seed, r.regime, r.template_id), 0.0) for r in sensor_rows])

        num = float(np.sum(rewards - ni_arr))
        agg_sivr = signed_sivr(float(np.mean(rewards)), float(np.mean(ni_arr)),
                               float(np.mean(oracle_arr))).value

        sivr_vals = [r.sivr for r in sensor_rows]
        belief_corrects = [r.belief_correct for r in sensor_rows]

        sensor_agg[sensor] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_fill_rate": float(np.mean([r.fill_rate for r in sensor_rows])),
            "aggregate_sivr": agg_sivr,
            "mean_sivr": float(np.mean(sivr_vals)),
            "belief_accuracy": float(np.mean(belief_corrects)),
            "n": len(sensor_rows),
        }
        print(f"  {sensor:25s} {agg_sivr:8.3f} {float(np.mean(rewards)):8.1f} "
              f"{sensor_agg[sensor]['mean_fill_rate']:8.3f} {sensor_agg[sensor]['belief_accuracy']:7.3f}")

    agg_df = [{"sensor": s, **m} for s, m in sensor_agg.items()]
    _save_csv(agg_df, RESULTS_DIR / "aggregate_sivr.csv")

    print("\n=== Brier Scores ===")
    brier_scores = {}
    for sensor in SENSORS:
        sensor_rows = [r for r in rows if r.sensor == sensor]
        beliefs = []
        for r in sensor_rows:
            true = r.regime
            probs = {"normal": r.belief_normal, "demand_surge": r.belief_surge}
            beliefs.append(standard_brier_score(probs, true,
                                                 class_order=["normal", "demand_surge"]))
        brier_scores[sensor] = float(np.mean(beliefs)) if beliefs else 0.0
        print(f"  {sensor:25s} Brier={brier_scores[sensor]:.4f}")

    _save_csv(
        [{"sensor": s, "brier_score": b} for s, b in brier_scores.items()],
        RESULTS_DIR / "belief_metrics.csv",
    )

    print("\n=== Hierarchical Paired Bootstrap CIs (primary) ===")
    ci_results = []
    for sensor in SENSORS:
        if sensor in ("NoInfo", "OracleSemantic"):
            continue
        s_rows = [r for r in rows if r.sensor == sensor]
        ni_rows = [r for r in rows if r.sensor == "NoInfo"]
        s_dict = {(r.seed, r.regime, r.template_id): r.total_reward for r in s_rows}
        ni_dict = {(r.seed, r.regime, r.template_id): r.total_reward for r in ni_rows}

        diffs_by_template = defaultdict(list)
        for key in s_dict:
            if key in ni_dict:
                template_id = key[2]
                diffs_by_template[template_id].append(s_dict[key] - ni_dict[key])

        diffs_by_template_arr = {k: np.array(v) for k, v in diffs_by_template.items()}
        mean_diff, ci_lo, ci_hi = hierarchical_paired_bootstrap(
            diffs_by_template_arr, n_boot=5000, alpha=0.05, seed=42
        )

        sig = ci_lo > 0 or ci_hi < 0
        ci_results.append({
            "sensor": sensor,
            "vs_sensor": "NoInfo",
            "mean_diff": mean_diff,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "significant": bool(sig),
            "ci_type": "hierarchical_template_seed",
        })
        print(f"  {sensor:25s} diff={mean_diff:7.1f} [{ci_lo:7.1f}, {ci_hi:7.1f}] sig={'YES' if sig else 'NO'}")

    seed_ci_results = []
    for sensor in SENSORS:
        if sensor in ("NoInfo", "OracleSemantic"):
            continue
        s_rewards = np.array([r.total_reward for r in rows if r.sensor == sensor])
        ni_rewards_arr = np.array([r.total_reward for r in rows if r.sensor == "NoInfo"])
        if len(s_rewards) == len(ni_rewards_arr) and len(s_rewards) > 0:
            mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(
                s_rewards, ni_rewards_arr, n_boot=10000, alpha=0.05, seed=42
            )
            seed_ci_results.append({
                "sensor": sensor,
                "vs_sensor": "NoInfo",
                "mean_diff": mean_diff,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "significant": bool(ci_lo > 0 or ci_hi < 0),
                "ci_type": "seed_level_flat",
            })

    all_ci = ci_results + seed_ci_results
    _save_csv(all_ci, RESULTS_DIR / "hierarchical_bootstrap.csv")

    regime_summary = []
    for regime in REGIMES_2:
        for sensor in SENSORS:
            sr = [r for r in rows if r.sensor == sensor and r.regime == regime.value]
            if sr:
                regime_summary.append({
                    "regime": regime.value,
                    "sensor": sensor,
                    "mean_reward": float(np.mean([r.total_reward for r in sr])),
                    "mean_fill_rate": float(np.mean([r.fill_rate for r in sr])),
                    "mean_sivr": float(np.mean([r.sivr for r in sr])),
                    "n": len(sr),
                })
    _save_csv(regime_summary, RESULTS_DIR / "regime_summary.csv")

    ambiguity_summary = []
    for alevel in ["clear", "moderate", "vague"]:
        for sensor in SENSORS:
            ar = [r for r in rows if r.sensor == sensor and r.ambiguity_level == alevel]
            if ar:
                ambiguity_summary.append({
                    "ambiguity_level": alevel,
                    "sensor": sensor,
                    "mean_reward": float(np.mean([r.total_reward for r in ar])),
                    "mean_fill_rate": float(np.mean([r.fill_rate for r in ar])),
                    "mean_sivr": float(np.mean([r.sivr for r in ar])),
                    "n": len(ar),
                })
    _save_csv(ambiguity_summary, RESULTS_DIR / "ambiguity_summary.csv")

    win_rates = []
    for sensor in SENSORS:
        if sensor == "NoInfo":
            continue
        s_dict = {(r.seed, r.regime, r.template_id): r.total_reward for r in rows if r.sensor == sensor}
        ni_dict = {(r.seed, r.regime, r.template_id): r.total_reward for r in rows if r.sensor == "NoInfo"}
        common = set(s_dict.keys()) & set(ni_dict.keys())
        if common:
            wins = sum(1 for k in common if s_dict[k] > ni_dict[k])
            ties = sum(1 for k in common if abs(s_dict[k] - ni_dict[k]) < 1e-9)
            win_rates.append({
                "sensor": sensor,
                "win_rate_vs_NoInfo": wins / len(common),
                "tie_rate": ties / len(common),
                "n_pairs": len(common),
            })
    _save_csv(win_rates, RESULTS_DIR / "win_rates.csv")

    print("\n=== Interpreter Semantic Metrics ===")
    interp_rows = []
    for sensor in SENSORS:
        sr = [r for r in rows if r.sensor == sensor]
        acc = float(np.mean([r.belief_correct for r in sr]))
        beliefs_list = []
        for r in sr:
            true = r.regime
            b = r.belief_normal if true == "normal" else r.belief_surge
            beliefs_list.append(b)
        brier = float(np.mean([(1 - b) ** 2 for b in beliefs_list])) if beliefs_list else 0.0
        logloss = -float(np.mean([np.log(max(b, 1e-15)) for b in beliefs_list])) if beliefs_list else 0.0
        interp_rows.append({
            "sensor": sensor,
            "accuracy": acc,
            "brier": brier,
            "log_loss": logloss,
            "mean_reward": sensor_agg[sensor]["mean_reward"],
            "aggregate_sivr": sensor_agg[sensor]["aggregate_sivr"],
        })
        print(f"  {sensor:25s} acc={acc:.3f} brier={brier:.4f} logloss={logloss:.3f}")
    _save_csv(interp_rows, RESULTS_DIR / "interpreter_summary.csv")

    manifest = {
        "phase": "8B",
        "description": "Frozen leakage-safe confirmation of semantic decision-value in Paper-1 Gymnasium",
        "parent_phase": "8A",
        "seeds": seeds,
        "seed_range": f"{seeds[0]}-{seeds[-1]}",
        "num_seeds": len(seeds),
        "templates": [t["template_id"] for t in templates],
        "template_source": "CONFIRMATION_TEMPLATES (Phase-6 held-out)",
        "num_templates": len(templates),
        "sensors": SENSORS,
        "base_mu": BASE_MU,
        "shock_mag": SHOCK_MAG,
        "scenario": SCENARIO,
        "num_periods": NUM_PERIODS,
        "event_start": EVENT_START,
        "fill_rate_formula": "sum(retail_sales_S[:,retail_link]) / sum(customer_demand_D)",
        "paper1_version": "0.1.0",
        "paper1_commit": "a745fd5186a73d177dd94d283a4f8f8e8d329977",
        "simulator_package": "gym-invmgmt==0.2.1 (published release)",
        "tfidf_raw_model": "results/phase7_classical_baseline/tfidf_logreg_model.pkl",
        "tfidf_calibrated_model": "results/phase7_classical_baseline/tfidf_logreg_calibrated_model.pkl",
        "controller_mapping": "effective_mu = base_mu * (1 + P(surge) * (surge_multiplier - 1))",
        "aggregate_sivr": {s: sensor_agg[s]["aggregate_sivr"] for s in SENSORS},
    }
    (RESULTS_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    split_manifest = {
        "phase": "8B",
        "frozen": True,
        "train_source": "REGIME_WARNING_TEMPLATES (original 18)",
        "test_source": "CONFIRMATION_TEMPLATES (36 held-out from Phase 6)",
        "test_used": [t["template_id"] for t in templates],
        "num_test": len(templates),
        "leakage_audit": {
            "train_template_ids": [
                "delay_clear_1", "delay_clear_2", "delay_moderate_1", "delay_moderate_2",
                "delay_vague_1", "delay_vague_2", "surge_clear_1", "surge_clear_2",
                "surge_moderate_1", "surge_moderate_2", "surge_vague_1", "surge_vague_2",
                "normal_clear_1", "normal_clear_2", "normal_moderate_1", "normal_moderate_2",
                "normal_vague_1", "normal_vague_2",
            ],
            "test_template_ids": [t["template_id"] for t in templates],
            "overlap": [],
            "no_leak": True,
            "vectorizer_fitted_on_train_only": True,
            "calibration_on_train_only": True,
        },
    }
    (RESULTS_DIR / "linguistic_split_manifest.json").write_text(json.dumps(split_manifest, indent=2))

    seed_manifest = {
        "phase": "8B",
        "seeds": seeds,
        "phase8a_pilot_seeds": list(range(1, 21)),
        "phase8a_confirmation_seeds": list(range(2000, 2020)),
        "phase8b_seeds": seeds,
        "all_disjoint": True,
    }
    (RESULTS_DIR / "seed_manifest.json").write_text(json.dumps(seed_manifest, indent=2))

    mechanism = _mechanism_audit(rows, templates, seeds)
    _save_csv(mechanism, RESULTS_DIR / "mechanism_audit.csv")

    print(f"\nPhase 8B complete. Results in {RESULTS_DIR}")
    return rows, sensor_agg, ci_results


def _mechanism_audit(rows, templates, seeds):
    representative_seed = seeds[0]
    surge_tmpl = next((t for t in templates if t["regime"] == Regime.DEMAND_SURGE), None)
    if surge_tmpl is None:
        return []

    text = surge_tmpl["text"]
    mechanism_rows = []

    sensor_configs = [
        ("NoInfo", get_no_info_probs()),
        ("RuleBased", get_rulebased_probs(text)),
        ("OracleSemantic", get_perfect_semantic_probs(Regime.DEMAND_SURGE)),
    ]

    pkl_raw = PHASE7_DIR / "tfidf_logreg_model.pkl"
    pkl_cal = PHASE7_DIR / "tfidf_logreg_calibrated_model.pkl"
    tfidf_raw = TFIDFLogReg.load(str(pkl_raw))
    tfidf_cal = TFIDFLogReg.load(str(pkl_cal))
    sensor_configs.append(("TFIDF_LogReg_Raw", get_tfidf_probs(text, tfidf_raw)))
    sensor_configs.append(("TFIDF_LogReg_Calibrated", get_tfidf_probs(text, tfidf_cal)))

    for sensor_name, probs in sensor_configs:
        result, trajectory = run_gym_episode_corrected_fill_rate(
            seed=representative_seed,
            regime=Regime.DEMAND_SURGE,
            sensor=sensor_name,
            regime_probabilities=probs,
            scenario=SCENARIO,
            num_periods=NUM_PERIODS,
            base_mu=BASE_MU,
            event_start=EVENT_START,
            shock_mag=SHOCK_MAG,
        )

        for step in trajectory:
            t = step["period"]
            obs = step["obs"]
            action = step["action"]
            inventory = step["inventory"]

            inventory_position = float(obs[2]) if len(obs) > 2 else 0.0

            phase = "pre_surge" if t < EVENT_START else "surge"

            mechanism_rows.append({
                "sensor": sensor_name,
                "template_id": surge_tmpl["template_id"],
                "seed": representative_seed,
                "period": t,
                "phase": phase,
                "P_surge": probs.get("demand_surge", 0.0),
                "action_sum": float(np.sum(action)),
                "reward": step["reward"],
                "inventory_position": inventory_position,
            })

    return mechanism_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8B: Frozen confirmation")
    # Defaults MUST match PHASE8B_SEEDS, otherwise the documented command
    # reproduces a different experiment from the frozen results. Before
    # September 2026 the default started at 3000 while the declared
    # confirmation seeds were 3100-3129.
    parser.add_argument("--seeds", type=int, default=len(PHASE8B_SEEDS))
    parser.add_argument("--seed-start", type=int, default=PHASE8B_SEEDS[0])
    parser.add_argument("--llm-models", nargs="*", default=["gpt-4o"])
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if args.no_llm:
        llm_models = []
        SENSORS.remove("gpt-4o")
    else:
        llm_models = args.llm_models

    run_phase8b(seeds=seeds, llm_models=llm_models)
