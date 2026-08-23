"""Phase 9A: Cross-Dynamics Generalization — SupplierCapacityDrop.

Verifies that the benchmark's semantic-value framework generalizes
to a different shock type (supply-side capacity disruption vs demand surge).

Uses divergent topology (Factory 4 with C=90), temporary capacity reduction
to C=30 during periods 10-25. Binary classification: Normal vs CapacityDrop.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
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
    RegimeInterpretation,
)
from src.gym_adapter9 import (
    run_phase9_episode,
    get_no_info_probs_9,
    get_perfect_semantic_probs_9,
    SupplyBeliefAdaptiveController,
    DEFAULT_BASE_MU,
    DEFAULT_NUM_PERIODS,
    DEFAULT_EVENT_START,
    DEFAULT_EVENT_END,
    DEFAULT_AFFECTED_FACTORY,
    DEFAULT_NORMAL_CAPACITY,
    DEFAULT_DISRUPTED_CAPACITY,
    DEFAULT_SCENARIO,
)
from src.classical_baseline import TFIDFLogReg
from src.capacity_drop_templates import (
    ALL_CAPACITY_DROP_TEMPLATES,
    TRAIN_TEMPLATES,
    TRAIN_TEMPLATE_IDS,
    CAPACITY_DROP_TEMPLATES,
    NORMAL_TEMPLATES,
    Regime9,
)
from src.metrics import signed_sivr, standard_brier_score

RESULTS_DIR = ROOT / "results" / "phase9a_capacity_confirmation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHASE9_SEEDS = list(range(4000, 4030))

SENSORS = [
    "NoInfo",
    "RuleBased",
    "TFIDF_LogReg_Raw",
    "TFIDF_LogReg_Calibrated",
    "gpt-4o",
    "OracleSemantic",
]

BASE_MU = DEFAULT_BASE_MU
NUM_PERIODS = DEFAULT_NUM_PERIODS
EVENT_START = DEFAULT_EVENT_START
EVENT_END = DEFAULT_EVENT_END
SCENARIO = DEFAULT_SCENARIO
AFFECTED_FACTORY = DEFAULT_AFFECTED_FACTORY
NORMAL_CAPACITY = DEFAULT_NORMAL_CAPACITY
DISRUPTED_CAPACITY = DEFAULT_DISRUPTED_CAPACITY

REGIMES_9 = [Regime.NORMAL, Regime.SUPPLIER_CAPACITY_DROP]


@dataclass
class Phase9ARow:
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
    belief_capacity_drop: float
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
    """Get test templates — those NOT used for TF-IDF training."""
    test_ids = [t["template_id"] for t in ALL_CAPACITY_DROP_TEMPLATES
                if t["template_id"] not in TRAIN_TEMPLATE_IDS]
    test_templates = [t for t in ALL_CAPACITY_DROP_TEMPLATES
                      if t["template_id"] in test_ids]
    return test_templates


def _get_sensor_belief(
    sensor: str,
    text: str,
    true_regime: Regime,
    tfidf_raw=None,
    tfidf_cal=None,
    llm_model: str = "gpt-4o",
) -> dict[str, float]:
    if sensor == "NoInfo":
        return get_no_info_probs_9()
    elif sensor == "OracleSemantic":
        return get_perfect_semantic_probs_9(true_regime)
    elif sensor == "RuleBased":
        return _rulebased_probs_9(text)
    elif sensor == "TFIDF_LogReg_Raw" and tfidf_raw is not None:
        return _tfidf_probs_9(text, tfidf_raw)
    elif sensor == "TFIDF_LogReg_Calibrated" and tfidf_cal is not None:
        return _tfidf_probs_9(text, tfidf_cal)
    elif sensor == "gpt-4o":
        return _llm_probs_9(text, model=llm_model)
    return get_no_info_probs_9()


def _rulebased_probs_9(text: str) -> dict[str, float]:
    text_lower = text.lower()
    cd_score = 0.0
    normal_score = 0.0

    if any(w in text_lower for w in [
        "capacity", "production", "output", "factory", "manufacturing",
    ]):
        cd_score += 0.3
    if any(w in text_lower for w in [
        "reduce", "drop", "cut", "constraint", "shutdown", "failure",
        "breakdown", "limited", "lower", "below normal", "reduced",
    ]):
        cd_score += 0.3
    if any(w in text_lower for w in [
        "confirmed", "urgent", "notified", "equipment", "machinery",
    ]):
        cd_score += 0.2

    if any(w in text_lower for w in [
        "on schedule", "normal", "no disruption", "no change",
        "within normal", "stable", "no material", "operating normally",
        "standard", "expected ranges", "no capacity",
    ]):
        normal_score += 0.4
    if any(w in text_lower for w in [
        "no unusual", "routine variability", "no significant",
    ]):
        normal_score += 0.2

    total = cd_score + normal_score
    if total < 1e-9:
        return {"normal": 0.5, "supplier_capacity_drop": 0.5}
    return {
        "normal": normal_score / total,
        "supplier_capacity_drop": cd_score / total,
    }


def _tfidf_probs_9(text: str, model: TFIDFLogReg) -> dict[str, float]:
    probs_arr = model.predict_proba([text])[0]
    classes = model.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs_arr)}

    p_normal = prob_dict.get("normal", 0.0)
    p_cd = prob_dict.get("supplier_capacity_drop", 0.0)
    total = p_normal + p_cd
    if total > 0:
        return {"normal": p_normal / total, "supplier_capacity_drop": p_cd / total}
    return {"normal": 0.5, "supplier_capacity_drop": 0.5}


def _llm_probs_9(text: str, model: str = "gpt-4o") -> dict[str, float]:
    prompt = (
        "You are an operational risk analyst. Given a textual factory or supply "
        "report, classify the likely operational regime. Respond ONLY with valid "
        'JSON: {"normal": <float 0-1>, "supplier_capacity_drop": <float 0-1>}\n'
        "The two regime probabilities should sum approximately to 1.0.\n"
        "Do not include any other text."
    )
    import openai
    import hashlib
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return {"normal": 0.5, "supplier_capacity_drop": 0.5}

    h = hashlib.sha256(f"{model}||{text}".encode()).hexdigest()[:16]
    cache_path = ROOT / ".llm_cache" / f"phase9_{h}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        return data

    import time
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=15.0, max_retries=2)
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        p_normal = float(data.get("normal", 0.5))
        p_cd = float(data.get("supplier_capacity_drop", 0.5))
        total = p_normal + p_cd
        if total > 0:
            result = {"normal": p_normal / total, "supplier_capacity_drop": p_cd / total}
        else:
            result = {"normal": 0.5, "supplier_capacity_drop": 0.5}
        cache_path.write_text(json.dumps(result))
        return result
    except Exception:
        return {"normal": 0.5, "supplier_capacity_drop": 0.5}


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


def run_phase9a(seeds=None, llm_models=None):
    if seeds is None:
        seeds = PHASE9_SEEDS
    if llm_models is None:
        llm_models = ["gpt-4o"]

    train_templates = TRAIN_TEMPLATES
    test_templates = _get_heldout_templates()
    all_templates = train_templates + test_templates

    print(f"Phase 9A: {len(seeds)} seeds x {len(all_templates)} templates x {len(SENSORS)} sensors")
    print(f"  Seeds: {seeds[0]}-{seeds[-1]}")
    print(f"  Train templates: {len(train_templates)} ({len([t for t in train_templates if t['regime']==Regime9.SUPPLIER_CAPACITY_DROP])} cd, {len([t for t in train_templates if t['regime']==Regime9.NORMAL])} normal)")
    print(f"  Test templates:  {len(test_templates)} ({len([t for t in test_templates if t['regime']==Regime9.SUPPLIER_CAPACITY_DROP])} cd, {len([t for t in test_templates if t['regime']==Regime9.NORMAL])} normal)")
    print(f"  Results dir: {RESULTS_DIR}")

    train_texts = [t["text"] for t in train_templates]
    train_labels = [t["regime"].value for t in train_templates]

    tfidf_raw = TFIDFLogReg(seed=42)
    tfidf_raw.fit(train_texts, train_labels)
    tfidf_raw.save(RESULTS_DIR / "tfidf_capacity_logreg_raw.pkl")
    print(f"  Trained TF-IDF raw model ({tfidf_raw.train_size_} train, {tfidf_raw.vocabulary_size_} vocab)")

    tfidf_cal = TFIDFLogReg(calibrate=True, seed=42)
    tfidf_cal.fit(train_texts, train_labels)
    tfidf_cal.save(RESULTS_DIR / "tfidf_capacity_logreg_calibrated.pkl")
    print(f"  Trained TF-IDF calibrated model")

    for model in llm_models:
        print(f"  Pre-caching LLM responses for {model}...")
        for tmpl in all_templates:
            try:
                _llm_probs_9(tmpl["text"], model=model)
                print(f"    {tmpl['template_id']}: OK")
            except Exception as e:
                print(f"    {tmpl['template_id']}: ERROR {e}")

    rows = []
    total_calls = len(seeds) * len(all_templates) * len(SENSORS)
    call_count = 0

    for regime in REGIMES_9:
        r_templates = [t for t in all_templates if t["regime"].value == regime.value]
        if not r_templates:
            r_templates = all_templates

        for seed in seeds:
            for tmpl in r_templates:
                text = tmpl["text"]
                tid = tmpl["template_id"]
                alevel = tmpl["ambiguity_level"]

                for sensor in SENSORS:
                    call_count += 1
                    probs = _get_sensor_belief(
                        sensor, text, regime,
                        tfidf_raw=tfidf_raw, tfidf_cal=tfidf_cal,
                        llm_model=llm_models[0] if llm_models else "gpt-4o",
                    )

                    result, trajectory = run_phase9_episode(
                        seed=seed,
                        regime=regime,
                        sensor=sensor,
                        regime_probabilities=probs,
                        scenario=SCENARIO,
                        num_periods=NUM_PERIODS,
                        base_mu=BASE_MU,
                        event_start=EVENT_START,
                        event_end=EVENT_END,
                        affected_factory=AFFECTED_FACTORY,
                        normal_capacity=NORMAL_CAPACITY,
                        disrupted_capacity=DISRUPTED_CAPACITY,
                    )

                    row = Phase9ARow(
                        seed=seed,
                        regime=regime.value,
                        sensor=sensor,
                        template_id=tid,
                        ambiguity_level=alevel,
                        total_reward=result.total_reward,
                        total_demand=result.total_demand,
                        total_retail_sales=result.total_retail_sales,
                        fill_rate=result.fill_rate,
                        avg_inventory=result.avg_inventory,
                        inventory_position_final=result.inventory_position_final,
                        belief_normal=result.belief_normal,
                        belief_capacity_drop=result.belief_capacity_drop,
                        most_likely_regime=result.most_likely_regime,
                        belief_correct=result.belief_correct,
                    )
                    rows.append(row)

                    if call_count % 100 == 0 or call_count == total_calls:
                        print(f"  [{call_count}/{total_calls}] seed={seed} regime={regime.value} "
                              f"sensor={sensor} reward={result.total_reward:.1f}")

    print(f"\nAll {call_count} episodes complete. Computing metrics...")
    return _compute_and_save(rows, seeds, all_templates)





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
            probs = {"normal": r.belief_normal,
                     "supplier_capacity_drop": r.belief_capacity_drop}
            beliefs.append(standard_brier_score(
                probs, true, class_order=["normal", "supplier_capacity_drop"],
            ))
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
        if not diffs_by_template_arr:
            continue
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
    for regime in REGIMES_9:
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
            b = r.belief_normal if true == "normal" else r.belief_capacity_drop
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

    mechanism = _mechanism_audit(rows, templates, seeds)
    _save_csv(mechanism, RESULTS_DIR / "mechanism_audit.csv")

    bullwhip_data = _compute_bullwhip(rows, seeds)
    _save_csv(bullwhip_data, RESULTS_DIR / "bullwhip.csv")

    manifest = {
        "phase": "9A",
        "description": "Cross-Dynamics Generalization: SupplierCapacityDrop on divergent topology",
        "topology": "divergent",
        "seeds": seeds,
        "seed_range": f"{seeds[0]}-{seeds[-1]}",
        "num_seeds": len(seeds),
        "train_templates": TRAIN_TEMPLATE_IDS,
        "test_templates": [t["template_id"] for t in _get_heldout_templates()],
        "num_templates": len(ALL_CAPACITY_DROP_TEMPLATES),
        "sensors": SENSORS,
        "base_mu": BASE_MU,
        "scenario": SCENARIO,
        "num_periods": NUM_PERIODS,
        "event_start": EVENT_START,
        "event_end": EVENT_END,
        "affected_factory": AFFECTED_FACTORY,
        "normal_capacity": NORMAL_CAPACITY,
        "disrupted_capacity": DISRUPTED_CAPACITY,
        "controller_mapping": "effective_capacity = P(Normal)*C_normal + P(CapacityDrop)*C_disrupted",
        "aggregate_sivr": {s: sensor_agg[s]["aggregate_sivr"] for s in SENSORS},
        "version_parity_audit": {
            "gym_invmgmt_v0.1.0_commit": "a745fd5",
            "gym_invmgmt_v0.2.0_commit": "a97a603",
            "parity_classification": "A",
            "core_dynamics_identical": True,
            "differences": "packaging/metadata only",
        },
    }
    (RESULTS_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nPhase 9A complete. Results in {RESULTS_DIR}")
    return rows, sensor_agg, ci_results


def _mechanism_audit(rows, templates, seeds):
    representative_seed = seeds[0]
    cd_tmpl = next((t for t in templates if t["regime"] == Regime9.SUPPLIER_CAPACITY_DROP), None)
    if cd_tmpl is None:
        return []

    text = cd_tmpl["text"]
    mechanism_rows = []

    sensor_configs = [
        ("NoInfo", get_no_info_probs_9()),
        ("RuleBased", _rulebased_probs_9(text)),
        ("OracleSemantic", get_perfect_semantic_probs_9(Regime.SUPPLIER_CAPACITY_DROP)),
    ]

    tfidf_raw = TFIDFLogReg.load(str(RESULTS_DIR / "tfidf_capacity_logreg_raw.pkl"))
    tfidf_cal = TFIDFLogReg.load(str(RESULTS_DIR / "tfidf_capacity_logreg_calibrated.pkl"))
    sensor_configs.append(("TFIDF_LogReg_Raw", _tfidf_probs_9(text, tfidf_raw)))
    sensor_configs.append(("TFIDF_LogReg_Calibrated", _tfidf_probs_9(text, tfidf_cal)))

    for sensor_name, probs in sensor_configs:
        result, trajectory = run_phase9_episode(
            seed=representative_seed,
            regime=Regime.SUPPLIER_CAPACITY_DROP,
            sensor=sensor_name,
            regime_probabilities=probs,
            scenario=SCENARIO,
            num_periods=NUM_PERIODS,
            base_mu=BASE_MU,
            event_start=EVENT_START,
            event_end=EVENT_END,
            affected_factory=AFFECTED_FACTORY,
            normal_capacity=NORMAL_CAPACITY,
            disrupted_capacity=DISRUPTED_CAPACITY,
        )

        for step in trajectory:
            t = step["period"]
            obs = step["obs"]
            action = step["action"]
            inventory = step["inventory"]

            inventory_position = float(obs[2]) if len(obs) > 2 else 0.0

            if t < EVENT_START:
                phase = "pre_disruption"
            elif t < EVENT_END:
                phase = "disruption"
            else:
                phase = "post_disruption"

            mechanism_rows.append({
                "sensor": sensor_name,
                "template_id": cd_tmpl["template_id"],
                "seed": representative_seed,
                "period": t,
                "phase": phase,
                "P_capacity_drop": probs.get("supplier_capacity_drop", 0.0),
                "action_sum": float(np.sum(action)),
                "reward": step["reward"],
                "inventory_position": inventory_position,
            })

    return mechanism_rows


def _compute_bullwhip(rows, seeds):
    bullwhip_rows = []
    for sensor in SENSORS:
        for regime in REGIMES_9:
            sr = [r for r in rows if r.sensor == sensor and r.regime == regime.value]
            if not sr:
                continue
            rewards = np.array([r.total_reward for r in sr])
            demands = np.array([r.total_demand for r in sr])
            fill_rates = np.array([r.fill_rate for r in sr])

            reward_cv = float(np.std(rewards) / np.mean(rewards)) if np.mean(rewards) > 0 else 0.0
            demand_cv = float(np.std(demands) / np.mean(demands)) if np.mean(demands) > 0 else 0.0
            fill_rate_cv = float(np.std(fill_rates) / np.mean(fill_rates)) if np.mean(fill_rates) > 0 else 0.0

            bullwhip_ratio = reward_cv / demand_cv if demand_cv > 0 else 0.0

            bullwhip_rows.append({
                "sensor": sensor,
                "regime": regime.value,
                "reward_cv": reward_cv,
                "demand_cv": demand_cv,
                "fill_rate_cv": fill_rate_cv,
                "bullwhip_ratio": bullwhip_ratio,
                "mean_reward": float(np.mean(rewards)),
                "mean_demand": float(np.mean(demands)),
                "n": len(sr),
            })
    return bullwhip_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 9A: CapacityDrop cross-dynamics")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--llm-models", nargs="*", default=["gpt-4o"])
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if args.no_llm:
        llm_models = []
        SENSORS.remove("gpt-4o")
    else:
        llm_models = args.llm_models

    run_phase9a(seeds=seeds, llm_models=llm_models)
