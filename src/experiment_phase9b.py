"""Phase 9B: One-Factor-at-a-Time Robustness for SupplierCapacityDrop.

Tests whether the semantic-value advantage persists across environmental
variations: lead times, lost sales, and demand noise.  Each variant
modifies ONE factor from the frozen Phase-9A baseline and runs its own
NoInfo/Oracle baselines so SIVR is self-contained.
"""

from __future__ import annotations

import csv
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
    Regime9,
)
from src.experiment_phase9a import (
    hierarchical_paired_bootstrap,
    _rulebased_probs_9,
)
from src.metrics import signed_sivr, standard_brier_score

ROBUSTNESS_DIR = ROOT / "results" / "phase9b_robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

PHASE9B_SEEDS = list(range(4100, 4110))

SENSORS = [
    "NoInfo",
    "Constant_p1.0",       # the strongest text-free control found in Experiments M/S
    "NoText_Tuned",
    "RuleBased",
    "TFIDF_LogReg_Raw",
    "TFIDF_LogReg_Calibrated",
    "gpt-4o",
    "OracleSemantic",
]

REGIME = Regime.SUPPLIER_CAPACITY_DROP

# --- Text-world pairing -------------------------------------------------
# Phase 9B iterates over all 16 templates, six of which are labelled `normal`.
# Until September 2026 every episode was nevertheless simulated in a
# SUPPLIER_CAPACITY_DROP world, so those six reports were false negatives.
# That is a legitimate experiment, but it is an *evidence intervention*, not
# the mechanical one-factor-at-a-time robustness study the results were
# presented as, and it silently broke the text-world relationship that
# Phase 9A establishes.
#
#   "matched"  (default)  each template is simulated in the world its own
#                         label describes -- the honest OFAT robustness study.
#   "false_negative"      normal reports are paired with a capacity-drop
#                         world -- reported separately as a misinformation /
#                         false-negative evidence intervention.
PAIRING_MATCHED = "matched"
PAIRING_FALSE_NEGATIVE = "false_negative"
PAIRINGS = (PAIRING_MATCHED, PAIRING_FALSE_NEGATIVE)
DEFAULT_PAIRING = PAIRING_MATCHED


def world_regime_for(template, pairing: str) -> Regime:
    """Resolve the simulated world for a template under the chosen pairing."""
    if pairing == PAIRING_FALSE_NEGATIVE:
        return REGIME
    tmpl_regime = template.get("regime", REGIME)
    return tmpl_regime if isinstance(tmpl_regime, Regime) else Regime(tmpl_regime)

VARIANTS = {
    "baseline": {
        "label": "Baseline (Phase-9A settings)",
        "scenario": DEFAULT_SCENARIO,
        "config_path": None,
        "backlog": True,
        "noise_scale": None,
    },
    "short_lead": {
        "label": "SHORT_LEAD (L×0.5)",
        "scenario": "custom",
        "config_path": str(ROBUSTNESS_DIR / "divergent_short_lead.yaml"),
        "backlog": True,
        "noise_scale": None,
    },
    "long_lead": {
        "label": "LONG_LEAD (L×2.0)",
        "scenario": "custom",
        "config_path": str(ROBUSTNESS_DIR / "divergent_long_lead.yaml"),
        "backlog": True,
        "noise_scale": None,
    },
    "lost_sales": {
        "label": "LOST_SALES (backlog=False)",
        "scenario": DEFAULT_SCENARIO,
        "config_path": None,
        "backlog": False,
        "noise_scale": None,
    },
    "low_noise": {
        "label": "LOW_NOISE (noise_scale=0.0)",
        "scenario": DEFAULT_SCENARIO,
        "config_path": None,
        "backlog": True,
        "noise_scale": 0.0,
    },
    "high_noise": {
        "label": "HIGH_NOISE (noise_scale=2.0)",
        "scenario": DEFAULT_SCENARIO,
        "config_path": None,
        "backlog": True,
        "noise_scale": 2.0,
    },
}


@dataclass
class Phase9BRow:
    variant: str
    seed: int
    regime: str                 # the world actually simulated
    template_regime: str        # the regime the warning text describes
    pairing: str                # matched | false_negative
    text_world_mismatch: bool   # True when the report contradicts the world
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
    test_ids = [t["template_id"] for t in ALL_CAPACITY_DROP_TEMPLATES
                if t["template_id"] not in TRAIN_TEMPLATE_IDS]
    return [t for t in ALL_CAPACITY_DROP_TEMPLATES if t["template_id"] in test_ids]


def _tfidf_probs(text, model):
    probs_arr = model.predict_proba([text])[0]
    classes = model.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs_arr)}
    p_normal = prob_dict.get("normal", 0.0)
    p_cd = prob_dict.get("supplier_capacity_drop", 0.0)
    total = p_normal + p_cd
    if total > 0:
        return {"normal": p_normal / total, "supplier_capacity_drop": p_cd / total}
    return {"normal": 0.5, "supplier_capacity_drop": 0.5}


def _llm_probs(text, model="gpt-4o"):
    """Reuse the cached LLM responses from Phase 9A."""
    import hashlib
    from src.offline_artifacts import find_cached_response, require_cached_response

    # Cache FIRST: an offline replay must not depend on credentials, and a
    # missing entry must never be replaced by a silent 50/50 prior recorded
    # under the model's own name.
    h = hashlib.sha256(f"{model}||{text}".encode()).hexdigest()[:16]
    cache_path = find_cached_response(f"phase9_{h}.json")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    if cache_path is None and not api_key:
        require_cached_response(f"phase9_{h}.json", model=model, text=text)
    if cache_path is not None:
        return json.loads(cache_path.read_text())

    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=15.0, max_retries=2)
    prompt = (
        "You are an operational risk analyst. Given a textual factory or supply "
        "report, classify the likely operational regime. Respond ONLY with valid "
        'JSON: {"normal": <float 0-1>, "supplier_capacity_drop": <float 0-1>}\n'
        "The two regime probabilities should sum approximately to 1.0.\n"
        "Do not include any other text."
    )
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
        from src.offline_artifacts import runtime_cache_dir
        target = runtime_cache_dir() / f"phase9_{h}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result))
        return result
    except Exception as exc:
        # Never degrade to an uninformative prior recorded under the model's
        # name: that produced episodes labelled `gpt-4o` containing no model
        # output at all.  Fail loudly instead.
        raise RuntimeError(
            f"Live interpretation failed for model={model!r} and no frozen "
            f"cache entry exists (phase9_{h}.json): {exc}"
        ) from exc


def _get_sensor_belief(sensor, text, tfidf_raw, tfidf_cal, llm_model="gpt-4o",
                       world_regime=None, tuned_p_event=None):
    """Resolve one sensor's belief.

    `world_regime` is the regime actually simulated for this episode. The oracle
    reference must describe THAT world; previously it always asserted
    SUPPLIER_CAPACITY_DROP, which under the matched pairing made the
    "oracle" wrong on every normal-world episode.
    """
    from src.notext_controls import constant_belief

    world = world_regime or REGIME
    if sensor == "NoInfo":
        return get_no_info_probs_9()
    if sensor.startswith("Constant_p"):
        return constant_belief(float(sensor.split("_p")[1]), REGIME)
    if sensor == "NoText_Tuned":
        if tuned_p_event is None:
            raise ValueError("NoText_Tuned needs a development-tuned operating point")
        return constant_belief(tuned_p_event, REGIME)
    if sensor == "OracleSemantic":
        return get_perfect_semantic_probs_9(world)
    if sensor == "RuleBased":
        return _rulebased_probs_9(text)
    if sensor == "TFIDF_LogReg_Raw" and tfidf_raw is not None:
        return _tfidf_probs(text, tfidf_raw)
    if sensor == "TFIDF_LogReg_Calibrated" and tfidf_cal is not None:
        return _tfidf_probs(text, tfidf_cal)
    if sensor == "gpt-4o":
        return _llm_probs(text, model=llm_model)
    raise ValueError(f"unknown sensor {sensor!r} (or its model was not supplied)")


def _compute_variant_metrics(rows, variant_name, variant_dir):
    """Compute aggregate SIVR, Brier, bootstrap CIs for one variant."""
    _save_csv([asdict(r) for r in rows], variant_dir / "operational_results.csv")

    ni_rewards = {}
    oracle_rewards = {}
    for r in rows:
        key = (r.seed, r.template_id)
        if r.sensor == "NoInfo":
            ni_rewards[key] = r.total_reward
        elif r.sensor == "OracleSemantic":
            oracle_rewards[key] = r.total_reward

    for r in rows:
        key = (r.seed, r.template_id)
        ni = ni_rewards.get(key, 0.0)
        oracle = oracle_rewards.get(key, 0.0)
        r.oiv = oracle - ni
        r.sivr = signed_sivr(r.total_reward, ni, oracle).value

    _save_csv([asdict(r) for r in rows], variant_dir / "operational_results.csv")

    sensor_agg = {}
    for sensor in SENSORS:
        s_rows = [r for r in rows if r.sensor == sensor]
        if not s_rows:
            continue
        rewards = np.array([r.total_reward for r in s_rows])
        ni_arr = np.array([ni_rewards.get((r.seed, r.template_id), 0.0) for r in s_rows])
        oracle_arr = np.array([oracle_rewards.get((r.seed, r.template_id), 0.0) for r in s_rows])

        agg_sivr = signed_sivr(float(np.mean(rewards)), float(np.mean(ni_arr)),
                               float(np.mean(oracle_arr))).value

        sensor_agg[sensor] = {
            "sensor": sensor,
            "aggregate_sivr": agg_sivr,
            "mean_reward": float(np.mean(rewards)),
            "mean_fill_rate": float(np.mean([r.fill_rate for r in s_rows])),
            "belief_accuracy": float(np.mean([r.belief_correct for r in s_rows])),
        }

    _save_csv(list(sensor_agg.values()), variant_dir / "aggregate_sivr.csv")

    brier_scores = {}
    for sensor in SENSORS:
        s_rows = [r for r in rows if r.sensor == sensor]
        beliefs = []
        for r in s_rows:
            probs = {"normal": r.belief_normal,
                     "supplier_capacity_drop": r.belief_capacity_drop}
            beliefs.append(standard_brier_score(
                probs, r.regime,
                class_order=["normal", "supplier_capacity_drop"],
            ))
        brier_scores[sensor] = float(np.mean(beliefs)) if beliefs else 0.0
    _save_csv(
        [{"sensor": s, "brier_score": b} for s, b in brier_scores.items()],
        variant_dir / "belief_metrics.csv",
    )

    ci_results = []
    for sensor in SENSORS:
        if sensor in ("NoInfo", "OracleSemantic"):
            continue
        s_rows = [r for r in rows if r.sensor == sensor]
        ni_rows = [r for r in rows if r.sensor == "NoInfo"]
        s_dict = {(r.seed, r.template_id): r.total_reward for r in s_rows}
        ni_dict = {(r.seed, r.template_id): r.total_reward for r in ni_rows}

        diffs_by_template = defaultdict(list)
        for key in s_dict:
            if key in ni_dict:
                tid = key[1]
                diffs_by_template[tid].append(s_dict[key] - ni_dict[key])

        diffs_arr = {k: np.array(v) for k, v in diffs_by_template.items()}
        if not diffs_arr:
            continue
        mean_diff, ci_lo, ci_hi = hierarchical_paired_bootstrap(
            diffs_arr, n_boot=5000, alpha=0.05, seed=42,
        )
        sig = ci_lo > 0 or ci_hi < 0
        ci_results.append({
            "sensor": sensor,
            "mean_diff": mean_diff,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "significant": bool(sig),
        })

    _save_csv(ci_results, variant_dir / "hierarchical_bootstrap.csv")
    return sensor_agg, ci_results


def run_phase9b(seeds=None, llm_models=None, pairing: str = DEFAULT_PAIRING):
    if seeds is None:
        seeds = PHASE9B_SEEDS
    if llm_models is None:
        llm_models = ["gpt-4o"]
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")

    train_templates = TRAIN_TEMPLATES
    test_templates = _get_heldout_templates()
    all_templates = train_templates + test_templates

    print(f"Phase 9B: {len(VARIANTS)} variants x {len(seeds)} seeds x "
          f"{len(all_templates)} templates x {len(SENSORS)} sensors")
    print(f"  Text-world pairing: {pairing}")
    n_mismatch = sum(
        1 for t in all_templates if world_regime_for(t, pairing) is not (
            t["regime"] if isinstance(t["regime"], Regime) else Regime(t["regime"])
        )
    )
    print(f"  Templates whose report contradicts the simulated world: {n_mismatch}"
          f"/{len(all_templates)}")

    train_texts = [t["text"] for t in train_templates]
    train_labels = [t["regime"].value for t in train_templates]

    tfidf_raw = TFIDFLogReg(seed=42)
    tfidf_raw.fit(train_texts, train_labels)
    print(f"  TF-IDF raw trained")

    tfidf_cal = TFIDFLogReg(calibrate=True, seed=42)
    tfidf_cal.fit(train_texts, train_labels)
    print(f"  TF-IDF calibrated trained")

    # Reuse the Experiment-S no-text operating point, selected on development
    # seeds. Boundary analysis must not re-tune per variant: that would let the
    # control see the very perturbation it is being tested against.
    tuning_path = ROOT / "results" / "phase9a_capacity_confirmation" / "notext_tuning.json"
    if tuning_path.exists():
        tuned_p_event = json.loads(tuning_path.read_text())["p_event"]
        print(f"  No-text control p(event)={tuned_p_event} "
              f"(from Experiment S development seeds, not re-tuned per variant)")
    else:
        tuned_p_event = 1.0
        print("  No-text tuning file absent; falling back to p(event)=1.0")

    for model in llm_models:
        print(f"  Pre-caching LLM responses for {model}...")
        for tmpl in all_templates:
            _llm_probs(tmpl["text"], model=model)

    all_rows = []
    variant_results = {}

    for vname, vcfg in VARIANTS.items():
        vdir = ROBUSTNESS_DIR / vname
        vdir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Variant: {vcfg['label']}")
        print(f"  Dir: {vdir}")

        rows = []
        total_calls = len(seeds) * len(all_templates) * len(SENSORS)
        call_count = 0

        for seed in seeds:
            for tmpl in all_templates:
                text = tmpl["text"]
                tid = tmpl["template_id"]
                alevel = tmpl["ambiguity_level"]

                # Resolve the simulated world BEFORE any sensor consults it:
                # the oracle reference must describe the world this episode
                # actually runs in.
                world = world_regime_for(tmpl, pairing)
                tmpl_regime = (
                    tmpl["regime"] if isinstance(tmpl["regime"], Regime)
                    else Regime(tmpl["regime"])
                )

                for sensor in SENSORS:
                    call_count += 1
                    probs = _get_sensor_belief(
                        sensor, text, tfidf_raw, tfidf_cal,
                        llm_model=llm_models[0] if llm_models else "gpt-4o",
                        world_regime=world,
                        tuned_p_event=tuned_p_event,
                    )

                    result, _ = run_phase9_episode(
                        seed=seed,
                        regime=world,
                        sensor=sensor,
                        regime_probabilities=probs,
                        scenario=vcfg["scenario"],
                        config_path=vcfg["config_path"],
                        backlog=vcfg["backlog"],
                        noise_scale=vcfg["noise_scale"],
                        base_mu=DEFAULT_BASE_MU,
                        num_periods=DEFAULT_NUM_PERIODS,
                        event_start=DEFAULT_EVENT_START,
                        event_end=DEFAULT_EVENT_END,
                        affected_factory=DEFAULT_AFFECTED_FACTORY,
                        normal_capacity=DEFAULT_NORMAL_CAPACITY,
                        disrupted_capacity=DEFAULT_DISRUPTED_CAPACITY,
                    )

                    row = Phase9BRow(
                        variant=vname,
                        seed=seed,
                        regime=world.value,
                        template_regime=tmpl_regime.value,
                        pairing=pairing,
                        text_world_mismatch=(world is not tmpl_regime),
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

                    if call_count % 200 == 0 or call_count == total_calls:
                        print(f"  [{call_count}/{total_calls}] seed={seed} "
                              f"sensor={sensor} reward={result.total_reward:.1f}")

        print(f"  Computing metrics for {vname}...")
        agg, cis = _compute_variant_metrics(rows, vname, vdir)
        variant_results[vname] = {"agg": agg, "cis": cis}
        all_rows.extend(rows)

        print(f"\n  Aggregate SIVR ({vname}):")
        for s in SENSORS:
            if s in agg:
                print(f"    {s:25s} SIVR={agg[s]['aggregate_sivr']:.3f}  "
                      f"Rew={agg[s]['mean_reward']:.1f}  "
                      f"FR={agg[s]['mean_fill_rate']:.3f}")

    print(f"\n{'='*60}")
    print("Cross-Variant Comparison (RuleBased SIVR)")
    print(f"{'='*60}")
    for vname in VARIANTS:
        agg = variant_results[vname]["agg"]
        if "RuleBased" in agg:
            print(f"  {vname:15s}  SIVR={agg['RuleBased']['aggregate_sivr']:.3f}  "
                  f"FR={agg['RuleBased']['mean_fill_rate']:.3f}")

    comparison_rows = []
    for vname in VARIANTS:
        agg = variant_results[vname]["agg"]
        cis_list = variant_results[vname]["cis"]
        rb_cis = next((c for c in cis_list if c["sensor"] == "RuleBased"), None)
        comparison_rows.append({
            "variant": vname,
            "rulebased_sivr": agg.get("RuleBased", {}).get("aggregate_sivr", 0.0),
            "rulebased_fill_rate": agg.get("RuleBased", {}).get("mean_fill_rate", 0.0),
            "rulebased_ci_lower": rb_cis["ci_lower"] if rb_cis else 0.0,
            "rulebased_ci_upper": rb_cis["ci_upper"] if rb_cis else 0.0,
            "rulebased_significant": rb_cis["significant"] if rb_cis else False,
            "tfidf_raw_sivr": agg.get("TFIDF_LogReg_Raw", {}).get("aggregate_sivr", 0.0),
            "tfidf_cal_sivr": agg.get("TFIDF_LogReg_Calibrated", {}).get("aggregate_sivr", 0.0),
            "gpt4o_sivr": agg.get("gpt-4o", {}).get("aggregate_sivr", 0.0),
            "oracle_sivr": agg.get("OracleSemantic", {}).get("aggregate_sivr", 0.0),
        })
    _save_csv(comparison_rows, ROBUSTNESS_DIR / "cross_variant_comparison.csv")

    manifest = {
        "phase": "9B",
        "description": "One-factor-at-a-time robustness for SupplierCapacityDrop",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_templates": len(all_templates),
        "n_sensors": len(SENSORS),
        "n_variants": len(VARIANTS),
        "total_episodes": len(seeds) * len(all_templates) * len(SENSORS) * len(VARIANTS),
        "variants": {k: v["label"] for k, v in VARIANTS.items()},
        "regime": REGIME.value,
        "llm_model": llm_models[0] if llm_models else None,
    }
    (ROBUSTNESS_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"\nPhase 9B complete. Results in {ROBUSTNESS_DIR}")
    return variant_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9B Robustness")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=4100)
    parser.add_argument("--llm-models", nargs="*", default=["gpt-4o"])
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Run only these variants (e.g., baseline short_lead)")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    llm_models = [] if args.no_llm else args.llm_models
    if args.no_llm and "gpt-4o" in SENSORS:
        SENSORS.remove("gpt-4o")

    if args.variants:
        VARIANTS = {k: v for k, v in VARIANTS.items() if k in args.variants}

    run_phase9b(seeds=seeds, llm_models=llm_models)
