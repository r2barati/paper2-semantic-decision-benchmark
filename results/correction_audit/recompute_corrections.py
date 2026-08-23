"""Recompute the corrected benchmark analysis from frozen episode CSVs.

This script deliberately does not rerun a simulator.  It defines two explicit
estimands, uses the signed SIVR in ``src.metrics``, performs a
regime/family/variant/seed bootstrap, and writes corrected audit artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics import (
    SIVR_NEGATIVE_ORACLE_REFERENCE_VALUE,
    SIVR_VALID,
    SIVR_ZERO_OR_NEAR_ZERO_REFERENCE_VALUE,
    family_seed_bootstrap_ci,
    signed_sivr,
    standard_brier_score,
)

OUT = ROOT / "results" / "correction_audit"
EPSILON = 1e-9

BALANCED = "balanced_benchmark"
PRIOR = "deployment_prior"

PHASE_SPECS = {
    "Phase 5.5": {
        "path": ROOT / "results" / "phase5_5" / "operational_results.csv",
        "reward": "total_profit", "noinfo": "NoInfo", "oracle": "PerfectSemantic",
        "declared_prior": {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30},
        "prior_note": "REGIME_PRIOR in src/events.py; capacity-drop mass is zero and absent",
    },
    "Phase 6": {
        "path": ROOT / "results" / "phase6" / "operational_results.csv",
        "reward": "total_profit", "noinfo": "NoInfo", "oracle": "PerfectSemantic",
        "declared_prior": {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30},
        "prior_note": "REGIME_PRIOR in src/events.py; saved file has 20 seeds",
    },
    "Phase 7": {
        "path": ROOT / "results" / "phase7_classical_baseline" / "operational_results.csv",
        "reward": "total_profit", "noinfo": "NoInfo", "oracle": "PerfectSemantic",
        "declared_prior": {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30},
        "prior_note": "REGIME_PRIOR in src/events.py",
    },
    "Phase 8A": {
        "path": ROOT / "results" / "phase8_gym_replication" / "operational_results.csv",
        "reward": "total_reward", "noinfo": "NoInfo", "oracle": "PerfectSemantic",
        "declared_prior": {"normal": 0.70, "demand_surge": 0.30},
        "prior_note": "get_no_info_probs in src/gym_adapter.py; exploratory phase",
    },
    "Phase 8B": {
        "path": ROOT / "results" / "phase8b_gym_confirmation" / "operational_results.csv",
        "reward": "total_reward", "noinfo": "NoInfo", "oracle": "OracleSemantic",
        "declared_prior": {"normal": 0.70, "demand_surge": 0.30},
        "prior_note": "get_no_info_probs in src/gym_adapter.py",
    },
    "Phase 9A": {
        "path": ROOT / "results" / "phase9a_capacity_confirmation" / "operational_results.csv",
        "reward": "total_reward", "noinfo": "NoInfo", "oracle": "OracleSemantic",
        "declared_prior": {"normal": 0.50, "supplier_capacity_drop": 0.50},
        "prior_note": "code-executed get_no_info_probs_9; no separate deployment prior declared",
    },
}

PHASE9B_VARIANTS = [
    "baseline", "short_lead", "long_lead", "lost_sales", "low_noise", "high_noise",
]
PHASE9A_TEST_IDS = {
    "cn_moderate_2", "cn_vague_2", "cd_clear_3", "cd_moderate_2",
    "cd_vague_1", "cd_vague_2", "cd_vague_3",
}


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def float_value(row, key):
    return float(row[key])


def bool_value(value):
    return str(value).lower() in {"1", "true", "yes"}


def family_variant(template_id):
    parts = str(template_id).split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2]), "_".join(parts[2:])
    return str(template_id), "1"


def regimes_in(rows):
    return list(dict.fromkeys(row["regime"] for row in rows))


def weights_for(spec, rows, estimand):
    regimes = regimes_in(rows)
    if estimand == BALANCED:
        return {regime: 1.0 / len(regimes) for regime in regimes}
    declared = {r: float(spec["declared_prior"].get(r, 0.0)) for r in regimes}
    total = sum(declared.values())
    if total <= 0:
        raise ValueError(f"No declared-prior mass for observed regimes: {regimes}")
    return {r: declared[r] / total for r in regimes}


def build_reward_index(rows, reward_key):
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for row in rows:
        family, variant = family_variant(row["template_id"])
        index[row["regime"]][family][variant][row["sensor"]][row["seed"]] = float_value(row, reward_key)
    return index


def build_value_index(rows, value_fn):
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for row in rows:
        family, variant = family_variant(row["template_id"])
        index[row["regime"]][family][variant][row["sensor"]][row["seed"]] = float(value_fn(row))
    return index


def mean_from_index(index, sensor, regime_weights):
    regime_means = {}
    for regime in sorted(index):
        family_means = []
        for family in sorted(index[regime]):
            variant_means = []
            for variant in sorted(index[regime][family]):
                values = index[regime][family][variant].get(sensor, {})
                if values:
                    variant_means.append(float(np.mean(list(values.values()))))
            if variant_means:
                family_means.append(float(np.mean(variant_means)))
        if family_means:
            regime_means[regime] = float(np.mean(family_means))
    return float(sum(regime_weights[r] * regime_means[r] for r in regime_means)), regime_means


def paired_diffs(index, sensor, noinfo):
    diffs = defaultdict(lambda: defaultdict(dict))
    for regime in index:
        for family in index[regime]:
            for variant in index[regime][family]:
                sensor_values = index[regime][family][variant].get(sensor, {})
                noinfo_values = index[regime][family][variant].get(noinfo, {})
                seeds = sorted(set(sensor_values) & set(noinfo_values))
                if seeds:
                    diffs[regime][family][variant] = np.asarray(
                        [sensor_values[s] - noinfo_values[s] for s in seeds], dtype=float
                    )
    return diffs


def family_bootstrap_for_sensor(index, sensor, noinfo, regime_weights, seed):
    return family_seed_bootstrap_ci(
        paired_diffs(index, sensor, noinfo),
        regime_weights=regime_weights,
        # 1,000 draws are sufficient for the frozen audit tables and keep the
        # complete phase-by-phase correction reproducible on a clean laptop.
        # The implementation remains configurable for higher-resolution work.
        n_boot=1000,
        alpha=0.05,
        seed=seed,
    )


def bootstrap_p_value(samples):
    samples = np.asarray(samples, dtype=float)
    return float(min(1.0, 2.0 * min(np.mean(samples <= 0), np.mean(samples >= 0))))


def holm_adjust(records):
    if not records:
        return
    ordered = sorted(records, key=lambda record: record["p_value"])
    previous = 0.0
    count = len(ordered)
    for index, record in enumerate(ordered):
        adjusted = min(1.0, max(previous, (count - index) * record["p_value"]))
        record["holm_p_value"] = adjusted
        record["significant_holm_0_05"] = adjusted < 0.05
        previous = adjusted


def calculate_sivr_rows(phase, variant, rows, spec, estimand):
    reward_key = spec["reward"]
    index = build_reward_index(rows, reward_key)
    weights = weights_for(spec, rows, estimand)
    sensors = sorted({row["sensor"] for row in rows})
    noinfo = spec["noinfo"]
    oracle = spec["oracle"]
    means = {}
    regime_means = {}
    for sensor in sensors:
        means[sensor], regime_means[sensor] = mean_from_index(index, sensor, weights)

    output = []
    for regime in sorted(regimes_in(rows)) + ["ALL"]:
        for sensor in sensors:
            if regime == "ALL":
                sensor_mean = means[sensor]
                noinfo_mean = means[noinfo]
                oracle_mean = means[oracle]
            else:
                sensor_mean = regime_means[sensor].get(regime, float("nan"))
                noinfo_mean = regime_means[noinfo].get(regime, float("nan"))
                oracle_mean = regime_means[oracle].get(regime, float("nan"))
            result = signed_sivr(sensor_mean, noinfo_mean, oracle_mean, EPSILON)
            legacy_denominator = abs(result.oiv)
            legacy_sivr = (
                (sensor_mean - noinfo_mean) / legacy_denominator
                if legacy_denominator > EPSILON else float("nan")
            )
            output.append({
                "phase": phase,
                "variant": variant,
                "regime": regime,
                "sensor": sensor,
                "noinfo_reward": noinfo_mean,
                "sensor_reward": sensor_mean,
                "oracle_reward": oracle_mean,
                "signed_oiv": result.oiv,
                "signed_sivr": result.value,
                "legacy_abs_denominator_sivr": legacy_sivr,
                "sivr_change_vs_abs_denominator": result.value - legacy_sivr
                if not (math.isnan(result.value) or math.isnan(legacy_sivr)) else float("nan"),
                "status": result.status,
                "aggregation_estimand": estimand,
            })
    return output, means, regime_means, index, weights


def calculate_effect_rows(phase, variant, rows, spec, estimand, index, weights):
    sensors = sorted({row["sensor"] for row in rows})
    noinfo = spec["noinfo"]
    records = []
    for sensor in sensors:
        if sensor in {noinfo, spec["oracle"]}:
            continue
        boot = family_bootstrap_for_sensor(index, sensor, noinfo, weights, seed=42)
        records.append({
            "phase": phase,
            "variant": variant,
            "sensor": sensor,
            "aggregation_estimand": estimand,
            "mean_delta_vs_noinfo": boot["mean"],
            "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"],
            "p_value": bootstrap_p_value(boot["bootstrap_samples"]),
            "bootstrap_hierarchy": boot["bootstrap_hierarchy"],
            "bootstrap_seed": boot["bootstrap_seed"],
        })
    return records


def quality_rows(phase, variant, rows, spec, estimand):
    # Phase 9B is a conditional operational boundary analysis; its saved
    # operational rows do not contain a separately evaluated belief vector.
    if phase == "Phase 9B":
        return []
    class_order_by_phase = {
        "Phase 5.5": ["normal", "supplier_delay", "demand_surge"],
        "Phase 6": ["normal", "supplier_delay", "demand_surge"],
        "Phase 7": ["normal", "supplier_delay", "demand_surge"],
        "Phase 8A": ["normal", "demand_surge"],
        "Phase 8B": ["normal", "demand_surge"],
        "Phase 9A": ["normal", "supplier_capacity_drop"],
    }
    class_order = class_order_by_phase[phase]

    def row_values(row):
        probs = {}
        for label in class_order:
            if label == "normal":
                probs[label] = float(row.get("belief_normal", 0.0))
            elif label == "supplier_delay":
                probs[label] = float(row.get("belief_delay", 0.0))
            elif label == "demand_surge":
                probs[label] = float(row.get("belief_surge", 0.0))
            elif label == "supplier_capacity_drop":
                probs[label] = float(row.get("belief_capacity_drop", 0.0))
        true = row["regime"]
        true_prob = max(probs.get(true, 0.0), 1e-15)
        correct = bool_value(row.get("belief_correct", False))
        return {
            "accuracy": float(correct),
            "brier": standard_brier_score(probs, true, class_order=class_order),
            "log_loss": -math.log(true_prob),
        }

    values = build_value_index(rows, lambda row: row_values(row)["brier"])
    accuracy_index = build_value_index(rows, lambda row: row_values(row)["accuracy"])
    logloss_index = build_value_index(rows, lambda row: row_values(row)["log_loss"])
    weights = weights_for(spec, rows, estimand)
    output = []
    for sensor in sorted({row["sensor"] for row in rows}):
        brier, _ = mean_from_index(values, sensor, weights)
        accuracy, _ = mean_from_index(accuracy_index, sensor, weights)
        logloss, _ = mean_from_index(logloss_index, sensor, weights)
        output.append({
            "phase": phase, "variant": variant, "sensor": sensor,
            "aggregation_estimand": estimand,
            "accuracy": accuracy, "brier_standard": brier, "log_loss": logloss,
            "brier_definition": "sum_k_(p_k-y_k)^2",
        })
    return output


def audit_row(phase, variant, rows, spec):
    regimes = regimes_in(rows)
    regime_counts = Counter(row["regime"] for row in rows)
    template_counts = Counter(row["template_id"] for row in rows)
    regime_template_counts = Counter((row["regime"], row["template_id"]) for row in rows)
    return {
        "phase": phase,
        "variant": variant,
        "declared_prior": json.dumps(spec["declared_prior"], sort_keys=True),
        "prior_note": spec["prior_note"],
        "observed_regime_mix": json.dumps({r: regime_counts[r] for r in regimes}, sort_keys=True),
        "observed_regime_proportions": json.dumps({r: regime_counts[r] / len(rows) for r in regimes}, sort_keys=True),
        "template_count": len(template_counts),
        "template_mix": json.dumps(dict(sorted(template_counts.items())), sort_keys=True),
        "regime_template_counts": json.dumps({f"{r}|{t}": n for (r, t), n in sorted(regime_template_counts.items())}, sort_keys=True),
        "episode_count": len(rows),
        "primary_aggregation": "equal regime -> equal template family -> equal variant -> equal paired seed",
        "secondary_aggregation": "declared/code-executed prior over regimes; equal family/variant/seed within regime",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_sivr = []
    all_effects = []
    all_quality = []
    estimand_rows = []
    boundary_rows = []

    phase_rows = []
    for phase, spec in PHASE_SPECS.items():
        if not spec["path"].exists():
            continue
        rows = read_csv(spec["path"])
        phase_rows.append((phase, "baseline", rows, spec))
        estimand_rows.append(audit_row(phase, "baseline", rows, spec))

    for variant in PHASE9B_VARIANTS:
        path = ROOT / "results" / "phase9b_robustness" / variant / "operational_results.csv"
        if path.exists():
            spec = {
                "reward": "total_reward", "noinfo": "NoInfo", "oracle": "OracleSemantic",
                "declared_prior": {"supplier_capacity_drop": 1.0},
                "prior_note": "conditional single-regime robustness analysis",
            }
            rows = read_csv(path)
            phase_rows.append(("Phase 9B", variant, rows, spec))
            estimand_rows.append(audit_row("Phase 9B", variant, rows, spec))

    for phase, variant, rows, spec in phase_rows:
        estimands = [BALANCED] if phase == "Phase 9B" else [BALANCED, PRIOR]
        for estimand in estimands:
            sivr, means, regime_means, index, weights = calculate_sivr_rows(
                phase, variant, rows, spec, estimand,
            )
            all_sivr.extend(sivr)
            all_effects.extend(calculate_effect_rows(
                phase, variant, rows, spec, estimand, index, weights,
            ))
            all_quality.extend(quality_rows(phase, variant, rows, spec, estimand))

            if phase == "Phase 9B":
                noinfo = means[spec["noinfo"]]
                oracle = means[spec["oracle"]]
                oracle_delta = oracle - noinfo
                oracle_status = signed_sivr(oracle, noinfo, oracle).status
                for sensor, reward in sorted(means.items()):
                    delta = reward - noinfo
                    if oracle_delta > EPSILON and delta > EPSILON:
                        interpretation = "semantic_value_supported_under_fixed_controller"
                    elif oracle_delta > EPSILON:
                        interpretation = "sensor_or_controller_underperforms_noinfo"
                    elif oracle_delta < -EPSILON:
                        interpretation = "controller_or_environment_mismatch; oracle_reference_not_positive"
                    else:
                        interpretation = "absent_or_near_zero_oracle_reference_value"
                    boundary_rows.append({
                        "phase": phase, "variant": variant, "sensor": sensor,
                        "mean_reward": reward, "noinfo_reward": noinfo,
                        "delta_vs_noinfo": delta, "oracle_reward": oracle,
                        "oracle_delta_vs_noinfo": oracle_delta,
                        "oracle_reference_status": oracle_status,
                        "interpretation": interpretation,
                    })

    # Holm correction within publication comparison families.
    for group_key in [("Phase 8B", "baseline", BALANCED), ("Phase 8B", "baseline", PRIOR),
                      ("Phase 9A", "baseline", BALANCED), ("Phase 9A", "baseline", PRIOR)]:
        group = [r for r in all_effects if (r["phase"], r["variant"], r["aggregation_estimand"]) == group_key]
        holm_adjust(group)
    phase9b_group = [r for r in all_effects if r["phase"] == "Phase 9B"]
    holm_adjust(phase9b_group)
    for record in all_effects:
        record.setdefault("holm_p_value", "")
        record.setdefault("significant_holm_0_05", "")
        record["inference_note"] = (
            "Holm-adjusted simultaneous sensor comparison" if record["phase"] in {"Phase 8B", "Phase 9A"}
            else "descriptive robustness; Holm correction shown for completeness"
        )

    write_csv(OUT / "estimand_audit.csv", estimand_rows)
    write_csv(OUT / "sivr_recomputed.csv", all_sivr)
    write_csv(OUT / "raw_reward_effects.csv", all_effects)
    write_csv(OUT / "belief_quality_recomputed.csv", all_quality)
    write_csv(OUT / "phase9b_boundary_analysis.csv", boundary_rows)

    # Phase 9A test-template-only evidence, using already-saved episodes.
    p9a_spec = PHASE_SPECS["Phase 9A"]
    p9a_rows = read_csv(p9a_spec["path"])
    p9a_test_rows = [row for row in p9a_rows if row["template_id"] in PHASE9A_TEST_IDS]
    write_csv(OUT / "phase9a_test_templates_only_operational_results.csv", p9a_test_rows)
    test_sivr, _, _, test_index, test_weights = calculate_sivr_rows(
        "Phase 9A test templates only", "held_out_only", p9a_test_rows,
        p9a_spec, BALANCED,
    )
    write_csv(OUT / "phase9a_test_templates_only.csv", test_sivr)

    # A concise machine-readable manifest for the corrected analysis.
    manifest = {
        "analysis": "post_correction_recomputed_from_frozen_raw_episode_csvs",
        "new_simulations": 0,
        "primary_estimand": BALANCED,
        "secondary_estimand": PRIOR,
        "sivr": "src.metrics.signed_sivr; signed denominator; NaN for near-zero OIV",
        "bootstrap": "src.metrics.family_seed_bootstrap_ci; regime -> family -> variant -> paired seed",
        "brier": "standard multiclass sum_k_(p_k-y_k)^2",
        "phase9a_test_template_ids": sorted(PHASE9A_TEST_IDS),
    }
    (OUT / "correction_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
