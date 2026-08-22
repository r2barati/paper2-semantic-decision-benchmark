"""Phase 7: Classical NLP baseline evaluation.

Trains a TF-IDF + Logistic Regression classifier on original Phase-5
templates, evaluates on held-out confirmation templates, and runs the
full downstream operational pipeline to compute SIVR and related metrics.

DO NOT modify Phase-6 outputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.events import (
    Regime, REGIME_PRIOR, REGIME_PARAMS, REGIME_WARNING_TEMPLATES,
    P5_HORIZON, P5_NORMAL_LEAD_TIME, P5_DEMAND_MEAN,
    P5_EVENT_START, P5_EVENT_DURATION, P5_WARNING_TIME,
)
from src.env import (
    InventoryEnv, InventoryState, CausalOptimizer,
    REVENUE_PER_UNIT, ORDERING_FIXED_COST, ORDERING_VARIABLE_COST,
    HOLDING_COST_PER_UNIT, STOCKOUT_COST_PER_UNIT, INITIAL_INVENTORY,
)
from src.interpreter import (
    RegimeInterpretation, no_info_regime_belief, perfect_semantic_regime_belief,
    rule_based_regime_extract,
)
from src.experiment_phase5 import _run_p5_episode
from src.experiment_phase6 import (
    brier_score, log_loss_safe, aggregate_sivr, macro_sivr,
    paired_bootstrap_ci, hierarchical_bootstrap_ci, calibration_analysis,
)
from src.confirmation_templates import CONFIRMATION_TEMPLATES
from src.classical_baseline import (
    TFIDFLogReg, build_dataset, grouped_train_test_split,
    evaluate_classification, leakage_audit, compute_dataset_manifest,
    cross_validate_on_original, REGIME_LABELS,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase7_classical_baseline"

NOINFO = "NoInfo"
RULEBASED = "RuleBased"
PERFECT = "PerfectSemantic"
TFIDF = "TFIDF_LogReg"
TFIDF_CAL = "TFIDF_LogReg_Calibrated"
CONTROLLER = "CausalOptimizer"


def _save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def run_phase7(
    num_seeds: int = 20,
    seed_base: int = 2000,
    C: float = 1.0,
    calibrate: bool = True,
    cal_method: str = "isotonic",
    max_features: int = 500,
    ngram_range: tuple = (1, 2),
) -> dict:
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    original_templates = REGIME_WARNING_TEMPLATES
    confirmation_templates = CONFIRMATION_TEMPLATES
    regimes = [Regime.NORMAL, Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]
    seeds = list(range(seed_base, seed_base + num_seeds))

    orig_texts, orig_labels, orig_ids, orig_families = build_dataset(original_templates)
    conf_texts, conf_labels, conf_ids, conf_families = build_dataset(confirmation_templates)

    all_texts = orig_texts + conf_texts
    all_labels = orig_labels + conf_labels
    all_ids = orig_ids + conf_ids
    all_families = orig_families + conf_families

    test_ids = set(conf_ids)

    (train_texts, train_labels, train_ids, train_families,
     eval_texts, eval_labels, eval_ids, eval_families) = \
        grouped_train_test_split(all_texts, all_labels, all_ids, all_families, test_ids)

    print("=" * 70)
    print("PHASE 7: CLASSICAL NLP BASELINE")
    print("=" * 70)
    print(f"Original templates: {len(original_templates)}")
    print(f"Confirmation templates: {len(confirmation_templates)}")
    print(f"Train templates: {len(train_texts)}")
    print(f"Test templates: {len(eval_texts)}")
    print()

    print("Cross-validation on original templates...")
    cv_results = cross_validate_on_original(original_templates, n_splits=3, seed=42)
    print(f"  CV accuracy: {cv_results['avg_accuracy']:.3f}")
    print(f"  CV Brier: {cv_results['avg_brier_macro']:.3f}")
    print(f"  CV log-loss: {cv_results['avg_log_loss']:.3f}")
    print()

    print("Training final model on original templates...")
    model = TFIDFLogReg(
        max_features=max_features, ngram_range=ngram_range,
        C=C, calibrate=False, seed=42,
    )
    model.fit(orig_texts, orig_labels)
    print(f"  Vocabulary size: {model.vocabulary_size_}")
    print()

    model_cal = None
    if calibrate:
        print("Training calibrated model...")
        model_cal = TFIDFLogReg(
            max_features=max_features, ngram_range=ngram_range,
            C=C, calibrate=True, cal_method=cal_method, seed=42,
        )
        model_cal.fit(orig_texts, orig_labels)
        print()

    model.save(output_dir / "tfidf_logreg_model.pkl")
    if model_cal is not None:
        model_cal.save(output_dir / "tfidf_logreg_calibrated_model.pkl")

    print("Evaluating on held-out confirmation templates...")
    eval_proba = model.predict_proba(eval_texts)
    eval_pred = model.predict(eval_texts)
    classes = model.classes_

    eval_metrics = evaluate_classification(eval_labels, eval_pred, eval_proba, classes)
    print(f"  Accuracy: {eval_metrics['accuracy']:.3f}")
    print(f"  Brier (macro): {eval_metrics['brier_macro']:.3f}")
    print(f"  Log-loss: {eval_metrics['log_loss']:.3f}")

    cal_metrics = None
    if model_cal is not None:
        eval_proba_cal = model_cal.predict_proba(eval_texts)
        eval_pred_cal = model_cal.predict(eval_texts)
        cal_metrics = evaluate_classification(eval_labels, eval_pred_cal, eval_proba_cal, classes)
        print(f"  Calibrated accuracy: {cal_metrics['accuracy']:.3f}")
        print(f"  Calibrated Brier (macro): {cal_metrics['brier_macro']:.3f}")
        print(f"  Calibrated log-loss: {cal_metrics['log_loss']:.3f}")
    print()

    audit = leakage_audit(train_ids, eval_ids, train_families, eval_families)
    manifest = compute_dataset_manifest(
        original_templates, confirmation_templates,
        train_texts, train_labels, train_ids, train_families,
        eval_texts, eval_labels, eval_ids, eval_families,
    )

    print("Leakage audit:")
    print(f"  No template ID overlap: {audit['no_template_leak']}")
    print(f"  Train families: {audit['train_families']}")
    print(f"  Test families: {audit['test_families']}")
    print(f"  Shared families: {audit['families_shared']}")
    print()

    sensors = [NOINFO, RULEBASED, TFIDF, PERFECT]
    if model_cal is not None:
        sensors.insert(3, TFIDF_CAL)

    all_episodes = []
    all_beliefs = []

    print("Running operational episodes...")
    for regime in regimes:
        r_templates = [t for t in confirmation_templates if t["regime"] == regime]
        for seed in seeds:
            for tmpl in r_templates:
                text = tmpl["text"]
                tid = tmpl["template_id"]
                alevel = tmpl["ambiguity_level"]

                for sensor in sensors:
                    if sensor == NOINFO:
                        ri = no_info_regime_belief()
                    elif sensor == PERFECT:
                        ri = perfect_semantic_regime_belief(regime)
                    elif sensor == RULEBASED:
                        ri = rule_based_regime_extract(text)
                    elif sensor == TFIDF:
                        result_baseline = model.predict_regime(text)
                        ri = result_baseline.to_regime_interpretation()
                    elif sensor == TFIDF_CAL and model_cal is not None:
                        result_baseline = model_cal.predict_regime(text)
                        ri = result_baseline.to_regime_interpretation()
                    else:
                        ri = no_info_regime_belief()

                    belief = ri.normalized()

                    episode_result, history = _run_p5_episode(
                        seed=seed, regime=regime, sensor=sensor,
                        controller=CONTROLLER, template_id=tid,
                        ambiguity_level=alevel, regime_interp=ri,
                    )

                    all_episodes.append({
                        "seed": seed, "regime": regime.value,
                        "template_id": tid, "ambiguity_level": alevel,
                        "sensor": sensor, "controller": CONTROLLER,
                        "total_profit": episode_result.total_profit,
                        "fill_rate": episode_result.fill_rate,
                        "total_lost_sales": episode_result.total_lost_sales,
                        "avg_inventory": episode_result.avg_inventory,
                        "periods_stockout": episode_result.periods_stockout,
                        "belief_normal": belief.regime_probabilities.get("normal", 0.0),
                        "belief_delay": belief.regime_probabilities.get("supplier_delay", 0.0),
                        "belief_surge": belief.regime_probabilities.get("demand_surge", 0.0),
                        "most_likely_regime": belief.most_likely_regime,
                        "belief_correct": belief.most_likely_regime == regime.value,
                        "est_lt_increase": belief.estimated_lt_increase,
                        "est_duration": belief.estimated_duration,
                        "est_demand_multiplier": belief.estimated_demand_multiplier,
                    })

                    confidence = max(belief.regime_probabilities.values())
                    all_beliefs.append({
                        "sensor": sensor, "regime": regime.value,
                        "template_id": tid, "ambiguity_level": alevel,
                        "seed": seed,
                        "belief_normal": belief.regime_probabilities.get("normal", 0.0),
                        "belief_delay": belief.regime_probabilities.get("supplier_delay", 0.0),
                        "belief_surge": belief.regime_probabilities.get("demand_surge", 0.0),
                        "most_likely": belief.most_likely_regime,
                        "correct": belief.most_likely_regime == regime.value,
                        "confidence": confidence,
                        "true_regime": regime.value,
                        "brier": brier_score(belief.regime_probabilities, regime.value),
                        "log_loss": log_loss_safe(belief.regime_probabilities, regime.value),
                    })

    _save_csv(all_episodes, output_dir / "operational_results.csv")
    _save_csv(all_beliefs, output_dir / "belief_metrics.csv")

    # Belief summary
    belief_summary = []
    for sensor in sensors:
        sensor_beliefs = [b for b in all_beliefs if b["sensor"] == sensor]
        if not sensor_beliefs:
            continue
        acc = np.mean([1.0 if b["correct"] else 0.0 for b in sensor_beliefs])
        brier = np.mean([b["brier"] for b in sensor_beliefs])
        ll = np.mean([b["log_loss"] for b in sensor_beliefs])
        ece_data = calibration_analysis(sensor_beliefs)
        belief_summary.append({
            "sensor": sensor,
            "accuracy": float(acc),
            "brier_score": float(brier),
            "log_loss": float(ll),
            "ece": ece_data["ece"],
            "count": len(sensor_beliefs),
        })
    _save_csv(belief_summary, output_dir / "belief_metrics_summary.csv")

    # SIVR by regime
    sivr_rows = []
    ni_episodes = [ep for ep in all_episodes if ep["sensor"] == NOINFO]
    ps_episodes = [ep for ep in all_episodes if ep["sensor"] == PERFECT]

    for regime in regimes:
        ni_regime = [ep for ep in ni_episodes if ep["regime"] == regime.value]
        ps_regime = [ep for ep in ps_episodes if ep["regime"] == regime.value]
        ni_mean = np.mean([ep["total_profit"] for ep in ni_regime]) if ni_regime else 0
        ps_mean = np.mean([ep["total_profit"] for ep in ps_regime]) if ps_regime else 0
        oiv = ps_mean - ni_mean

        for sensor in sensors:
            if sensor in (NOINFO, PERFECT):
                continue
            s_regime = [ep for ep in all_episodes
                        if ep["sensor"] == sensor and ep["regime"] == regime.value]
            s_mean = np.mean([ep["total_profit"] for ep in s_regime]) if s_regime else 0
            rov = s_mean - ni_mean
            sr = ps_mean - s_mean
            denom = ps_mean - ni_mean
            sivr_val = rov / denom if abs(denom) > 1e-9 else 0.0

            sivr_rows.append({
                "sensor": sensor, "regime": regime.value,
                "mean_profit": float(s_mean),
                "ni_mean_profit": float(ni_mean),
                "ps_mean_profit": float(ps_mean),
                "oiv": float(oiv),
                "rov": float(rov),
                "semantic_regret": float(sr),
                "sivr": float(sivr_val),
            })
    _save_csv(sivr_rows, output_dir / "sivr_by_regime.csv")

    # Aggregate SIVR
    agg_sivr_rows = []
    for sensor in sensors:
        if sensor in (NOINFO, PERFECT):
            continue
        sensor_eps = np.array([ep["total_profit"] for ep in all_episodes if ep["sensor"] == sensor])
        ni_eps = np.array([ep["total_profit"] for ep in all_episodes if ep["sensor"] == NOINFO])
        ps_eps = np.array([ep["total_profit"] for ep in all_episodes if ep["sensor"] == PERFECT])
        agg = aggregate_sivr(sensor_eps, ni_eps, ps_eps)

        sensor_mean = float(np.mean(sensor_eps))
        ni_mean = float(np.mean(ni_eps))
        ps_mean = float(np.mean(ps_eps))

        agg_sivr_rows.append({
            "sensor": sensor,
            "mean_profit": sensor_mean,
            "ni_mean_profit": ni_mean,
            "ps_mean_profit": ps_mean,
            "oiv": ps_mean - ni_mean,
            "rov": sensor_mean - ni_mean,
            "semantic_regret": ps_mean - sensor_mean,
            "aggregate_sivr": agg,
        })
    _save_csv(agg_sivr_rows, output_dir / "aggregate_sivr.csv")

    # Ambiguity analysis
    ambiguity_rows = []
    for sensor in sensors:
        if sensor in (NOINFO, PERFECT):
            continue
        for amb in ["clear", "moderate", "vague"]:
            ni_amb = [ep for ep in all_episodes
                      if ep["sensor"] == NOINFO and ep["ambiguity_level"] == amb]
            ps_amb = [ep for ep in all_episodes
                      if ep["sensor"] == PERFECT and ep["ambiguity_level"] == amb]
            s_amb = [ep for ep in all_episodes
                     if ep["sensor"] == sensor and ep["ambiguity_level"] == amb]
            if not s_amb:
                continue
            ni_m = np.mean([ep["total_profit"] for ep in ni_amb]) if ni_amb else 0
            ps_m = np.mean([ep["total_profit"] for ep in ps_amb]) if ps_amb else 0
            s_m = np.mean([ep["total_profit"] for ep in s_amb])
            denom = ps_m - ni_m
            sivr_val = (s_m - ni_m) / denom if abs(denom) > 1e-9 else 0.0

            ambiguity_rows.append({
                "sensor": sensor, "ambiguity_level": amb,
                "mean_profit": float(s_m),
                "oiv": float(ps_m - ni_m),
                "rov": float(s_m - ni_m),
                "semantic_regret": float(ps_m - s_m),
                "sivr": float(sivr_val),
                "count": len(s_amb),
            })
    _save_csv(ambiguity_rows, output_dir / "ambiguity_analysis.csv")

    # Bootstrap comparisons
    bootstrap_rows = []
    for sensor in sensors:
        if sensor in (NOINFO, PERFECT):
            continue
        sensor_profits = np.array([ep["total_profit"] for ep in all_episodes if ep["sensor"] == sensor])
        ni_profits = np.array([ep["total_profit"] for ep in all_episodes if ep["sensor"] == NOINFO])
        diffs = sensor_profits - ni_profits
        mean_diff, ci_low, ci_high = paired_bootstrap_ci(diffs, np.zeros_like(diffs), n_boot=10000, seed=42)

        template_profits = defaultdict(list)
        for ep in all_episodes:
            if ep["sensor"] == sensor:
                template_profits[ep["template_id"]].append(ep["total_profit"])
        templ_dict = {k: np.array(v) for k, v in template_profits.items()}
        h_mean, h_low, h_high = hierarchical_bootstrap_ci(templ_dict, n_boot=5000, seed=42)

        bootstrap_rows.append({
            "sensor": sensor,
            "paired_mean_diff": float(mean_diff),
            "paired_ci_low": float(ci_low),
            "paired_ci_high": float(ci_high),
            "hierarchical_mean": float(h_mean),
            "hierarchical_ci_low": float(h_low),
            "hierarchical_ci_high": float(h_high),
        })
    _save_csv(bootstrap_rows, output_dir / "bootstrap_comparisons.csv")

    # Classification metrics CSV
    classification_rows = []
    for sensor in [TFIDF, TFIDF_CAL, RULEBASED]:
        sensor_beliefs = [b for b in all_beliefs if b["sensor"] == sensor]
        if not sensor_beliefs:
            continue
        y_true = [b["true_regime"] for b in sensor_beliefs]
        y_pred = [b["most_likely"] for b in sensor_beliefs]
        from sklearn.metrics import accuracy_score, confusion_matrix
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=REGIME_LABELS)
        classification_rows.append({
            "sensor": sensor,
            "accuracy": float(acc),
            "confusion_matrix": json.dumps(cm.tolist()),
            "labels": json.dumps(REGIME_LABELS),
            "n_samples": len(y_true),
        })
    _save_csv(classical_rows if 'classical_rows' in dir() else classification_rows,
              output_dir / "classification_metrics.csv")

    # Save manifests
    experiment_manifest = {
        "phase": "7",
        "frozen": True,
        "model": "TFIDF_LogReg",
        "calibrated_model": "TFIDF_LogReg_Calibrated" if model_cal else None,
        "train_source": "REGIME_WARNING_TEMPLATES (original 18)",
        "test_source": "CONFIRMATION_TEMPLATES (36 held-out)",
        "num_train": len(train_texts),
        "num_test": len(eval_texts),
        "num_seeds": len(seeds),
        "seed_base": seed_base,
        "sensors": sensors,
        "controller": CONTROLLER,
        "cv_results": cv_results,
        "eval_metrics": eval_metrics,
        "cal_metrics": cal_metrics,
        "config": {
            "C": C, "max_features": max_features,
            "ngram_range": list(ngram_range),
            "calibrate": calibrate, "cal_method": cal_method,
        },
    }
    with open(output_dir / "experiment_manifest.json", "w") as f:
        json.dump(_to_json_serializable(experiment_manifest), f, indent=2)

    split_manifest = {
        "phase": "7",
        "frozen": True,
        "leakage_audit": audit,
        "dataset_manifest": manifest,
    }
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(_to_json_serializable(split_manifest), f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("PHASE 7 RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("SIVR by Sensor and Regime:")
    for row in sivr_rows:
        print(f"  {row['sensor']:25s} {row['regime']:15s} SIVR={row['sivr']:.3f} "
              f"(OIV={row['oiv']:.1f} ROV={row['rov']:.1f} SR={row['semantic_regret']:.1f})")
    print()
    print("Aggregate SIVR:")
    for row in agg_sivr_rows:
        print(f"  {row['sensor']:25s} AggregateSIVR={row['aggregate_sivr']:.3f}")
    print()
    print("Belief Quality:")
    for row in belief_summary:
        print(f"  {row['sensor']:25s} acc={row['accuracy']:.3f} brier={row['brier_score']:.3f} "
              f"logloss={row['log_loss']:.3f} ECE={row['ece']:.3f}")
    print()
    print(f"Outputs saved to {output_dir}")

    return {
        "eval_metrics": eval_metrics,
        "cal_metrics": cal_metrics,
        "cv_results": cv_results,
        "sivr_rows": sivr_rows,
        "agg_sivr_rows": agg_sivr_rows,
        "ambiguity_rows": ambiguity_rows,
        "belief_summary": belief_summary,
        "bootstrap_rows": bootstrap_rows,
        "audit": audit,
        "manifest": manifest,
        "all_episodes": all_episodes,
        "all_beliefs": all_beliefs,
    }


if __name__ == "__main__":
    run_phase7()
