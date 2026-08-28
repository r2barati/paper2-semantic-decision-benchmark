"""Generate corrected publication tables from correction-audit CSVs.

No experimental values are embedded here: every numeric cell is read from the
authoritative frozen-data recomputation outputs.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results" / "correction_audit"
OUT = ROOT / "results" / "publication"


def read(name):
    with (AUDIT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write(name, text):
    (OUT / name).write_text(text.rstrip() + "\n")


def phase8b_tables():
    sivr = [r for r in read("sivr_recomputed.csv")
            if r["phase"] == "Phase 8B" and r["regime"] == "ALL"]
    effects = [r for r in read("raw_reward_effects.csv")
               if r["phase"] == "Phase 8B"]
    by = {(r["sensor"], r["aggregation_estimand"]): r for r in sivr}
    eb = {(r["sensor"], r["aggregation_estimand"]): r for r in effects}
    sensors = sorted({r["sensor"] for r in sivr})
    lines = [
        "# Phase 8B corrected main table",
        "",
        "Primary estimand: balanced benchmark; secondary: declared 70/30 deployment prior.",
        "",
        "| Sensor | Balanced reward | Balanced Δ vs NoInfo | Balanced SIVR | Prior reward | Prior Δ vs NoInfo | Prior SIVR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sensor in sensors:
        b, p = by[(sensor, "balanced_benchmark")], by[(sensor, "deployment_prior")]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            sensor, f(b["sensor_reward"], 1), f(float(b["sensor_reward"]) - float(b["noinfo_reward"]), 1),
            f(b["signed_sivr"]), f(p["sensor_reward"], 1),
            f(float(p["sensor_reward"]) - float(p["noinfo_reward"]), 1), f(p["signed_sivr"])))
    write("corrected_phase8b_main.md", "\n".join(lines))

    lines = [
        "# Phase 8B regime-specific reward effects",
        "",
        "Rewards are means over template families, variants, and paired seeds within regime.",
        "",
        "| Sensor | Regime | NoInfo | Sensor | Oracle-Belief Reference | Δ vs NoInfo | Signed OIV | SIVR status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(sivr, key=lambda x: (x["aggregation_estimand"], x["regime"], x["sensor"])):
        if r["aggregation_estimand"] != "balanced_benchmark":
            continue
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["sensor"], r["regime"], f(r["noinfo_reward"], 1), f(r["sensor_reward"], 1),
            f(r["oracle_reward"], 1), f(float(r["sensor_reward"]) - float(r["noinfo_reward"]), 1),
            f(r["signed_oiv"], 1), r["status"]))
    write("corrected_phase8b_regime_rewards.md", "\n".join(lines))


def phase9_tables():
    sivr = [r for r in read("sivr_recomputed.csv")
            if r["phase"] == "Phase 9A" and r["regime"] == "ALL"
            and r["aggregation_estimand"] == "balanced_benchmark"]
    lines = ["# Phase 9A corrected cross-event operational-transfer table", "",
             "The all-template table is not a clean linguistic-generalization estimate.", "",
             "| Sensor | Reward | Δ vs NoInfo | Signed SIVR | Status |", "|---|---:|---:|---:|---|"]
    for r in sorted(sivr, key=lambda x: x["sensor"]):
        lines.append("| {} | {} | {} | {} | {} |".format(
            r["sensor"], f(r["sensor_reward"], 1),
            f(float(r["sensor_reward"]) - float(r["noinfo_reward"]), 1), f(r["signed_sivr"]), r["status"]))
    write("corrected_phase9a_main.md", "\n".join(lines))

    test = read("phase9a_test_templates_only.csv")
    lines = ["# Phase 9A test-template-only analysis", "",
             "This is the linguistic-generalization subset computed from already-saved episodes.", "",
             "| Sensor | Regime | Reward | Δ vs NoInfo | Signed SIVR | Status |", "|---|---|---:|---:|---:|---|"]
    for r in sorted(test, key=lambda x: (x["regime"], x["sensor"])):
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            r["sensor"], r["regime"], f(r["sensor_reward"], 1),
            f(float(r["sensor_reward"]) - float(r["noinfo_reward"]), 1), f(r["signed_sivr"]), r["status"]))
    write("corrected_phase9a_test_templates_only.md", "\n".join(lines))


def quality_and_boundary():
    quality = [r for r in read("belief_quality_recomputed.csv")
               if r["aggregation_estimand"] == "balanced_benchmark"]
    lines = ["# Corrected belief-quality table", "",
             "Brier is conventional multiclass sum_k (p_k-y_k)^2.", "",
             "| Phase | Sensor | Accuracy | Brier | Log-loss |", "|---|---|---:|---:|---:|"]
    for r in sorted(quality, key=lambda x: (x["phase"], x["sensor"])):
        lines.append("| {} | {} | {} | {} | {} |".format(
            r["phase"], r["sensor"], f(r["accuracy"]), f(r["brier_standard"]), f(r["log_loss"])))
    write("corrected_belief_quality.md", "\n".join(lines))

    boundary = read("phase9b_boundary_analysis.csv")
    lines = ["# Phase 9B boundary analysis", "",
             "Descriptive robustness/boundary analysis of the fixed controller.", "",
             "| Variant | Sensor | Reward | NoInfo | Δ vs NoInfo | Oracle | Oracle Δ | OIV status | Interpretation |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for r in sorted(boundary, key=lambda x: (x["variant"], x["sensor"])):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["variant"], r["sensor"], f(r["mean_reward"], 1), f(r["noinfo_reward"], 1),
            f(r["delta_vs_noinfo"], 1), f(r["oracle_reward"], 1), f(r["oracle_delta_vs_noinfo"], 1),
            r["oracle_reference_status"], r["interpretation"]))
    write("corrected_phase9b_boundary.md", "\n".join(lines))


def main():
    phase8b_tables()
    phase9_tables()
    quality_and_boundary()


if __name__ == "__main__":
    main()
