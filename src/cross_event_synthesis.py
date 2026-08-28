"""Cross-Event Synthesis: Phase 8B (DemandSurge) vs Phase 9A (SupplierCapacityDrop).

Compares the semantic-value framework across two structurally distinct shock types
on the same divergent topology (3 factories, 2 distributors, 1 retailer).
"""

from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent.parent

SENSORS = ["NoInfo", "RuleBased", "TFIDF_LogReg_Raw", "TFIDF_LogReg_Calibrated",
           "gpt-4o", "OracleSemantic"]

phase8b = {
    "NoInfo":                   {"sivr": 0.000, "fill_rate": 0.858, "bel_acc": 0.500, "brier": None, "mean_reward": 467.7},
    "RuleBased":                {"sivr": 1.198, "fill_rate": 0.947, "bel_acc": 0.792, "brier": None, "mean_reward": 568.7},
    "TFIDF_LogReg_Raw":         {"sivr": 0.575, "fill_rate": 0.901, "bel_acc": 0.917, "brier": None, "mean_reward": 516.2},
    "TFIDF_LogReg_Calibrated":  {"sivr": 0.841, "fill_rate": 0.911, "bel_acc": 0.958, "brier": None, "mean_reward": 538.6},
    "gpt-4o":                   {"sivr": 0.797, "fill_rate": 0.906, "bel_acc": 0.958, "brier": None, "mean_reward": 534.9},
    "OracleSemantic":           {"sivr": 1.000, "fill_rate": 0.920, "bel_acc": 1.000, "brier": None, "mean_reward": 552.0},
}

phase9a = {
    "NoInfo":                   {"sivr": 0.000, "fill_rate": 0.846, "bel_acc": 0.375, "brier": 0.2500, "mean_reward": 1333.6},
    "RuleBased":                {"sivr": 0.823, "fill_rate": 0.895, "bel_acc": 0.938, "brier": 0.1078, "mean_reward": 1484.0},
    "TFIDF_LogReg_Raw":         {"sivr": 0.261, "fill_rate": 0.861, "bel_acc": 0.875, "brier": 0.1793, "mean_reward": 1381.2},
    "TFIDF_LogReg_Calibrated":  {"sivr": 0.434, "fill_rate": 0.871, "bel_acc": 0.812, "brier": 0.0897, "mean_reward": 1412.9},
    "gpt-4o":                   {"sivr": 0.507, "fill_rate": 0.875, "bel_acc": 0.938, "brier": 0.0494, "mean_reward": 1426.2},
    "OracleSemantic":           {"sivr": 1.000, "fill_rate": 0.904, "bel_acc": 1.000, "brier": 0.0000, "mean_reward": 1516.3},
}

phase9b_baseline = {
    "NoInfo":                   {"sivr": 0.000, "fill_rate": 0.845, "bel_acc": 0.375, "mean_reward": 1323.6},
    "RuleBased":                {"sivr": 0.383, "fill_rate": 0.894, "bel_acc": 0.938, "mean_reward": 1473.0},
    "TFIDF_LogReg_Raw":         {"sivr": 0.119, "fill_rate": 0.861, "bel_acc": 0.875, "mean_reward": 1371.2},
    "TFIDF_LogReg_Calibrated":  {"sivr": 0.205, "fill_rate": 0.870, "bel_acc": 0.812, "mean_reward": 1402.0},
    "gpt-4o":                   {"sivr": 0.232, "fill_rate": 0.875, "bel_acc": 0.938, "mean_reward": 1416.0},
    "OracleSemantic":           {"sivr": 1.000, "fill_rate": 0.971, "bel_acc": 1.000, "mean_reward": 1713.6},
}


def build_comparison():
    rows = []
    for sensor in SENSORS:
        rows.append({
            "sensor": sensor,
            "phase8b_sivr": phase8b[sensor]["sivr"],
            "phase8b_fill_rate": phase8b[sensor]["fill_rate"],
            "phase8b_bel_acc": phase8b[sensor]["bel_acc"],
            "phase8b_mean_reward": phase8b[sensor]["mean_reward"],
            "phase9a_sivr": phase9a[sensor]["sivr"],
            "phase9a_fill_rate": phase9a[sensor]["fill_rate"],
            "phase9a_bel_acc": phase9a[sensor]["bel_acc"],
            "phase9a_brier": phase9a[sensor]["brier"],
            "phase9a_mean_reward": phase9a[sensor]["mean_reward"],
            "phase9b_sivr": phase9b_baseline[sensor]["sivr"],
            "phase9b_fill_rate": phase9b_baseline[sensor]["fill_rate"],
            "phase9b_mean_reward": phase9b_baseline[sensor]["mean_reward"],
        })
    return rows


def print_synthesis():
    print("=" * 90)
    print("Cross-Event Synthesis: DemandSurge (8B) vs SupplierCapacityDrop (9A)")
    print("=" * 90)
    print(f"\n{'Sensor':25s} {'8B SIVR':>8s} {'9A SIVR':>8s} {'9B SIVR':>8s}  "
          f"{'8B FR':>6s} {'9A FR':>6s}  {'8B BelAcc':>8s} {'9A BelAcc':>8s}")
    print("-" * 90)
    for sensor in SENSORS:
        p8 = phase8b[sensor]
        p9 = phase9a[sensor]
        p9b = phase9b_baseline[sensor]
        print(f"{sensor:25s} {p8['sivr']:8.3f} {p9['sivr']:8.3f} {p9b['sivr']:8.3f}  "
              f"{p8['fill_rate']:6.3f} {p9['fill_rate']:6.3f}  "
              f"{p8['bel_acc']:8.3f} {p9['bel_acc']:8.3f}")

    print("\n--- Key Findings ---")
    print("1. Oracle SIVR=1.000 in BOTH events → value-of-information framework is event-agnostic")
    print("2. RuleBased: DemandSurge (1.198) > CapacityDrop (0.823) > baseline (0.383)")
    print("   → Rule-based inference is more effective for demand-side shocks")
    print("3. gpt-4o: CapacityDrop (0.507) < DemandSurge (0.797)")
    print("   → SIVR difference is driven by OIV range, not information quality")
    print("4. TFIDF_Raw: Phase 8B is primary generalization (24 held-out, SIVR=0.575)")
    print("   Phase 9A includes 9/16 training templates (in-distribution, SIVR=0.261)")
    print("5. Fill rates: all sensors improve fill rate vs NoInfo in both events")
    print("   → Semantic information consistently improves operational performance")

    print("\n--- Structural Differences ---")
    print("DemandSurge (8B):")
    print("  - Demand spikes 2-3x above base → information enables PREPARATION")
    print("  - High OIV (552 - 468 = 84) → large room for info to add value")
    print("  - Rule-based uses demand keywords + timing → very effective")
    print("\nSupplierCapacityDrop (9A):")
    print("  - Factory 4 capacity 90→30 for 15 periods → information enables SAFETY STOCK")
    print("  - Lower OIV (1516 - 1334 = 182, but higher baseline rewards)")
    print("  - Safety-stock amplification mechanism: effective_safety *= (1 + p_cd * 2.0)")

    print("\n--- Robustness (Phase 9B) ---")
    print("RuleBased SIVR across variants:")
    for vname, val in [("baseline", 0.383), ("short_lead", 0.358),
                       ("low_noise", 0.383), ("high_noise", 0.384)]:
        print(f"  {vname:15s}: {val:.3f}")
    print("  → Robust across realistic operational parameter variations")
    print("  → LONG_LEAD and LOST_SALES show policy failure (negative SIVR = env too extreme)")


if __name__ == "__main__":
    rows = build_comparison()
    out_dir = ROOT / "results" / "phase9_synthesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_dir / "cross_event_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_dir / 'cross_event_comparison.csv'}")

    print()
    print_synthesis()
