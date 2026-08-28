"""Generate the scientific ledger from authoritative corrected CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results" / "correction_audit"
OUT = ROOT / "results" / "publication" / "SCIENTIFIC_LEDGER_CORRECTED.md"
CANONICAL_OUT = ROOT / "SCIENTIFIC_LEDGER.md"


def read(name):
    with (AUDIT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    quality = [r for r in read("belief_quality_recomputed.csv")
               if r["aggregation_estimand"] == "balanced_benchmark"]
    effects = [r for r in read("raw_reward_effects.csv")
               if r["aggregation_estimand"] == "balanced_benchmark"]
    lines = [
        "# Scientific Ledger (corrected, generated)", "",
        "Every value below is read from `results/correction_audit`; no experimental value is hardcoded.", "",
        "## Phase 7 belief quality", "",
        "| Sensor | Accuracy | Standard Brier | Log-loss |", "|---|---:|---:|---:|"]
    for r in quality:
        if r["phase"] == "Phase 7":
            lines.append(f"| {r['sensor']} | {float(r['accuracy']):.6f} | {float(r['brier_standard']):.6f} | {float(r['log_loss']):.6f} |")
    lines += ["", "## Primary paired reward effects", "",
              "| Phase | Sensor | Δ vs NoInfo | 95% CI | Holm p |", "|---|---|---:|---|---:|"]
    for r in effects:
        if r["phase"] in {"Phase 8B", "Phase 9A"}:
            lines.append(f"| {r['phase']} | {r['sensor']} | {float(r['mean_delta_vs_noinfo']):.6f} | [{float(r['ci_lower']):.6f}, {float(r['ci_upper']):.6f}] | {r.get('holm_p_value', '')} |")
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    CANONICAL_OUT.write_text(text)


if __name__ == "__main__":
    main()
