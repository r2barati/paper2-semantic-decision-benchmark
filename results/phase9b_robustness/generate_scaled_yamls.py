#!/usr/bin/env python3
"""Generate lead-time-scaled YAML topologies for Phase 9B robustness."""
import shutil
from pathlib import Path

SRC = Path("/Users/sanamimani/Paper 1/final-github-clean/gym-invmgmt-paper/gym_invmgmt/topologies/divergent.yaml")
OUT_DIR = Path(__file__).parent


def scale_yaml(multiplier: float, out_name: str):
    """Copy divergent.yaml and scale all L values by *multiplier* (int)."""
    lines = SRC.read_text().splitlines()
    out_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("L:"):
            indent = line[: len(line) - len(stripped)]
            orig_val = int(stripped.split(":")[1].strip())
            new_val = max(0, int(orig_val * multiplier))
            out_lines.append(f"{indent}L: {new_val}")
        else:
            out_lines.append(line)
    out_path = OUT_DIR / out_name
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"  Wrote {out_path}  (L×{multiplier})")


if __name__ == "__main__":
    scale_yaml(0.5, "divergent_short_lead.yaml")
    scale_yaml(2.0, "divergent_long_lead.yaml")
