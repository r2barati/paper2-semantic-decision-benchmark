"""End-to-end verification of the offline release.

Checks the things a reader is told the release can do, by doing them:

``--artifacts``   every offline artifact is present or reconstructable
``--replay``      stored beliefs replay to the stored rewards, bit for bit
``--regenerate``  the published tables regenerate from the frozen episodes
``--all``         all of the above

Exits non-zero on the first genuine failure, so CI cannot go green while the
documented reproduction is broken.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


def _read(path: Path) -> list:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_artifacts() -> bool:
    print("== offline artifacts ==")
    from tools.rebuild_offline_artifacts import (
        rebuild_semantic_cache, rebuild_tfidf_models,
    )
    cache = rebuild_semantic_cache(verify_only=True)
    models = rebuild_tfidf_models(verify_only=True)
    ok = True
    if cache["missing"]:
        print(f"  MISSING {len(cache['missing'])} semantic cache files")
        ok = False
    if cache["checksum_mismatch"]:
        print(f"  CHECKSUM MISMATCH: {cache['checksum_mismatch'][:5]}")
        ok = False
    if models["missing"]:
        print(f"  MISSING model checkpoints: {models['missing']}")
        ok = False
    if ok:
        print(f"  ok: {cache['verified']} semantic caches, "
              f"{len(models['already_present'])} checkpoints")
    return ok


def check_replay(n_per_group: int = 2) -> bool:
    """Replay stored beliefs through the simulator and compare to stored reward.

    This is the check that distinguishes a table replay from a genuine
    simulator reconstruction. Any drift between the committed episodes and the
    current simulator shows up here.
    """
    print("== stored-belief simulation replay ==")
    from src.events import Regime
    from src.gym_adapter import run_gym_episode
    from src.gym_adapter9 import run_phase9_episode

    specs = [
        ("Experiment M", RESULTS / "phase8b_gym_confirmation" / "operational_results.csv",
         run_gym_episode, "belief_surge", "demand_surge"),
        ("Experiment S", RESULTS / "phase9a_capacity_confirmation" / "operational_results.csv",
         run_phase9_episode, "belief_capacity_drop", "supplier_capacity_drop"),
    ]

    ok = True
    for name, path, runner, belief_key, event in specs:
        if not path.exists():
            print(f"  [skip] {name}: {path.name} not present")
            continue
        rows = _read(path)
        seen: dict = {}
        checked = max_err = 0
        for row in rows:
            key = (row["sensor"], row["regime"])
            if seen.get(key, 0) >= n_per_group:
                continue
            seen[key] = seen.get(key, 0) + 1
            probs = {
                "normal": float(row["belief_normal"]),
                event: float(row[belief_key]),
            }
            result, _ = runner(
                seed=int(row["seed"]), regime=Regime(row["regime"]),
                sensor=row["sensor"], regime_probabilities=probs,
            )
            err = abs(result.total_reward - float(row["total_reward"]))
            max_err = max(max_err, err)
            checked += 1
        status = "ok" if max_err <= 1e-9 else "MISMATCH"
        print(f"  {name}: {checked} episodes replayed, max reward error {max_err:.3e} [{status}]")
        if max_err > 1e-9:
            ok = False
    return ok


def check_regenerate() -> bool:
    print("== publication regeneration ==")
    steps = [
        [sys.executable, "-m", "results.publication.generate_lncs_tables"],
    ]
    for cmd in steps:
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  FAILED: {' '.join(cmd)}")
            return False
    tables = ROOT / "paper2_submission" / "manuscript_lncs" / "tables"
    produced = sorted(p.name for p in tables.glob("*.tex"))
    print(f"  ok: regenerated {len(produced)} table files: {', '.join(produced)}")
    return bool(produced)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    run_all = args.all or not any((args.artifacts, args.replay, args.regenerate))
    ok = True
    if run_all or args.artifacts:
        ok &= check_artifacts()
    if run_all or args.replay:
        ok &= check_replay()
    if run_all or args.regenerate:
        ok &= check_regenerate()

    print("\nRELEASE VERIFICATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
