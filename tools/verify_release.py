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
    """Regenerate the paper's tables and require them to be UNCHANGED.

    Rewriting the files and reporting that they exist proves nothing. The
    contract is that the committed tables are exactly what the committed data
    and analysis produce, so this snapshots them, regenerates, and diffs. Any
    drift between the paper and the data fails here.
    """
    print("== publication regeneration ==")
    tables = ROOT / "paper2_submission" / "manuscript_lncs" / "tables"
    before = {p.name: p.read_text() for p in sorted(tables.glob("*.tex"))}
    if not before:
        print("  FAILED: no committed tables to compare against")
        return False

    figures = ROOT / "paper2_submission" / "manuscript_lncs" / "figures"
    fig_before = {p.name: p.read_bytes()
                  for p in sorted(figures.glob("*")) if p.is_file()}

    for cmd in (
        [sys.executable, "-m", "results.publication.generate_lncs_tables"],
        [sys.executable, "-m", "results.publication.generate_lncs_figures"],
    ):
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"  FAILED: {' '.join(cmd)}")
            return False

    # Both the PNG and the PDF are compared. The generator suppresses
    # matplotlib's creation timestamp, so a figure that has not changed is
    # byte-identical in either format.
    fig_after = {p.name: p.read_bytes()
                 for p in sorted(figures.glob("*")) if p.is_file()}
    fig_drift = sorted(n for n in fig_before if fig_before[n] != fig_after.get(n))
    if fig_drift:
        for name in fig_drift:
            print(f"  DRIFT: figure {name} differs from the committed version")
        return False
    if fig_before:
        print(f"  ok: {len(fig_before)} figure(s) regenerate identically")

    after = {p.name: p.read_text() for p in sorted(tables.glob("*.tex"))}
    drifted = sorted(n for n in before if before[n] != after.get(n))
    missing = sorted(set(before) - set(after))
    if drifted or missing:
        for name in drifted:
            print(f"  DRIFT: {name} differs from the committed version")
            b, a = before[name].splitlines(), after[name].splitlines()
            for i, (x, y) in enumerate(zip(b, a)):
                if x != y:
                    print(f"    line {i + 1}\n      committed: {x}\n      regenerated: {y}")
                    break
        for name in missing:
            print(f"  MISSING after regeneration: {name}")
        return False

    print(f"  ok: {len(after)} table files regenerate byte-identically")
    return True


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
