#!/usr/bin/env python3
"""Reproducible benchmark runner.

Usage:
    # Offline mode (default — uses cached LLM predictions)
    python3 -m benchmark.run --experiment controlled --sensor tfidf_calibrated
    python3 -m benchmark.run --experiment gym --sensor oracle_semantic

    # Live LLM inference (requires OPENAI_API_KEY in .env)
    python3 -m benchmark.run --experiment controlled --sensor gpt-4o --live

    # Run all tests
    python3 -m benchmark.run --test
"""

from __future__ import annotations

import argparse
import sys
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONTROLLED_SENSORS = [
    "noinfo", "rulebased", "tfidf_raw", "tfidf_calibrated",
    "gpt4o", "perfect_semantic",
]
GYM_SENSORS = [
    "noinfo", "rulebased", "tfidf_raw", "tfidf_calibrated",
    "gpt4o", "oracle_semantic",
]


SENSOR_LABELS = {
    "noinfo": "NoInfo",
    "rulebased": "RuleBased",
    "tfidf_raw": "TFIDF_LogReg",
    "tfidf_calibrated": "TFIDF_LogReg_Calibrated",
    "gpt4o": "gpt-4o",
    "perfect_semantic": "PerfectSemantic",
    "oracle_semantic": "OracleSemantic",
}


def run_offline(experiment: str, sensor: str):
    """Replay the publication analysis from frozen episode-level outputs."""
    correction_script = ROOT / "results" / "correction_audit" / "recompute_corrections.py"
    sivr_path = ROOT / "results" / "correction_audit" / "sivr_recomputed.csv"
    if not sivr_path.exists():
        subprocess.run([sys.executable, str(correction_script)], cwd=str(ROOT), check=True)

    phase = "Phase 7" if experiment == "controlled" else "Phase 8B"
    label = SENSOR_LABELS.get(sensor, sensor)
    with sivr_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["phase"] == phase and row["regime"] == "ALL"
                and row["sensor"] == label]
    if not rows:
        raise SystemExit(f"No frozen offline result for {experiment}/{sensor} ({label}).")
    print(f"Offline replay from frozen episode outputs: {experiment}/{sensor}")
    for row in rows:
        print("  {aggregation_estimand}: reward={sensor_reward}, "
              "delta_vs_noinfo={delta}, signed_sivr={sivr}, status={status}".format(
                  aggregation_estimand=row["aggregation_estimand"],
                  sensor_reward=row["sensor_reward"],
                  delta=float(row["sensor_reward"]) - float(row["noinfo_reward"]),
                  sivr=row["signed_sivr"], status=row["status"]))


def run_controlled(sensor: str, live: bool = False, seeds: int = 20, seed_start: int = 2000):
    """Run controlled benchmark (Experiment A) in explicitly live mode."""
    if sensor in ("noinfo", "rulebased", "tfidf_raw", "tfidf_calibrated", "perfect_semantic"):
        print(f"Running Phase-7 classical baseline for sensor={sensor}...")
        subprocess.run([sys.executable, "-m", "src.experiment_phase7"], cwd=str(ROOT), check=True)
    elif sensor == "gpt4o":
        if live:
            print(f"Running live LLM inference for gpt-4o...")
            subprocess.run([sys.executable, "-m", "src.experiment_phase5_5", "gpt-4o"], cwd=str(ROOT), check=True)
        else:
            print(f"Using cached LLM predictions for gpt-4o...")
            subprocess.run([sys.executable, "-m", "src.experiment_phase5_5", "gpt-4o"], cwd=str(ROOT), check=True)
    else:
        print(f"Unknown sensor for controlled experiment: {sensor}")
        sys.exit(1)


def run_gym(sensor: str, live: bool = False, seeds: int = 30, seed_start: int = 3100):
    """Run Paper-1 Gymnasium experiment (Experiment B)."""
    args = [sys.executable, "-m", "src.experiment_phase8b",
            "--seeds", str(seeds), "--seed-start", str(seed_start)]
    if not live:
        args.append("--no-llm")
        if sensor in ("gpt4o",):
            print(f"Sensor '{sensor}' requires --live mode for LLM inference.")
            print("Falling back to available offline sensors.")
            return
    else:
        model_map = {"gpt4o": "gpt-4o"}
        if sensor in model_map:
            args.extend(["--llm-models", model_map[sensor]])
    print(f"Running Phase-8B gym experiment for sensor={sensor}...")
    subprocess.run(args, cwd=str(ROOT), check=True)


def run_tests():
    """Run the full test suite."""
    print("Running full test suite...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(ROOT)
    )
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Decision Value Benchmark Runner"
    )
    parser.add_argument("--experiment", choices=["controlled", "gym"],
                        help="Experiment type")
    parser.add_argument("--sensor", type=str,
                        help="Sensor/interpreter to evaluate")
    parser.add_argument("--live", action="store_true",
                        help="Use live LLM inference (requires API key)")
    parser.add_argument("--offline", action="store_true",
                        help="Replay frozen publication outputs (default unless --live is set)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of operational seeds")
    parser.add_argument("--seed-start", type=int, default=None,
                        help="Starting seed value")
    parser.add_argument("--test", action="store_true",
                        help="Run the full test suite")

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if not args.experiment or not args.sensor:
        parser.print_help()
        sys.exit(1)

    sensor = args.sensor.lower()
    if args.offline or not args.live:
        run_offline(args.experiment, sensor)
        return
    if args.experiment == "controlled":
        run_controlled(sensor, live=args.live,
                       seeds=args.seeds or 20, seed_start=args.seed_start or 2000)
    elif args.experiment == "gym":
        run_gym(sensor, live=args.live,
                seeds=args.seeds or 30, seed_start=args.seed_start or 3100)


if __name__ == "__main__":
    main()
