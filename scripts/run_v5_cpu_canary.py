"""Run the four non-lockbox CPU canaries in the frozen order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--reranker-model-path", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=18765)
    args = parser.parse_args()
    commands = [
        [
            sys.executable,
            "versions/sem2act-v5/compute/cpu/canary.py",
            "--llama-server", str(args.llama_server),
            "--host-manifest", "versions/sem2act-v5/manifests/cpu_host_preflight.json",
            "--base-port", str(args.base_port),
        ],
        [
            sys.executable,
            "scripts/run_v5_cpu_reranker_canary.py",
            "--model-path", str(args.reranker_model_path),
            "--host-manifest", "versions/sem2act-v5/manifests/cpu_host_preflight.json",
        ],
    ]
    codes = [run(command) for command in commands]
    combined = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-cpu-canary-gate-v1",
        "status": "pass" if codes == [0, 0] else "failed",
        "result_bearing_execution_started": False,
        "consumer_canary": "versions/sem2act-v5/manifests/cpu_canary.json",
        "reranker_canary": "versions/sem2act-v5/manifests/cpu_reranker_canary.json",
        "subprocess_exit_codes": {"consumer": codes[0], "reranker": codes[1]},
    }
    out = ROOT / "versions/sem2act-v5/manifests/cpu_canary_gate.json"
    out.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    print(json.dumps(combined, indent=2, sort_keys=True))
    return 0 if combined["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
