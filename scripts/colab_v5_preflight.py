"""Record an exact runtime preflight for the Muse/Colab V5 pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/colab_runtime_lock_v1.json"
EXPECTED_LOCK_SHA256 = "b40454497dd6a365aa1531b691eb515bab06a156e1a158d774af92d9cb4c3c4b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    account = os.environ.get("SEM2ACT_ACCOUNT_ID", "").strip()
    if not account:
        raise SystemExit("SEM2ACT_ACCOUNT_ID must contain a non-secret Colab profile alias")
    if sha(LOCK) != EXPECTED_LOCK_SHA256:
        raise SystemExit("Colab runtime lock hash differs from this execution source")
    import torch
    import transformers
    memory = None
    name = None
    failures = []
    if platform.python_version_tuple()[:2] != ("3", "12"):
        failures.append("Python must be 3.12")
    if torch.__version__ != "2.11.0+cu128" or str(torch.version.cuda or "") != "12.8":
        failures.append("Torch/CUDA differs from the explicitly accepted Colab runtime")
    if transformers.__version__ != "4.57.6":
        failures.append("Transformers must be 4.57.6")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        failures.append("exactly one visible CUDA device is required")
    else:
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory
        if "t4" not in name.lower() or memory < 14 * 1024**3:
            failures.append("expected one Tesla T4 with at least 14 GiB visible memory")
    record = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-colab-preflight-v1",
        "status": "pass" if not failures else "blocked",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022",
        "runtime_lock": "versions/sem2act-v5/manifests/colab_runtime_lock_v1.json",
        "runtime_lock_sha256": sha(LOCK),
        "account_identity": account,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "device_name": name,
        "device_memory_bytes": memory,
        "cuda_driver_report": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:2000],
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "result_bearing_execution_started": False,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
