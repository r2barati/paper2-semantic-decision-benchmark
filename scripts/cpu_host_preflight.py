"""Record the non-result CPU host profile used to freeze v5 thread settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "versions/sem2act-v5/manifests/cpu_host_preflight.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict:
    physical = None
    logical = os.cpu_count() or 1
    memory_bytes = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True) or logical
        memory_bytes = int(psutil.virtual_memory().total)
    except Exception:
        if platform.system() == "Darwin":
            try:
                physical = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
                logical = int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True).strip())
                memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            except Exception:
                pass
    physical = int(physical or logical)
    logical = int(logical or physical)
    threads = max(1, min(physical, 8))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": platform.processor() or "unknown",
        "physical_cores": physical,
        "logical_cores": logical,
        "memory_bytes": memory_bytes,
        "memory_gib": None if memory_bytes is None else round(memory_bytes / 2**30, 3),
        "frozen_threads": threads,
        "frozen_threads_batch": threads,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "GGML_CUDA": "0",
            "GGML_METAL": "0",
            "GGML_VULKAN": "0",
            "GGML_SYCL": "0",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--llama-server", type=Path)
    args = parser.parse_args()
    data = snapshot()
    binary = args.llama_server
    if binary is None:
        found = shutil.which("llama-server")
        binary = None if found is None else Path(found)
    data["llama_server"] = None if binary is None else str(binary.resolve())
    data["llama_server_sha256"] = None if binary is None else sha(binary)
    data["status"] = "pass"
    data["schema_version"] = 1
    data["manifest_id"] = "sem2act-v5-cpu-host-preflight-v1"
    data["generated_utc"] = datetime.now(timezone.utc).isoformat()
    data["result_bearing_execution_started"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
