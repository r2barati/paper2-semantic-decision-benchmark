"""Non-result Kaggle runtime/device canary for Sem2Act V5."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
AMENDMENT_ID = "sem2act-v5-kaggle-backend-v3"
EXPECTED_TORCH = "2.7.1+cu126"
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_CUDA = "12.6"
EXPECTED_RUNTIME_LOCK_SHA256 = "df2618f13337b37d59ae4e21cda4eb3423569190869db27614b651f07234ca6b"
# Kaggle's launcher replaces this placeholder with the exact handoff SHA.
KAGGLE_HANDOFF_SHA = None


def install_pins():
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        "--index-url", "https://download.pytorch.org/whl/cu126",
        f"torch=={EXPECTED_TORCH}",
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        f"transformers=={EXPECTED_TRANSFORMERS}",
    ], check=True)


def main() -> int:
    install_pins()
    import torch
    import transformers

    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise SystemExit(f"Python runtime drift: {platform.python_version()}")
    if torch.__version__ != EXPECTED_TORCH:
        raise SystemExit(f"Torch runtime drift: {torch.__version__}")
    cuda = str(torch.version.cuda or "")
    if not cuda.startswith(EXPECTED_CUDA):
        raise SystemExit(f"Torch CUDA runtime drift: {cuda}")
    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise SystemExit(f"Transformers runtime drift: {transformers.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible CUDA device")
    name = torch.cuda.get_device_name(0)
    if "t4" not in name.lower():
        raise SystemExit(f"expected a Tesla T4, got {name}")
    device_memory = torch.cuda.get_device_properties(0).total_memory
    if device_memory < 14 * 1024**3:
        raise SystemExit("T4 device exposes less than 14 GiB of memory")
    x = torch.ones((512, 512), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()):
        raise SystemExit("CUDA canary produced non-finite output")
    try:
        driver = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        driver = f"unavailable: {type(exc).__name__}"
    manifest = {
        "schema_version": 1,
        "stage": "kaggle_cuda_runtime_canary",
        "status": "pass",
        "protocol_hash": PROTOCOL_HASH,
        "amendment_id": AMENDMENT_ID,
        "runtime_lock_sha256": EXPECTED_RUNTIME_LOCK_SHA256,
        "execution_sha": KAGGLE_HANDOFF_SHA,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": cuda,
        "transformers": transformers.__version__,
        "device_count": torch.cuda.device_count(),
        "device_name": name,
        "device_memory_bytes": device_memory,
        "driver_devices": driver,
        "real_cuda_matmul": True,
        "result_bearing_execution_started": False,
    }
    Path("/kaggle/working/canary_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
