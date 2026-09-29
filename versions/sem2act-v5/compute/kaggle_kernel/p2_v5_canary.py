"""Non-lockbox Kaggle CUDA canary for the frozen Sem2Act v5 backend."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
AMENDMENT_ID = "sem2act-v5-kaggle-backend-v3"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_CUDA = "12.8"


def main() -> int:
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        raise SystemExit(f"cannot verify allocated Kaggle GPUs: {type(exc).__name__}: {exc}") from exc
    physical_devices = [line for line in nvidia_smi.splitlines() if line.startswith("GPU ")]
    if len(physical_devices) != 2 or any("t4" not in line.lower() for line in physical_devices):
        raise SystemExit(f"expected exactly two allocated T4 GPUs, got {nvidia_smi}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected one CUDA device, got {torch.cuda.device_count()}")
    device_name = torch.cuda.get_device_name(0)
    if "t4" not in device_name.lower():
        raise SystemExit(f"expected Nvidia T4, got {device_name}")
    if torch.__version__ != EXPECTED_TORCH:
        raise SystemExit(f"torch pin drift: {torch.__version__} != {EXPECTED_TORCH}")
    cuda_version = str(torch.version.cuda or "")
    if cuda_version != EXPECTED_CUDA:
        raise SystemExit(f"CUDA pin drift: {cuda_version} != {EXPECTED_CUDA}")

    x = torch.ones((512, 512), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()):
        raise SystemExit("CUDA canary produced non-finite output")

    manifest = {
        "schema_version": 1,
        "stage": "cuda_canary",
        "status": "pass",
        "protocol_hash": PROTOCOL_HASH,
        "amendment_id": AMENDMENT_ID,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda_version,
        "device_count": torch.cuda.device_count(),
        "physical_device_count": len(physical_devices),
        "device_name": device_name,
        "nvidia_smi": nvidia_smi,
        "real_cuda_matmul": True,
    }
    Path("/kaggle/working/canary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
