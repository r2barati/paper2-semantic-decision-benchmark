"""Minimal account-entitlement canary for the Kaggle web UI.

Run this exact script in a fresh private Kaggle notebook under the
siavashsimin account after manually selecting a GPU accelerator. It contains
no Sem2Act code, data, model, package pin, or result-bearing input.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone

import torch


def main() -> int:
    cuda_available = bool(torch.cuda.is_available())
    result = {
        "schema_version": 1,
        "canary": "kaggle-ui-account-entitlement",
        "status": "gpu-available" if cuda_available else "no-gpu",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "NO_GPU",
        "device_count": torch.cuda.device_count(),
    }
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        result["nvidia_smi"] = f"unavailable:{type(exc).__name__}"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
