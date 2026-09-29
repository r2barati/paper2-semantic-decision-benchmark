"""Fail-closed host/runtime preflight for the Moon CPU reranker pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/moon_cpu_reranker_runtime_v1.json"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
GIB = 1024 ** 3
LOCAL_FILESYSTEMS = {"ext4", "xfs", "btrfs", "zfs", "tmpfs"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_limit() -> int:
    values = []
    try:
        values.append(int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")))
    except (ValueError, OSError, AttributeError):
        pass
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(name).read_text().strip()
            if raw != "max":
                value = int(raw)
                if value < 2**60:
                    values.append(value)
        except (OSError, ValueError):
            pass
    return min(values) if values else 0


def cpu_flags() -> set[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith(("flags", "features")) and ":" in line:
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def mount_info(path: Path) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["findmnt", "-n", "-o", "FSTYPE,SOURCE", "-T", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip().split(None, 1)
        return (result[0], result[1] if len(result) > 1 else None)
    except Exception:
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    root = args.node_local_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    home = Path.home().resolve()
    if root == home or home in root.parents:
        raise SystemExit("node-local root must not be inside the home/NFS tree")
    hf_home = root / "hf"
    torch_home = root / "torch"
    hf_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TORCH_HOME"] = str(torch_home)
    for key, value in lock["runtime"]["environment"].items():
        os.environ[key] = value

    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 0
    memory = memory_limit()
    disk_free = shutil.disk_usage(root).free
    fs_type, fs_source = mount_info(root)
    flags = cpu_flags()
    account_identity = os.environ.get("TMU_USER", "").strip()
    torch_version = transformers_version = None
    cuda_available = None
    try:
        import torch
        import transformers
        torch_version = torch.__version__
        transformers_version = transformers.__version__
        cuda_available = bool(torch.cuda.is_available())
        torch.set_num_threads(16)
        torch.set_num_interop_threads(1)
        thread_count = torch.get_num_threads()
        interop_threads = torch.get_num_interop_threads()
        cuda_runtime = torch.version.cuda
    except Exception as exc:
        raise SystemExit(f"pinned CPU runtime import failed: {type(exc).__name__}: {exc}") from exc

    failures = []
    if platform.python_version_tuple()[:2] != ("3", "12"):
        failures.append("python must be 3.12")
    if torch_version != "2.7.1+cpu":
        failures.append("torch must be 2.7.1+cpu")
    if transformers_version != "4.57.6":
        failures.append("transformers must be 4.57.6")
    if affinity < 16:
        failures.append("fewer than 16 CPU vCPUs are visible")
    if memory < 48 * GIB:
        failures.append("effective memory limit is below 48 GiB")
    if disk_free < 20 * GIB:
        failures.append("node-local free space is below 20 GiB")
    if not fs_type or fs_type.lower() not in LOCAL_FILESYSTEMS:
        failures.append("node-local root filesystem could not be verified as local")
    if not flags:
        failures.append("CPU feature flags could not be read")
    if bool({"avx2"} & flags) or bool({"avx512f"} & flags):
        failures.append("AVX2/AVX512 visibility differs from the recorded Moon profile")
    if cuda_available:
        failures.append("CUDA is visible on the CPU-only Moon route")
    if cuda_runtime is not None:
        failures.append("CPU Torch build unexpectedly exposes a CUDA runtime")
    if thread_count != 16:
        failures.append("PyTorch failed to apply the 16-thread setting")
    if interop_threads != 1:
        failures.append("PyTorch failed to apply the one inter-op thread setting")
    if not account_identity:
        failures.append("TMU_USER account identity is missing")

    manifest = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-moon-cpu-reranker-host-preflight-v1",
        "status": "pass" if not failures else "blocked",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "TMU moon CPU server",
        "account_identity": account_identity,
        "host_id": platform.node(),
        "platform": platform.platform(),
        "cpu_model": platform.processor() or "unknown",
        "visible_vcpus": affinity,
        "memory_limit_bytes": memory,
        "memory_limit_gib": round(memory / GIB, 3) if memory else None,
        "avx2_visible": "avx2" in flags,
        "avx512_visible": "avx512f" in flags,
        "node_local_root": str(root).replace(str(home), "~"),
        "node_local_filesystem": fs_type,
        "node_local_mount_source": fs_source,
        "node_local_free_bytes": disk_free,
        "home_tree_used_for_model_cache": False,
        "hf_home": str(hf_home),
        "torch_home": str(torch_home),
        "python": platform.python_version(),
        "torch": torch_version,
        "transformers": transformers_version,
        "torch_cuda_available": cuda_available,
        "torch_cuda_runtime": cuda_runtime,
        "torch_num_threads": thread_count,
        "torch_num_interop_threads": interop_threads,
        "frozen_threads": thread_count,
        "frozen_threads_batch": thread_count,
        "runtime_lock": "versions/sem2act-v5/manifests/moon_cpu_reranker_runtime_v1.json",
        "runtime_lock_sha256": sha(LOCK),
        "protocol_hash": PROTOCOL_HASH,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "result_bearing_execution_started": False,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
