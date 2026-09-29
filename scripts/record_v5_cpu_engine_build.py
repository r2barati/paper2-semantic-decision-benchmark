"""Record the pinned llama.cpp CPU build provenance without loading models."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "versions/sem2act-v5/manifests/cpu_engine_build.json"
COMMIT = "9f70b2cecd1a9a3f73ac525c47ca22a6ee9a7b69"
FLAGS = [
    "-DGGML_NATIVE=OFF",
    "-DGGML_OPENMP=ON",
    "-DGGML_CUDA=OFF",
    "-DGGML_METAL=OFF",
    "-DGGML_VULKAN=OFF",
    "-DGGML_SYCL=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--llama-quantize", type=Path, required=True)
    parser.add_argument("--openmp-root", type=Path)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.engine_root), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise SystemExit(f"llama.cpp commit drift: expected {COMMIT}, got {head}")
    for binary in (args.llama_server, args.llama_quantize):
        if not binary.exists():
            raise SystemExit(f"missing CPU engine binary: {binary}")
    manifest = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-cpu-engine-build-v1",
        "status": "pass-before-model-conversion",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/ggml-org/llama.cpp",
        "tag": "b11194",
        "commit": head,
        "engine_root": str(args.engine_root.resolve()),
        "build_root": str(args.build_root.resolve()),
        "build_flags": FLAGS,
        "openmp_root": None if args.openmp_root is None else str(args.openmp_root.resolve()),
        "binaries": {
            "llama-server": {"path": str(args.llama_server.resolve()), "sha256": sha(args.llama_server)},
            "llama-quantize": {"path": str(args.llama_quantize.resolve()), "sha256": sha(args.llama_quantize)},
        },
        "result_bearing_execution_started": False,
    }
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
