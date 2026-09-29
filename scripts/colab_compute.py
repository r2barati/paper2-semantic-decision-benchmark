"""Fail-closed Colab launcher for the frozen Sem2Act v5 Qwen stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_REL = "versions/sem2act-v5/manifests/colab_runtime_lock.json"
FREEZE_REL = "versions/sem2act-v5/manifests/colab_runtime_freeze.json"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
EXPECTED_TORCH = "2.11.0+cu128"
EXPECTED_CUDA = "12.8"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_checkout(expected_sha: str) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "-uall"], cwd=ROOT, text=True
    )
    if head != expected_sha:
        raise SystemExit(f"checkout SHA mismatch: expected {expected_sha}, got {head}")
    if status:
        raise SystemExit("checkout is dirty; refusing Colab result-stage execution")


def require_input_bundle(root: Path, expected_manifest_sha: str) -> dict:
    manifest_path = root / "consumer_inputs_qwen.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise SystemExit("Qwen input bundle and consumer_inputs_qwen.json manifest are required")
    if sha(manifest_path) != expected_manifest_sha:
        raise SystemExit("Qwen input manifest SHA-256 differs from the Codex handoff")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("family") != "qwen" or manifest.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("Qwen input manifest family/protocol mismatch")
    expected_files = manifest.get("files", {})
    if any(path.is_dir() for path in root.iterdir()):
        raise SystemExit("Colab Qwen input bundle must contain only manifested files")
    actual_files = {
        path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in root.iterdir()
        if path.is_file() and path != manifest_path
    }
    if expected_files != actual_files:
        raise SystemExit("Qwen input files do not match consumer_inputs_qwen.json")
    required = {"evidence_inputs.jsonl", "model_spec.json", "runtime_smoke_v1.json",
                "consumers_v3.py", "interpreter.py", "events.py"}
    if not required.issubset(actual_files):
        raise SystemExit(f"Qwen input bundle is missing files: {sorted(required - set(actual_files))}")
    spec = json.loads((root / "model_spec.json").read_text())
    if (spec.get("model_family") != "qwen"
            or spec.get("model_id") != "Qwen/Qwen3-8B-AWQ"
            or spec.get("revision") != "4da05a8edb55c6046cce958586c33b61da07bb79"
            or spec.get("protocol_hash") != PROTOCOL_HASH):
        raise SystemExit("Qwen model specification does not match the frozen lock")
    for path in root.iterdir():
        if path.is_file() and ("qrel" in path.name.lower()
                               or "qrel" in path.read_text(errors="ignore").lower()):
            raise SystemExit(f"qrel-bearing input refused: {path.name}")
    return {"manifest_sha256": sha(manifest_path), "files": actual_files,
            "model_spec_runtime_lock_sha256": spec.get("runtime_lock_sha256")}


def require_colab_runtime() -> dict:
    import torch

    if torch.__version__ != EXPECTED_TORCH or str(torch.version.cuda or "") != EXPECTED_CUDA:
        raise SystemExit(
            f"Torch/CUDA mismatch: expected {(EXPECTED_TORCH, EXPECTED_CUDA)}, "
            f"got {(torch.__version__, str(torch.version.cuda or ''))}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Colab policy requires exactly one available CUDA device")
    name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if "t4" not in name.lower() or vram_gib < 14:
        raise SystemExit(f"Colab policy requires one T4 with >=14 GiB VRAM; got {name}, {vram_gib:.2f} GiB")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN must be supplied through Colab Secrets")
    return {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": str(torch.version.cuda), "device": name,
        "device_count": torch.cuda.device_count(),
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("qwen",))
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_sha) != 40 or any(c not in "0123456789abcdef" for c in args.expected_sha):
        parser.error("--expected-sha must be a lowercase full Git SHA")
    if len(args.expected_input_manifest_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.expected_input_manifest_sha256
    ):
        parser.error("--expected-input-manifest-sha256 must be a lowercase SHA-256")

    require_checkout(args.expected_sha)
    os.environ["SEM2ACT_RUNTIME_LOCK"] = str(ROOT / LOCK_REL)
    os.environ["SEM2ACT_RUNTIME_FREEZE"] = str(ROOT / FREEZE_REL)
    os.environ.setdefault("HF_HOME", "/content/sem2act-v5/cache/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/content/sem2act-v5/cache/huggingface/hub")
    os.environ.setdefault("TORCH_HOME", "/content/sem2act-v5/cache/torch")

    sys.path.insert(0, str(ROOT / "versions/sem2act-v5/compute/lightning"))
    from runtime_contract import load_freeze_manifest, validate_lock  # noqa: E402

    lock = validate_lock(ROOT)
    freeze = load_freeze_manifest(ROOT)
    runtime = require_colab_runtime()
    bundle = require_input_bundle(args.input_root.resolve(), args.expected_input_manifest_sha256)
    out = args.output_root.resolve()
    drive_root = Path("/content/drive/MyDrive").resolve()
    if not drive_root.is_dir() or drive_root not in out.parents:
        raise SystemExit("Colab output must be under the mounted /content/drive/MyDrive directory")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema_version": 1, "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": args.expected_sha, "protocol_hash": lock["protocol_hash"],
        "runtime_lock_sha256": sha(ROOT / LOCK_REL),
        "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
        "input_manifest_sha256": bundle["manifest_sha256"],
        "input_files": bundle["files"], "runtime": runtime,
        "qrels_read": False,
    }
    (out / "colab_preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    command = [
        sys.executable, str(ROOT / "versions/sem2act-v5/compute/lightning/job_entrypoint.py"),
        "--stage", "qwen", "--family", "qwen",
        "--input-root", str(args.input_root.resolve()), "--output-root", str(out),
    ]
    print(json.dumps({"status": "preflight-pass", "execution_sha": args.expected_sha,
                      "protocol_hash": PROTOCOL_HASH,
                      "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
                      "input_manifest_sha256": bundle["manifest_sha256"],
                      "output_root": str(out), "stage": "qwen"}, indent=2), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
