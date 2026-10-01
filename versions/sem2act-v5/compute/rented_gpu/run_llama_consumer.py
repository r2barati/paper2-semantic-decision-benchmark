"""Rented-GPU launcher for the frozen Llama cross-family consumer (DEV-prepared).

Runs ON a user-controlled rented host (RunPod T4 recommended). Executes the
FROZEN kernel file versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py
verbatim by recreating the Kaggle filesystem layout it expects. No scientific
code path differs from the Kaggle route.

Credential model (infrastructure only, see
versions/sem2act-v5/manifests/llama_rented_credential_amendment_1.json):
  - HF token arrives ONLY via the SEM2ACT_HF_TOKEN process environment
    variable, injected as a platform secret at container start.
  - The launcher sets SEM2ACT_RENTED_RUN=1 so the kernel takes its explicit
    rented-secret branch and records credential_source=rented_runtime_secret
    (no Kaggle-secrets masquerade; provenance states the actual channel).
  - The token is used solely for exact pinned-revision download and the
    kernel's authenticated transformers load.
  - It is never printed, never written to disk/datasets/logs/manifests/repo,
    and never embedded in URLs that are logged (all logged URLs redacted).
  - It lives only in process memory on the single-tenant rented host; host
    teardown destroys it.

Usage (on the rented host):
  export SEM2ACT_HF_TOKEN="..."   # via platform secret, never on a shell history file
  python3 run_llama_consumer.py --repo /path/to/checkout --bundle <bundle-dir> --out <out-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_BNB = "0.48.1"
EXPECTED_HF_HUB = "0.36.2"
BUNDLE_FILES = ("evidence_inputs.jsonl", "model_spec.json", "runtime_smoke_v1.json")
SNAPSHOT_ALLOW = ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json",
                  "model.safetensors.index.json", "model-00001-of-00004.safetensors",
                  "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors",
                  "model-00004-of-00004.safetensors")


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"rented-llama gate failed: {message}")


def redact(url: str) -> str:
    """Strip any embedded credentials from a URL before logging."""
    try:
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(url)
        netloc = parts.hostname or ""
        if parts.port:
            netloc += f":{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "<unloggable-url>"


def require_token() -> str:
    token = os.environ.get("SEM2ACT_HF_TOKEN", "")
    if not token:
        fail("SEM2ACT_HF_TOKEN is absent or empty")
    return token


def assert_package_pins() -> None:
    import torch

    if (torch.__version__, str(torch.version.cuda or "")) != (EXPECTED_TORCH, EXPECTED_CUDA):
        fail(f"torch/cuda mismatch: {(torch.__version__, str(torch.version.cuda or ''))}")
    if not torch.cuda.is_available():
        fail("CUDA unavailable")
    if torch.cuda.device_count() < 1:
        fail("no CUDA device")


def assert_file_identities(root: Path, expected: dict[str, str]) -> None:
    """Fail closed unless every expected relative path hashes exactly."""
    for rel, want in sorted(expected.items()):
        path = root / rel
        if not path.is_file():
            fail(f"missing verified file: {rel}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            fail(f"identity drift: {rel}")


def build_kaggle_layout(bundle_dir: Path, kaggle_root: Path) -> tuple[Path, Path]:
    """Recreate the Kaggle input/working layout the frozen kernel expects."""
    for name in BUNDLE_FILES:
        if not (bundle_dir / name).is_file():
            fail(f"bundle file missing: {name}")
    input_dir = kaggle_root / "input" / "llama-inputs"
    working_dir = kaggle_root / "working"
    input_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        shutil.copy2(bundle_dir / name, input_dir / name)
    return input_dir, working_dir


def arm_rented_channel() -> None:
    """Opt the kernel into its explicit rented-secret branch (no stubs)."""
    os.environ["SEM2ACT_RENTED_RUN"] = "1"


def snapshot_and_verify(token: str, recovery_manifest: Path, cache_dir: Path) -> Path:
    """Download the exact pinned revision, then verify file identities pre-inference."""
    from huggingface_hub import snapshot_download

    manifest = json.loads(recovery_manifest.read_text())
    snap = snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=list(SNAPSHOT_ALLOW),
        cache_dir=str(cache_dir),
        token=token,
    )
    expected = {k: v["sha256"] if isinstance(v, dict) else v
                for k, v in manifest["small_file_sha256"].items()}
    expected.update({
        f"model-0000{i}-of-00004.safetensors": manifest["weight_shard_lfs_oid_sha256"][
            f"model-0000{i}-of-00004.safetensors"]["sha256"]
        for i in (1, 2, 3, 4)
    })
    missing = [k for k in expected if not (Path(snap) / k).is_file()]
    if missing:
        fail(f"snapshot missing files: {missing}")
    assert_file_identities(Path(snap), expected)
    return Path(snap)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rented-GPU runner for the frozen Llama consumer.")
    ap.add_argument("--repo", required=True, help="path to the repo checkout at the authorized SHA")
    ap.add_argument("--bundle", required=True, help="dir with evidence_inputs.jsonl, model_spec.json, runtime_smoke_v1.json")
    ap.add_argument("--out", required=True, help="output directory for fetched-style artifacts")
    ap.add_argument("--cache-dir", required=True, help="huggingface cache directory")
    ap.add_argument("--kaggle-root", default="/kaggle", help="Kaggle-layout root to recreate")
    args = ap.parse_args()

    repo = Path(args.repo)
    kernel = repo / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py"
    recovery = repo / "versions/sem2act-v5/manifests/cpu_source_probes/llama/recovery_manifest.json"
    for path in (kernel, recovery):
        if not path.is_file():
            fail(f"repo file missing: {path}")
    if json.loads((repo / "versions/sem2act-v5/manifests/protocol_freeze.json").read_text())["protocol_hash"] != "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022":
        fail("protocol hash drift in repo checkout")

    token = require_token()
    assert_package_pins()
    snap = snapshot_and_verify(token, recovery, Path(args.cache_dir))
    print(f"snapshot verified: {snap}", flush=True)
    input_dir, working_dir = build_kaggle_layout(Path(args.bundle), Path(args.kaggle_root))
    print(f"staged: {input_dir} -> {working_dir}", flush=True)
    arm_rented_channel()
    old_argv, old_cwd = sys.argv, os.getcwd()
    sys.argv = [str(kernel)]
    neutral_cwd = tempfile.mkdtemp(prefix="llama-run-")
    os.chdir(neutral_cwd)
    try:
        runpy.run_path(str(kernel), run_name="__main__")
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv
        try:
            os.chdir(old_cwd)
        except Exception:
            pass
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("raw_outputs.jsonl", "beliefs.jsonl", "run_manifest.json",
                 "failures.json", "runtime_versions.json"):
        src = working_dir / name
        if src.is_file():
            shutil.copy2(src, out / name)
            print(f"artifact: {name} sha={hashlib.sha256((out / name).read_bytes()).hexdigest()}", flush=True)
    print(f"kernel exit code: {code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
