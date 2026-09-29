"""Managed-job entrypoint for the frozen Sem2Act v5 Lightning sequence."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from runtime_contract import (
    FREEZE_REL,
    ROOT,
    environment_snapshot,
    load_freeze_manifest,
    load_lock,
    package_versions,
    sha256,
    validate_lock,
)


KERNELS = {
    "reranker": ROOT / "versions/sem2act-v5/compute/lightning/kernels/p2_v5_rerank.py",
    "qwen": ROOT / "versions/sem2act-v5/compute/lightning/kernels/p2_v5_qwen_consumer.py",
    "llama": ROOT / "versions/sem2act-v5/compute/lightning/kernels/p2_v5_consumer.py",
    "mistral": ROOT / "versions/sem2act-v5/compute/lightning/kernels/p2_v5_consumer.py",
}


def ensure_runtime_packages() -> dict[str, str | None]:
    lock = load_lock()
    pins = lock["runtime"]["package_pins"]
    requirements = [
        f"vllm=={pins['vllm']}",
        f"transformers=={pins['transformers']}",
        f"openai=={pins['openai']}",
        f"pydantic=={pins['pydantic']}",
        f"PyYAML=={pins['PyYAML']}",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *requirements],
        check=True,
    )
    versions = package_versions()
    for name, expected in (
        ("vllm", pins["vllm"]),
        ("transformers", pins["transformers"]),
        ("openai", pins["openai"]),
        ("pydantic", pins["pydantic"]),
        ("yaml", pins["PyYAML"]),
    ):
        if versions.get(name) != expected:
            raise RuntimeError(
                f"package pin mismatch for {name}: "
                f"expected {expected}, got {versions.get(name)}"
            )
    return versions


def qrel_free(root: Path) -> None:
    if not root.exists():
        raise RuntimeError(f"missing job input root: {root}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "qrel" in path.name.lower():
            raise RuntimeError(f"qrel-bearing filename: {path}")
        if "qrel" in path.read_text(errors="ignore").lower():
            raise RuntimeError(f"qrel token in input: {path}")


def verify_model_spec(family: str, input_root: Path, lock: dict) -> dict:
    spec_path = input_root / "model_spec.json"
    if family == "reranker":
        return {"model_id": lock["models"]["reranker"]["model_id"],
                "revision": lock["models"]["reranker"]["revision"]}
    if not spec_path.exists():
        raise RuntimeError(f"{family}: model_spec.json is missing")
    spec = json.loads(spec_path.read_text())
    locked = lock["models"][family]
    for key in ("model_id", "revision"):
        if spec.get(key) != locked[key]:
            raise RuntimeError(f"{family}: model pin drift in model_spec.json")
    if spec.get("protocol_hash") != lock["protocol_hash"]:
        raise RuntimeError(f"{family}: protocol hash drift in model_spec.json")
    return spec


def output_root(args: argparse.Namespace) -> Path:
    path = Path(args.output_root or os.environ.get(
        "SEM2ACT_OUTPUT_ROOT",
        str(ROOT / "versions/sem2act-v5/runtime/lightning_outputs"),
    ))
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_smoke_outputs(root: Path, family: str | None = None) -> None:
    required = [root / "cuda_smoke_manifest.json"]
    if family is None:
        required += [
            root / "reranker_smoke_manifest.json",
            root / "qwen_smoke_manifest.json",
            root / "llama_smoke_manifest.json",
            root / "mistral_smoke_manifest.json",
        ]
    else:
        required.append(root / f"{family}_smoke_manifest.json")
    for path in required:
        if not path.exists():
            raise RuntimeError(f"required preflight output missing: {path}")
        payload = json.loads(path.read_text())
        if payload.get("status") != "pass":
            raise RuntimeError(f"preflight did not pass: {path}")


def _materialized_root(staging: Path) -> Path:
    if not staging.exists():
        raise RuntimeError(f"Lightning Drive copy produced no directory: {staging}")
    marker_names = ("model_spec.json", "evidence_inputs.jsonl", "queries.jsonl")
    if any((staging / name).exists() for name in marker_names):
        return staging
    children = [path for path in staging.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    raise RuntimeError("Lightning Drive copy did not yield one frozen input bundle")


def materialize_input_root(input_root: str | Path, out: Path, family: str) -> tuple[Path, str | None]:
    source = str(input_root)
    if not source.startswith("lit://"):
        return Path(input_root), None
    executable = shutil.which("lightning")
    if not executable:
        raise RuntimeError("Lightning CLI is required to materialize a lit:// input bundle")
    staging = out / "_inputs" / family
    if staging.exists():
        raise RuntimeError(f"refusing to overwrite input staging directory: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [executable, "cp", "-r", source, str(staging)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Lightning Drive input materialization failed with exit code {result.returncode}"
        )
    return _materialized_root(staging), source


def write_job_manifest(path: Path, *, stage: str, status: str,
                       input_root: Path | None, freeze: dict,
                       input_source_uri: str | None = None,
                       error: str | None = None) -> None:
    input_files = {}
    if input_root is not None and input_root.exists() and input_root.is_dir():
        input_files = {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(input_root.iterdir()) if p.is_file()
        }
    payload = {
        "schema_version": 1,
        "experiment_id": f"v5-lightning-{stage}",
        "stage": stage,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": freeze["protocol_hash"],
        "runtime_freeze_manifest_sha256": sha256(ROOT / FREEZE_REL),
        "input_root": str(input_root) if input_root else None,
        "input_source_uri": input_source_uri,
        "input_files": input_files,
        "environment": environment_snapshot(),
        "package_versions": package_versions(),
        "qrels_read": False,
        "error": error,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_result_stage(family: str, input_root: str | Path, out: Path) -> int:
    lock = validate_lock()
    freeze = load_freeze_manifest()
    manifest_path = out / f"{family}_lightning_job_manifest.json"
    local_input: Path | None = Path(input_root) if not str(input_root).startswith("lit://") else None
    source_uri = str(input_root) if str(input_root).startswith("lit://") else None
    try:
        local_input, source_uri = materialize_input_root(input_root, out, family)
        qrel_free(local_input)
        verify_model_spec(family, local_input, lock)
        os.environ["SEM2ACT_INPUT_ROOT"] = str(local_input)
        os.environ["SEM2ACT_OUTPUT_ROOT"] = str(out)
        os.environ["SEM2ACT_RUNTIME_LOCK"] = str(ROOT / "versions/sem2act-v5/manifests/lightning_runtime_lock.json")
        os.environ["SEM2ACT_RUNTIME_FREEZE"] = str(ROOT / FREEZE_REL)
        # Jobs do not share prior job home files. Re-run the same fixed,
        # non-lockbox family smoke in this job immediately before the result
        # kernel, so result execution still has a local fail-closed gate.
        from model_smoke import run as run_smoke
        run_smoke(family)
        smoke_manifest = out / f"{family}_smoke_manifest.json"
        if not smoke_manifest.exists() or json.loads(smoke_manifest.read_text()).get("status") != "pass":
            raise RuntimeError(f"in-job {family} smoke did not pass")
        runpy.run_path(str(KERNELS[family]), run_name="__main__")
    except Exception as exc:
        write_job_manifest(manifest_path, stage=family, status="failed",
                           input_root=local_input, freeze=freeze,
                           input_source_uri=source_uri,
                           error=f"{type(exc).__name__}: {exc}")
        raise
    write_job_manifest(manifest_path, stage=family, status="pass",
                       input_root=local_input, freeze=freeze,
                       input_source_uri=source_uri)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("canary", "smoke", "reranker", "qwen", "llama", "mistral"))
    parser.add_argument("--family", choices=("reranker", "qwen", "llama", "mistral"))
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_lock()
    freeze = load_freeze_manifest()
    out = Path(args.output_root or os.environ.get(
        "SEM2ACT_OUTPUT_ROOT",
        str(ROOT / "versions/sem2act-v5/runtime/lightning_outputs"),
    ))
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({
            "status": "dry-run-pass",
            "stage": args.stage,
            "family": args.family,
            "protocol_hash": freeze["protocol_hash"],
            "runtime_freeze_manifest_sha256": sha256(ROOT / FREEZE_REL),
        }, indent=2))
        return 0

    if args.stage != "canary" and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN Lightning secret is missing")
    ensure_runtime_packages()
    if args.stage == "canary":
        from model_smoke import run
        result = run("cuda_canary")
        # Keep the filename stable for the downstream gates.
        (out / "cuda_smoke_manifest.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return 0
    if args.stage == "smoke":
        if args.family is None:
            raise SystemExit("--family is required for --stage smoke")
        from model_smoke import run
        result = run(args.family)
        (out / f"{args.family}_smoke_manifest.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return 0
    if args.input_root is None:
        raise SystemExit("--input-root is required for result-bearing stages")
    return run_result_stage(args.stage, args.input_root, out)


if __name__ == "__main__":
    raise SystemExit(main())
