"""Fetch and verify one private Kaggle CPU non-result canary.

This verifier accepts only the fixed smoke preflight output. It records the
remote manifest and raw response hashes locally, and refuses to promote a
failed, incomplete, lockbox-bearing, or nondeterministic run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ROOT / "versions/sem2act-v5/manifests/execution_attempts"
RUN_ROOT = ROOT / "versions/sem2act-v5/manifests/cpu_canary_runs"
EXPECTED_FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
FAMILIES = {"qwen", "llama", "mistral", "reranker"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify(family: str, output_dir: Path, package: dict) -> dict:
    manifest_path = output_dir / "preflight_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing remote preflight_manifest.json in {output_dir}")
    remote = read_json(manifest_path)
    if remote.get("family") != family:
        raise SystemExit("remote family does not match requested family")
    if remote.get("result_bearing_execution_started") is not False:
        raise SystemExit("remote canary claims result-bearing execution")
    if remote.get("qrels_read") is not False:
        raise SystemExit("remote canary claims qrels access")
    if remote.get("fixture_sha256") != sha(EXPECTED_FIXTURE):
        raise SystemExit("remote smoke fixture hash mismatch")
    if remote.get("model_id") != package["model_id"]:
        raise SystemExit("remote model id mismatch")
    if remote.get("revision") != package["revision"]:
        raise SystemExit("remote model revision mismatch")
    if remote.get("tokenizer_revision") != package["tokenizer_revision"]:
        raise SystemExit("remote tokenizer revision mismatch")
    if remote.get("runtime") != package["runtime"]:
        raise SystemExit("remote CPU inference representation mismatch")
    if remote.get("source_script_sha256") != package["source_script_sha256"]:
        raise SystemExit("remote canary source hash mismatch")
    expected_source = package.get("model_source")
    if expected_source:
        remote_source = remote.get("model_source")
        if not isinstance(remote_source, dict):
            raise SystemExit("remote official model source metadata is missing")
        for key, value in expected_source.items():
            if remote_source.get(key) != value:
                raise SystemExit(f"remote official model source mismatch for {key}")
        resolved = str(remote_source.get("resolved_model_path", ""))
        if remote_source.get("type") == "verified_kaggle_dataset_assembly":
            if not resolved.startswith("/kaggle/working/hf_model_assembled"):
                raise SystemExit("verified source assembly was not loaded from the pinned working assembly")
        elif not resolved.startswith("/kaggle/input/"):
            raise SystemExit("official model was not loaded from a mounted Kaggle source")
        snapshot = remote.get("source_model_snapshot")
        if not isinstance(snapshot, dict) or not snapshot.get("files_sha256"):
            raise SystemExit("official model content snapshot is missing")
    canary = remote.get("canary")
    if not isinstance(canary, dict):
        raise SystemExit("remote canary result is missing")
    if remote.get("status") != "pass" or canary.get("status") != "pass":
        raise SystemExit("remote canary failed; preserving failed output without promotion")
    if canary.get("schema_valid") is not True:
        raise SystemExit("remote canary schema check failed")
    if canary.get("deterministic_repeat_equality") is not True:
        raise SystemExit("remote canary repeat equality failed")
    if canary.get("actual_calls") != canary.get("expected_calls"):
        raise SystemExit("remote canary call count mismatch")
    if family == "reranker" and canary.get("actual_calls") != 2:
        raise SystemExit("reranker canary call count mismatch")
    if family != "reranker" and canary.get("actual_calls") != 8:
        raise SystemExit("consumer canary call count mismatch")
    return remote


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=sorted(FAMILIES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kaggle", default="/private/tmp/kaggle-cli-v2/bin/kaggle")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    attempt_path = ATTEMPTS / f"cpu-canary-package-{args.family}.json"
    if not attempt_path.exists():
        raise SystemExit(f"missing package manifest: {attempt_path}")
    attempt = read_json(attempt_path)
    if attempt.get("status") != "submitted-non-result-canary":
        raise SystemExit("package must be submitted before remote verification")
    package_files = attempt.get("files", {})
    if set(package_files) != {"canary_config.json", "canary_job.py", "kernel-metadata.json", "runtime_smoke_v1.json"}:
        raise SystemExit("unexpected package file set")
    if attempt.get("result_bearing_execution_started") is not False:
        raise SystemExit("local package is not marked non-result")
    if attempt.get("protocol_hash") != "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022":
        raise SystemExit("package protocol hash mismatch")
    if sha(EXPECTED_FIXTURE) != attempt.get("fixture_sha256"):
        raise SystemExit("local fixture differs from package manifest")
    lock = read_json(ROOT / "versions/sem2act-v5/manifests/cpu_runtime_lock.json")
    spec = lock["models"][args.family]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        subprocess.run([
            args.kaggle, "kernels", "output", attempt["slug"],
            "-p", str(args.output_dir), "--force",
        ], check=True)

    package_for_check = {
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "tokenizer_revision": spec["tokenizer_revision"],
        "runtime": {
            key: spec[key]
            for key in (
                "backend", "dtype", "bitsandbytes", "device", "device_map",
                "max_context_tokens", "batch_size", "ubatch_size", "attention_backend",
            )
            if key in spec
        },
        "source_script_sha256": package_files["canary_job.py"]["sha256"],
    }
    if "model_source" in spec:
        package_for_check["model_source"] = spec["model_source"]
    remote = verify(args.family, args.output_dir, package_for_check)

    destination = RUN_ROOT / args.family
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(args.output_dir, destination)
    local_manifest = {
        "schema_version": 1,
        "manifest_id": f"sem2act-v5-kaggle-cpu-canary-result-{args.family}-v1",
        "status": "accepted-non-result-canary",
        "accepted_utc": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "slug": attempt["slug"],
        "protocol_hash": attempt["protocol_hash"],
        "package_manifest_sha256": sha(attempt_path),
        "package_files": package_files,
        "fixture_sha256": attempt["fixture_sha256"],
        "remote_manifest_sha256": sha(destination / "preflight_manifest.json"),
        "remote_manifest": remote,
        "downloaded_files": {
            path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in sorted(destination.iterdir()) if path.is_file()
        },
        "result_bearing_execution_started": False,
        "qrels_read": False,
    }
    (destination / "acceptance_manifest.json").write_text(
        json.dumps(local_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(local_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
