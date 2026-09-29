"""Fail-closed Lightning launcher for Sem2Act v5.

This command validates the frozen backend contract locally and prepares one
managed single-T4 job at a time. It never places platform or HF credentials
in command arguments, manifests, or logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIGHTNING_DIR = ROOT / "versions/sem2act-v5/compute/lightning"
PREFLIGHT_ROOT = ROOT / "versions/sem2act-v5/runtime/lightning_preflight"
STAGE_MANIFEST_ROOT = ROOT / "versions/sem2act-v5/manifests/lightning_stages"
ARTIFACT_ARCHIVE_ROOT = ROOT / "versions/sem2act-v5/runtime/lightning_artifacts"
UPLOAD_MANIFEST_ROOT = ROOT / "versions/sem2act-v5/manifests/lightning_uploads"
ATTEMPT_MANIFEST_ROOT = ROOT / "versions/sem2act-v5/manifests/lightning_attempts"
TERMINAL_STATUSES = {"completed", "failed", "stopped", "cancelled", "canceled", "error"}
RESULT_STAGES = {"reranker", "qwen", "llama", "mistral"}
MODEL_FAMILIES = ("reranker", "qwen", "llama", "mistral")
sys.path.insert(0, str(LIGHTNING_DIR))
from runtime_contract import (  # noqa: E402
    FIXTURE_REL,
    FREEZE_REL,
    environment_snapshot,
    load_fixture,
    load_freeze_manifest,
    load_lock,
    package_versions,
    sha256,
    validate_lock,
)


def validate() -> dict:
    lock = validate_lock(ROOT)
    freeze = load_freeze_manifest(ROOT)
    fixture = load_fixture(ROOT)
    if not os.environ.get("HF_TOKEN"):
        # Local validation may run without the remote secret.  The remote job
        # entrypoint performs the same check before model access.
        secret_status = "remote-secret-required"
    else:
        secret_status = "present-in-process-only"
    report = {
        "status": "pass",
        "protocol_hash": lock["protocol_hash"],
        "lock_id": lock["lock_id"],
        "runtime_freeze_manifest_sha256": sha256(ROOT / FREEZE_REL),
        "smoke_fixture_sha256": sha256(ROOT / FIXTURE_REL),
        "smoke_fixture_id": fixture["fixture_id"],
        "hf_token_status": secret_status,
        "environment": environment_snapshot(),
        "package_versions": package_versions(),
    }
    print(json.dumps(report, indent=2))
    return report


def qrel_free_manifest(path: Path) -> dict:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"upload source must be a directory: {path}")
    files = {}
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if "qrel" in candidate.name.lower():
            raise SystemExit(f"qrel-bearing filename refused: {candidate}")
        text = candidate.read_text(errors="ignore")
        if "qrel" in text.lower():
            raise SystemExit(f"qrel token in upload file refused: {candidate}")
        files[str(candidate.relative_to(path))] = {
            "sha256": sha256(candidate),
            "bytes": candidate.stat().st_size,
        }
    if not files:
        raise SystemExit("refusing to upload an empty directory")
    return {
        "source": str(path),
        "files": files,
        "qrels_included": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _component(value: str | None, label: str) -> str:
    value = (value or "").strip().strip("/")
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise SystemExit(f"{label} must be a single non-empty path component")
    return value


def _expected_upload_source(purpose: str, family: str | None) -> Path:
    if purpose == "reranker-input":
        if family:
            raise SystemExit("--family is forbidden for --purpose reranker-input")
        return ROOT / "versions/sem2act-v5/runtime/kaggle_rerank_inputs"
    if purpose == "consumer-input":
        if family not in ("qwen", "llama", "mistral"):
            raise SystemExit("--family must be qwen, llama, or mistral for consumer-input")
        return ROOT / f"versions/sem2act-v5/runtime/kaggle_{family}_inputs"
    raise SystemExit(f"unknown upload purpose: {purpose}")


def _local_bundle_manifest(purpose: str, family: str | None) -> Path:
    if purpose == "reranker-input":
        return ROOT / "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json"
    return ROOT / f"versions/sem2act-v5/manifests/consumer_inputs_{family}.json"


def _verify_frozen_bundle_manifest(purpose: str, family: str | None,
                                  source: Path, current: dict) -> dict:
    manifest_path = _local_bundle_manifest(purpose, family)
    if not manifest_path.exists():
        raise SystemExit(f"frozen bundle manifest is missing: {manifest_path}")
    frozen = _load_manifest(manifest_path)
    lock = load_lock(ROOT)
    if frozen.get("protocol_hash") != lock["protocol_hash"]:
        raise SystemExit(f"bundle manifest protocol hash mismatch: {manifest_path}")
    if purpose == "reranker-input" and frozen.get("qrels_uploaded") is not False:
        raise SystemExit("reranker bundle manifest is not qrel-free")
    frozen_files = frozen.get("files", {})
    frozen_hashes = {
        name: (value.get("sha256") if isinstance(value, dict) else value)
        for name, value in frozen_files.items()
    }
    current_hashes = {name: value["sha256"] for name, value in current["files"].items()}
    if frozen_hashes != current_hashes:
        raise SystemExit(f"local bundle differs from its frozen manifest: {manifest_path}")
    return frozen


def _require_accepted_stage(key: str) -> dict:
    path = STAGE_MANIFEST_ROOT / f"{key}.json"
    if not path.exists():
        raise SystemExit(f"accepted Lightning stage is required first: {path}")
    payload = _load_manifest(path)
    if payload.get("status") != "accepted":
        raise SystemExit(f"Lightning stage is not accepted: {path}")
    return payload


def _verified_upload_for(remote_input: str, purpose: str, family: str | None = None) -> dict:
    matches = []
    if UPLOAD_MANIFEST_ROOT.exists():
        for path in sorted(UPLOAD_MANIFEST_ROOT.glob("*.json")):
            payload = _load_manifest(path)
            if (payload.get("status") == "verified"
                    and payload.get("lit_destination") == remote_input
                    and payload.get("purpose") == purpose
                    and payload.get("family") == family):
                matches.append(payload)
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one verified {purpose} upload for {remote_input}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _roundtrip_upload(executable: str, remote: str,
                      expected: dict[str, dict[str, int | str]]) -> dict:
    with tempfile.TemporaryDirectory(prefix="sem2act-v5-upload-") as temp:
        destination = Path(temp) / "download"
        result = subprocess.run(
            [executable, "cp", "-r", remote, str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"Lightning upload round-trip verification failed with exit code {result.returncode}"
            )
        root = _find_download_root(destination)
        actual = _hash_tree(root)
        if actual != expected:
            raise SystemExit("remote upload file list or SHA-256 hashes do not match local manifest")
        return actual


def upload(args: argparse.Namespace) -> int:
    validate()
    _preflight_gate()
    executable = require_cli()
    source = Path(args.local_input).resolve()
    expected_source = _expected_upload_source(args.purpose, args.family).resolve()
    if source != expected_source:
        raise SystemExit(
            f"--local-input must be the frozen {args.purpose} bundle: {expected_source}"
        )
    manifest = qrel_free_manifest(source)
    _verify_frozen_bundle_manifest(args.purpose, args.family, source, manifest)
    remote_path = args.remote_path.strip("/")
    if not remote_path:
        raise SystemExit("--remote-path must not be empty")
    owner, teamspace = _lifecycle_scope(args)
    if args.purpose == "consumer-input":
        _require_accepted_stage("reranker")
    lit_destination = f"lit://{owner}/{teamspace}/uploads/{remote_path}"
    output_manifest = UPLOAD_MANIFEST_ROOT / f"{_component(args.name, '--name')}.json"
    if output_manifest.exists():
        raise SystemExit(f"refusing to overwrite upload manifest: {output_manifest}")
    print(json.dumps({
        "status": "uploading",
        "purpose": args.purpose,
        "family": args.family,
        "teamspace": teamspace,
        "owner": owner,
        "remote_path": remote_path,
        "lit_destination": lit_destination,
        "file_count": len(manifest["files"]),
        "qrels_included": False,
        "credential_values_logged": False,
    }, indent=2))
    result = subprocess.run(
        [executable, "cp", "-r", str(source), lit_destination],
        check=False,
    )
    if result.returncode != 0:
        return result.returncode
    remote_files = _roundtrip_upload(executable, lit_destination, manifest["files"])
    manifest.update({
        "status": "verified",
        "purpose": args.purpose,
        "family": args.family,
        "owner": owner,
        "teamspace": teamspace,
        "remote_path": remote_path,
        "lit_destination": lit_destination,
        "remote_files": remote_files,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "credential_values_logged": False,
    })
    UPLOAD_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "verified",
        "purpose": args.purpose,
        "family": args.family,
        "lit_destination": lit_destination,
        "file_count": len(remote_files),
        "qrels_included": False,
    }, indent=2))
    return 0


def require_cli() -> str:
    executable = shutil.which("lightning")
    if not executable:
        raise SystemExit(
            "Lightning CLI is not installed; install lightning-sdk in the "
            "disposable execution environment before submitting."
        )
    if not os.environ.get("LIGHTNING_USER_ID") or not os.environ.get("LIGHTNING_API_KEY"):
        raise SystemExit(
            "LIGHTNING_USER_ID and LIGHTNING_API_KEY must be supplied through "
            "the process environment; plaintext credentials are never accepted "
            "as command arguments."
        )
    return executable


def _parse_cli_json(raw: str) -> dict:
    """Parse JSON emitted by Lightning inspect without retaining raw logs."""
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw).strip()
    try:
        value = json.loads(clean)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(clean):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(clean[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    # Older Lightning CLI builds hard-wrap long JSON string fields such as
    # command. Recover only the short fields required by the lifecycle gate.
    recovered = {}
    for field in ("status", "artifact_path"):
        match = re.search(rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"', clean)
        if match:
            recovered[field] = json.loads('"' + match.group(1) + '"')
    if "status" in recovered:
        return recovered
    raise RuntimeError("Lightning inspect did not return a JSON object")


def _find_field(value, name: str):
    if isinstance(value, dict):
        if name in value:
            return value[name]
        for child in value.values():
            found = _find_field(child, name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_field(child, name)
            if found is not None:
                return found
    return None


def _lifecycle_scope(args: argparse.Namespace) -> tuple[str, str]:
    owner = _component(args.owner, "--owner")
    teamspace = _component(args.teamspace, "--teamspace")
    return owner, teamspace


def _capture_cli(executable: str, command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Lightning command failed with exit code {result.returncode}")
    return result.stdout


def inspect_job(args: argparse.Namespace) -> dict:
    executable = require_cli()
    owner, teamspace = _lifecycle_scope(args)
    raw = _capture_cli(
        executable,
        [executable, "inspect", "job", "--name", args.name,
         "--teamspace", f"{owner}/{teamspace}"],
    )
    payload = _parse_cli_json(raw)
    status = _find_field(payload, "status")
    if status is None:
        raise RuntimeError("Lightning job inspection omitted status")
    return {
        "name": args.name,
        "owner": owner,
        "teamspace": teamspace,
        "status": str(status).lower(),
        "artifact_path": _find_field(payload, "artifact_path"),
    }


def status_job(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_job(args), indent=2))
    return 0


def wait_job(args: argparse.Namespace) -> int:
    deadline = time.time() + args.timeout
    while True:
        record = inspect_job(args)
        print(json.dumps(record), flush=True)
        if record["status"] in TERMINAL_STATUSES:
            if record["status"] != "completed":
                raise SystemExit(f"Lightning job ended unsuccessfully: {record['status']}")
            return 0
        if time.time() >= deadline:
            raise SystemExit("timed out waiting for Lightning job")
        time.sleep(args.interval)


def _hash_tree(root: Path) -> dict[str, dict[str, int | str]]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[str(path.relative_to(root))] = {
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
    return files


def _find_download_root(root: Path) -> Path:
    if not root.exists():
        raise RuntimeError(f"download destination does not exist: {root}")
    candidate = root / "sem2act-v5"
    if candidate.is_dir():
        return candidate
    if any(root.glob("*_smoke_manifest.json")) or (root / "cuda_smoke_manifest.json").exists():
        return root
    if (root / "run_manifest.json").exists():
        return root
    if any(path.is_file() for path in root.iterdir()):
        return root
    children = [path for path in root.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    raise RuntimeError("could not locate the downloaded Lightning artifact root")


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing manifest: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"manifest is not an object: {path}")
    return value


def _require_provenance(payload: dict, *, fixture: bool = False) -> None:
    lock = load_lock(ROOT)
    freeze_sha = sha256(ROOT / FREEZE_REL)
    lock_sha = sha256(ROOT / "versions/sem2act-v5/manifests/lightning_runtime_lock.json")
    if payload.get("protocol_hash") != lock["protocol_hash"]:
        raise RuntimeError("artifact protocol hash mismatch")
    if payload.get("runtime_freeze_manifest_sha256") != freeze_sha:
        raise RuntimeError("artifact runtime-freeze hash mismatch")
    if fixture:
        if payload.get("fixture_sha256") != sha256(ROOT / FIXTURE_REL):
            raise RuntimeError("smoke fixture hash mismatch")
        if payload.get("runtime_lock_sha256") != lock_sha:
            raise RuntimeError("smoke runtime-lock hash mismatch")
        if payload.get("smoke_fixture_id") != load_fixture(ROOT)["fixture_id"]:
            raise RuntimeError("smoke fixture identity mismatch")


def _verify_smoke(root: Path, family: str) -> dict:
    name = "cuda_smoke_manifest.json" if family == "cuda" else f"{family}_smoke_manifest.json"
    manifest = _load_manifest(root / name)
    if manifest.get("status") != "pass":
        raise RuntimeError(f"smoke did not pass: {family}")
    _require_provenance(manifest, fixture=True)
    if family == "cuda":
        environment = manifest.get("environment", {})
        devices = environment.get("devices", [])
        if not environment.get("cuda_available") or environment.get("device_count") != 1:
            raise RuntimeError("CUDA canary device gate failed")
        if len(devices) != 1 or "t4" not in str(devices[0]).lower():
            raise RuntimeError("CUDA canary did not run on exactly one T4")
    else:
        spec = load_lock(ROOT)["models"][family]
        if manifest.get("model_id") != spec["model_id"]:
            raise RuntimeError(f"{family} smoke model mismatch")
        if manifest.get("revision") != spec["revision"]:
            raise RuntimeError(f"{family} smoke revision mismatch")
        if manifest.get("tokenizer_revision") != spec["tokenizer_revision"]:
            raise RuntimeError(f"{family} smoke tokenizer revision mismatch")
    return manifest


def _preflight_gate() -> dict:
    accepted = {"cuda": _verify_smoke(PREFLIGHT_ROOT, "cuda")}
    for family in ("reranker", "qwen", "llama", "mistral"):
        accepted[family] = _verify_smoke(PREFLIGHT_ROOT, family)
    return {
        "status": "pass",
        "protocol_hash": load_lock(ROOT)["protocol_hash"],
        "runtime_freeze_manifest_sha256": sha256(ROOT / FREEZE_REL),
        "accepted": sorted(accepted),
    }


def _verify_job_manifest(root: Path, family: str) -> dict:
    manifest = _load_manifest(root / f"{family}_lightning_job_manifest.json")
    if manifest.get("status") != "pass" or manifest.get("qrels_read") is not False:
        raise RuntimeError(f"{family} Lightning job manifest failed")
    _require_provenance(manifest)
    return manifest


def _verify_reranker(root: Path) -> dict:
    manifest = _load_manifest(root / "run_manifest.json")
    _verify_job_manifest(root, "reranker")
    if manifest.get("experiment_id") != "v5-lockbox-rerank":
        raise RuntimeError("wrong reranker experiment id")
    if manifest.get("qrels_read") is not False or manifest.get("n_queries") != 240 or manifest.get("n_fail") != 0:
        raise RuntimeError("reranker coverage or qrel gate failed")
    rows = {}
    for line in (root / "rerank.trec").read_text().splitlines():
        qid, _, did, rank, *_ = line.split()
        rows.setdefault(qid, []).append((int(rank), did))
    expected = {f"v5-lb-q{i:04d}" for i in range(1, 241)}
    if set(rows) != expected:
        raise RuntimeError("reranker query coverage failed")
    if any(len(values) != 50 or [rank for rank, _ in values] != list(range(1, 51))
           or len({did for _, did in values}) != 50 for values in rows.values()):
        raise RuntimeError("reranker ranking schema failed")
    return {"status": "pass", "n_queries": 240, "n_fail": 0}


def _verify_consumer(root: Path, family: str) -> dict:
    manifest = _load_manifest(root / "run_manifest.json")
    _verify_job_manifest(root, family)
    expected_systems = ("bm25", "rerank", "oracle") if family == "qwen" else ("rerank", "oracle")
    expected_calls = 2880 if family == "qwen" else 1920
    expected_beliefs = 1440 if family == "qwen" else 960
    if manifest.get("status") != "pass" or manifest.get("qrels_read") is not False:
        raise RuntimeError(f"{family} failure or qrel gate failed")
    if manifest.get("expected_calls") != expected_calls or manifest.get("actual_calls") != expected_calls:
        raise RuntimeError(f"{family} call-count gate failed")
    if manifest.get("n_cache_files") != expected_calls or manifest.get("n_fail") != 0:
        raise RuntimeError(f"{family} cache/failure gate failed")
    if manifest.get("prompt_shas") != {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}:
        raise RuntimeError(f"{family} prompt hash drift")
    if len((root / "raw_outputs.jsonl").read_text().splitlines()) != expected_calls:
        raise RuntimeError(f"{family} raw-output coverage failed")
    rows = [json.loads(line) for line in (root / "beliefs.jsonl").read_text().splitlines() if line]
    if len(rows) != expected_beliefs:
        raise RuntimeError(f"{family} belief coverage failed")
    expected = {
        (f"v5-lb-q{i:04d}", system, consumer)
        for i in range(1, 241)
        for system in expected_systems
        for consumer in ("C1", "C3")
    }
    keys = {(row["query_id"], row["system"], row["consumer"]) for row in rows}
    if keys != expected or len(keys) != len(rows):
        raise RuntimeError(f"{family} belief keyspace failed")
    for row in rows:
        values = [float(row[name]) for name in ("p_normal", "p_supplier_delay", "p_demand_surge")]
        if any(value < 0 or value > 1 for value in values) or abs(sum(values) - 1) > 1e-6:
            raise RuntimeError(f"{family} belief simplex failed")
    return {"status": "pass", "n_calls": expected_calls, "n_beliefs": expected_beliefs}


def _verify_result_artifact(root: Path, family: str) -> dict:
    _verify_smoke(root, family)
    if family == "reranker":
        return _verify_reranker(root)
    return _verify_consumer(root, family)


def _accepted_files(root: Path, family: str) -> list[str]:
    if family == "reranker":
        return ["rerank.trec", "run_manifest.json", "runtime_versions.json", "reranker_lightning_job_manifest.json"]
    return [
        "beliefs.jsonl", "raw_outputs.jsonl", "run_manifest.json",
        "runtime_versions.json", f"{family}_lightning_job_manifest.json",
    ]


def _copy_accepted_files(source: Path, family: str) -> dict:
    if family == "reranker":
        destination = ROOT / "versions/sem2act-v5/runtime/retrieval_bundle"
    else:
        destination = ROOT / f"versions/sem2act-v5/runtime/model_outputs/{family}"
    destination.mkdir(parents=True, exist_ok=True)
    names = _accepted_files(source, family)
    for name in names:
        src = source / name
        if not src.exists():
            raise RuntimeError(f"accepted artifact file missing: {name}")
        if (destination / name).exists():
            raise RuntimeError(f"refusing to overwrite accepted output: {destination / name}")
    copied = {}
    for name in names:
        src = source / name
        dst = destination / name
        shutil.copy2(src, dst)
        copied[name] = {"sha256": sha256(dst), "bytes": dst.stat().st_size}
    return {"destination": str(destination.relative_to(ROOT)), "files": copied}


def _fetch_identity(args: argparse.Namespace) -> tuple[str, str | None, Path, Path | None]:
    if args.stage == "canary":
        key = "cuda"
        family = "cuda"
        preflight_path = PREFLIGHT_ROOT / "cuda.json"
    elif args.stage == "smoke":
        family = args.family
        if not family:
            raise SystemExit("--family is required for smoke fetch verification")
        key = family
        preflight_path = PREFLIGHT_ROOT / f"{family}.json"
    else:
        if args.family and args.family != args.stage:
            raise SystemExit("--family must match --stage for result fetch verification")
        family = args.family or args.stage
        key = family
        preflight_path = None
    return key, family, STAGE_MANIFEST_ROOT / f"{key}.json", preflight_path


def fetch_verify(args: argparse.Namespace) -> int:
    key, family, stage_manifest, preflight_path = _fetch_identity(args)
    if stage_manifest.exists():
        raise SystemExit(f"refusing to overwrite stage acceptance manifest: {stage_manifest}")
    if preflight_path is not None and preflight_path.exists():
        raise SystemExit(f"refusing to overwrite accepted preflight: {preflight_path}")
    archive = ARTIFACT_ARCHIVE_ROOT / _component(args.name, "--name")
    if archive.exists():
        raise SystemExit(f"refusing to overwrite archived Lightning artifact: {archive}")

    record = inspect_job(args)
    if record["status"] != "completed":
        raise SystemExit(f"job is not successfully completed: {record['status']}")
    expected_artifact = f"/teamspace/jobs/{args.name}/artifacts"
    artifact_path = str(record.get("artifact_path") or "").rstrip("/")
    if artifact_path != expected_artifact:
        raise SystemExit(f"unexpected Lightning artifact path: {artifact_path!r}")
    executable = require_cli()
    owner, teamspace = _lifecycle_scope(args)
    remote = f"lit://{owner}/{teamspace}/jobs/{args.name}/artifacts"
    with tempfile.TemporaryDirectory(prefix=f"sem2act-v5-{args.name}-") as temp:
        downloaded = Path(temp) / "download"
        result = subprocess.run(
            [executable, "cp", "-r", remote, str(downloaded)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"Lightning artifact download failed with exit code {result.returncode}")
        source = _find_download_root(downloaded)
        if args.stage == "canary":
            manifest = _verify_smoke(source, "cuda")
            accepted = {"status": "pass", "manifest": manifest}
        elif args.stage == "smoke":
            manifest = _verify_smoke(source, family)
            accepted = {"status": "pass", "manifest": manifest}
        else:
            for filename in _accepted_files(source, family):
                if not (source / filename).exists():
                    raise SystemExit(f"accepted artifact file missing: {filename}")
            verification = _verify_result_artifact(source, family)
            accepted = {"status": "pass", "verification": verification}

        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, archive)
        if args.stage in RESULT_STAGES:
            accepted["staged"] = _copy_accepted_files(source, family)
        if preflight_path is not None:
            preflight_path.parent.mkdir(parents=True, exist_ok=True)
            preflight_path.write_text(json.dumps(accepted["manifest"], indent=2) + "\n")

        record = {
            "schema_version": 1,
            "status": "accepted",
            "stage": args.stage,
            "family": family,
            "job_name": args.name,
            "owner": owner,
            "teamspace": teamspace,
            "remote_artifact": remote,
            "artifact_path": artifact_path,
            "protocol_hash": load_lock(ROOT)["protocol_hash"],
            "runtime_freeze_manifest_sha256": sha256(ROOT / FREEZE_REL),
            "artifact_files": _hash_tree(archive),
            "verification": accepted,
        }
        STAGE_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
        stage_manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


def job_command(stage: str, *, family: str | None, remote_input: str | None,
                remote_output: str) -> str:
    command = [
        "python3",
        "versions/sem2act-v5/compute/lightning/job_entrypoint.py",
        "--stage", stage,
        "--output-root", remote_output,
    ]
    if family:
        command.extend(("--family", family))
    if remote_input:
        command.extend(("--input-root", remote_input))
    rendered = []
    artifact_marker = chr(36) + "{LIGHTNING_ARTIFACTS_DIR}"
    for item in command:
        if artifact_marker in item:
            rendered.append(item)
        else:
            rendered.append(shlex.quote(item))
    return " ".join(rendered)


def _require_shared_output(remote_output: str) -> None:
    expected = "${LIGHTNING_ARTIFACTS_DIR}/sem2act-v5"
    if remote_output != expected:
        raise SystemExit(f"--remote-output is frozen at {expected}")


def _require_result_inputs(stage: str, remote_input: str) -> dict:
    if not remote_input.startswith("lit://"):
        raise SystemExit("result-bearing --remote-input must be a verified lit:// upload")
    if stage == "reranker":
        return _verified_upload_for(remote_input, "reranker-input", None)
    _require_accepted_stage("reranker")
    family = stage
    local_input = ROOT / f"versions/sem2act-v5/runtime/kaggle_{family}_inputs"
    current = qrel_free_manifest(local_input)
    _verify_frozen_bundle_manifest("consumer-input", family, local_input, current)
    return _verified_upload_for(remote_input, "consumer-input", family)


def _write_attempt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def submit(args: argparse.Namespace) -> int:
    validate()
    _component(args.name, "--name")
    _require_shared_output(args.remote_output)
    executable = require_cli()
    if args.stage in RESULT_STAGES:
        if args.family and args.family != args.stage:
            raise SystemExit("--family must match --stage for result-bearing jobs")
        family = args.stage
        if not args.remote_input:
            raise SystemExit("result-bearing stages require --remote-input")
        _require_preflight = _preflight_gate()
        _require_result_inputs(family, args.remote_input)
    elif args.stage == "smoke":
        if not args.family:
            raise SystemExit("--family is required for --stage smoke")
        _verify_smoke(PREFLIGHT_ROOT, "cuda")
        family = args.family
        if args.remote_input:
            raise SystemExit("smoke jobs do not accept --remote-input")
    else:
        family = args.family
        if args.remote_input:
            raise SystemExit("canary jobs do not accept --remote-input")

    if not args.teamspace:
        raise SystemExit("--teamspace is required for remote submission")
    command = job_command(
        args.stage,
        family=family,
        remote_input=args.remote_input,
        remote_output=args.remote_output,
    )
    cli = [
        executable, "run", "job",
        "--name", args.name,
        "--machine", "T4",
        "--command", command,
    ]
    if args.studio:
        cli.extend(("--studio", args.studio))
    cli.extend(("--teamspace", args.teamspace))
    attempt_path = ATTEMPT_MANIFEST_ROOT / f"{_component(args.name, '--name')}.json"
    if attempt_path.exists():
        raise SystemExit(f"refusing to overwrite submission attempt manifest: {attempt_path}")
    attempt = {
        "schema_version": 1,
        "status": "submitting",
        "stage": args.stage,
        "family": family,
        "job_name": args.name,
        "machine": "T4",
        "job_type": "single-machine",
        "teamspace": args.teamspace,
        "studio": args.studio,
        "remote_input": args.remote_input,
        "remote_output": args.remote_output,
        "command": command,
        "protocol_hash": load_lock(ROOT)["protocol_hash"],
        "runtime_freeze_manifest_sha256": sha256(ROOT / FREEZE_REL),
        "credential_values_logged": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_attempt(attempt_path, attempt)
    print(json.dumps({
        "status": "submitting",
        "stage": args.stage,
        "family": family,
        "machine": "T4",
        "job_type": "single-machine",
        "command": command,
        "credential_values_logged": False,
    }, indent=2))
    result = subprocess.run(cli, check=False, capture_output=True, text=True)
    attempt["status"] = "submitted" if result.returncode == 0 else "submit-failed"
    attempt["returncode"] = result.returncode
    attempt["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _write_attempt(attempt_path, attempt)
    print(json.dumps({
        "status": attempt["status"],
        "stage": args.stage,
        "family": family,
        "job_name": args.name,
        "returncode": result.returncode,
        "credential_values_logged": False,
    }, indent=2))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    dry = sub.add_parser("dry-run")
    dry.add_argument("--stage", required=True,
                     choices=("canary", "smoke", "reranker", "qwen", "llama", "mistral"))
    dry.add_argument("--family", choices=MODEL_FAMILIES)
    dry.add_argument("--remote-input")
    dry.add_argument("--remote-output", default="${LIGHTNING_ARTIFACTS_DIR}/sem2act-v5")

    upload_parser = sub.add_parser("upload")
    upload_parser.add_argument("--purpose", required=True,
                               choices=("reranker-input", "consumer-input"))
    upload_parser.add_argument("--family", choices=("qwen", "llama", "mistral"))
    upload_parser.add_argument("--local-input", required=True)
    upload_parser.add_argument("--owner", required=True)
    upload_parser.add_argument("--teamspace", required=True)
    upload_parser.add_argument("--remote-path", required=True)
    upload_parser.add_argument("--name", required=True)

    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--stage", required=True,
                               choices=("canary", "smoke", "reranker", "qwen", "llama", "mistral"))
    submit_parser.add_argument("--family", choices=MODEL_FAMILIES)
    submit_parser.add_argument("--remote-input")
    submit_parser.add_argument("--remote-output", default="${LIGHTNING_ARTIFACTS_DIR}/sem2act-v5")
    submit_parser.add_argument("--name", required=True)
    submit_parser.add_argument("--studio")
    submit_parser.add_argument("--teamspace")

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--name", required=True)
    status_parser.add_argument("--owner", required=True)
    status_parser.add_argument("--teamspace", required=True)

    wait_parser = sub.add_parser("wait")
    wait_parser.add_argument("--name", required=True)
    wait_parser.add_argument("--owner", required=True)
    wait_parser.add_argument("--teamspace", required=True)
    wait_parser.add_argument("--interval", type=float, default=15.0)
    wait_parser.add_argument("--timeout", type=float, default=86400.0)

    fetch_parser = sub.add_parser("fetch-verify")
    fetch_parser.add_argument("--stage", required=True,
                              choices=("canary", "smoke", "reranker", "qwen", "llama", "mistral"))
    fetch_parser.add_argument("--family", choices=MODEL_FAMILIES)
    fetch_parser.add_argument("--name", required=True)
    fetch_parser.add_argument("--owner", required=True)
    fetch_parser.add_argument("--teamspace", required=True)

    args = parser.parse_args()

    if args.command == "validate":
        validate()
        return 0
    if args.command == "upload":
        return upload(args)
    if args.command == "status":
        return status_job(args)
    if args.command == "wait":
        if args.interval <= 0 or args.timeout <= 0:
            raise SystemExit("--interval and --timeout must be positive")
        return wait_job(args)
    if args.command == "fetch-verify":
        return fetch_verify(args)
    if args.command == "dry-run":
        validate()
        if args.stage == "smoke" and not args.family:
            raise SystemExit("--family is required for --stage smoke")
        if args.stage in RESULT_STAGES and args.family and args.family != args.stage:
            raise SystemExit("--family must match --stage for result-bearing jobs")
        family = args.family or (args.stage if args.stage in RESULT_STAGES else None)
        print(json.dumps({
            "status": "dry-run-pass",
            "command": job_command(
                args.stage, family=family,
                remote_input=args.remote_input,
                remote_output=args.remote_output,
            ),
            "machine": "T4",
            "job_type": "single-machine",
            "credentials_logged": False,
        }, indent=2))
        return 0
    return submit(args)


if __name__ == "__main__":
    raise SystemExit(main())
