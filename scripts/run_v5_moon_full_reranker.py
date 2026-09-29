"""Fail-closed, checkpointed Moon full-reranker runner.

This tool is not an authorization. It requires an external manifest with
status=authorized and a PASS pilot verifier receipt whose exact SHA-256 is
named in that manifest. It is never invoked by repository tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
PILOT_SHA = "6d97cd0fe11cc87cd38661ac3a3646ebb36b7599"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INPUT_HASHES = {
    "corpus.jsonl": "6cab24bc7420635674856f35345cd4c5c30a23d2d0ee243a79b69d36bc8ab0e1",
    "queries.jsonl": "c457d93ea195d82a738219ad152382380d89b4b5e0c3fabde5cf4d6d33434986",
    "rerank_candidates.jsonl": "11dcac88ad210447ac16f78decbb5dd09776648e14b083aa3d89f67c8b701803",
}
LOCK = ROOT / "versions/sem2act-v5/manifests/moon_reranker_runtime_lock.json"
FREEZE = ROOT / "versions/sem2act-v5/manifests/moon_reranker_runtime_freeze.json"
INPUT_MANIFEST = ROOT / "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json"
MODEL_REFERENCE = ROOT / "versions/sem2act-v5/manifests/cpu_reranker_canary.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_map(root: Path, omit: set[str] | None = None) -> dict[str, str]:
    omit = omit or set()
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in omit
    }


def snapshot(path: Path) -> dict:
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and ".cache" not in item.parts:
            files.append({"path": str(item.relative_to(path)), "sha256": sha(item), "bytes": item.stat().st_size})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"file_count": len(files), "files_sha256": digest, "files": files}


def probe_nfs_capacity(directory: Path, required_bytes: int = 10_000_000) -> int:
    fd, filename = tempfile.mkstemp(prefix=".sem2act-v5-full-quota-probe-", dir=directory)
    path = Path(filename)
    written = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            block = b"\0" * (1024 * 1024)
            while written < required_bytes:
                count = min(len(block), required_bytes - written)
                handle.write(block[:count])
                written += count
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        path.unlink(missing_ok=True)
    return written


def require_clean_checkout(expected_sha: str) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "-uall"], cwd=ROOT, text=True)
    if head != expected_sha:
        raise SystemExit(f"execution SHA mismatch: expected {expected_sha}, got {head}")
    if status:
        raise SystemExit("checkout is dirty; refusing full reranker execution")


def verify_pilot_gate(authorization: dict, verification_path: Path,
                      pilot_root: Path) -> tuple[dict, dict, dict]:
    if authorization.get("status") != "authorized":
        raise SystemExit("full-reranker authorization status must be 'authorized'")
    if authorization.get("pilot_execution_sha") != PILOT_SHA:
        raise SystemExit("full authorization pilot SHA drift")
    verification = json.loads(verification_path.read_text())
    if sha(verification_path) != authorization.get("pilot_verification_sha256"):
        raise SystemExit("accepted pilot-verifier result SHA-256 mismatch")
    if verification.get("status") != "PASS" or verification.get("execution_sha") != PILOT_SHA:
        raise SystemExit("full reranker requires a PASS verifier result for the exact 20-query pilot")
    pilot_manifest_path = pilot_root / "pilot_manifest.json"
    host_path = pilot_root / "moon_preflight.json"
    parity_path = pilot_root / "moon_reranker_canary.json"
    host_identity_path = pilot_root / "moon_host_provenance.json"
    for path in (pilot_manifest_path, host_path, parity_path, host_identity_path):
        if not path.is_file():
            raise SystemExit(f"accepted pilot provenance artifact is missing: {path.name}")
    pilot = json.loads(pilot_manifest_path.read_text())
    host = json.loads(host_path.read_text())
    parity = json.loads(parity_path.read_text())
    identity = json.loads(host_identity_path.read_text())
    recorded = verification.get("checks", {}).get("output_sha256", {})
    for name, digest in recorded.items():
        if not (pilot_root / name).is_file() or sha(pilot_root / name) != digest:
            raise SystemExit(f"accepted pilot output changed after verifier pass: {name}")
    if verification.get("checks", {}).get("host_provenance_sha256") != sha(host_identity_path):
        raise SystemExit("accepted Moon hostname/runtime receipt changed after pilot verification")
    if identity.get("hostname") != socket.gethostname():
        raise SystemExit("full reranker host differs from the accepted Moon pilot host")
    if host.get("status") != "pass" or parity.get("status") != "pass":
        raise SystemExit("Moon pilot host/parity evidence is not passing")
    return verification, host, {"host": host_path, "parity": parity_path, "identity": host_identity_path}


def validate_live_host(output_root: Path, input_root: Path, model_path: Path,
                       pilot_identity: dict, execution_sha: str) -> dict:
    import torch
    import transformers

    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise SystemExit("Moon full reranker requires Linux x86_64")
    if socket.gethostname() != pilot_identity.get("hostname"):
        raise SystemExit("Moon hostname changed after pilot acceptance")
    if sys.version.split()[0] != "3.9.19" or torch.__version__ != "2.8.0+cpu" or transformers.__version__ != "4.57.6":
        raise SystemExit("Moon Python/Torch/Transformers runtime differs from the accepted pilot")
    if torch.cuda.is_available() or os.cpu_count() != 16:
        raise SystemExit("Moon full reranker requires CPU-only execution and 16 visible vCPUs")
    memory_bytes = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            memory_bytes = int(line.split()[1]) * 1024
            break
    if memory_bytes < 60_000_000_000:
        raise SystemExit("Moon full reranker requires at least 60 GB RAM")
    per_cpu_flags = []
    for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
        flags_line = next((line for line in block.splitlines()
                           if line.startswith("flags") or line.startswith("Features")), None)
        if flags_line:
            flags = set(flags_line.split(":", 1)[1].split())
            if {"avx2", "avx512f"} & flags:
                raise SystemExit("Moon AVX2/AVX512 masking differs from the accepted pilot")
            per_cpu_flags.append(sorted(flags))
    scratch = Path("/tmp/sem2act-v5").resolve()
    for label, path in (("inputs", input_root), ("model", model_path)):
        if not path.resolve().is_relative_to(scratch):
            raise SystemExit(f"Moon {label} must be on node-local {scratch}")
    if shutil.disk_usage(scratch).free < 5_000_000_000:
        raise SystemExit("Moon node-local scratch has less than 5 GB free")
    if not input_root.is_dir() or {p.name for p in input_root.iterdir()} != set(INPUT_HASHES):
        raise SystemExit("Moon input directory must contain exactly the three frozen files")
    for name, expected in INPUT_HASHES.items():
        if sha(input_root / name) != expected:
            raise SystemExit(f"Moon input hash mismatch: {name}")
        if "qrel" in (input_root / name).read_text(errors="ignore").lower():
            raise SystemExit(f"qrel token detected in Moon input {name}")
    fs = subprocess.run(["findmnt", "-n", "-o", "FSTYPE", "--target", str(output_root.parent)],
                        text=True, capture_output=True, check=False)
    if fs.returncode or not fs.stdout.strip().lower().startswith("nfs"):
        raise SystemExit("Moon full-reranker output must be on the pilot-verified NFS filesystem")
    output_free = shutil.disk_usage(output_root.parent).free
    if output_free < 10_000_000:
        raise SystemExit("Moon NFS has less than 10 MB free for the full-reranker artifacts")
    quota_probe = probe_nfs_capacity(output_root.parent, 10_000_000)
    return {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-moon-full-host-provenance-v1",
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": execution_sha,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "transformers": transformers.__version__,
        "logical_vcpus": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "cpu_flags_records": len(per_cpu_flags),
        "cpu_flags_sha256": hashlib.sha256(json.dumps(per_cpu_flags, separators=(",", ":")).encode()).hexdigest(),
        "avx2_masked": True,
        "avx512f_masked": True,
        "scratch_root": str(scratch),
        "scratch_free_bytes": shutil.disk_usage(scratch).free,
        "output_filesystem": fs.stdout.strip().lower(),
        "output_filesystem_free_bytes": output_free,
        "nfs_quota_probe_bytes_written_and_removed": quota_probe,
        "runtime_lock_sha256": sha(LOCK),
        "runtime_freeze_sha256": sha(FREEZE),
        "input_sha256": INPUT_HASHES,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "qrels_read": False,
    }


def verify_authorization(args: argparse.Namespace) -> tuple[dict, dict, dict, dict, dict, dict]:
    authorization = json.loads(args.authorization.read_text())
    if authorization.get("status") != "authorized":
        raise SystemExit("full-reranker authorization status must be 'authorized'")
    if authorization.get("execution_sha") != args.execution_sha:
        raise SystemExit("full authorization execution SHA does not match the requested checkout")
    if authorization.get("run_id") != "sem2act-v5-moon-reranker-240q-001":
        raise SystemExit("full authorization run ID drift")
    pilot_gate = authorization.get("pilot_gate", {})
    if (pilot_gate.get("pilot_execution_sha") != PILOT_SHA
            or pilot_gate.get("verifier") != "scripts/verify_v5_moon_reranker_pilot.py"
            or pilot_gate.get("required_verifier_status") != "PASS"
            or pilot_gate.get("codex_acceptance_required") is not True
            or pilot_gate.get("pilot_verification_sha256") != authorization.get("pilot_verification_sha256")):
        raise SystemExit("full authorization does not require the exact accepted pilot verifier receipt")
    if authorization.get("pilot_execution_sha") != PILOT_SHA:
        raise SystemExit("full authorization pilot execution SHA drift")
    expected_scope = {
        "queries": 240, "candidate_depth": 50, "candidate_scores": 12000,
        "queries_per_shard": 20, "shards": 12, "threads": 16,
        "dtype": "float32", "device": "cpu", "qrels_read": False,
        "scientific_protocol_changed": False,
    }
    if authorization.get("scope") != expected_scope:
        raise SystemExit("full authorization scope differs from the frozen 240-query reranker stage")
    if (authorization.get("checkpoint_policy", {}).get("one_shard_directory_per_20_queries") is not True
            or authorization.get("checkpoint_policy", {}).get("stop_on_first_failed_shard") is not True):
        raise SystemExit("full authorization must checkpoint per 20-query shard and stop on failure")
    require_clean_checkout(args.execution_sha)
    freeze_check = subprocess.run(
        [sys.executable, str(ROOT / "scripts/freeze_v5_moon_runtime.py"), "--check"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if freeze_check.returncode != 0:
        raise SystemExit("Moon static runtime/source freeze check failed")
    verification, host, evidence_paths = verify_pilot_gate(
        authorization, args.pilot_verification, args.pilot_root.resolve()
    )
    input_root = args.input_root.resolve()
    model_path = args.model_path.resolve()
    output_root = args.output_root.resolve()
    expected_paths = authorization.get("paths", {})
    for label, actual in (("input_root", str(input_root)), ("model_path", str(model_path)),
                          ("output_root", str(output_root))):
        if expected_paths.get(label) != actual:
            raise SystemExit(f"full authorization {label} does not match supplied path")
    if authorization.get("input_sha256") != INPUT_HASHES:
        raise SystemExit("full authorization expected input hashes drift")
    if authorization.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("full authorization protocol hash drift")
    if authorization.get("model", {}).get("model_id") != MODEL_ID or authorization.get("model", {}).get("revision") != MODEL_REVISION:
        raise SystemExit("full authorization reranker model/revision drift")
    if sha(LOCK) != authorization.get("runtime_lock_sha256") or sha(FREEZE) != authorization.get("runtime_freeze_sha256"):
        raise SystemExit("full authorization Moon lock/freeze SHA-256 mismatch")
    manifest = json.loads(INPUT_MANIFEST.read_text())
    if sha(INPUT_MANIFEST) != authorization.get("input_manifest_sha256") or manifest.get("files") != INPUT_HASHES:
        raise SystemExit("full authorization source dataset manifest hash mismatch")
    if authorization.get("pilot_verification_sha256") != sha(args.pilot_verification):
        raise SystemExit("authorization is not bound to this accepted pilot verifier output")
    reference = json.loads(MODEL_REFERENCE.read_text())
    actual_snapshot = snapshot(model_path)
    if reference.get("model_snapshot") != actual_snapshot:
        raise SystemExit("local reranker model snapshot differs from the accepted model revision")
    actual_snapshot_sha = hashlib.sha256(
        json.dumps(actual_snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_snapshot_sha != authorization.get("model_snapshot_sha256"):
        raise SystemExit("model snapshot hash differs from the full authorization")
    live_host = validate_live_host(
        output_root, input_root, model_path,
        json.loads(evidence_paths["identity"].read_text()), args.execution_sha,
    )
    return authorization, verification, host, evidence_paths, actual_snapshot, live_host


def expected_state(authorization: dict, args: argparse.Namespace,
                   plan_root: Path) -> dict:
    return {
        "schema_version": 1,
        "run_id": authorization["run_id"],
        "execution_sha": args.execution_sha,
        "protocol_hash": PROTOCOL_HASH,
        "authorization_sha256": sha(args.authorization),
        "pilot_verification_sha256": sha(args.pilot_verification),
        "input_sha256": INPUT_HASHES,
        "plan_root": str(plan_root),
        "completed_shards": [],
        "status": "in-progress",
        "result_bearing_execution_started": False,
        "qrels_read": False,
    }


def validate_existing_shard(shard_dir: Path, shard: dict, query_sha: dict) -> bool:
    run_path = shard_dir / "run_manifest.json"
    rankings_path = shard_dir / "rankings.jsonl"
    failures_path = shard_dir / "failures.json"
    if not (run_path.is_file() and rankings_path.is_file() and failures_path.is_file()):
        return False
    run = json.loads(run_path.read_text())
    failures = json.loads(failures_path.read_text())
    rows = [json.loads(line) for line in rankings_path.read_text().splitlines() if line.strip()]
    expected = set(shard["expected_query_keys"])
    return (
        run.get("status") == "pass" and run.get("n_fail") == 0 and failures == []
        and run.get("expected_queries") == len(expected)
        and run.get("accepted_queries") == len(expected)
        and {row.get("key") for row in rows} == expected and len(rows) == len(expected)
        and all(row.get("valid") is True for row in rows)
    )


def run(args: argparse.Namespace) -> int:
    authorization, verification, host, evidence_paths, model_snapshot, live_host = verify_authorization(args)
    out = args.output_root.resolve()
    plan_root = out / "plans"
    shard_root = out / "shards"
    state_path = out / "full_run_state.json"
    recovery = authorization.get("recovery", {"mode": "fresh"})
    mode = recovery.get("mode", "fresh")
    if mode == "fresh":
        if out.exists():
            raise SystemExit("fresh full-reranker authorization refuses an existing output root")
        out.mkdir(parents=True)
        plan_root.mkdir(parents=True)
        state = expected_state(authorization, args, plan_root)
        (out / "provenance").mkdir()
        shutil.copyfile(args.authorization, out / "provenance/full_authorization.json")
        shutil.copyfile(args.pilot_verification, out / "provenance/pilot_verification.json")
        for name, path in evidence_paths.items():
            shutil.copyfile(path, out / f"provenance/{name}_evidence.json")
        write_json(out / "provenance/full_host_provenance.json", live_host)
        (out / "provenance/reranker_model_snapshot.json").write_text(
            json.dumps(model_snapshot, indent=2, sort_keys=True) + "\n"
        )
    elif mode == "resume-missing-keys":
        if not out.is_dir() or not state_path.is_file():
            raise SystemExit("recovery authorization requires the existing full-run state checkpoint")
        if sha(state_path) != recovery.get("parent_state_sha256"):
            raise SystemExit("recovery authorization does not bind the current checkpoint state hash")
        state = json.loads(state_path.read_text())
        if (state.get("run_id") != authorization.get("run_id")
                or state.get("execution_sha") != args.execution_sha
                or state.get("pilot_verification_sha256") != sha(args.pilot_verification)
                or state.get("input_sha256") != INPUT_HASHES):
            raise SystemExit("recovery checkpoint does not match the authorized full run")
        parent_state_copy = out / f"provenance/checkpoint_state_{recovery['parent_state_sha256']}.json"
        if parent_state_copy.exists():
            raise SystemExit("refusing to overwrite an existing parent checkpoint receipt")
        shutil.copyfile(state_path, parent_state_copy)
        plan_root = Path(state["plan_root"])
        if not plan_root.is_dir():
            raise SystemExit("recovery checkpoint is missing the original shard plan")
        recovery_auth_path = out / "provenance" / f"recovery_authorization_{sha(args.authorization)}.json"
        if recovery_auth_path.exists():
            raise SystemExit("refusing to overwrite a prior recovery authorization receipt")
        write_json(recovery_auth_path, authorization)
        state["recovery_authorization_sha256"] = sha(args.authorization)
        state["authorization_sha256"] = sha(args.authorization)
        write_json(out / "provenance/full_host_provenance.json", live_host)
    else:
        raise SystemExit("recovery mode must be fresh or resume-missing-keys")

    if mode == "fresh":
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/plan_v5_cpu_rerank_shards.py"),
            "--queries", str(args.input_root / "queries.jsonl"),
            "--corpus", str(args.input_root / "corpus.jsonl"),
            "--candidates", str(args.input_root / "rerank_candidates.jsonl"),
            "--queries-per-shard", "20", "--output-root", str(plan_root),
        ], cwd=ROOT, check=False)
        if result.returncode != 0:
            state.update({"status": "failed-before-result", "failure": "shard planning failed"})
            write_json(state_path, state)
            return result.returncode
    index_path = plan_root / "index.json"
    if not index_path.is_file():
        raise SystemExit("full reranker shard index missing")
    index = json.loads(index_path.read_text())
    if index.get("shard_count") != 12 or index.get("expected_total_queries") != 240:
        raise SystemExit("planned full reranker keyspace is not 12 x 20 = 240 queries")
    state["plan_index_sha256"] = sha(index_path)
    state["result_bearing_execution_started"] = any(
        (shard_root / f"shard-{i:03d}" / "result_bearing_started.json").is_file()
        for i in range(12)
    )
    write_json(state_path, state)
    plans = [plan_root / f"shard-{i:03d}.json" for i in range(12)]
    if any(not p.is_file() for p in plans):
        raise SystemExit("full reranker plan is missing one or more shard manifests")
    plan_hashes = {str(i): sha(plans[i]) for i in range(12)}
    completed = set(state.get("completed_shards", []))
    allowed_resume = set(recovery.get("approved_shard_indexes", [])) if mode != "fresh" else set(range(12))
    shard_root.mkdir(parents=True, exist_ok=True)

    if mode != "fresh":
        incomplete = {
            i for i, plan_path in enumerate(plans)
            if i not in completed or not validate_existing_shard(
                shard_root / f"shard-{i:03d}", json.loads(plan_path.read_text()), INPUT_HASHES
            )
        }
        if allowed_resume != incomplete:
            raise SystemExit(
                "recovery authorization must approve exactly every incomplete shard; "
                f"needed={sorted(incomplete)}, approved={sorted(allowed_resume)}"
            )

    for index_number, plan_path in enumerate(plans):
        shard = json.loads(plan_path.read_text())
        shard_dir = shard_root / f"shard-{index_number:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        if index_number in completed and validate_existing_shard(shard_dir, shard, INPUT_HASHES):
            continue
        if mode != "fresh" and index_number not in allowed_resume:
            raise SystemExit(f"recovery manifest did not approve incomplete shard {index_number:03d}")
        if mode != "fresh" and shard_dir.exists() and any(shard_dir.iterdir()) and index_number not in allowed_resume:
            raise SystemExit(f"partial shard {index_number:03d} requires explicit missing-key-only recovery authorization")
        shard_auth = {
            "schema_version": 1,
            "manifest_id": f"sem2act-v5-moon-reranker-derived-shard-auth-{index_number:03d}",
            "status": "authorized-after-gates",
            "execution_sha": args.execution_sha,
            "protocol_hash": PROTOCOL_HASH,
            "runtime_lock_sha256": sha(LOCK),
            "runtime_freeze_sha256": sha(FREEZE),
            "thread_count": 16,
            "parity_status": "pass",
            "host_preflight": {"path": str(evidence_paths["host"]), "sha256": sha(evidence_paths["host"])},
            "thread_canary": {"path": str(evidence_paths["parity"]), "sha256": sha(evidence_paths["parity"])},
            "input_manifest_sha256": sha(INPUT_MANIFEST),
            "input_files": INPUT_HASHES,
            "model_snapshot_sha256": model_snapshot["files_sha256"],
            "pilot_shard_sha256": plan_hashes[str(index_number)],
            "accepted_pilot_verification_sha256": sha(args.pilot_verification),
            "result_bearing_execution_started": False,
            "qrels_read": False,
        }
        auth_path = shard_dir / "moon_shard_authorization.json"
        write_json(auth_path, shard_auth)
        command = [
            sys.executable, str(ROOT / "scripts/run_v5_cpu_rerank_shard.py"),
            "--moon-pilot-manifest", str(auth_path),
            "--shard", str(plan_path),
            "--queries", str(args.input_root / "queries.jsonl"),
            "--corpus", str(args.input_root / "corpus.jsonl"),
            "--candidates", str(args.input_root / "rerank_candidates.jsonl"),
            "--model-path", str(args.model_path), "--output-dir", str(shard_dir),
        ]
        child = subprocess.run(command, cwd=ROOT, check=False)
        if child.returncode != 0 or not validate_existing_shard(shard_dir, shard, INPUT_HASHES):
            state.update({
                "status": "failed-stop",
                "failed_shard_index": index_number,
                "failed_shard_manifest_sha256": plan_hashes[str(index_number)],
                "completed_shards": sorted(completed),
                "result_bearing_execution_started": state.get("result_bearing_execution_started") or
                    (shard_dir / "result_bearing_started.json").is_file(),
            })
            write_json(state_path, state)
            return child.returncode or 1
        completed.add(index_number)
        state.update({
            "status": "in-progress",
            "completed_shards": sorted(completed),
            "completed_shard_output_sha256": {
                str(i): file_map(shard_root / f"shard-{i:03d}")
                for i in sorted(completed)
            },
            "result_bearing_execution_started": True,
        })
        write_json(state_path, state)

    all_rows = []
    for i, plan_path in enumerate(plans):
        shard = json.loads(plan_path.read_text())
        shard_dir = shard_root / f"shard-{i:03d}"
        if not validate_existing_shard(shard_dir, shard, INPUT_HASHES):
            raise SystemExit(f"completed shard {i:03d} no longer passes its checkpoint")
        all_rows.extend(json.loads(line) for line in (shard_dir / "rankings.jsonl").read_text().splitlines() if line.strip())
    all_rows.sort(key=lambda row: row["query_id"])
    if len(all_rows) != 240 or len({row["query_id"] for row in all_rows}) != 240:
        raise SystemExit("merged reranker rankings do not cover the exact 240 query IDs")
    rankings_path = out / "rankings.jsonl"
    rankings_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    trec_path = out / "rerank.trec"
    lines = []
    for row in all_rows:
        for rank, doc_id in enumerate(row["ranking"], 1):
            # The frozen consumer reads TREC order and does not use score values.
            # Native score payloads remain auditable through scores_sha256 per query.
            rank_encoding = 51 - rank
            lines.append(f"{row['query_id']} Q0 {doc_id} {rank} {rank_encoding} sem2act-v5-moon-reranker\n")
    trec_path.write_text("".join(lines))
    failures_path = out / "failures.json"
    write_json(failures_path, {"failures": [], "n_fail": 0})
    state.update({"status": "complete", "completed_shards": list(range(12)),
                  "result_bearing_execution_started": True, "qrels_read": False})
    write_json(state_path, state)
    outputs = file_map(out, omit={"full_run_manifest.json"})
    final = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-moon-full-reranker-run-v1",
        "status": "pass",
        "run_id": authorization["run_id"],
        "execution_sha": args.execution_sha,
        "pilot_execution_sha": PILOT_SHA,
        "pilot_verification_sha256": sha(args.pilot_verification),
        "authorization_sha256": sha(args.authorization),
        "protocol_hash": PROTOCOL_HASH,
        "runtime_lock_sha256": sha(LOCK),
        "runtime_freeze_sha256": sha(FREEZE),
        "hostname": live_host["hostname"],
        "full_host_provenance_sha256": sha(out / "provenance/full_host_provenance.json"),
        "input_manifest_sha256": sha(INPUT_MANIFEST),
        "input_sha256": INPUT_HASHES,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": "cpu",
        "dtype": "float32",
        "thread_count": 16,
        "expected_queries": 240,
        "accepted_queries": 240,
        "candidate_depth": 50,
        "expected_candidate_scores": 12000,
        "shard_count": 12,
        "queries_per_shard": 20,
        "n_fail": 0,
        "qrels_read": False,
        "result_bearing_execution_started": True,
        "checkpoint_policy": "one immutable 20-query shard directory per NFS checkpoint; stop on any failed shard",
        "trec_score_semantics": "strictly monotonic rank encoding for downstream order-only assembler; native scorer values are represented by per-query scores_sha256 in rankings.jsonl",
        "output_sha256": outputs,
        "stop_after_this_stage": True,
        "next_action": "run scripts/verify_v5_moon_full_reranker.py and return to Codex for explicit acceptance",
    }
    write_json(out / "full_run_manifest.json", final)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--pilot-verification", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
