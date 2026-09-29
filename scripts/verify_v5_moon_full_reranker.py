"""Verify completed 240-query Moon reranker output and its checkpoint lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
EXPECTED_QIDS = [f"v5-lb-q{i:04d}" for i in range(1, 241)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_map(root: Path, verifier_output: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() != (root / "full_run_manifest.json").resolve()
        and path.resolve() != verifier_output.resolve()
    }


def verify(args: argparse.Namespace) -> dict:
    fail_reasons: list[dict[str, str]] = []
    review_reasons: list[dict[str, str]] = []
    checks: dict[str, object] = {}

    def fail(code: str, message: str) -> None:
        fail_reasons.append({"code": code, "message": message})

    def review(code: str, message: str) -> None:
        review_reasons.append({"code": code, "message": message})

    root, inputs = args.run_root.resolve(), args.input_root.resolve()
    manifest_path = root / "full_run_manifest.json"
    if not manifest_path.is_file():
        return result(args, "REVIEW", fail_reasons,
                      [{"code": "manifest_missing", "message": "full_run_manifest.json is missing"}], checks)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "pass":
        fail("run_status", f"full reranker status is {manifest.get('status')!r}")
    if manifest.get("execution_sha") != args.execution_sha:
        fail("execution_sha", "full reranker result is not bound to the expected execution SHA")
    if manifest.get("pilot_execution_sha") != PILOT_SHA:
        fail("pilot_sha", "full reranker result is not bound to the canonical pilot SHA")
    if manifest.get("protocol_hash") != PROTOCOL_HASH:
        fail("protocol_hash", "full reranker protocol hash drift")
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        fail("model_revision", "full reranker model/revision drift")
    if manifest.get("device") != "cpu" or manifest.get("dtype") != "float32" or manifest.get("thread_count") != 16:
        fail("runtime_policy", "Moon full-reranker CPU/float32/16-thread policy drift")
    exact_counts = {
        "expected_queries": 240, "accepted_queries": 240,
        "candidate_depth": 50, "expected_candidate_scores": 12000,
        "shard_count": 12, "queries_per_shard": 20, "n_fail": 0,
    }
    for key, expected in exact_counts.items():
        if manifest.get(key) != expected:
            fail("scope", f"full reranker {key}={manifest.get(key)!r}, expected {expected}")
    if manifest.get("qrels_read") is not False or manifest.get("result_bearing_execution_started") is not True:
        fail("qrels_or_execution", "full reranker must prove qrels_read=false and result-bearing execution started")
    if manifest.get("stop_after_this_stage") is not True:
        fail("stop_gate", "full reranker manifest must require STOP after the 240-query stage")

    actual_inputs = {}
    for name, expected in sorted(INPUT_HASHES.items()):
        path = inputs / name
        if not path.is_file():
            fail("input_missing", f"missing frozen input {name}")
            continue
        digest = sha(path)
        actual_inputs[name] = digest
        if digest != expected:
            fail("input_hash", f"frozen input hash mismatch for {name}")
    if set(actual_inputs) != set(INPUT_HASHES):
        return result(args, "FAIL", fail_reasons, review_reasons, checks)
    if {p.name for p in inputs.iterdir() if p.is_file()} != set(INPUT_HASHES):
        fail("input_file_set", "Moon input root must contain exactly the three frozen files")
    for name in INPUT_HASHES:
        if "qrel" in (inputs / name).read_text(errors="ignore").lower():
            fail("qrels_input", f"qrel token detected in frozen input {name}")
    if manifest.get("input_sha256") != INPUT_HASHES:
        fail("input_manifest", "full run manifest input hash contract drift")
    lock_path = ROOT / "versions/sem2act-v5/manifests/moon_reranker_runtime_lock.json"
    freeze_path = ROOT / "versions/sem2act-v5/manifests/moon_reranker_runtime_freeze.json"
    input_manifest_path = ROOT / "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json"
    if (manifest.get("runtime_lock_sha256") != sha(lock_path)
            or manifest.get("runtime_freeze_sha256") != sha(freeze_path)):
        fail("runtime_manifest_hash", "full-run Moon runtime lock/freeze hashes drift")
    if (not input_manifest_path.is_file()
            or manifest.get("input_manifest_sha256") != sha(input_manifest_path)):
        fail("input_manifest_hash", "full run does not bind the frozen source dataset manifest")
    checks["input_sha256"] = actual_inputs

    auth_copy = root / "provenance/full_authorization.json"
    verify_copy = root / "provenance/pilot_verification.json"
    if not verify_copy.is_file():
        review("pilot_verification_copy_missing", "full run does not include the accepted pilot verifier receipt")
    else:
        pilot_verification = json.loads(verify_copy.read_text())
        if (pilot_verification.get("status") != "PASS"
                or pilot_verification.get("execution_sha") != PILOT_SHA
                or sha(verify_copy) != manifest.get("pilot_verification_sha256")):
            fail("pilot_verification", "full reranker pilot verifier receipt is not the accepted exact-SHA PASS")
    if not auth_copy.is_file():
        review("authorization_copy_missing", "full run does not include its full-run authorization record")
    else:
        auth = json.loads(auth_copy.read_text())
        if auth.get("status") != "authorized" or auth.get("execution_sha") != args.execution_sha:
            fail("authorization", "full run authorization is missing or bound to another SHA")
        if (auth.get("pilot_execution_sha") != PILOT_SHA
                or auth.get("pilot_verification_sha256") != manifest.get("pilot_verification_sha256")
                or auth.get("protocol_hash") != PROTOCOL_HASH
                or auth.get("input_sha256") != INPUT_HASHES
                or auth.get("model", {}).get("model_id") != MODEL_ID
                or auth.get("model", {}).get("revision") != MODEL_REVISION
                or auth.get("scope", {}).get("queries") != 240
                or auth.get("scope", {}).get("candidate_scores") != 12000
                or auth.get("scope", {}).get("threads") != 16
                or auth.get("scope", {}).get("scientific_protocol_changed") is not False):
            fail("authorization_scope", "full authorization does not bind the accepted pilot and frozen reranker scope")
        if auth.get("paths", {}).get("input_root") != str(inputs):
            fail("authorization_input_path", "full authorization input path differs from the verified input root")
        if auth.get("paths", {}).get("output_root") != str(root):
            fail("authorization_output_path", "full authorization output path differs from the verified run root")
        if auth.get("paths", {}).get("model_path") != "/tmp/sem2act-v5/models/qwen3-reranker-0.6b":
            fail("authorization_model_path", "full authorization model path differs from the frozen node-local path")

    full_host_path = root / "provenance/full_host_provenance.json"
    pilot_identity_path = root / "provenance/identity_evidence.json"
    if not full_host_path.is_file() or not pilot_identity_path.is_file():
        review("host_provenance_missing", "full-run and accepted-pilot hostname provenance are required")
    else:
        full_host = json.loads(full_host_path.read_text())
        pilot_identity = json.loads(pilot_identity_path.read_text())
        if (full_host.get("status") != "pass"
                or full_host.get("execution_sha") != args.execution_sha
                or full_host.get("hostname") != pilot_identity.get("hostname")
                or full_host.get("qrels_read") is not False
                or full_host.get("runtime_lock_sha256") != manifest.get("runtime_lock_sha256")
                or full_host.get("runtime_freeze_sha256") != manifest.get("runtime_freeze_sha256")
                or full_host.get("python") != "3.9.19"
                or full_host.get("torch") != "2.8.0+cpu"
                or full_host.get("transformers") != "4.57.6"
                or full_host.get("torch_cuda_available") is not False
                or full_host.get("logical_vcpus") != 16
                or int(full_host.get("memory_bytes", 0)) < 60_000_000_000
                or full_host.get("avx2_masked") is not True
                or full_host.get("avx512f_masked") is not True
                or not str(full_host.get("output_filesystem", "")).startswith("nfs")
                or int(full_host.get("nfs_quota_probe_bytes_written_and_removed", 0)) < 10_000_000
                or full_host.get("input_sha256") != INPUT_HASHES
                or full_host.get("model_id") != MODEL_ID
                or full_host.get("revision") != MODEL_REVISION):
            fail("host_provenance", "full-run host receipt differs from the accepted same-host CPU/runtime profile")
        if (manifest.get("hostname") != full_host.get("hostname")
                or manifest.get("full_host_provenance_sha256") != sha(full_host_path)):
            fail("host_provenance_hash", "full-run manifest hostname/host receipt hash drift")

    queries = [json.loads(line) for line in (inputs / "queries.jsonl").read_text().splitlines() if line.strip()]
    candidate_rows = [json.loads(line) for line in (inputs / "rerank_candidates.jsonl").read_text().splitlines() if line.strip()]
    qids = [row.get("_id") for row in queries]
    candidates = {row.get("query_id"): row.get("doc_ids", []) for row in candidate_rows}
    if qids != EXPECTED_QIDS or len(candidates) != 240 or set(candidates) != set(EXPECTED_QIDS):
        fail("input_keyspace", "full reranker inputs do not have the exact ordered 240 query keys")
    if any(len(candidates[qid]) != 50 or len(set(candidates[qid])) != 50 for qid in EXPECTED_QIDS):
        fail("candidate_depth", "every query must have exactly 50 unique candidates")

    plan_root = root / "plans"
    index_path = plan_root / "index.json"
    if not index_path.is_file():
        fail("plan_missing", "full run has no persisted 12-shard plan")
        index = {}
    else:
        index = json.loads(index_path.read_text())
        if index.get("shard_count") != 12 or index.get("expected_total_queries") != 240:
            fail("plan_index", "persisted plan is not exactly 12 disjoint 20-query shards")
    all_rankings = []
    for shard_index in range(12):
        plan_path = plan_root / f"shard-{shard_index:03d}.json"
        shard_dir = root / "shards" / f"shard-{shard_index:03d}"
        if not plan_path.is_file():
            fail("shard_plan_missing", f"missing shard plan {shard_index:03d}")
            continue
        shard = json.loads(plan_path.read_text())
        expected_qids = EXPECTED_QIDS[shard_index * 20:(shard_index + 1) * 20]
        if shard.get("shard_index") != shard_index or shard.get("query_ids") != expected_qids or shard.get("expected_queries") != 20:
            fail("shard_plan_scope", f"shard {shard_index:03d} query scope drift")
        run_path = shard_dir / "run_manifest.json"
        rankings_path = shard_dir / "rankings.jsonl"
        failures_path = shard_dir / "failures.json"
        start_path = shard_dir / "result_bearing_started.json"
        if not all(path.is_file() for path in (run_path, rankings_path, failures_path, start_path)):
            fail("shard_files_missing", f"shard {shard_index:03d} output set is incomplete")
            continue
        run = json.loads(run_path.read_text())
        shard_failures = json.loads(failures_path.read_text())
        shard_rows = [json.loads(line) for line in rankings_path.read_text().splitlines() if line.strip()]
        if (run.get("status") != "pass" or run.get("expected_queries") != 20
                or run.get("accepted_queries") != 20 or run.get("n_fail") != 0
                or run.get("execution_sha") != args.execution_sha or run.get("qrels_read") is not False):
            fail("shard_run_manifest", f"shard {shard_index:03d} run manifest failed or drifted")
        if (run.get("model_id") != MODEL_ID or run.get("revision") != MODEL_REVISION
                or run.get("device") != "cpu" or run.get("dtype") != "float32"
                or run.get("runtime_profile") != "moon-reranker-only-v1"
                or run.get("runtime_lock_sha256") != manifest.get("runtime_lock_sha256")
                or run.get("runtime_freeze_sha256") != manifest.get("runtime_freeze_sha256")):
            fail("shard_runtime", f"shard {shard_index:03d} runtime/model provenance drift")
        host_evidence_path = root / "provenance/host_evidence.json"
        parity_evidence_path = root / "provenance/parity_evidence.json"
        if (not host_evidence_path.is_file() or not parity_evidence_path.is_file()
                or run.get("host_preflight_sha256") != sha(host_evidence_path)
                or run.get("thread_canary_sha256") != sha(parity_evidence_path)):
            fail("shard_gate_provenance", f"shard {shard_index:03d} is not bound to accepted pilot host/parity evidence")
        if shard_failures != []:
            fail("shard_failures", f"shard {shard_index:03d} failures.json is not empty")
        start_marker = json.loads(start_path.read_text())
        if (start_marker.get("qrels_read") is not False
                or start_marker.get("execution_sha") != args.execution_sha
                or start_marker.get("protocol_hash") != PROTOCOL_HASH):
            fail("shard_start_provenance", f"shard {shard_index:03d} start marker does not prove qrel-free execution")
        expected_keys = {f"reranker|{qid}" for qid in expected_qids}
        if len(shard_rows) != 20 or {row.get("key") for row in shard_rows} != expected_keys:
            fail("shard_coverage", f"shard {shard_index:03d} ranking keyspace is incomplete")
        for row in shard_rows:
            qid, ranking = row.get("query_id"), row.get("ranking", [])
            if (row.get("valid") is not True or qid not in expected_qids
                    or len(ranking) != 50 or len(set(ranking)) != 50
                    or set(ranking) != set(candidates.get(qid, []))):
                fail("shard_ranking_schema", f"shard {shard_index:03d}, query {qid}: candidate coverage drift")
            if not re.fullmatch(r"[0-9a-f]{64}", row.get("scores_sha256", "")):
                fail("shard_score_hash", f"shard {shard_index:03d}, query {qid}: missing native score digest")
            expected_rank_hash = hashlib.sha256(
                json.dumps(ranking, separators=(",", ":")).encode()
            ).hexdigest()
            if row.get("ranking_sha256") != expected_rank_hash:
                fail("shard_ranking_hash", f"shard {shard_index:03d}, query {qid}: ranking digest drift")
        all_rankings.extend(shard_rows)

    all_rankings.sort(key=lambda row: row.get("query_id", ""))
    if len(all_rankings) != 240 or [row.get("query_id") for row in all_rankings] != EXPECTED_QIDS:
        fail("full_coverage", "merged rankings do not cover exactly 240 ordered queries")
    merged_path = root / "rankings.jsonl"
    trec_path = root / "rerank.trec"
    if not merged_path.is_file() or not trec_path.is_file():
        fail("aggregate_outputs_missing", "merged rankings.jsonl and rerank.trec are required")
    else:
        merged_rows = [json.loads(line) for line in merged_path.read_text().splitlines() if line.strip()]
        if merged_rows != all_rankings:
            fail("merged_rankings", "aggregate rankings.jsonl differs from accepted shard rankings")
        expected_trec = []
        for row in all_rankings:
            for rank, did in enumerate(row["ranking"], 1):
                expected_trec.append(f"{row['query_id']} Q0 {did} {rank} {51-rank} sem2act-v5-moon-reranker")
        actual_trec = trec_path.read_text().splitlines()
        if actual_trec != expected_trec:
            fail("trec_order", "rerank.trec does not preserve exact ranking order and documented rank encoding")
        if len(actual_trec) != 12000:
            fail("trec_count", f"expected 12,000 TREC rows, got {len(actual_trec)}")
        checks["source_reranker_output_sha256"] = sha(trec_path)

    state_path = root / "full_run_state.json"
    state = {}
    if not state_path.is_file():
        fail("checkpoint_state_missing", "full run checkpoint state is missing")
    else:
        state = json.loads(state_path.read_text())
        if state.get("status") != "complete" or state.get("completed_shards") != list(range(12)):
            fail("checkpoint_state", "checkpoint state does not record all 12 completed shards")
        if state.get("qrels_read") is not False:
            fail("checkpoint_qrels", "checkpoint state does not prove qrels_read=false")
        if state.get("authorization_sha256") != manifest.get("authorization_sha256"):
            fail("checkpoint_authorization", "final checkpoint does not bind the run manifest's latest authorization")
    failure_path = root / "failures.json"
    if not failure_path.is_file():
        fail("failures_manifest_missing", "aggregate failures.json is missing")
    elif json.loads(failure_path.read_text()) != {"failures": [], "n_fail": 0}:
        fail("aggregate_failures", "aggregate failures.json does not report zero failures")

    expected_outputs = manifest.get("output_sha256", {})
    actual_outputs = output_map(root, args.output)
    if expected_outputs != actual_outputs:
        fail("output_hash_manifest", "full run manifest file hash map is incomplete or differs from output artifacts")
    expected_paths = {
        "full_run_state.json", "rankings.jsonl", "rerank.trec", "failures.json",
        "plans/index.json",
        *(f"plans/shard-{i:03d}.json" for i in range(12)),
        "provenance/full_authorization.json", "provenance/pilot_verification.json",
        "provenance/host_evidence.json", "provenance/parity_evidence.json",
        "provenance/identity_evidence.json", "provenance/full_host_provenance.json",
        "provenance/reranker_model_snapshot.json",
        *(f"shards/shard-{i:03d}/{name}" for i in range(12) for name in (
            "moon_shard_authorization.json", "run_manifest.json", "rankings.jsonl",
            "failures.json", "result_bearing_started.json")),
    }
    recovery_files = sorted((root / "provenance").glob("recovery_authorization_*.json"))
    recovery_receipts: dict[str, dict] = {}
    recovery_parents: set[str] = set()
    for recovery_file in recovery_files:
        recovery_sha = sha(recovery_file)
        expected_name = f"recovery_authorization_{recovery_sha}.json"
        if recovery_file.name != expected_name:
            fail("recovery_receipt_name", f"recovery authorization receipt is not named by its SHA-256: {recovery_file.name}")
        recovery_auth = json.loads(recovery_file.read_text())
        recovery = recovery_auth.get("recovery", {})
        parent_sha = recovery.get("parent_state_sha256")
        parent_path = root / "provenance" / f"checkpoint_state_{parent_sha}.json"
        if (recovery_auth.get("status") != "authorized"
                or recovery_auth.get("execution_sha") != args.execution_sha
                or recovery_auth.get("pilot_execution_sha") != PILOT_SHA
                or recovery_auth.get("pilot_verification_sha256") != manifest.get("pilot_verification_sha256")
                or recovery_auth.get("protocol_hash") != PROTOCOL_HASH
                or recovery_auth.get("input_sha256") != INPUT_HASHES
                or recovery_auth.get("scope", {}).get("queries") != 240
                or recovery_auth.get("scope", {}).get("candidate_scores") != 12000
                or recovery_auth.get("scope", {}).get("threads") != 16
                or recovery.get("mode") != "resume-missing-keys"
                or not re.fullmatch(r"[0-9a-f]{64}", str(parent_sha))
                or not parent_path.is_file() or sha(parent_path) != parent_sha):
            fail("recovery_authorization", f"recovery receipt {recovery_file.name} or its exact parent checkpoint is invalid")
        else:
            parent_state = json.loads(parent_path.read_text())
            completed_at_parent = parent_state.get("completed_shards")
            approved = recovery.get("approved_shard_indexes")
            if not isinstance(completed_at_parent, list) or any(
                    not isinstance(i, int) or i < 0 or i > 11 for i in completed_at_parent):
                fail("recovery_parent_checkpoint", f"parent checkpoint for {recovery_file.name} has invalid completed_shards")
            if not isinstance(approved, list):
                fail("recovery_shard_scope", f"recovery receipt {recovery_file.name} has no shard allowlist")
            elif (any(not isinstance(i, int) or i < 0 or i > 11 for i in approved)
                    or len(set(approved)) != len(approved)
                    or (isinstance(completed_at_parent, list)
                        and sorted(approved) != sorted(set(range(12)) - set(completed_at_parent)))):
                fail("recovery_shard_scope", f"recovery receipt {recovery_file.name} has an invalid shard allowlist")
            if receipt_parent := parent_state.get("authorization_sha256"):
                if receipt_parent in recovery_parents:
                    fail("recovery_chain_branch", "multiple recovery receipts fork from the same authorization checkpoint")
                recovery_parents.add(receipt_parent)
            recovery_receipts[recovery_sha] = {
                "auth": recovery_auth,
                "parent_authorization_sha256": receipt_parent,
                "parent_state_sha256": parent_sha,
            }
            expected_paths.add(f"provenance/{recovery_file.name}")
            expected_paths.add(f"provenance/checkpoint_state_{parent_sha}.json")

    base_auth_sha = sha(auth_copy) if auth_copy.is_file() else None
    authorization_chain = {base_auth_sha} if base_auth_sha else set()
    remaining = dict(recovery_receipts)
    progressed = True
    while progressed:
        progressed = False
        for recovery_sha, receipt in list(remaining.items()):
            if receipt["parent_authorization_sha256"] in authorization_chain:
                authorization_chain.add(recovery_sha)
                del remaining[recovery_sha]
                progressed = True
    if remaining:
        fail("recovery_chain", "one or more recovery receipts do not continue the base authorization/checkpoint chain")
    latest_authorization_sha = manifest.get("authorization_sha256")
    if latest_authorization_sha not in authorization_chain:
        fail("authorization_hash", "run manifest authorization SHA is not present in the validated authorization chain")
    if state.get("recovery_authorization_sha256") != (None if latest_authorization_sha == base_auth_sha else latest_authorization_sha):
        fail("checkpoint_recovery_authorization", "checkpoint does not identify the latest recovery authorization consistently")
    if latest_authorization_sha != base_auth_sha and recovery_files:
        if not any(sha(path) == latest_authorization_sha for path in recovery_files):
            fail("authorization_hash", "run manifest authorization SHA does not identify a saved recovery authorization")
    if set(actual_outputs) != expected_paths:
        fail("output_manifest_completeness", "full run output tree contains missing or unmanifested artifacts")
    checks["output_sha256"] = actual_outputs
    checks["full_run_manifest_sha256"] = sha(manifest_path)
    checks["expected_queries"] = 240
    checks["expected_candidate_scores"] = 12000
    checks["shard_count"] = 12
    checks["qrels_read"] = False
    status = "FAIL" if fail_reasons else "REVIEW" if review_reasons else "PASS"
    return result(args, status, fail_reasons, review_reasons, checks)


def result(args: argparse.Namespace, status: str, fails: list[dict],
           reviews: list[dict], checks: dict) -> dict:
    return {
        "schema_version": 1,
        "verifier": "sem2act-v5-moon-full-reranker",
        "status": status,
        "execution_sha": args.execution_sha,
        "pilot_execution_sha": PILOT_SHA,
        "protocol_hash": PROTOCOL_HASH,
        "reasons": fails + reviews,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        verified = verify(args)
    except Exception as exc:
        verified = result(args, "FAIL", [{
            "code": "verifier_exception",
            "message": f"{type(exc).__name__}: {exc}",
        }], [], {})
    payload = json.dumps(verified, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")
    return {"PASS": 0, "FAIL": 1, "REVIEW": 2}[verified["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
