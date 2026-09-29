"""Deterministically verify the single authorized Moon reranker pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXECUTION_SHA = "6d97cd0fe11cc87cd38661ac3a3646ebb36b7599"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INPUT_HASHES = {
    "corpus.jsonl": "6cab24bc7420635674856f35345cd4c5c30a23d2d0ee243a79b69d36bc8ab0e1",
    "queries.jsonl": "c457d93ea195d82a738219ad152382380d89b4b5e0c3fabde5cf4d6d33434986",
    "rerank_candidates.jsonl": "11dcac88ad210447ac16f78decbb5dd09776648e14b083aa3d89f67c8b701803",
}
LOCK_REL = "versions/sem2act-v5/manifests/moon_reranker_runtime_lock.json"
FREEZE_REL = "versions/sem2act-v5/manifests/moon_reranker_runtime_freeze.json"
INPUT_MANIFEST_REL = "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def git_blob_sha(commit: str, relative: str) -> str:
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest()


def planned_pilot_shard_sha(inputs: dict[str, str]) -> str:
    qids = [f"v5-lb-q{i:04d}" for i in range(1, 21)]
    shard = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-cpu-reranker-shard-000",
        "status": "planned-before-result-bearing-execution",
        "protocol_hash": PROTOCOL_HASH,
        "stage": "reranker",
        "shard_index": 0,
        "query_ids": qids,
        "expected_query_keys": [f"reranker|{qid}" for qid in qids],
        "expected_queries": 20,
        "candidate_depth": 50,
        "queries_sha256": inputs["queries.jsonl"],
        "corpus_sha256": inputs["corpus.jsonl"],
        "candidates_sha256": inputs["rerank_candidates.jsonl"],
        "resume_policy": "missing-key-only",
    }
    payload = json.dumps(shard, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def verify(args: argparse.Namespace) -> dict:
    reasons: list[dict[str, str]] = []
    review_reasons: list[dict[str, str]] = []
    checks: dict[str, object] = {}

    def fail(code: str, message: str) -> None:
        reasons.append({"code": code, "message": message})

    def review(code: str, message: str) -> None:
        review_reasons.append({"code": code, "message": message})

    if args.execution_sha != EXPECTED_EXECUTION_SHA:
        fail("execution_sha", f"expected {EXPECTED_EXECUTION_SHA}, got {args.execution_sha}")

    input_root = args.input_root.resolve()
    actual_input_hashes: dict[str, str] = {}
    for name, expected in sorted(INPUT_HASHES.items()):
        path = input_root / name
        if not path.is_file():
            fail("input_missing", f"missing input file: {name}")
            continue
        actual = sha256(path)
        actual_input_hashes[name] = actual
        if actual != expected:
            fail("input_hash", f"{name} SHA-256 does not match the frozen input bundle")
    if input_root.is_dir():
        actual_names = {p.name for p in input_root.iterdir() if p.is_file()}
        if actual_names != set(INPUT_HASHES):
            fail("input_file_set", f"expected only {sorted(INPUT_HASHES)}, got {sorted(actual_names)}")
    checks["input_sha256"] = actual_input_hashes
    if reasons:
        return report(args, "FAIL", reasons, review_reasons, checks)

    pilot_dir = args.pilot_dir.resolve()
    required = {
        "moon_preflight.json", "moon_reranker_canary.json", "pilot_authorization.json",
        "result_bearing_started.json", "rankings.jsonl", "failures.json", "run_manifest.json",
        "pilot_manifest.json",
    }
    missing = sorted(name for name in required if not (pilot_dir / name).is_file())
    if missing:
        review("pilot_files_missing", f"pilot output is incomplete: {missing}")
        return report(args, "FAIL" if reasons else "REVIEW", reasons, review_reasons, checks)

    pilot = read_json(pilot_dir / "pilot_manifest.json")
    host = read_json(pilot_dir / "moon_preflight.json")
    parity = read_json(pilot_dir / "moon_reranker_canary.json")
    authorization = read_json(pilot_dir / "pilot_authorization.json")
    started = read_json(pilot_dir / "result_bearing_started.json")
    run = read_json(pilot_dir / "run_manifest.json")
    failures = read_json(pilot_dir / "failures.json")

    for label, record in (("pilot_manifest", pilot), ("moon_preflight", host),
                          ("thread_canary", parity), ("pilot_authorization", authorization),
                          ("result_bearing_started", started), ("run_manifest", run)):
        if record.get("execution_sha") != EXPECTED_EXECUTION_SHA:
            fail(f"{label}_execution_sha", f"{label} does not bind the canonical execution SHA")
        if record.get("protocol_hash") != PROTOCOL_HASH:
            fail(f"{label}_protocol", f"{label} protocol hash drift")
        if record.get("qrels_read") is not False:
            fail(f"{label}_qrels", f"{label} does not prove qrels_read=false")

    if pilot.get("status") != "pass":
        fail("pilot_status", f"pilot status is {pilot.get('status')!r}, expected 'pass'")
    if pilot.get("result_bearing_execution_started") is not True or started.get("status") != "started":
        fail("result_marker", "result-bearing start marker is absent or inconsistent")
    if pilot.get("stop_after_this_pilot") is not True:
        fail("stop_gate", "pilot manifest does not require STOP after this pilot")
    if pilot.get("expected_queries") != 20 or pilot.get("expected_candidate_scores") != 1000:
        fail("pilot_scope", "pilot manifest does not authorize exactly 20 queries / 1,000 scores")

    lock_path = ROOT / LOCK_REL
    freeze_path = ROOT / FREEZE_REL
    input_manifest_path = ROOT / INPUT_MANIFEST_REL
    for relative in (LOCK_REL, FREEZE_REL, INPUT_MANIFEST_REL,
                     "versions/sem2act-v5/manifests/cpu_reranker_canary.json",
                     "scripts/run_v5_cpu_rerank_shard.py",
                     "scripts/run_v5_moon_reranker_pilot.py"):
        path = ROOT / relative
        try:
            canonical_blob_sha = git_blob_sha(EXPECTED_EXECUTION_SHA, relative)
        except subprocess.CalledProcessError:
            fail("canonical_commit_missing", f"canonical execution commit does not contain {relative}")
            continue
        if path.is_file() and sha256(path) != canonical_blob_sha:
            fail("canonical_source_drift", f"{relative} differs from canonical execution SHA")
    for label, path, field in (("runtime_lock", lock_path, "runtime_lock_sha256"),
                               ("runtime_freeze", freeze_path, "runtime_freeze_sha256")):
        if not path.is_file():
            fail(f"{label}_missing", f"repository {label} is missing")
            continue
        digest = sha256(path)
        checks[f"{label}_sha256"] = digest
        if pilot.get(field) != digest or host.get(field) != digest or authorization.get(field) != digest:
            fail(f"{label}_hash", f"pilot {label} hash does not match the repository freeze")
    if input_manifest_path.is_file():
        input_manifest = read_json(input_manifest_path)
        input_manifest_sha = sha256(input_manifest_path)
        if pilot.get("input_manifest_sha256") != input_manifest_sha:
            fail("input_manifest_hash", "pilot input-manifest SHA-256 drift")
        if (input_manifest.get("status") != "remote-verified"
                or input_manifest.get("protocol_hash") != PROTOCOL_HASH
                or input_manifest.get("qrels_uploaded") is not False
                or input_manifest.get("files") != INPUT_HASHES):
            fail("frozen_input_contract", "repository input manifest differs from the frozen expected hashes")
    else:
        fail("input_manifest_missing", "repository frozen input manifest is missing")

    lock = read_json(lock_path)
    freeze = read_json(freeze_path)
    for relative, expected_sha in freeze.get("source_contract_sha256", {}).items():
        source_path = ROOT / relative
        if not source_path.is_file() or sha256(source_path) != expected_sha:
            fail("source_contract", f"Moon execution source contract drift: {relative}")
    if lock.get("model", {}).get("model_id") != MODEL_ID or lock.get("model", {}).get("revision") != MODEL_REVISION:
        fail("model_lock", "Moon runtime lock model/revision differs from the authorized reranker")
    if lock.get("inputs", {}).get("files") != INPUT_HASHES:
        fail("runtime_input_lock", "Moon runtime lock input hashes differ from expected")
    if host.get("status") != "pass":
        fail("host_preflight", "Moon host preflight did not pass")
    expected_host = {
        "python": "3.9.19", "torch": "2.8.0+cpu", "transformers": "4.57.6",
        "logical_vcpus": 16, "machine": "x86_64", "torch_cuda_available": False,
        "avx2_masked": True, "avx512f_masked": True,
    }
    for field, value in expected_host.items():
        if host.get(field) != value:
            fail("host_runtime", f"Moon host preflight {field}={host.get(field)!r}, expected {value!r}")
    if int(host.get("memory_bytes", 0)) < 60_000_000_000:
        fail("host_memory", "Moon preflight reports less than 60 GB RAM")
    if not str(host.get("output_filesystem", "")).lower().startswith("nfs"):
        fail("host_output_fs", "Moon output filesystem is not NFS")
    if int(host.get("nfs_quota_probe_bytes_written_and_removed", 0)) < 10_000_000:
        fail("host_nfs_quota", "Moon did not pass the 10 MB NFS write probe")
    if host.get("input_files") != INPUT_HASHES:
        fail("host_input_hashes", "Moon host preflight input hashes differ from frozen inputs")
    if host.get("input_manifest_sha256") != sha256(input_manifest_path):
        fail("host_input_manifest", "Moon host preflight does not bind the frozen input manifest")
    model_reference_path = ROOT / "versions/sem2act-v5/manifests/cpu_reranker_canary.json"
    model_reference = read_json(model_reference_path)
    expected_snapshot_sha = hashlib.sha256(json.dumps(
        model_reference.get("model_snapshot"), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if (model_reference.get("status") != "pass"
            or model_reference.get("model_id") != MODEL_ID
            or model_reference.get("revision") != MODEL_REVISION
            or host.get("model_snapshot_sha256") != expected_snapshot_sha):
        fail("model_snapshot_hash", "Moon preflight model snapshot differs from the pinned accepted snapshot")

    # The original pilot runner did not record a hostname. Require a post-run
    # operator receipt; absence is REVIEW and cannot authorize scale-up.
    provenance_path = args.host_provenance.resolve() if args.host_provenance else pilot_dir / "moon_host_provenance.json"
    if not provenance_path.is_file():
        review("hostname_provenance_missing", "provide moon_host_provenance.json with hostname and matching runtime facts")
    else:
        provenance = read_json(provenance_path)
        required_host_fields = (
            "hostname", "execution_sha", "protocol_hash", "platform", "machine",
            "python", "torch", "transformers", "logical_vcpus", "memory_bytes",
            "torch_cuda_available", "output_filesystem", "qrels_read",
            "runtime_lock_sha256", "runtime_freeze_sha256", "avx2_masked",
            "avx512f_masked", "nfs_quota_probe_bytes_written_and_removed",
            "input_files", "input_manifest_sha256", "model_id", "revision",
            "model_snapshot_sha256",
        )
        absent = [field for field in required_host_fields if field not in provenance]
        if absent:
            review("hostname_provenance_incomplete", f"operator hostname/runtime receipt lacks {absent}")
        else:
            if provenance.get("status") != "pass" or not str(provenance.get("hostname", "")).strip():
                fail("hostname_provenance", "Moon hostname receipt is not a passing named host")
            if provenance.get("execution_sha") != EXPECTED_EXECUTION_SHA or provenance.get("protocol_hash") != PROTOCOL_HASH:
                fail("hostname_execution_binding", "Moon hostname receipt is not bound to the canonical pilot/protocol")
            for field in ("platform", "machine", "python", "torch", "transformers",
                          "logical_vcpus", "memory_bytes", "torch_cuda_available", "output_filesystem",
                          "runtime_lock_sha256", "runtime_freeze_sha256", "avx2_masked",
                          "avx512f_masked", "nfs_quota_probe_bytes_written_and_removed",
                          "input_files", "input_manifest_sha256", "model_id", "revision",
                          "model_snapshot_sha256"):
                if provenance.get(field) != host.get(field):
                    fail("hostname_runtime_mismatch", f"host receipt {field} differs from moon_preflight.json")
            if int(provenance.get("nfs_quota_probe_bytes_written_and_removed", 0)) < 10_000_000:
                fail("hostname_nfs_quota", "Moon hostname receipt lacks the required 10 MB NFS write probe")
            if provenance.get("model_id") != MODEL_ID or provenance.get("revision") != MODEL_REVISION:
                fail("hostname_model", "Moon hostname receipt does not bind the pinned reranker revision")
            if provenance.get("input_files") != INPUT_HASHES or provenance.get("input_manifest_sha256") != sha256(input_manifest_path):
                fail("hostname_inputs", "Moon hostname receipt does not bind the frozen qrel-free inputs")
            if provenance.get("qrels_read") is not False:
                fail("hostname_qrels", "Moon hostname receipt does not prove qrels_read=false")
            checks["hostname"] = provenance["hostname"]
            checks["host_provenance_sha256"] = sha256(provenance_path)

    if authorization.get("status") != "authorized-after-gates":
        fail("pilot_authorization", "pilot authorization did not pass its pre-result gates")
    if authorization.get("thread_count") != 16 or authorization.get("parity_status") != "pass":
        fail("pilot_thread_authorization", "pilot authorization does not bind the 16-thread parity pass")
    expected_shard_sha = planned_pilot_shard_sha(INPUT_HASHES)
    if authorization.get("pilot_shard_sha256") != expected_shard_sha:
        fail("pilot_shard_hash", "pilot shard hash differs from deterministic shard 000 plan")
    if authorization.get("input_files") != INPUT_HASHES:
        fail("authorization_inputs", "pilot authorization input hashes drift")
    if (authorization.get("host_preflight", {}).get("sha256") != sha256(pilot_dir / "moon_preflight.json")
            or authorization.get("thread_canary", {}).get("sha256") != sha256(pilot_dir / "moon_reranker_canary.json")):
        fail("authorization_evidence", "pilot authorization host/parity evidence hashes drift")
    if pilot.get("host_preflight_sha256") != sha256(pilot_dir / "moon_preflight.json"):
        fail("pilot_host_evidence", "pilot manifest host-preflight hash drift")
    if pilot.get("thread_canary_sha256") != sha256(pilot_dir / "moon_reranker_canary.json"):
        fail("pilot_parity_evidence", "pilot manifest thread-canary hash drift")
    if pilot.get("pilot_authorization_sha256") != sha256(pilot_dir / "pilot_authorization.json"):
        fail("pilot_authorization_hash", "pilot manifest authorization hash drift")
    if host.get("model_snapshot_sha256") != authorization.get("model_snapshot_sha256"):
        fail("model_snapshot", "Moon preflight and authorization do not bind the same model snapshot")

    if parity.get("status") != "pass" or parity.get("model_revision") != MODEL_REVISION:
        fail("thread_parity_status", "4/16-thread canary status or model revision drift")
    if (parity.get("fixture_sha256") != freeze.get("smoke_fixture_sha256")
            or parity.get("runtime_lock_sha256") != sha256(lock_path)
            or parity.get("authorized_threads") != 16
            or parity.get("device") != "cpu" or parity.get("dtype") != "float32"
            or parity.get("exact_repeat_equality") is not True
            or parity.get("score_tolerance") != 1e-6):
        fail("thread_parity_contract", "4/16-thread canary does not match the frozen fixture/runtime policy")
    profiles = parity.get("profiles", {})
    p4, p16 = profiles.get("4", {}), profiles.get("16", {})
    s4, s16 = p4.get("scores", []), p16.get("scores", [])
    if len(s4) != 3 or len(s16) != 3 or p4.get("repeat_equal") is not True or p16.get("repeat_equal") is not True:
        fail("thread_parity_coverage", "each thread profile must repeat the three fixture scores exactly")
    if p4.get("ranking") != p16.get("ranking") or parity.get("ranking_equal") is not True:
        fail("thread_parity_ranking", "4-thread and 16-thread fixture rankings differ")
    if len(s4) == len(s16) == 3:
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in s4 + s16):
            fail("thread_parity_nonfinite", "4/16-thread canary contains a non-finite score")
        max_delta = max(abs(float(a) - float(b)) for a, b in zip(s4, s16))
        if max_delta > 1e-6 or float(parity.get("max_absolute_score_delta", float("inf"))) > 1e-6:
            fail("thread_parity_scores", f"4/16-thread score delta exceeds tolerance: {max_delta}")
        checks["thread_parity_max_absolute_delta"] = max_delta

    if run.get("status") != "pass" or run.get("stage") != "reranker":
        fail("run_status", "reranker run manifest is not a passing reranker run")
    if run.get("model_id") != MODEL_ID or run.get("revision") != MODEL_REVISION:
        fail("run_model", "reranker run model/revision drift")
    if run.get("runtime_profile") != "moon-reranker-only-v1" or run.get("device") != "cpu" or run.get("dtype") != "float32":
        fail("run_runtime", "reranker run profile/device/dtype drift")
    if (run.get("runtime_lock_sha256") != sha256(lock_path)
            or run.get("runtime_freeze_sha256") != sha256(freeze_path)
            or run.get("host_preflight_sha256") != sha256(pilot_dir / "moon_preflight.json")
            or run.get("thread_canary_sha256") != sha256(pilot_dir / "moon_reranker_canary.json")
            or run.get("result_bearing_execution_started") is not True):
        fail("run_provenance", "reranker run does not bind the accepted host/parity gates and runtime freeze")
    if run.get("expected_queries") != 20 or run.get("accepted_queries") != 20 or run.get("n_fail") != 0:
        fail("run_coverage", "reranker run does not report exact 20-query coverage and zero failures")
    if pilot.get("launcher_exit_code") != 0:
        fail("launcher_exit", "pilot launcher did not exit successfully")
    if failures != []:
        fail("failures", "failures.json must be an empty list")

    queries = [json.loads(line) for line in (input_root / "queries.jsonl").read_text().splitlines() if line.strip()]
    candidates = [json.loads(line) for line in (input_root / "rerank_candidates.jsonl").read_text().splitlines() if line.strip()]
    expected_qids = {f"v5-lb-q{i:04d}" for i in range(1, 21)}
    candidate_map = {row["query_id"]: row["doc_ids"] for row in candidates if row.get("query_id") in expected_qids}
    if [row.get("_id") for row in queries[:20]] != [f"v5-lb-q{i:04d}" for i in range(1, 21)] or set(candidate_map) != expected_qids:
        fail("pilot_input_coverage", "frozen inputs do not contain the first 20 authorized pilot queries")
    if any(len(candidate_map.get(qid, [])) != 50 for qid in expected_qids):
        fail("pilot_candidate_depth", "each pilot query must have exactly 50 candidate documents")
    if any(len(set(candidate_map.get(qid, []))) != 50 for qid in expected_qids):
        fail("pilot_candidate_duplicates", "each pilot query must have 50 unique candidate documents")

    fixture_path = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
    fixture = read_json(fixture_path)
    fixture_doc_ids = [row.get("doc_id") for row in fixture.get("documents", [])]
    if len(fixture_doc_ids) != 3 or len(set(fixture_doc_ids)) != 3:
        fail("parity_fixture", "frozen parity fixture must contain exactly three unique documents")
    for profile_name, profile in (("4", p4), ("16", p16)):
        if set(profile.get("ranking", [])) != set(fixture_doc_ids):
            fail("thread_parity_fixture_coverage", f"{profile_name}-thread canary does not rank exactly the three fixture documents")

    ranking_rows = []
    try:
        ranking_rows = [json.loads(line) for line in (pilot_dir / "rankings.jsonl").read_text().splitlines() if line.strip()]
    except Exception as exc:
        fail("rankings_jsonl", f"rankings.jsonl cannot be parsed: {type(exc).__name__}")
    if len(ranking_rows) != 20:
        fail("ranking_row_count", f"expected 20 ranking rows, got {len(ranking_rows)}")
    keys = set()
    for row in ranking_rows:
        qid = row.get("query_id")
        key = row.get("key")
        ranking = row.get("ranking", [])
        if key != f"reranker|{qid}" or qid not in expected_qids or key in keys:
            fail("ranking_key", f"invalid or duplicate reranker key {key!r}")
        keys.add(key)
        if row.get("valid") is not True or len(ranking) != 50 or len(set(ranking)) != 50:
            fail("ranking_coverage", f"{qid}: ranking must contain 50 unique candidates")
        if set(ranking) != set(candidate_map.get(qid, [])):
            fail("ranking_candidates", f"{qid}: output ranking does not cover exactly its 50 input candidates")
        if not valid_sha(row.get("scores_sha256")) or not valid_sha(row.get("ranking_sha256")):
            fail("ranking_hash_fields", f"{qid}: missing score/ranking digest")
    if keys != {f"reranker|{qid}" for qid in expected_qids}:
        fail("ranking_keyspace", "ranking rows do not cover exactly the 20 authorized query keys")
    started_path = pilot_dir / "result_bearing_started.json"
    if (started.get("shard_sha256") != expected_shard_sha
            or started.get("thread_count") != 16
            or started.get("runtime_lock_sha256") != sha256(lock_path)
            or started.get("runtime_freeze_sha256") != sha256(freeze_path)):
        fail("result_start_provenance", "result-bearing marker does not bind the exact pilot shard and 16-thread runtime")
    checks["authorized_candidate_score_count"] = 20 * 50

    outputs_declared = pilot.get("outputs", {})
    expected_output_names = {
        "moon_preflight.json", "moon_reranker_canary.json", "pilot_authorization.json",
        "rankings.jsonl", "failures.json", "run_manifest.json", "result_bearing_started.json",
    }
    provenance_path = args.host_provenance.resolve() if args.host_provenance else pilot_dir / "moon_host_provenance.json"
    allowed_output_names = expected_output_names | {"pilot_manifest.json"}
    if provenance_path.parent == pilot_dir:
        allowed_output_names.add(provenance_path.name)
    if args.output.resolve().parent == pilot_dir:
        allowed_output_names.add(args.output.resolve().name)
    extra_names = sorted(path.name for path in pilot_dir.iterdir()
                         if path.is_dir() or path.name not in allowed_output_names)
    if extra_names:
        fail("unmanifested_outputs", f"unexpected files in pilot directory: {extra_names}")
    if set(outputs_declared) != expected_output_names:
        fail("output_manifest_completeness", f"pilot output hash map must contain {sorted(expected_output_names)}")
    actual_output_hashes = {}
    for name in sorted(expected_output_names):
        path = pilot_dir / name
        if not path.is_file():
            fail("output_missing", f"missing output artifact {name}")
            continue
        actual = sha256(path)
        actual_output_hashes[name] = actual
        if outputs_declared.get(name) != actual:
            fail("output_hash", f"pilot manifest SHA-256 mismatch for {name}")
    checks["output_sha256"] = actual_output_hashes
    checks["expected_queries"] = 20
    checks["expected_candidate_scores"] = 1000
    checks["candidate_score_count_basis"] = "20 exact query rankings times the frozen 50-candidate input depth, with one score digest per query"
    checks["score_loop_source_sha256"] = freeze.get("source_contract_sha256", {}).get(
        "scripts/run_v5_cpu_rerank_shard.py"
    )
    checks["qrels_read"] = False
    checks["model_revision"] = MODEL_REVISION
    checks["model_snapshot_sha256"] = expected_snapshot_sha
    checks["runtime_source_contract_sha256"] = freeze.get("source_contract_sha256", {})

    status = "FAIL" if reasons else "REVIEW" if review_reasons else "PASS"
    return report(args, status, reasons, review_reasons, checks)


def report(args: argparse.Namespace, status: str, reasons: list[dict],
           review_reasons: list[dict], checks: dict) -> dict:
    return {
        "schema_version": 1,
        "verifier": "sem2act-v5-moon-reranker-pilot",
        "status": status,
        "execution_sha": args.execution_sha,
        "protocol_hash": PROTOCOL_HASH,
        "reasons": reasons + review_reasons,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--host-provenance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args)
    except Exception as exc:
        result = report(args, "FAIL", [{
            "code": "verifier_exception",
            "message": f"{type(exc).__name__}: {exc}",
        }], [], {})
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")
    return {"PASS": 0, "FAIL": 1, "REVIEW": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
