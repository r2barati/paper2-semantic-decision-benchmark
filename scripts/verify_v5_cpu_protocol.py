"""Verify CPU_BACKEND_V1 invariants without running inference."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def main() -> int:
    lock = load("versions/sem2act-v5/manifests/cpu_runtime_lock.json")
    artifacts = load("versions/sem2act-v5/manifests/cpu_model_artifacts.json")
    protocol = load("versions/sem2act-v5/manifests/protocol_freeze.json")
    status = load("versions/sem2act-v5/manifests/EXECUTION_STATUS.json")
    preflight = load("versions/sem2act-v5/manifests/cpu_preflight_status.json")
    if lock["protocol_hash"] != PROTOCOL_HASH or protocol["protocol_hash"] != PROTOCOL_HASH:
        raise SystemExit("CPU branch protocol hash drift")
    if lock["status"] != "prospective-before-cpu-preflight":
        raise SystemExit("CPU runtime lock is not prospective")
    if lock["platform"]["accelerator"] != "cpu_only" or lock["platform"]["cuda"] or lock["platform"]["metal"]:
        raise SystemExit("CPU branch permits an accelerator")
    if lock["engine"]["commit"] != "9f70b2cecd1a9a3f73ac525c47ca22a6ee9a7b69":
        raise SystemExit("llama.cpp commit drift")
    if lock["engine"]["quantization_type"] != "Q4_K_M":
        raise SystemExit("common CPU quantization drift")
    if artifacts["status"] != "pending-non-result-preflight":
        raise SystemExit("artifact manifest was marked complete before conversion")
    for family in ("qwen", "llama", "mistral"):
        spec = lock["models"][family]
        if spec["revision"] != spec["tokenizer_revision"]:
            raise SystemExit(f"{family}: model/tokenizer revision drift")
        if artifacts["models"][family]["sha256"] is not None:
            raise SystemExit(f"{family}: artifact hash unexpectedly present in pending manifest")
    if status.get("remote_jobs_started"):
        raise SystemExit("a result-bearing remote job is recorded before CPU freeze")
    if preflight.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("CPU preflight status protocol hash drift")
    if preflight.get("result_bearing_execution_started") is not False or preflight.get("qrels_read") is not False:
        raise SystemExit("CPU preflight status is not non-result/qrel-free")
    if preflight.get("status") not in {
        "blocked-before-cpu-runtime-freeze",
        "blocked-before-cpu-runtime-freeze; model-source-access-pending",
        "blocked-before-cpu-runtime-freeze;llama-source-access-pending",
        "blocked-before-cpu-runtime-freeze;llama-source-probe-failed-no-mount",
        "blocked-before-cpu-runtime-freeze;qwen-awq-conversion-failed;llama-source-probe-failed-no-mount;mistral-source-assembly-verified",
        "pass-before-cpu-runtime-freeze",
    }:
        raise SystemExit("unexpected CPU preflight status")
    for relative in lock["source_contract"]:
        if not (ROOT / relative).exists():
            raise SystemExit(f"missing CPU source contract file: {relative}")
    fixture = load("versions/sem2act-v5/fixtures/runtime_smoke_v1.json")
    if fixture["query_id"].startswith("v5-lb-"):
        raise SystemExit("CPU smoke fixture is a lockbox query")
    print(json.dumps({
        "status": "pass",
        "protocol_hash": PROTOCOL_HASH,
        "cpu_lock_status": lock["status"],
        "artifact_status": artifacts["status"],
        "remote_jobs_started": 0,
        "result_bearing_execution_started": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
