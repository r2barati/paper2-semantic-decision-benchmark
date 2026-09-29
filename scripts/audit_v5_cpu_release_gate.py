"""Audit the prospective CPU_BACKEND_V1 release gate without inference or qrel access."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"

def load(rel: str) -> dict | None:
    path = ROOT / rel
    return json.loads(path.read_text()) if path.exists() else None

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--json", action="store_true"); args=parser.parse_args()
    lock=load("versions/sem2act-v5/manifests/cpu_runtime_lock.json")
    artifacts=load("versions/sem2act-v5/manifests/cpu_model_artifacts.json")
    preflight=load("versions/sem2act-v5/manifests/cpu_preflight_status.json")
    source=load("versions/sem2act-v5/manifests/cpu_model_source_preflight.json")
    host=load("versions/sem2act-v5/manifests/cpu_host_preflight.json")
    engine=load("versions/sem2act-v5/manifests/cpu_engine_build.json")
    consumer=load("versions/sem2act-v5/manifests/cpu_canary.json")
    reranker=load("versions/sem2act-v5/manifests/cpu_reranker_canary.json")
    freeze=load("versions/sem2act-v5/manifests/cpu_runtime_freeze.json")
    execution=load("versions/sem2act-v5/manifests/EXECUTION_STATUS.json")
    missing=[]
    checks={}
    checks["protocol_hash"] = bool(lock and lock.get("protocol_hash")==PROTOCOL_HASH and preflight and preflight.get("protocol_hash")==PROTOCOL_HASH)
    if not checks["protocol_hash"]: missing.append("protocol_hash")
    checks["cpu_lock"] = bool(lock and lock.get("platform",{}).get("accelerator")=="cpu_only" and lock.get("engine",{}).get("quantization_type")=="Q4_K_M")
    if not checks["cpu_lock"]: missing.append("cpu_runtime_lock")
    checks["source_hashes"] = bool(source and all(source.get("models",{}).get(f,{}).get("source",{}).get("content_sha256") for f in ("qwen","llama")))
    if not checks["source_hashes"]: missing.append("official_model_source_hashes")
    checks["artifacts"] = bool(artifacts and artifacts.get("status")=="complete-before-result-bearing-execution" and all(artifacts.get("models",{}).get(f,{}).get("sha256") for f in ("qwen","llama","mistral")) and artifacts.get("conversion_script_sha256") and artifacts.get("quantizer_binary_sha256"))
    if not checks["artifacts"]: missing.append("gguf_artifacts_and_conversion_hashes")
    checks["host"] = bool(host and host.get("status")=="pass" and int(host.get("frozen_threads",0))>0 and host.get("llama_server_sha256"))
    if not checks["host"]: missing.append("cpu_host_preflight")
    checks["engine"] = bool(engine and engine.get("status")=="pass-before-model-conversion" and engine.get("commit")=="9f70b2cecd1a9a3f73ac525c47ca22a6ee9a7b69")
    if not checks["engine"]: missing.append("llama_cpp_engine_build")
    checks["reranker_canary"] = bool(reranker and reranker.get("status")=="pass" and reranker.get("schema_valid") is True and reranker.get("deterministic_repeat_equality") is True)
    if not checks["reranker_canary"]: missing.append("reranker_canary")
    checks["consumer_canaries"] = bool(consumer and consumer.get("status")=="pass" and set(consumer.get("models",{}))=={"qwen","llama","mistral"} and all(v.get("status")=="pass" and v.get("schema_valid") is True and v.get("deterministic_repeat_equality") is True for v in consumer.get("models",{}).values()))
    if not checks["consumer_canaries"]: missing.append("consumer_canaries")
    checks["runtime_freeze"] = bool(freeze and freeze.get("status")=="frozen-before-result-bearing-execution" and freeze.get("protocol_hash")==PROTOCOL_HASH)
    if not checks["runtime_freeze"]: missing.append("cpu_runtime_freeze_manifest")
    remote_jobs=[]
    if execution:
        backend=execution.get("backend",{})
        remote_jobs=backend.get("remote_jobs_started", execution.get("remote_jobs_started", []))
    checks["no_result_execution"] = bool(preflight and preflight.get("result_bearing_execution_started") is False and preflight.get("qrels_read") is False and not remote_jobs)
    if not checks["no_result_execution"]: missing.append("result_execution_invariant")
    checks["sharding_implementation"] = all((ROOT/f).exists() for f in ("scripts/plan_v5_cpu_shards.py","scripts/plan_v5_cpu_rerank_shards.py","scripts/run_v5_cpu_shard.py","scripts/verify_v5_cpu_shards.py"))
    if not checks["sharding_implementation"]: missing.append("deterministic_sharding_implementation")
    report={"schema_version":1,"manifest_id":"sem2act-v5-cpu-release-gate-audit-v1","protocol_hash":PROTOCOL_HASH,"status":"pass" if not missing else "blocked-before-result-bearing-execution","checks":checks,"missing":missing,"result_bearing_execution_started":False,"qrels_read":False}
    if args.json: print(json.dumps(report,indent=2,sort_keys=True))
    else: print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not missing else 2
if __name__ == "__main__": raise SystemExit(main())
