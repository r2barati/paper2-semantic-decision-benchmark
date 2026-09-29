"""Run one frozen CPU consumer shard through llama.cpp.

This entry point refuses to run until CPU_BACKEND_V1 has been fully frozen.
It resumes only missing call keys, stores raw responses by content hash, and
fails closed on any unresolved key or schema error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CPU_DIR = ROOT / "versions/sem2act-v5/compute/cpu"
sys.path.insert(0, str(CPU_DIR))
from contract import (  # noqa: E402
    PROMPT_C1,
    PROMPT_C3_DOC,
    ROOT as CONTRACT_ROOT,
    belief_key,
    canonical_sha256,
    call_keys_for_row,
    expected_belief_keys,
    expected_call_keys,
    load_artifacts,
    load_cpu_freeze,
    load_cpu_lock,
    model_server_id,
    post_chat,
    response_content,
    server_command,
    sha256_bytes,
    sha256_file,
    wait_for_server,
    cpu_subprocess_env,
)


def load_module():
    path = ROOT / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py"
    spec = importlib.util.spec_from_file_location("sem2act_v5_consumer_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen consumer contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any("qrel" in json.dumps(row, sort_keys=True).lower() for row in rows):
        raise RuntimeError("qrel token detected in CPU model input")
    return rows


def load_shard(path: Path) -> dict[str, Any]:
    shard = json.loads(path.read_text())
    if shard.get("status") != "planned-before-result-bearing-execution":
        raise RuntimeError("shard is not a frozen preplanned shard")
    return shard


def load_accepted(raw_dir: Path, index_path: Path) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    if not index_path.exists():
        return accepted
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = item["key"]
        if key in accepted:
            raise RuntimeError(f"duplicate accepted key in raw index: {key}")
        raw_path = raw_dir / item["content_address"]
        if not raw_path.exists():
            raise RuntimeError(f"accepted raw artifact missing: {raw_path}")
        record = json.loads(raw_path.read_text())
        if record.get("key") != key:
            raise RuntimeError(f"raw key mismatch: {key}")
        if sha256_file(raw_path) != item["file_sha256"]:
            raise RuntimeError(f"raw artifact hash mismatch: {key}")
        accepted[key] = record
    return accepted


def append_raw(raw_dir: Path, index_path: Path, record: dict[str, Any]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    content_hash = sha256_bytes(payload.encode("utf-8"))
    raw_path = raw_dir / f"{content_hash}.json"
    if raw_path.exists() and raw_path.read_text() != payload:
        raise RuntimeError(f"content-address collision: {raw_path}")
    raw_path.write_text(payload)
    with index_path.open("a") as handle:
        handle.write(json.dumps({
            "key": record["key"],
            "content_address": raw_path.name,
            "file_sha256": content_hash,
            "raw_sha256": record["raw_sha256"],
        }, sort_keys=True) + "\n")


def user_messages(row: dict[str, Any], consumer: str, doc: dict[str, Any] | None = None):
    if consumer == "C1":
        user = (
            f"Operator information need: {row['query_text']}\n\nEvidence:\n"
            + "\n\n".join(f"[DOC {item['doc_id']}] {item['text']}" for item in row["documents"])
        )
        return [{"role": "system", "content": PROMPT_C1}, {"role": "user", "content": user}]
    if doc is None:
        raise ValueError("C3 requires a document")
    user = f"Operator's own node: {row['metadata']['entity_node']}\n\nEvidence document:\n{doc['text']}"
    return [{"role": "system", "content": PROMPT_C3_DOC}, {"role": "user", "content": user}]


def call_one(base_url: str, model_name: str, module, key: str, row: dict[str, Any],
             consumer: str, doc: dict[str, Any] | None, family: str) -> dict[str, Any]:
    messages = user_messages(row, consumer, doc)
    request_hash = canonical_sha256({"model": model_name, "messages": messages, "family": family, "key": key})
    response = post_chat(base_url, model_name, messages)
    raw = response_content(response)
    parsed = module.parse_json(raw)
    parsed = module.validate_c1(parsed) if consumer == "C1" else module.validate_c3(parsed)
    record = {
        "schema_version": 1,
        "key": key,
        "family": family,
        "query_id": row["query_id"],
        "system": row["system"],
        "consumer": consumer,
        "doc_id": None if doc is None else doc["doc_id"],
        "model_server_id": model_name,
        "prompt_sha": module.PROMPT_SHAS[consumer],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "request_sha256": request_hash,
        "raw": raw,
        "raw_sha256": sha256_bytes(raw.encode("utf-8")),
        "parsed": parsed,
        "valid": True,
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("qwen", "llama", "mistral"), required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18865)
    args = parser.parse_args()

    # This is the result-bearing gate.
    freeze = load_cpu_freeze(require_frozen=True)
    lock = load_cpu_lock()
    artifacts = load_artifacts(require_complete=True)
    family_artifact = artifacts["models"][args.family]
    if Path(family_artifact["path"]).resolve() != args.model.resolve():
        raise SystemExit("model path does not match frozen artifact manifest")
    if sha256_file(args.model) != family_artifact["sha256"]:
        raise SystemExit("model artifact hash does not match frozen manifest")
    shard = load_shard(args.shard)
    if shard["family"] != args.family:
        raise SystemExit("shard family mismatch")
    rows = load_rows(args.inputs)
    selected = [row for row in rows if row["query_id"] in set(shard["query_ids"])]
    actual_calls = expected_call_keys(selected, args.family)
    actual_beliefs = expected_belief_keys(selected, args.family)
    if actual_calls != shard["expected_call_keys"] or actual_beliefs != shard["expected_belief_keys"]:
        raise SystemExit("shard expected keyspace mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    index_path = args.output_dir / "raw_index.jsonl"
    accepted = load_accepted(raw_dir, index_path)
    if not set(accepted).issubset(set(actual_calls)):
        raise SystemExit("accepted raw key outside shard keyspace")
    failures: list[dict[str, Any]] = []
    process = None
    log_path = args.output_dir / "llama-server.log"
    base_url = f"http://127.0.0.1:{args.port}"
    module = load_module()
    host = json.loads((ROOT / "versions/sem2act-v5/manifests/cpu_host_preflight.json").read_text())
    threads = int(freeze["frozen_threads"])
    if threads != int(host["frozen_threads"]):
        raise SystemExit("CPU thread count drift")
    command = server_command(args.llama_server, args.model, threads, threads, args.port)
    env = cpu_subprocess_env()
    env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads)})
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_for_server(base_url, process, timeout=1800)
        model_name = model_server_id(base_url)
        for row in sorted(selected, key=lambda item: (item["query_id"], item["system"])):
            doc_ids = [doc["doc_id"] for doc in row["documents"]]
            c1_key = f"{args.family}|{row['query_id']}|{row['system']}|C1|"
            if c1_key not in accepted:
                try:
                    record = call_one(base_url, model_name, module, c1_key, row, "C1", None, args.family)
                    append_raw(raw_dir, index_path, record)
                    accepted[c1_key] = record
                except Exception as exc:
                    failures.append({"key": c1_key, "error": f"{type(exc).__name__}: {exc}"})
            for doc in row["documents"]:
                key = f"{args.family}|{row['query_id']}|{row['system']}|C3|{doc['doc_id']}"
                if key in accepted:
                    continue
                try:
                    record = call_one(base_url, model_name, module, key, row, "C3", doc, args.family)
                    append_raw(raw_dir, index_path, record)
                    accepted[key] = record
                except Exception as exc:
                    failures.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()

    if set(accepted) != set(actual_calls):
        failures.append({"error": "unresolved or missing call keys", "missing": sorted(set(actual_calls) - set(accepted))})
    beliefs = []
    for row in sorted(selected, key=lambda item: (item["query_id"], item["system"])):
        if row["system"] not in lock["models"].get(args.family, {}):
            pass
        doc_ids = [doc["doc_id"] for doc in row["documents"]]
        c1_key = f"{args.family}|{row['query_id']}|{row['system']}|C1|"
        c3_keys = [f"{args.family}|{row['query_id']}|{row['system']}|C3|{doc_id}" for doc_id in doc_ids]
        if c1_key in accepted and all(key in accepted for key in c3_keys):
            c1_belief = module.aggregate_c1(accepted[c1_key]["parsed"], doc_ids)
            c3_items = [(doc_id, accepted[key]["parsed"]) for doc_id, key in zip(doc_ids, c3_keys)]
            c3_belief = module.aggregate_c3(c3_items, doc_ids)
            for consumer, belief in (("C1", c1_belief), ("C3", c3_belief)):
                beliefs.append({
                    "belief_key": f"{args.family}|{row['query_id']}|{row['system']}|{consumer}",
                    "query_id": row["query_id"],
                    "system": row["system"],
                    "consumer": consumer,
                    "model_family": args.family,
                    "k": 3,
                    "p_normal": belief["p_normal"],
                    "p_supplier_delay": belief["p_supplier_delay"],
                    "p_demand_surge": belief["p_demand_surge"],
                    "abstain": belief["abstain"],
                    "confidence": belief["confidence"],
                    "evidence_ids": belief["evidence_ids"],
                })
    belief_keys = {row["belief_key"] for row in beliefs}
    if belief_keys != set(actual_beliefs):
        failures.append({"error": "belief keyspace failed", "missing": sorted(set(actual_beliefs) - belief_keys)})
    manifest = {
        "schema_version": 1,
        "manifest_id": shard["manifest_id"],
        "status": "pass" if not failures else "failed",
        "protocol_hash": lock["protocol_hash"],
        "runtime_freeze_sha256": sha256_file(ROOT / "versions/sem2act-v5/manifests/cpu_runtime_freeze.json"),
        "family": args.family,
        "shard": str(args.shard.relative_to(ROOT)),
        "expected_calls": len(actual_calls),
        "accepted_calls": len(accepted),
        "expected_beliefs": len(actual_beliefs),
        "accepted_beliefs": len(beliefs),
        "n_fail": len(failures),
        "qrels_read": False,
        "raw_content_addressed": True,
        "accepted_key_regeneration": False,
        "beliefs": "beliefs.jsonl",
        "failures": "failures.json",
    }
    (args.output_dir / "beliefs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(beliefs, key=lambda item: item["belief_key"]))
    )
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
