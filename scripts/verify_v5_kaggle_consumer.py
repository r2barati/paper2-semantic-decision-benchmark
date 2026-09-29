"""Verify one fetched Llama or Mistral consumer run against its exact bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}
SYSTEMS = ("rerank", "oracle")
QUERY_IDS = tuple(f"v5-lb-q{i:04d}" for i in range(1, 241))
FAMILIES = {
    "llama": {
        "job": "sem2act-v5-llama",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
    "mistral": {
        "job": "sem2act-v5-mistral",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef",
    },
}
LOCK_REL = "versions/sem2act-v5/manifests/kaggle_runtime_lock.json"
FREEZE_REL = "versions/sem2act-v5/manifests/kaggle_runtime_freeze.json"
FIXTURE_REL = "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_raw(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text.split("\n", 1)[1].rsplit(fence, 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("raw response is not a JSON object")
    return value


def validate_response(value: dict, consumer: str) -> None:
    if consumer == "C1":
        expected = {
            "normal", "supplier_delay", "demand_surge",
            "estimated_lt_increase", "estimated_duration", "estimated_demand_multiplier",
        }
        if set(value) != expected:
            raise ValueError("C1 response schema keys drift")
        probs = [float(value[key]) for key in ("normal", "supplier_delay", "demand_surge")]
        if any(not math.isfinite(item) or not 0 <= item <= 1 for item in probs) or abs(sum(probs) - 1) > 0.05:
            raise ValueError("C1 probability schema invalid")
        for key in ("estimated_lt_increase", "estimated_duration"):
            number = int(value[key])
            if number < 0 or float(value[key]) != number:
                raise ValueError(f"C1 {key} is not a nonnegative integer")
        if not math.isfinite(float(value["estimated_demand_multiplier"])) or float(value["estimated_demand_multiplier"]) < 0:
            raise ValueError("C1 demand multiplier invalid")
    else:
        expected = {"entity_match", "event", "fresh", "stance", "confidence"}
        if set(value) != expected:
            raise ValueError("C3 response schema keys drift")
        if not isinstance(value["entity_match"], bool) or not isinstance(value["fresh"], bool):
            raise ValueError("C3 boolean schema invalid")
        if value["event"] not in {"supplier_delay", "demand_surge", "normal", "none"}:
            raise ValueError("C3 event schema invalid")
        if value["stance"] not in {"support", "refute", "na"}:
            raise ValueError("C3 stance schema invalid")
        confidence = float(value["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("C3 confidence schema invalid")


def verify(args: argparse.Namespace) -> dict:
    failures: list[dict[str, str]] = []
    checks: dict[str, object] = {}

    def fail(code: str, message: str) -> None:
        failures.append({"code": code, "message": message})

    if args.family not in FAMILIES:
        fail("family", "family must be llama or mistral")
        return make_report(args, "FAIL", failures, checks)
    if not re.fullmatch(r"[0-9a-f]{40}", args.execution_sha):
        fail("execution_sha_format", "execution SHA must be a full lowercase Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", args.bundle_manifest_sha256):
        fail("bundle_sha_format", "bundle manifest SHA must be lowercase SHA-256")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "-uall"], cwd=ROOT, text=True)
        if head != args.execution_sha or dirty:
            fail("execution_checkout", "verifier checkout must be clean at the exact handoff SHA")
    except (OSError, subprocess.CalledProcessError) as exc:
        fail("execution_checkout", f"could not verify exact clean Git checkout: {type(exc).__name__}")

    family = FAMILIES[args.family]
    bundle_root, output_root = args.bundle_root.resolve(), args.output_root.resolve()
    bundle_path = bundle_root / f"consumer_inputs_{args.family}.json"
    if not bundle_path.is_file():
        return make_report(args, "REVIEW", [{"code": "bundle_manifest_missing", "message": str(bundle_path)}], checks)
    bundle_sha = sha(bundle_path)
    if bundle_sha != args.bundle_manifest_sha256:
        fail("bundle_manifest_sha", "consumer bundle does not match its accepted manifest SHA")
    bundle = read_json(bundle_path)
    if (bundle.get("family") != args.family
            or bundle.get("status") != "built-from-accepted-reranker-output"
            or bundle.get("execution_sha") != args.execution_sha
            or bundle.get("protocol_hash") != PROTOCOL_HASH
            or bundle.get("model_id") != family["model_id"] or bundle.get("revision") != family["revision"]
            or bundle.get("expected_workload") != {"queries": 240, "evidence_rows": 720, "calls": 1920}
            or bundle.get("evaluation_labels_included") is not True
            or bundle.get("model_prompt_label_exposure") is not False):
        fail("bundle_contract", "bundle family, execution, model, protocol, workload, or label provenance drift")
    actual_bundle_files = {
        path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in bundle_root.iterdir() if path.is_file() and path != bundle_path
    }
    if bundle.get("files") != actual_bundle_files:
        fail("bundle_files", "consumer bundle artifacts differ from the manifested hash/size map")
    if (any(path.is_dir() for path in bundle_root.iterdir())
            or {path.name for path in bundle_root.iterdir() if path.is_file()} != set(actual_bundle_files) | {bundle_path.name}):
        fail("bundle_tree", "consumer bundle has unexpected files")

    spec_path = bundle_root / "model_spec.json"
    evidence_path = bundle_root / "evidence_inputs.jsonl"
    input_manifest_path = bundle_root / "model_inputs_manifest.json"
    required_inputs = (spec_path, evidence_path, input_manifest_path, bundle_root / "runtime_smoke_v1.json")
    if any(not path.is_file() for path in required_inputs):
        return make_report(args, "FAIL", failures + [{"code": "bundle_inputs_missing", "message": "bundle input set is incomplete"}], checks)
    spec, input_manifest = read_json(spec_path), read_json(input_manifest_path)
    lock_path, freeze_path = ROOT / LOCK_REL, ROOT / FREEZE_REL
    lock, freeze = read_json(lock_path), read_json(freeze_path)
    if (sha(lock_path) != bundle.get("runtime_lock_sha256")
            or lock.get("status") != "prospective-before-result-bearing-execution"
            or lock.get("lock_id") != "sem2act-v5-kaggle-runtime-v3"
            or freeze.get("lock_sha256") != sha(lock_path)
            or freeze.get("status") != "prospective-before-result-bearing-execution"):
        fail("runtime_freeze", "Kaggle consumer bundle/runtime freeze differs from the handoff policy")
    source_contract = freeze.get("source_contract_sha256", {})
    if not source_contract:
        fail("runtime_source_contract", "Kaggle runtime freeze has no source contract")
    for relative, expected_sha in source_contract.items():
        source_path = ROOT / relative
        if not source_path.is_file() or sha(source_path) != expected_sha:
            fail("runtime_source_contract", f"Kaggle runtime source contract drift: {relative}")
    locked = lock.get("models", {}).get(args.family, {})
    if (spec.get("execution_sha") != args.execution_sha
            or spec.get("model_family") != args.family
            or spec.get("model_id") != family["model_id"]
            or spec.get("revision") != family["revision"]
            or spec.get("tokenizer_revision") != family["revision"]
            or spec.get("protocol_hash") != PROTOCOL_HASH
            or spec.get("expected_protocol_hash") != PROTOCOL_HASH
            or spec.get("runtime_lock_id") != lock.get("lock_id")
            or spec.get("runtime_lock_sha256") != sha(lock_path)
            or spec.get("runtime") != {
                key: locked[key] for key in (
                    "backend", "dtype", "quantization", "device_map", "attention_backend",
                    "max_context_tokens", "max_input_tokens", "max_new_tokens", "batch_size",
                    "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
                    "enforce_eager", "enable_thinking", "do_sample", "num_beams", "use_cache", "tokenizer",
                ) if key in locked
            }
            or bundle.get("model_spec_sha256") != sha(spec_path)
            or bundle.get("protocol_file_sha256") != sha(ROOT / "versions/sem2act-v5/protocol/confirmation.yaml")):
        fail("model_spec", "model spec does not match the exact family revision, runtime lock, or protocol hashes")
    if (input_manifest.get("status") != "accepted-reranker-inputs-only"
            or input_manifest.get("n_queries") != 240 or input_manifest.get("n_rows") != 720
            or input_manifest.get("evaluation_labels_included") is not True
            or input_manifest.get("model_prompt_label_exposure") is not False
            or input_manifest.get("source_reranker_output_sha256") != bundle.get("source_reranker_output_sha256")
            or spec.get("source_reranker_output_sha256") != bundle.get("source_reranker_output_sha256")
            or spec.get("source_manifest_sha256") != sha(input_manifest_path)):
        fail("model_input_source", "model input manifest does not bind the accepted full reranker output")
    for name in ("source_retrieval_manifest_sha256", "source_reranker_manifest_sha256",
                 "source_reranker_verification_sha256"):
        digest = bundle.get(name)
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or input_manifest.get(name) != digest):
            fail("bundle_source_lineage", f"consumer bundle does not bind a valid {name}")

    rows = [json.loads(line) for line in evidence_path.read_text().splitlines() if line.strip()]
    expected_all = {(qid, system) for qid in QUERY_IDS for system in ("bm25", "rerank", "oracle")}
    if (len(rows) != 720 or len({(row.get("query_id"), row.get("system")) for row in rows}) != 720
            or {(row.get("query_id"), row.get("system")) for row in rows} != expected_all):
        fail("bundle_evidence_coverage", "bundle must contain exactly the 240 x 3 evidence-row keyspace")
    selected = {(row.get("query_id"), row.get("system")): row for row in rows if row.get("system") in SYSTEMS}
    if len(selected) != 480 or any(len(row.get("documents", [])) != 3 for row in selected.values()):
        fail("bundle_selected_scope", "Llama/Mistral consumer needs exactly 480 rerank/oracle rows at depth 3")

    upload = read_json(args.dataset_manifest)
    expected_dataset = f"rezabarati2/sem2act-v5-{args.family}-inputs"
    expected_files = {name: item["sha256"] for name, item in bundle.get("files", {}).items()}
    expected_files[bundle_path.name] = bundle_sha
    if (upload.get("status") != "remote-verified" or upload.get("dataset") != expected_dataset
            or upload.get("protocol_hash") != PROTOCOL_HASH or upload.get("qrels_uploaded") is not False
            or upload.get("execution_sha") != args.execution_sha
            or upload.get("bundle_manifest_sha256") != bundle_sha
            or upload.get("source_reranker_output_sha256") != bundle.get("source_reranker_output_sha256")
            or upload.get("runtime_lock_sha256") != sha(lock_path)
            or upload.get("runtime_freeze_sha256") != sha(freeze_path)
            or upload.get("files") != expected_files or upload.get("remote_sha256") != expected_files
            or upload.get("remote_file_list_match") is not True or upload.get("remote_hash_match") is not True):
        fail("kaggle_dataset_provenance", "Kaggle upload manifest does not prove the exact qrel-free bundle was remotely verified")

    preflight, canary = read_json(args.kaggle_preflight), read_json(args.canary_manifest)
    if (preflight.get("status") != "canary-passed" or preflight.get("owner") != "rezabarati2"
            or preflight.get("owner_verified") is not True
            or float(preflight.get("gpu_remaining_hours", 0)) <= 0
            or preflight.get("runtime_lock_sha256") != sha(lock_path)
            or preflight.get("runtime_freeze_sha256") != sha(freeze_path)
            or preflight.get("canary_output_sha256") != sha(args.canary_manifest)):
        fail("kaggle_preflight", "Kaggle identity, quota/canary state, or runtime-freeze preflight drift")
    if (canary.get("status") != "pass" or canary.get("protocol_hash") != PROTOCOL_HASH
            or canary.get("torch") != "2.10.0+cu128" or canary.get("cuda") != "12.8"
            or canary.get("device_count") != 1 or canary.get("physical_device_count") != 2
            or "t4" not in canary.get("device_name", "").lower()
            or canary.get("real_cuda_matmul") is not True):
        fail("kaggle_canary", "Kaggle must prove the frozen 2x-T4 entitlement and one-visible-T4 runtime")

    output_names = {
        "raw_outputs.jsonl", "beliefs.jsonl", "run_manifest.json", "failures.json",
        "runtime_versions.json", "kaggle_fetch_manifest.json",
    }
    outputs = {name: output_root / name for name in output_names}
    missing = sorted(name for name, path in outputs.items() if not path.is_file())
    if missing:
        return make_report(args, "REVIEW", failures + [{"code": "output_files_missing", "message": str(missing)}], checks)
    if {path.name for path in output_root.iterdir()} != output_names:
        fail("output_tree", "fetched Kaggle consumer directory has missing or unmanifested files")
    run = read_json(outputs["run_manifest.json"])
    runtime = read_json(outputs["runtime_versions.json"])
    fetch = read_json(outputs["kaggle_fetch_manifest.json"])
    failures_json = read_json(outputs["failures.json"])
    if fetch.get("job") != family["job"]:
        fail("fetch_job", "Kaggle fetch manifest job slug differs from the requested family")
    job_key = f"v5-{args.family}"
    submission_path = ROOT / "versions/sem2act-v5/manifests/kaggle_submissions" / job_key / f"{args.execution_sha}.json"
    if not submission_path.is_file():
        fail("submission_receipt_missing", "matching Kaggle kernel-source submission receipt is missing")
    else:
        submission = read_json(submission_path)
        expected_kernel = ROOT / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py"
        if (submission.get("status") != "kernel-source-saved"
                or submission.get("execution_sha") != args.execution_sha
                or submission.get("job_key") != job_key
                or submission.get("job_slug") != family["job"]
                or submission.get("protocol_hash") != PROTOCOL_HASH
                or submission.get("kernel_script") != str(expected_kernel.relative_to(ROOT))
                or submission.get("kernel_source_sha256") != sha(expected_kernel)
                or submission.get("runtime_lock_sha256") != sha(lock_path)
                or submission.get("runtime_freeze_sha256") != sha(freeze_path)
                or submission.get("dataset_upload_manifest_sha256") != sha(args.dataset_manifest)
                or sha(submission_path) != fetch.get("submission_manifest_sha256")):
            fail("submission_receipt", "Kaggle kernel submission receipt does not bind this source, bundle, and output fetch")
        if (fetch.get("execution_sha") != args.execution_sha
                or fetch.get("kernel_script") != submission.get("kernel_script")
                or fetch.get("kernel_source_sha256") != submission.get("kernel_source_sha256")
                or fetch.get("runtime_lock_sha256") != sha(lock_path)
                or fetch.get("runtime_freeze_sha256") != sha(freeze_path)
                or fetch.get("dataset_upload_manifest_sha256") != sha(args.dataset_manifest)):
            fail("fetch_provenance", "Kaggle fetch manifest does not bind the submitted execution, source, and data bundle")
    expected_fetch_hashes = {
        name: sha(outputs[name]) for name in ("raw_outputs.jsonl", "beliefs.jsonl", "failures.json", "runtime_versions.json")
    }
    if fetch.get("files") != expected_fetch_hashes:
        fail("fetch_hashes", "Kaggle fetch manifest hashes differ from the fetched output artifacts")
    verification = fetch.get("verification", {})
    if (verification.get("status") != "pass" or verification.get("n_calls") != 1920
            or verification.get("n_beliefs") != 960):
        fail("fetch_verification", "Kaggle's fetch-stage verifier did not pass exact Llama/Mistral coverage")

    if (run.get("experiment_id") != "v5-cross-family-consumer" or run.get("status") != "pass"
            or run.get("model") != family["model_id"] or run.get("revision") != family["revision"]
            or run.get("runtime_lock_sha256") != sha(lock_path)
            or run.get("n_input_rows") != 720 or run.get("n_selected_rows") != 480
            or run.get("expected_calls") != 1920 or run.get("actual_calls") != 1920
            or run.get("n_fail") != 0 or run.get("qrels_read") is not False
            or run.get("prompt_shas") != PROMPT_SHAS
            or run.get("decoding") != {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "num_beams": 1, "seed": 0}
            or run.get("quantization") != locked.get("quantization")
            or run.get("runtime_pins") != {"transformers": "4.57.6", "bitsandbytes": "0.48.1"}
            or run.get("input_sha256") != {"evidence_inputs.jsonl": sha(evidence_path), "model_spec.json": sha(spec_path)}):
        fail("run_manifest", "Kaggle consumer run manifest does not match exact model, workload, runtime, or inputs")
    if (run.get("smoke", {}).get("status") != "pass"
            or run.get("smoke", {}).get("fixture_sha256") != sha(ROOT / FIXTURE_REL)
            or run.get("smoke", {}).get("n_calls") != 2):
        fail("consumer_smoke", "the separate two-call non-lockbox model smoke did not pass")
    if failures_json != []:
        fail("failures", "Kaggle consumer failures.json must be an empty list")
    if (runtime.get("torch") != "2.10.0+cu128" or runtime.get("cuda") != "12.8"
            or runtime.get("transformers") != "4.57.6" or runtime.get("bitsandbytes") != "0.48.1"
            or not str(runtime.get("python", "")).startswith("3.12")):
        fail("runtime_versions", "Kaggle runtime package/image facts differ from the frozen lock")

    raw_rows = [json.loads(line) for line in outputs["raw_outputs.jsonl"].read_text().splitlines() if line.strip()]
    if len(raw_rows) != 1920 or {row.get("call_id") for row in raw_rows} != set(range(1, 1921)):
        fail("raw_call_coverage", "raw outputs must contain exactly call IDs 1 through 1,920")
    raw_counts: Counter = Counter()
    per_row_calls: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in raw_rows:
        qid, system, consumer = row.get("query_id"), row.get("system"), row.get("consumer")
        key = (qid, system)
        raw_counts[consumer] += 1
        per_row_calls.setdefault(key, {"C1": [], "C3": []}).setdefault(consumer, []).append(row)
        if (key not in selected or consumer not in {"C1", "C3"}
                or row.get("model") != family["model_id"] or row.get("revision") != family["revision"]
                or row.get("prompt_sha") != PROMPT_SHAS.get(consumer)
                or row.get("temperature") != 0.0 or row.get("valid") is not True
                or not isinstance(row.get("raw"), str) or not isinstance(row.get("parsed"), dict)):
            fail("raw_output_schema", f"invalid raw consumer call row {row.get('call_id')!r}")
            continue
        try:
            parsed = parse_raw(row["raw"])
            validate_response(parsed, consumer)
            if parsed != row["parsed"]:
                fail("raw_output_parse", f"parsed response differs from raw JSON at call {row.get('call_id')}")
        except Exception as exc:
            fail("raw_output_parse", f"call {row.get('call_id')}: {type(exc).__name__}: {exc}")
    if raw_counts != Counter({"C1": 480, "C3": 1440}):
        fail("raw_call_counts", "expected exactly 480 C1 and 1,440 C3 model calls")
    for key, input_row in selected.items():
        calls = per_row_calls.get(key, {})
        c1, c3 = calls.get("C1", []), calls.get("C3", [])
        expected_doc_ids = [doc["doc_id"] for doc in input_row["documents"]]
        if len(c1) != 1 or len(c3) != 3 or {row.get("doc_id") for row in c3} != set(expected_doc_ids):
            fail("raw_row_calls", f"{key[0]}/{key[1]} must have one C1 and three document-specific C3 calls")

    beliefs = [json.loads(line) for line in outputs["beliefs.jsonl"].read_text().splitlines() if line.strip()]
    expected_belief_keys = {(qid, system, consumer) for qid in QUERY_IDS for system in SYSTEMS for consumer in ("C1", "C3")}
    belief_keys = [(row.get("query_id"), row.get("system"), row.get("consumer")) for row in beliefs]
    if len(beliefs) != 960 or len(set(belief_keys)) != 960 or set(belief_keys) != expected_belief_keys:
        fail("belief_coverage", "belief outputs must cover exactly 240 queries x 2 systems x 2 consumers")
    for row in beliefs:
        probabilities = [float(row.get(name, -1)) for name in ("p_normal", "p_supplier_delay", "p_demand_surge")]
        if (any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-6):
            fail("belief_schema", f"invalid probability simplex for {row.get('query_id')}/{row.get('system')}/{row.get('consumer')}")
        input_row = selected.get((row.get("query_id"), row.get("system")), {})
        if (row.get("model") != family["model_id"] or row.get("k") != 3
                or row.get("true_regime") != input_row.get("metadata", {}).get("true_regime")
                or row.get("evidence_ids") != input_row.get("doc_ids")):
            fail("belief_provenance", "belief output model, evaluation label, or evidence IDs drift")

    output_hashes = {name: sha(path) for name, path in outputs.items()}
    checks.update({
        "bundle_manifest_sha256": bundle_sha,
        "execution_sha": args.execution_sha,
        "source_reranker_output_sha256": bundle.get("source_reranker_output_sha256"),
        "expected_queries": 240,
        "selected_evidence_rows": len(selected),
        "expected_calls": 1920,
        "actual_calls": run.get("actual_calls"),
        "belief_rows": len(beliefs),
        "output_sha256": output_hashes,
        "qrels_read": False,
    })
    return make_report(args, "FAIL" if failures else "PASS", failures, checks)


def make_report(args: argparse.Namespace, status: str,
                failures: list[dict[str, str]], checks: dict) -> dict:
    return {
        "schema_version": 1,
        "verifier": "sem2act-v5-kaggle-cross-family-consumer",
        "family": args.family,
        "status": status,
        "execution_sha": args.execution_sha,
        "bundle_manifest_sha256": args.bundle_manifest_sha256,
        "protocol_hash": PROTOCOL_HASH,
        "reasons": failures,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=tuple(FAMILIES))
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--kaggle-preflight", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args)
    except Exception as exc:
        result = make_report(args, "FAIL", [{
            "code": "verifier_exception",
            "message": f"{type(exc).__name__}: {exc}",
        }], {})
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")
    return {"PASS": 0, "FAIL": 1, "REVIEW": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
