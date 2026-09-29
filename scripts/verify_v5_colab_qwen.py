"""Verify Muse's frozen Colab Qwen result-stage outputs without inference."""

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
EXPECTED_SHA256 = "9ab69722e0062fc38eb27721f24e9cd83ca7944655973cbbe85f079372ccd74b"
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
MODEL_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}
SYSTEMS = ("bm25", "rerank", "oracle")
QUERY_IDS = tuple(f"v5-lb-q{i:04d}" for i in range(1, 241))
LOCK_REL = "versions/sem2act-v5/manifests/colab_runtime_lock.json"
FREEZE_REL = "versions/sem2act-v5/manifests/colab_runtime_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cache_files(rows: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        docs = row["documents"]
        c1_user = (
            f"Operator information need: {row['query_text']}\n\nEvidence:\n"
            + "\n\n".join(f"[DOC {doc['doc_id']}] {doc['text']}" for doc in docs)
        )
        users = [("C1", c1_user)]
        users.extend(("C3", f"Operator's own node: {row['metadata']['entity_node']}\n\n"
                              f"Evidence document:\n{doc['text']}") for doc in docs)
        for consumer, user in users:
            prompt_sha = PROMPT_SHAS[consumer]
            user_sha = hashlib.sha256(user.encode()).hexdigest()[:16]
            key_source = f"{MODEL_ID}||{prompt_sha}||{user_sha}"
            filename = hashlib.sha256(key_source.encode()).hexdigest()[:16] + ".json"
            if filename in result and result[filename] != prompt_sha:
                raise ValueError("cache-key collision across consumer prompts")
            result[filename] = prompt_sha
    return result


def validate_raw_response(raw: str, prompt_sha: str) -> None:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text.split("\n", 1)[1].rsplit(fence, 1)[0].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("response JSON is not an object")
    if prompt_sha == PROMPT_SHAS["C1"]:
        required = {
            "normal", "supplier_delay", "demand_surge",
            "estimated_lt_increase", "estimated_duration", "estimated_demand_multiplier",
        }
        if set(obj) != required:
            raise ValueError("C1 raw response schema keys drift")
        probs = [float(obj[key]) for key in ("normal", "supplier_delay", "demand_surge")]
        if any(not 0 <= value <= 1 for value in probs) or abs(sum(probs) - 1.0) > 0.05:
            raise ValueError("C1 raw response probability schema invalid")
        for key in ("estimated_lt_increase", "estimated_duration"):
            value = int(obj[key])
            if value < 0 or float(obj[key]) != value:
                raise ValueError(f"C1 raw response {key} is not a nonnegative integer")
        if float(obj["estimated_demand_multiplier"]) < 0:
            raise ValueError("C1 raw response multiplier is negative")
    else:
        required = {"entity_match", "event", "fresh", "stance", "confidence"}
        if set(obj) != required:
            raise ValueError("C3 raw response schema keys drift")
        if not isinstance(obj["entity_match"], bool) or not isinstance(obj["fresh"], bool):
            raise ValueError("C3 raw response boolean field invalid")
        if obj["event"] not in {"supplier_delay", "demand_surge", "normal", "none"}:
            raise ValueError("C3 raw response event invalid")
        if obj["stance"] not in {"support", "refute", "na"}:
            raise ValueError("C3 raw response stance invalid")
        if not 0 <= float(obj["confidence"]) <= 1:
            raise ValueError("C3 raw response confidence invalid")


def verify(args: argparse.Namespace) -> dict:
    fails: list[dict[str, str]] = []
    reviews: list[dict[str, str]] = []
    checks: dict[str, object] = {}

    def fail(code: str, message: str) -> None:
        fails.append({"code": code, "message": message})

    def review(code: str, message: str) -> None:
        reviews.append({"code": code, "message": message})

    if not re.fullmatch(r"[0-9a-f]{40}", args.execution_sha):
        fail("execution_sha_format", "execution SHA must be a full lowercase Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", args.bundle_manifest_sha256):
        fail("bundle_sha_format", "Qwen bundle manifest SHA must be lowercase SHA-256")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "-uall"], cwd=ROOT, text=True)
        if head != args.execution_sha or dirty:
            fail("execution_checkout", "Qwen verifier checkout must be clean at the exact handoff SHA")
    except (OSError, subprocess.CalledProcessError) as exc:
        fail("execution_checkout", f"could not verify exact clean Git checkout: {type(exc).__name__}")
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    bundle_manifest_path = bundle_root / "consumer_inputs_qwen.json"
    if not bundle_manifest_path.is_file():
        return make_report(args, "FAIL" if fails else "REVIEW", fails, [{"code": "bundle_manifest_missing", "message": str(bundle_manifest_path)}], checks)
    bundle_manifest_sha = sha256(bundle_manifest_path)
    if bundle_manifest_sha != args.bundle_manifest_sha256:
        fail("bundle_manifest_hash", "Qwen bundle manifest does not match the handoff SHA")
    bundle = json.loads(bundle_manifest_path.read_text())
    if "qrel" in bundle_manifest_path.read_text().lower():
        fail("bundle_manifest_qrel", "Qwen bundle manifest contains qrel material")
    if (bundle.get("family") != "qwen"
            or bundle.get("status") != "built-from-accepted-reranker-output"
            or bundle.get("protocol_hash") != PROTOCOL_HASH):
        fail("bundle_identity", "Qwen bundle family or protocol hash drift")
    if bundle.get("execution_sha") != args.execution_sha:
        fail("bundle_execution_sha", "Qwen bundle was not built from the handoff execution SHA")
    bundle_files = bundle.get("files", {})
    if any(path.is_dir() for path in bundle_root.iterdir()):
        fail("bundle_directory", "Qwen bundle root must contain files only")
    actual_bundle_files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in bundle_root.iterdir() if path.is_file() and path != bundle_manifest_path
    }
    if bundle_files != actual_bundle_files:
        fail("bundle_file_hashes", "Qwen bundle files differ from the manifested file hashes")
    for name in actual_bundle_files:
        if "qrel" in (bundle_root / name).read_text(errors="ignore").lower():
            fail("bundle_qrels", f"qrel token found in manifested input {name}")
    for required in ("evidence_inputs.jsonl", "model_inputs_manifest.json", "model_spec.json", "runtime_smoke_v1.json",
                    "consumers_v3.py", "interpreter.py", "events.py"):
        if required not in actual_bundle_files:
            fail("bundle_file_missing", f"Qwen bundle is missing {required}")

    spec_path = bundle_root / "model_spec.json"
    if not spec_path.is_file():
        return make_report(args, "FAIL" if fails else "REVIEW", fails, reviews + [{"code": "model_spec_missing", "message": "Qwen model spec is missing"}], checks)
    spec = json.loads(spec_path.read_text())
    model_inputs_manifest_path = bundle_root / "model_inputs_manifest.json"
    if not model_inputs_manifest_path.is_file():
        return make_report(args, "FAIL" if fails else "REVIEW", fails,
                           reviews + [{"code": "model_inputs_manifest_missing", "message": "model_inputs_manifest.json is missing"}], checks)
    model_inputs_manifest = json.loads(model_inputs_manifest_path.read_text())
    if (spec.get("model_id") != MODEL_ID or spec.get("revision") != MODEL_REVISION
            or spec.get("tokenizer_revision") != MODEL_REVISION
            or spec.get("protocol_hash") != PROTOCOL_HASH
            or spec.get("expected_protocol_hash") != PROTOCOL_HASH
            or spec.get("execution_sha") != args.execution_sha):
        fail("model_revision", "Qwen model/tokenizer revision or protocol pin drift")
    if bundle.get("revision") != MODEL_REVISION or bundle.get("model_id") != MODEL_ID:
        fail("bundle_model_pin", "Qwen bundle manifest model/revision drift")
    if (bundle.get("runtime_lock_sha256") != spec.get("runtime_lock_sha256")
            or bundle.get("model_spec_sha256") != sha256(spec_path)):
        fail("bundle_config_hash", "Qwen bundle manifest does not bind the model spec/runtime lock")
    protocol_path = ROOT / "versions/sem2act-v5/protocol/confirmation.yaml"
    if bundle.get("protocol_file_sha256") != sha256(protocol_path):
        fail("bundle_protocol_file", "Qwen bundle does not bind the frozen confirmation protocol file")
    source_rerank_sha = bundle.get("source_reranker_output_sha256")
    if (not isinstance(source_rerank_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_rerank_sha) is None
            or spec.get("source_reranker_output_sha256") != source_rerank_sha
            or model_inputs_manifest.get("source_reranker_output_sha256") != source_rerank_sha
            or spec.get("source_manifest_sha256") != sha256(model_inputs_manifest_path)):
        fail("bundle_source_hash", "Qwen bundle does not bind the accepted reranker output/source manifest")
    for name in ("source_retrieval_manifest_sha256", "source_reranker_manifest_sha256",
                 "source_reranker_verification_sha256"):
        digest = bundle.get(name)
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or model_inputs_manifest.get(name) != digest):
            fail("bundle_source_lineage", f"Qwen bundle does not bind a valid {name}")
    if (model_inputs_manifest.get("n_queries") != 240
            or model_inputs_manifest.get("status") != "accepted-reranker-inputs-only"
            or model_inputs_manifest.get("n_rows") != 720
            or model_inputs_manifest.get("evaluation_labels_included") is not True
            or model_inputs_manifest.get("evaluation_label_field") != "metadata.true_regime"
            or model_inputs_manifest.get("model_prompt_label_exposure") is not False
            or model_inputs_manifest.get("relevance_judgments_included") is not False
            or model_inputs_manifest.get("source_reranker_output_sha256") != source_rerank_sha
            or model_inputs_manifest.get("rows_per_system") != {name: 240 for name in SYSTEMS}
            or model_inputs_manifest.get("files") != {"evidence_inputs.jsonl": {
                "sha256": sha256(bundle_root / "evidence_inputs.jsonl"),
                "bytes": (bundle_root / "evidence_inputs.jsonl").stat().st_size,
            }}
            or bundle.get("source_input_sha256") != model_inputs_manifest.get("input_sha256")):
        fail("model_inputs_manifest", "Qwen model-input source manifest workload/provenance drift")
    if (bundle.get("evaluation_labels_included") is not True
            or bundle.get("model_prompt_label_exposure") is not False):
        fail("bundle_label_provenance", "Qwen bundle must disclose evaluation labels and assert they are not prompt inputs")
    if spec.get("expected_calls") != 2880 or bundle.get("expected_workload") != {
            "queries": 240, "evidence_rows": 720, "calls": 2880}:
        fail("bundle_workload", "Qwen bundle does not bind 240 queries / 720 rows / 2,880 calls")
    if (spec.get("prompt_shas") != PROMPT_SHAS or bundle.get("prompt_shas") != PROMPT_SHAS
            or spec.get("temperature") != 0.0 or spec.get("top_p") != 1.0
            or spec.get("seed") != 0 or spec.get("max_new_tokens") != 256):
        fail("bundle_decoding", "Qwen prompt or decoding configuration drift")
    evidence_path = bundle_root / "evidence_inputs.jsonl"
    if not evidence_path.is_file():
        return make_report(args, "FAIL" if fails else "REVIEW", fails, reviews + [{"code": "evidence_missing", "message": "evidence_inputs.jsonl is missing"}], checks)
    rows = [json.loads(line) for line in evidence_path.read_text().splitlines() if line.strip()]
    expected_row_keys = {(qid, system) for qid in QUERY_IDS for system in SYSTEMS}
    row_keys = [(row.get("query_id"), row.get("system")) for row in rows]
    if len(rows) != 720 or len(set(row_keys)) != 720 or set(row_keys) != expected_row_keys:
        fail("evidence_coverage", "evidence bundle must contain exactly 720 unique query/system rows")
    for row in rows:
        if len(row.get("documents", [])) != 3 or row.get("doc_ids") != [d.get("doc_id") for d in row.get("documents", [])]:
            fail("evidence_schema", f"{row.get('query_id')}/{row.get('system')}: evidence must have three consistent documents")
        if (not isinstance(row.get("metadata"), dict)
                or row["metadata"].get("true_regime") not in {"normal", "supplier_delay", "demand_surge"}
                or not row["metadata"].get("entity_node")):
            fail("evaluation_metadata", f"{row.get('query_id')}/{row.get('system')}: frozen evaluation metadata is incomplete")

    preflight_path = output_root / "colab_preflight.json"
    job_path = output_root / "qwen_lightning_job_manifest.json"
    smoke_path = output_root / "qwen_smoke_manifest.json"
    run_path = output_root / "run_manifest.json"
    failures_path = output_root / "failures.json"
    raw_path = output_root / "raw_outputs.jsonl"
    beliefs_path = output_root / "beliefs.jsonl"
    runtime_path = output_root / "runtime_versions.json"
    required = (preflight_path, job_path, smoke_path, run_path, failures_path,
                raw_path, beliefs_path, runtime_path)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return make_report(args, "FAIL" if fails else "REVIEW", fails, reviews + [{"code": "output_files_missing", "message": str(missing)}], checks)
    preflight = json.loads(preflight_path.read_text())
    job = json.loads(job_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    run = json.loads(run_path.read_text())
    failures = json.loads(failures_path.read_text())
    runtime_versions = json.loads(runtime_path.read_text())

    lock_path = ROOT / LOCK_REL
    freeze_path = ROOT / FREEZE_REL
    if not lock_path.is_file() or not freeze_path.is_file():
        fail("runtime_contract_missing", "Colab runtime lock/freeze missing from checkout")
    else:
        lock_sha, freeze_sha = sha256(lock_path), sha256(freeze_path)
        colab_lock = json.loads(lock_path.read_text())
        colab_freeze = json.loads(freeze_path.read_text())
        checks["colab_runtime_lock_sha256"] = lock_sha
        checks["colab_runtime_freeze_sha256"] = freeze_sha
        if preflight.get("runtime_lock_sha256") != lock_sha or preflight.get("runtime_freeze_sha256") != freeze_sha:
            fail("runtime_contract_hash", "Colab preflight does not bind the checked-in runtime lock/freeze")
        if spec.get("runtime_lock_sha256") != lock_sha:
            fail("bundle_colab_lock", "Qwen bundle must be built against the frozen Colab runtime lock")
        locked_model = colab_lock.get("models", {}).get("qwen", {})
        runtime_fields = (
            "backend", "dtype", "quantization", "device_map", "attention_backend",
            "max_context_tokens", "max_input_tokens", "max_new_tokens", "batch_size",
            "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
            "enforce_eager", "enable_thinking", "do_sample", "num_beams", "use_cache", "tokenizer",
        )
        expected_model_runtime = {key: locked_model[key] for key in runtime_fields if key in locked_model}
        if spec.get("runtime") != expected_model_runtime:
            fail("bundle_runtime_config", "Qwen model configuration differs from the Colab runtime lock")
        if (spec.get("runtime_lock_id") != colab_lock.get("lock_id")
                or bundle.get("runtime_lock_id") != colab_lock.get("lock_id")):
            fail("bundle_runtime_lock_id", "Qwen bundle runtime-lock ID drift")
        if (colab_lock.get("status") != "frozen-before-result-bearing-execution"
                or colab_lock.get("lock_id") != "sem2act-v5-colab-runtime-v1"):
            fail("colab_lock_policy", "Qwen execution does not use the frozen Colab runtime policy")
        source_contract = colab_freeze.get("source_contract_sha256", {})
        if not source_contract:
            fail("colab_source_contract", "Colab runtime freeze has no source contract")
        for relative, expected_sha in source_contract.items():
            source_path = ROOT / relative
            if not source_path.is_file() or sha256(source_path) != expected_sha:
                fail("colab_source_contract", f"Colab source contract drift: {relative}")

    if preflight.get("status") != "pass" or preflight.get("execution_sha") != args.execution_sha:
        fail("colab_preflight", "Colab preflight status or execution SHA mismatch")
    if preflight.get("input_manifest_sha256") != bundle_manifest_sha:
        fail("preflight_bundle_hash", "Colab preflight does not bind the Qwen bundle manifest SHA")
    if preflight.get("input_files") != bundle_files or preflight.get("qrels_read") is not False:
        fail("preflight_inputs", "Colab preflight input hashes or qrel-free assertion drift")
    runtime = preflight.get("runtime", {})
    if (runtime.get("torch") != "2.11.0+cu128" or runtime.get("cuda") != "12.8"
            or runtime.get("device_count") != 1 or "t4" not in runtime.get("device", "").lower()
            or int(runtime.get("device_total_memory_bytes", 0)) < 14 * 2**30
            or not runtime.get("python")):
        fail("colab_runtime", "Colab runtime is not the frozen one-T4 torch 2.11.0+cu128 profile")
    if job.get("status") != "pass" or job.get("stage") != "qwen":
        fail("job_manifest", "Lightning Qwen job manifest did not pass")
    if job.get("protocol_hash") != PROTOCOL_HASH or job.get("qrels_read") is not False:
        fail("job_provenance", "Qwen job protocol/qrel provenance drift")
    if job.get("execution_runtime_lock_sha256") != preflight.get("runtime_lock_sha256"):
        fail("job_runtime_lock", "Qwen job manifest runtime lock differs from Colab preflight")
    if job.get("runtime_freeze_manifest_sha256") != preflight.get("runtime_freeze_sha256"):
        fail("job_runtime_freeze", "Qwen job manifest freeze differs from Colab preflight")
    expected_job_inputs = dict(bundle_files)
    expected_job_inputs[bundle_manifest_path.name] = {
        "sha256": bundle_manifest_sha,
        "bytes": bundle_manifest_path.stat().st_size,
    }
    if job.get("input_files") != expected_job_inputs:
        fail("job_input_hashes", "Qwen job manifest does not bind every transferred bundle file")
    if (smoke.get("status") != "pass" or smoke.get("fixture_sha256") != EXPECTED_SHA256
            or smoke.get("model_id") != MODEL_ID or smoke.get("revision") != MODEL_REVISION
            or smoke.get("tokenizer_revision") != MODEL_REVISION
            or smoke.get("protocol_hash") != PROTOCOL_HASH
            or smoke.get("runtime_lock_sha256") != preflight.get("runtime_lock_sha256")
            or smoke.get("runtime_freeze_manifest_sha256") != preflight.get("runtime_freeze_sha256")):
        fail("qwen_smoke", "Qwen non-lockbox smoke did not pass against the frozen fixture")

    if run.get("experiment_id") != "v5-primary-qwen-consumer" or run.get("status") != "pass":
        fail("qwen_run_status", "Qwen kernel run manifest did not pass")
    if run.get("model") != MODEL_ID or run.get("revision") != MODEL_REVISION:
        fail("qwen_run_model", "Qwen kernel output model revision drift")
    if run.get("n_input_rows") != 720 or run.get("expected_calls") != 2880 or run.get("actual_calls") != 2880:
        fail("qwen_call_count", "Qwen stage must bind 720 inputs and exactly 2,880 calls")
    if run.get("n_cache_files") != 2880 or run.get("n_fail") != 0 or failures != []:
        fail("qwen_failures", "Qwen stage must have 2,880 cache files and zero failures")
    if run.get("qrels_read") is not False or run.get("prompt_shas") != PROMPT_SHAS:
        fail("qwen_protocol", "Qwen run qrel or prompt hash drift")
    if run.get("input_sha256") != {
            "evidence_inputs.jsonl": sha256(evidence_path),
            "model_spec.json": sha256(spec_path)}:
        fail("qwen_input_hashes", "Qwen run manifest input SHA-256 values drift")
    if run.get("decoding") != {"temperature": 0.0, "top_p": 1.0, "enable_thinking": False}:
        fail("qwen_decoding", "Qwen run decoding configuration differs from the frozen policy")
    if (runtime_versions.get("torch") != "2.11.0+cu128"
            or runtime_versions.get("vllm") != "0.11.0"
            or runtime_versions.get("transformers") != "4.57.6"):
        fail("qwen_package_versions", "Qwen runtime package versions differ from frozen pins")
    if run.get("runtime_pins") != {"vllm": "0.11.0", "openai": "2.48.0", "pydantic": "2.12.5"}:
        fail("qwen_runtime_pins", "Qwen run manifest package pins drift")
    package_versions = job.get("package_versions", {})
    if any(package_versions.get(name) != version for name, version in (
            ("vllm", "0.11.0"), ("openai", "2.48.0"), ("pydantic", "2.12.5"),
            ("transformers", "4.57.6"))):
        fail("qwen_job_packages", "Qwen job package versions differ from the frozen pins")
    if job.get("environment", {}).get("torch") != "2.11.0+cu128" or job.get("environment", {}).get("cuda_version") != "12.8":
        fail("job_environment", "Qwen job environment Torch/CUDA evidence drift")
    if (job.get("environment", {}).get("python") != runtime.get("python")
            or runtime_versions.get("python") != runtime.get("python")):
        fail("python_image_policy", "Colab base Python image version was not consistently recorded")
    expected_source_hashes = {
        name: actual_bundle_files[name]["sha256"]
        for name in ("consumers_v3.py", "interpreter.py", "events.py")
        if name in actual_bundle_files
    }
    if run.get("source_sha256") != expected_source_hashes:
        fail("qwen_source_hashes", "Qwen kernel source files differ from the manifested input bundle")

    try:
        expected_caches = expected_cache_files(rows)
        raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        actual_raw_caches = [row.get("cache_file") for row in raw_rows]
        if len(raw_rows) != 2880 or len(set(actual_raw_caches)) != 2880 or set(actual_raw_caches) != set(expected_caches):
            fail("raw_call_coverage", "raw output cache keys do not cover the exact 2,880 expected model calls")
        call_prompt_counts = Counter()
        for row in raw_rows:
            if row.get("model") != MODEL_ID or row.get("prompt_sha") not in PROMPT_SHAS.values() or not row.get("raw"):
                fail("raw_output_schema", "raw output row lacks pinned model/prompt or non-empty response")
            filename = row.get("cache_file")
            if expected_caches.get(filename) != row.get("prompt_sha"):
                fail("raw_output_binding", f"raw output {filename!r} is not bound to its expected prompt")
            call_prompt_counts[row.get("prompt_sha")] += 1
            validate_raw_response(row.get("raw", ""), row.get("prompt_sha", ""))
        if call_prompt_counts != Counter({PROMPT_SHAS["C1"]: 720, PROMPT_SHAS["C3"]: 2160}):
            fail("consumer_call_counts", "Qwen output does not contain 720 C1 and 2,160 C3 calls")
        cache_dir = output_root / "cache_out"
        cache_files = {path.name: path for path in cache_dir.glob("*.json")} if cache_dir.is_dir() else {}
        if set(cache_files) != set(expected_caches):
            fail("cache_checkpoint_coverage", "cache directory is not the exact set of 2,880 expected call checkpoints")
        for name, path in cache_files.items():
            item = json.loads(path.read_text())
            if item.get("model") != MODEL_ID or item.get("prompt_sha") != expected_caches.get(name):
                fail("cache_model", f"cache model pin drift in {name}")
    except Exception as exc:
        fail("raw_output_parse", f"could not validate Qwen raw outputs: {type(exc).__name__}: {exc}")

    belief_rows = [json.loads(line) for line in beliefs_path.read_text().splitlines() if line.strip()]
    expected_belief_keys = {
        (qid, system, consumer) for qid in QUERY_IDS for system in SYSTEMS for consumer in ("C1", "C3")
    }
    belief_keys = [(row.get("query_id"), row.get("system"), row.get("consumer")) for row in belief_rows]
    if len(belief_rows) != 1440 or len(set(belief_keys)) != 1440 or set(belief_keys) != expected_belief_keys:
        fail("belief_coverage", "Qwen belief output must cover 240 x 3 systems x 2 consumers exactly")
    for row in belief_rows:
        probs = [float(row.get(name, -1)) for name in ("p_normal", "p_supplier_delay", "p_demand_surge")]
        if (any(not math.isfinite(value) or value < 0 or value > 1 for value in probs)
                or abs(sum(probs) - 1.0) > 1e-6):
            fail("belief_schema", f"invalid probability simplex for {row.get('query_id')}/{row.get('system')}/{row.get('consumer')}")

    output_hashes = {
        path.relative_to(output_root).as_posix(): sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.resolve() != args.output.resolve()
    }
    if len([name for name in output_hashes if name.startswith("cache_out/") and name.endswith(".json")]) != 2880:
        fail("output_checkpoint_count", "all 2,880 cache checkpoints must be included in the output manifest")
    expected_root_files = {
        "colab_preflight.json", "qwen_lightning_job_manifest.json", "qwen_smoke_manifest.json",
        "qwen_vllm_smoke.log", "run_manifest.json", "failures.json", "raw_outputs.jsonl",
        "beliefs.jsonl", "runtime_versions.json",
    }
    expected_nested_files = {
        *(f"cache_out/{name}" for name in expected_caches),
        "v5src/src/__init__.py", "v5src/src/consumers_v3.py",
        "v5src/src/interpreter.py", "v5src/src/events.py",
    }
    if set(output_hashes) != expected_root_files | expected_nested_files:
        fail("output_manifest_completeness", "Qwen output tree contains missing or unmanifested files")
    expected_directories = {"cache_out", "v5src", "v5src/src"}
    actual_directories = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*") if path.is_dir()
    }
    if actual_directories != expected_directories:
        fail("output_directory_completeness", "Qwen output tree contains missing or unmanifested directories")
    checks.update({
        "bundle_manifest_sha256": bundle_manifest_sha,
        "source_reranker_output_sha256": source_rerank_sha,
        "expected_queries": 240,
        "evidence_rows": len(rows),
        "expected_calls": 2880,
        "actual_calls": run.get("actual_calls"),
        "belief_rows": len(belief_rows),
        "qrels_read": False,
        "output_sha256": output_hashes,
    })
    return make_report(args, "FAIL" if fails else "REVIEW" if reviews else "PASS", fails, reviews, checks)


def make_report(args: argparse.Namespace, status: str, fails: list[dict],
                reviews: list[dict], checks: dict) -> dict:
    return {
        "schema_version": 1,
        "verifier": "sem2act-v5-colab-qwen",
        "status": status,
        "execution_sha": args.execution_sha,
        "bundle_manifest_sha256": args.bundle_manifest_sha256,
        "protocol_hash": PROTOCOL_HASH,
        "reasons": fails + reviews,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args)
    except Exception as exc:
        result = make_report(args, "FAIL", [{
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
