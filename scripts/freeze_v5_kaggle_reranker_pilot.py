"""Freeze or verify the Kaggle reranker pilot execution contract; launches no job.

The pilot kernel is authorized only against this freeze.  Every gate here is
local: no quota call, no Kaggle submission, and no model load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/kaggle_reranker_runtime_lock.json"
AMENDMENT = ROOT / "versions/sem2act-v5/amendments/kaggle_reranker_pilot_v1.yaml"
PARENT_LOCK = ROOT / "versions/sem2act-v5/manifests/kaggle_runtime_lock.json"
BLOCKED = ROOT / "versions/sem2act-v5/manifests/moon_reranker_pilot_blocked.json"
FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
MODEL_REFERENCE = ROOT / "versions/sem2act-v5/manifests/cpu_reranker_canary.json"
UPLOAD = ROOT / "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json"
PROTOCOL = ROOT / "versions/sem2act-v5/protocol/confirmation.yaml"
PROTOCOL_FREEZE = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
PILOT_KERNEL = ROOT / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank_pilot.py"
OUT = ROOT / "versions/sem2act-v5/manifests/kaggle_reranker_runtime_freeze.json"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
OWNER = "rezabarati2"
INPUT_FILES = ("corpus.jsonl", "queries.jsonl", "rerank_candidates.jsonl")
PILOT_QUERY_IDS = tuple(f"v5-lb-q{index:04d}" for index in range(1, 21))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def expected_manifest() -> dict:
    for path in (LOCK, AMENDMENT, PARENT_LOCK, BLOCKED, FIXTURE, MODEL_REFERENCE,
                 UPLOAD, PROTOCOL, PROTOCOL_FREEZE, PILOT_KERNEL):
        if not path.is_file():
            raise SystemExit(f"required Kaggle reranker pilot freeze input missing: {path}")
    lock = load(LOCK)
    parent = load(PARENT_LOCK)
    blocked = load(BLOCKED)
    reference = load(MODEL_REFERENCE)
    upload = load(UPLOAD)
    frozen_protocol = load(PROTOCOL_FREEZE)
    for label, value in (("lock", lock), ("parent lock", parent), ("blocked record", blocked),
                         ("upload manifest", upload), ("protocol freeze", frozen_protocol)):
        if value.get("protocol_hash") != PROTOCOL_HASH:
            raise SystemExit(f"Kaggle reranker pilot {label} protocol hash drift")
    # The accepted CPU canary predates the protocol-hash field; it is pinned by
    # its own manifest id, model id, revision, and per-file snapshot hashes.
    if reference.get("manifest_id") != "sem2act-v5-cpu-reranker-canary-v1":
        raise SystemExit("Kaggle reranker pilot model reference manifest id drift")
    if lock.get("lock_id") != "sem2act-v5-kaggle-reranker-pilot-v1":
        raise SystemExit("Kaggle reranker pilot lock id drift")
    if lock.get("status") != "authorized-before-result-bearing-execution":
        raise SystemExit("Kaggle reranker pilot lock must remain pre-result")

    platform = lock.get("platform", {})
    if platform.get("provider") != "kaggle" or platform.get("owner") != OWNER:
        raise SystemExit("Kaggle reranker pilot provider/owner drift")
    if platform.get("machine") != "NvidiaTeslaT4" or platform.get("machine_shape_request") != "NvidiaTeslaT4":
        raise SystemExit("Kaggle reranker pilot machine drift")
    if platform.get("gpu_count") != 2 or platform.get("visible_gpu_count") != 1:
        raise SystemExit("Kaggle reranker pilot allocation/visible-GPU policy drift")
    if platform.get("secondary_gpu_policy") != "allocated-but-masked;not-used":
        raise SystemExit("Kaggle reranker pilot secondary-GPU policy drift")
    if platform.get("qrel_free_inputs_only") is not True:
        raise SystemExit("Kaggle reranker pilot must remain qrel-free")

    runtime = lock.get("runtime", {})
    for key, value in (("python", "3.12"), ("cuda", "12.8"), ("torch", "2.10.0+cu128"),
                       ("transformers", "4.57.6"), ("torch_dtype", "float16")):
        if runtime.get(key) != value:
            raise SystemExit(f"Kaggle reranker pilot runtime drift: {key}")
    parent_runtime = parent.get("runtime", {})
    for key in ("python", "cuda", "torch", "transformers", "torch_dtype"):
        if runtime.get(key) != parent_runtime.get(key):
            raise SystemExit(f"Kaggle reranker pilot runtime no longer matches the v3 parent: {key}")
    if runtime.get("package_install_policy") != "exact-pins-preserve-image-torch-cuda-tuple":
        raise SystemExit("Kaggle reranker pilot package policy drift")

    model = lock.get("model", {})
    parent_model = parent.get("models", {}).get("reranker", {})
    for key in ("model_id", "revision", "tokenizer_revision", "dtype", "attention_backend",
                "max_context_tokens", "batch_size"):
        if model.get(key) != parent_model.get(key):
            raise SystemExit(f"Kaggle reranker pilot reranker parameter changed from v3: {key}")
    if model.get("device_map") != parent_model.get("device_map"):
        raise SystemExit("Kaggle reranker pilot device placement changed from v3")
    if model.get("quantization", {}).get("method") != "none":
        raise SystemExit("Kaggle reranker pilot quantization drift")
    accepted = reference.get("model_snapshot", {})
    if reference.get("status") != "pass" or accepted.get("file_count") != 12:
        raise SystemExit("accepted reranker snapshot reference is not the frozen 12-file pass")
    if model.get("snapshot_file_count") != 12:
        raise SystemExit("Kaggle reranker pilot must require exactly 12 model files")
    if model.get("snapshot_files_sha256") != accepted.get("files_sha256"):
        raise SystemExit("Kaggle reranker pilot model snapshot aggregate drift")

    inputs = lock.get("inputs", {})
    if inputs.get("dataset") != f"{OWNER}/sem2act-v5-rerank-inputs" or inputs.get("version") != 1:
        raise SystemExit("Kaggle reranker pilot input dataset drift")
    if sorted(inputs.get("files", {})) != sorted(INPUT_FILES):
        raise SystemExit("Kaggle reranker pilot input file set drift")
    if inputs.get("files") != upload.get("remote_sha256") or upload.get("remote_sha256") != upload.get("files"):
        raise SystemExit("Kaggle reranker pilot input hashes disagree with the remote-verified upload manifest")
    if upload.get("status") != "remote-verified" or upload.get("qrels_uploaded") is not False:
        raise SystemExit("Kaggle reranker pilot input dataset is not remote-verified qrel-free")

    pilot = lock.get("pilot", {})
    if tuple(pilot.get("query_ids", ())) != PILOT_QUERY_IDS:
        raise SystemExit("Kaggle reranker pilot query keyspace drift")
    if pilot.get("queries") != 20 or pilot.get("candidate_depth") != 50:
        raise SystemExit("Kaggle reranker pilot query/depth drift")
    if pilot.get("expected_scores") != 1000 or pilot.get("expected_ranking_rows") != 1000:
        raise SystemExit("Kaggle reranker pilot expected-score drift")
    if pilot.get("shard_index") != 0 or pilot.get("execution_order") != "frozen_query_order_sequential":
        raise SystemExit("Kaggle reranker pilot shard/order drift")

    full = lock.get("full_reranker_conditional_authorization", {})
    if full.get("status") != "not-authorized-until-pilot-accepted-by-codex":
        raise SystemExit("full reranker conditional authorization must stay locked behind pilot acceptance")
    if full.get("job") != "v5-rerank":
        raise SystemExit("conditional authorization job drift")

    if blocked.get("scientific_output_accepted") is not False:
        raise SystemExit("Moon blocked record must not claim accepted scientific output")
    if blocked.get("accepted_pairs") != 0 or blocked.get("accepted_queries") != 0:
        raise SystemExit("Moon blocked record must record zero accepted work")
    if blocked.get("qrels_read") is not False:
        raise SystemExit("Moon blocked record must record that no qrels were read")

    source_hashes = {}
    for relative in lock.get("source_contract", []):
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Kaggle reranker pilot source contract file missing: {relative}")
        source_hashes[relative] = sha(path)
    return {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-kaggle-reranker-runtime-freeze-v1",
        "status": "frozen-before-result-bearing-execution;owner-quota-and-canary-gated",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "lock_id": lock["lock_id"],
        "lock_sha256": sha(LOCK),
        "amendment_sha256": sha(AMENDMENT),
        "parent_lock_id": parent.get("lock_id"),
        "parent_lock_sha256": sha(PARENT_LOCK),
        "pilot_kernel": str(PILOT_KERNEL.relative_to(ROOT)),
        "pilot_kernel_sha256": sha(PILOT_KERNEL),
        "moon_blocked_record_sha256": sha(BLOCKED),
        "model_snapshot_files_sha256": accepted["files_sha256"],
        "smoke_fixture_sha256": sha(FIXTURE),
        "upload_manifest_sha256": sha(UPLOAD),
        "protocol_sha256": sha(PROTOCOL),
        "authorized_pilot_queries": 20,
        "authorized_pilot_scores": 1000,
        "full_reranker_authorized": False,
        "source_contract_sha256": source_hashes,
        "failure_policy": lock["failure_policy"],
        "no_result_dependent_runtime_changes": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = expected_manifest()
    if args.check:
        if not OUT.is_file():
            raise SystemExit(f"Kaggle reranker pilot freeze missing: {OUT}")
        actual = json.loads(OUT.read_text())
        for key, value in expected.items():
            if key != "frozen_utc" and actual.get(key) != value:
                raise SystemExit(f"Kaggle reranker pilot freeze drift: {key}")
        print(json.dumps(actual, indent=2, sort_keys=True))
    else:
        OUT.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
