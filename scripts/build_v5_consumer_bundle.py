"""Build an immutable qrel-free consumer bundle from accepted v5 retrieval data.

The command-line path is deliberately gated on both a passing full-reranker
verification record and a separate Codex acceptance receipt. The pure builder
is exposed for synthetic fixture tests and never reads repository result data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
PROTOCOL_PATH = ROOT / "versions/sem2act-v5/protocol/confirmation.yaml"
SMOKE_FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}
SYSTEMS = ("bm25", "rerank", "oracle")
FAMILIES = {
    "qwen": {"model_id": "Qwen/Qwen3-8B-AWQ", "revision": "4da05a8edb55c6046cce958586c33b61da07bb79", "calls": 2880},
    "llama": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "calls": 1920},
    "mistral": {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "revision": "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef", "calls": 1920},
}
EXPECTED_QUERY_IDS = tuple(f"v5-lb-q{i:04d}" for i in range(1, 241))
EXPECTED_RERANK_INPUT_HASHES = {
    "queries.jsonl": "c457d93ea195d82a738219ad152382380d89b4b5e0c3fabde5cf4d6d33434986",
    "corpus.jsonl": "6cab24bc7420635674856f35345cd4c5c30a23d2d0ee243a79b69d36bc8ab0e1",
    "rerank_candidates.jsonl": "11dcac88ad210447ac16f78decbb5dd09776648e14b083aa3d89f67c8b701803",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_trec(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"{path.name}:{line_number}: expected six TREC columns")
        qid, _, doc_id, rank_text = parts[:4]
        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid rank") from exc
        grouped.setdefault(qid, []).append((rank, doc_id))
    result = {}
    for qid, pairs in grouped.items():
        if len({rank for rank, _ in pairs}) != len(pairs) or len({doc for _, doc in pairs}) != len(pairs):
            raise ValueError(f"{path.name}: duplicate rank or document for {qid}")
        if sorted(rank for rank, _ in pairs) != list(range(1, len(pairs) + 1)):
            raise ValueError(f"{path.name}: ranks for {qid} are not contiguous from one")
        result[qid] = [doc for _, doc in sorted(pairs)]
    return result


def clean_execution_sha() -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "-uall"], cwd=ROOT, text=True)
    if len(head) != 40 or dirty:
        raise ValueError("accepted consumer bundles require a clean full-SHA execution checkout")
    return head


def build_bundle(*, family: str, output_dir: Path, queries: list[dict],
                 corpus: dict[str, dict], rankings: dict[str, dict[str, list[str]]],
                 model_spec: dict, runtime_lock_sha256: str,
                 retrieval_manifest_sha256: str, reranker_manifest_sha256: str,
                 reranker_verification_sha256: str, reranker_output_sha256: str,
                 input_hashes: dict[str, str], execution_sha: str) -> dict:
    """Write deterministic artifacts. Callers supply data; this function does no inference."""
    if family not in FAMILIES:
        raise ValueError(f"unsupported family: {family}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty consumer bundle: {output_dir}")
    qids = [row.get("_id") for row in queries]
    if len(qids) != 240 or set(qids) != set(EXPECTED_QUERY_IDS) or len(set(qids)) != 240:
        raise ValueError("consumer bundle requires the exact 240-query v5 workload")
    if set(rankings) != set(SYSTEMS):
        raise ValueError("rankings must contain bm25, rerank, and oracle")
    for system in SYSTEMS:
        if set(rankings[system]) != set(EXPECTED_QUERY_IDS):
            raise ValueError(f"{system} rankings do not cover the exact 240 query IDs")
        if any(len(rankings[system][qid]) < 3 or len(set(rankings[system][qid])) != len(rankings[system][qid])
               for qid in EXPECTED_QUERY_IDS):
            raise ValueError(f"{system} rankings must have at least three unique documents per query")
    rows = []
    for query in sorted(queries, key=lambda row: row["_id"]):
        qid = query["_id"]
        for system in SYSTEMS:
            top = rankings[system][qid][:3]
            if any(doc_id not in corpus for doc_id in top):
                raise ValueError(f"{system}/{qid}: ranking refers to a missing corpus document")
            rows.append({
                "query_id": qid,
                "query_text": query["text"],
                "system": system,
                "doc_ids": top,
                "documents": [{"doc_id": did, "text": corpus[did]["text"]} for did in top],
                "metadata": query["metadata"],
            })
    if len(rows) != 720:
        raise AssertionError(f"expected 720 evidence rows, got {len(rows)}")
    if any("qrel" in json.dumps(row, sort_keys=True).lower() for row in rows):
        raise ValueError("qrel token detected in consumer evidence rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence_inputs.jsonl"
    evidence_path.write_bytes(b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode() for row in rows
    ))
    shutil.copyfile(SMOKE_FIXTURE, output_dir / "runtime_smoke_v1.json")

    model_input_manifest = {
        "schema_version": 1,
        "experiment_id": "v5-model-inputs",
        "status": "accepted-reranker-inputs-only",
        "evaluation_labels_included": True,
        "evaluation_label_field": "metadata.true_regime",
        "model_prompt_label_exposure": False,
        "relevance_judgments_included": False,
        "n_queries": 240,
        "n_rows": len(rows),
        "rows_per_system": {system: 240 for system in SYSTEMS},
        "source_retrieval_manifest_sha256": retrieval_manifest_sha256,
        "source_reranker_manifest_sha256": reranker_manifest_sha256,
        "source_reranker_verification_sha256": reranker_verification_sha256,
        "source_reranker_output_sha256": reranker_output_sha256,
        "input_sha256": dict(sorted(input_hashes.items())),
        "files": {"evidence_inputs.jsonl": {
            "sha256": sha256_file(evidence_path), "bytes": evidence_path.stat().st_size,
        }},
    }
    model_input_path = output_dir / "model_inputs_manifest.json"
    model_input_path.write_bytes(json_bytes(model_input_manifest))

    revision_lock = model_spec["revision"]
    bundle_spec = {
        "schema_version": 1,
        "experiment_id": "v5-cross-family-consumer",
        "model_family": family,
        "execution_sha": execution_sha,
        "model_id": model_spec["model_id"],
        "revision": revision_lock,
        "tokenizer_revision": model_spec["tokenizer_revision"],
        "protocol_hash": PROTOCOL_HASH,
        "expected_protocol_hash": PROTOCOL_HASH,
        "prompt_shas": PROMPT_SHAS,
        "runtime_lock_id": model_spec["runtime_lock_id"],
        "runtime_lock_sha256": runtime_lock_sha256,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_new_tokens": 256,
        "runtime": model_spec["runtime"],
        "source_manifest_sha256": sha256_file(model_input_path),
        "source_reranker_output_sha256": reranker_output_sha256,
        "expected_calls": FAMILIES[family]["calls"],
    }
    spec_path = output_dir / "model_spec.json"
    spec_path.write_bytes(json_bytes(bundle_spec))
    if family == "qwen":
        for name in ("consumers_v3.py", "interpreter.py", "events.py"):
            shutil.copyfile(ROOT / "src" / name, output_dir / name)

    files = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    bundle_manifest = {
        "schema_version": 1,
        "family": family,
        "execution_sha": execution_sha,
        "status": "built-from-accepted-reranker-output",
        "model_id": bundle_spec["model_id"],
        "revision": revision_lock,
        "protocol_hash": PROTOCOL_HASH,
        "runtime_lock_id": model_spec["runtime_lock_id"],
        "runtime_lock_sha256": runtime_lock_sha256,
        "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
        "model_spec_sha256": sha256_file(spec_path),
        "source_retrieval_manifest_sha256": retrieval_manifest_sha256,
        "source_reranker_manifest_sha256": reranker_manifest_sha256,
        "source_reranker_verification_sha256": reranker_verification_sha256,
        "source_reranker_output_sha256": reranker_output_sha256,
        "source_input_sha256": dict(sorted(input_hashes.items())),
        "expected_workload": {
            "queries": 240, "evidence_rows": 720,
            "calls": FAMILIES[family]["calls"],
        },
        "evaluation_labels_included": True,
        "model_prompt_label_exposure": False,
        "prompt_shas": PROMPT_SHAS,
        "files": files,
    }
    manifest_path = output_dir / f"consumer_inputs_{family}.json"
    manifest_path.write_bytes(json_bytes(bundle_manifest))
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "source_reranker_output_sha256": reranker_output_sha256,
        "expected_workload": bundle_manifest["expected_workload"],
    }


def _qrel_free_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and "qrel" in path.name.lower():
            raise ValueError(f"qrel-bearing filename refused: {path.name}")
        # Provenance manifests explicitly declare qrels_read=false and may
        # describe the upstream oracle build. Scan only data-bearing artifacts.
        if (path.is_file() and path.name in {
                "queries.jsonl", "corpus.jsonl", "bm25.trec", "oracle.trec",
                "rerank_candidates.jsonl", "rerank.trec", "rankings.jsonl"
        } and "qrel" in path.read_text(errors="ignore").lower()):
            raise ValueError(f"qrel token detected in {path.name}")


def build_accepted(args: argparse.Namespace) -> dict:
    execution_sha = clean_execution_sha()
    retrieval_root = args.retrieval_root.resolve()
    reranker_root = args.reranker_root.resolve()
    verifier_path = args.reranker_verification.resolve()
    acceptance_path = args.acceptance_receipt.resolve()
    runtime_lock_path = args.runtime_lock.resolve()
    retrieval_manifest_path = retrieval_root / "retrieval_manifest.json"
    reranker_manifest_path = reranker_root / "full_run_manifest.json"
    rerank_path = reranker_root / "rerank.trec"
    for path in (retrieval_manifest_path, reranker_manifest_path, rerank_path,
                 verifier_path, acceptance_path, runtime_lock_path):
        if not path.is_file():
            raise ValueError(f"required accepted input is missing: {path}")
    _qrel_free_tree(retrieval_root)
    _qrel_free_tree(reranker_root)
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text())
    full_manifest = json.loads(reranker_manifest_path.read_text())
    verification = json.loads(verifier_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    if verification.get("status") != "PASS":
        raise ValueError("full-reranker verifier result is not PASS")
    if verification.get("execution_sha") != full_manifest.get("execution_sha"):
        raise ValueError("full-reranker verifier and output manifest execution SHAs differ")
    if verification.get("checks", {}).get("full_run_manifest_sha256") != sha256_file(reranker_manifest_path):
        raise ValueError("full-reranker verifier does not bind the supplied run manifest")
    if verification.get("checks", {}).get("output_sha256", {}).get("rerank.trec") != sha256_file(rerank_path):
        raise ValueError("full-reranker verifier does not bind the supplied rerank.trec")
    if acceptance.get("status") != "accepted":
        raise ValueError("a separate Codex acceptance receipt is required")
    if acceptance.get("full_verification_sha256") != sha256_file(verifier_path):
        raise ValueError("Codex acceptance receipt is for a different verifier result")
    if acceptance.get("source_reranker_output_sha256") != sha256_file(rerank_path):
        raise ValueError("Codex acceptance receipt is for a different reranker output")
    if acceptance.get("execution_sha") != full_manifest.get("execution_sha"):
        raise ValueError("Codex acceptance receipt is for a different execution SHA")
    if (retrieval_manifest.get("n_queries") != 240
            or retrieval_manifest.get("qrels_in_emitted_bundle") is not False
            or retrieval_manifest.get("rerank_revision") != "e61197ed45024b0ed8a2d74b80b4d909f1255473"
            or retrieval_manifest.get("rerank_candidate_depth") != 50):
        raise ValueError("retrieval manifest is not a qrel-free 240-query emitted bundle")
    runtime_lock = json.loads(runtime_lock_path.read_text())
    if runtime_lock.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError("consumer runtime lock protocol hash drift")
    if args.family == "qwen":
        runtime = runtime_lock.get("runtime", {})
        if (runtime_lock.get("lock_id") != "sem2act-v5-colab-runtime-v1"
                or runtime_lock.get("status") != "frozen-before-result-bearing-execution"
                or runtime.get("torch") != "2.11.0+cu128"
                or runtime.get("cuda") != "12.8"
                or runtime.get("python_policy") != "record-the-base-image-version;do-not-replace"):
            raise ValueError("Qwen consumer bundle requires the frozen Colab T4/Torch/CUDA policy")
    else:
        platform_lock = runtime_lock.get("platform", {})
        runtime = runtime_lock.get("runtime", {})
        if (runtime_lock.get("lock_id") != "sem2act-v5-kaggle-runtime-v3"
                or runtime_lock.get("status") != "prospective-before-result-bearing-execution"
                or platform_lock.get("provider") != "kaggle"
                or platform_lock.get("owner") != "rezabarati2"
                or platform_lock.get("gpu_count") != 2
                or platform_lock.get("visible_gpu_count") != 1
                or runtime.get("torch") != "2.10.0+cu128"
                or runtime.get("cuda") != "12.8"):
            raise ValueError("Llama/Mistral consumer bundles require the frozen rezabarati2 Kaggle runtime policy")
    model = FAMILIES[args.family]
    locked = runtime_lock.get("models", {}).get(args.family, {})
    if locked.get("model_id") != model["model_id"] or locked.get("revision") != model["revision"]:
        raise ValueError(f"{args.family} model revision differs from the frozen runtime lock")
    runtime_fields = (
        "backend", "dtype", "quantization", "device_map", "attention_backend",
        "max_context_tokens", "max_input_tokens", "max_new_tokens", "batch_size",
        "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
        "enforce_eager", "enable_thinking", "do_sample", "num_beams", "use_cache", "tokenizer",
    )
    spec = {
        "model_id": locked["model_id"],
        "revision": locked["revision"],
        "tokenizer_revision": locked["tokenizer_revision"],
        "runtime_lock_id": runtime_lock["lock_id"],
        "runtime": {key: locked[key] for key in runtime_fields if key in locked},
    }
    if spec["revision"] != spec["tokenizer_revision"]:
        raise ValueError("model and tokenizer revisions must be identical")

    expected_source_hashes = {
        name: entry["sha256"] for name, entry in retrieval_manifest.get("files", {}).items()
    }
    input_paths = {name: retrieval_root / name for name in (
        "queries.jsonl", "corpus.jsonl", "rerank_candidates.jsonl", "bm25.trec", "oracle.trec")}
    input_hashes = {}
    for name, path in input_paths.items():
        if not path.is_file():
            raise ValueError(f"missing retrieval source artifact: {name}")
        actual = sha256_file(path)
        input_hashes[name] = actual
        if expected_source_hashes.get(name) != actual:
            raise ValueError(f"retrieval-manifest hash mismatch: {name}")
    for name, expected in EXPECTED_RERANK_INPUT_HASHES.items():
        if input_hashes.get(name) != expected:
            raise ValueError(f"retrieval input differs from the frozen reranker source: {name}")
    queries = read_jsonl(input_paths["queries.jsonl"])
    docs = {row["_id"]: row for row in read_jsonl(input_paths["corpus.jsonl"])}
    rankings = {
        "bm25": read_trec(input_paths["bm25.trec"]),
        "oracle": read_trec(input_paths["oracle.trec"]),
        "rerank": read_trec(rerank_path),
    }
    if {row.get("_id") for row in queries} != set(EXPECTED_QUERY_IDS):
        raise ValueError("retrieval query set differs from the frozen 240-query workload")
    if len(docs) != len(read_jsonl(input_paths["corpus.jsonl"])):
        raise ValueError("duplicate corpus document IDs")
    out = args.output_root.resolve()
    result = build_bundle(
        family=args.family, output_dir=out, queries=queries, corpus=docs,
        rankings=rankings, model_spec=spec,
        runtime_lock_sha256=sha256_file(runtime_lock_path),
        retrieval_manifest_sha256=sha256_file(retrieval_manifest_path),
        reranker_manifest_sha256=sha256_file(reranker_manifest_path),
        reranker_verification_sha256=sha256_file(verifier_path),
        reranker_output_sha256=sha256_file(rerank_path), input_hashes=input_hashes,
        execution_sha=execution_sha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=tuple(FAMILIES))
    parser.add_argument("--retrieval-root", type=Path, required=True,
                        help="accepted qrel-free queries/corpus/BM25/oracle bundle")
    parser.add_argument("--reranker-root", type=Path, required=True,
                        help="accepted Moon full-reranker output directory")
    parser.add_argument("--reranker-verification", type=Path, required=True)
    parser.add_argument("--acceptance-receipt", type=Path, required=True,
                        help="separate Codex acceptance receipt; not created by this command")
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    build_accepted(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
