"""Build deterministic qrel-free v5 consumer upload bundles.

This script only runs after the reranker output is installed. It creates one
identical evidence bundle plus one immutable official model specification per
cross-family job. It refuses placeholders and refuses qrel-bearing files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL = ROOT / "versions/sem2act-v5/runtime/retrieval_bundle"
MODEL_INPUTS = ROOT / "versions/sem2act-v5/runtime/model_inputs"
PROTOCOL_HASH = json.loads(
    (ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json").read_text()
)["protocol_hash"]
MODELS = {
    "qwen": {
        "model_id": "Qwen/Qwen3-8B-AWQ",
        "revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef",
    },
}
PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}
RUNTIME_LOCK = ROOT / "versions/sem2act-v5/manifests/kaggle_runtime_lock_v3.json"
SMOKE_FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_qrel_free(path: Path) -> None:
    for candidate in path.rglob("*"):
        if candidate.is_file() and "qrel" in candidate.name.lower():
            raise SystemExit(f"qrel-bearing file in staging tree: {candidate}")
        if candidate.is_file() and "qrel" in candidate.read_text(errors="ignore").lower():
            raise SystemExit(f"qrel token in staged file: {candidate}")


def main() -> int:
    rerank_path = RETRIEVAL / "rerank.trec"
    run_manifest_path = RETRIEVAL / "run_manifest.json"
    if not rerank_path.exists() or not run_manifest_path.exists():
        raise SystemExit("rerank.trec is required; fetch and verify v5-rerank first")
    rerank_manifest = json.loads(run_manifest_path.read_text())
    if (rerank_manifest.get("experiment_id") != "v5-lockbox-rerank"
            or rerank_manifest.get("n_queries") != 240
            or rerank_manifest.get("qrels_read") is not False
            or rerank_manifest.get("n_fail") != 0):
        raise SystemExit("consumer bundles require a verified complete 240-query reranker run")
    # The existing assembler emits bm25, rerank, and oracle rows; the kernel
    # selects only rerank/oracle while the bundle remains shared and qrel-free.
    import runpy
    runpy.run_path(
        str(ROOT / "versions/sem2act-v5/experiments/lockbox/build_model_inputs.py"),
        run_name="__main__",
    )
    source = MODEL_INPUTS / "evidence_inputs.jsonl"
    if not source.exists():
        raise SystemExit("model input assembler did not emit evidence_inputs.jsonl")
    MODEL_INPUTS_MANIFEST = MODEL_INPUTS / "model_inputs_manifest.json"
    if not MODEL_INPUTS_MANIFEST.exists():
        raise SystemExit("model input manifest missing")
    ensure_qrel_free(source)
    for family, model in MODELS.items():
        dest = ROOT / f"versions/sem2act-v5/runtime/kaggle_{family}_inputs"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        shutil.copy2(source, dest / "evidence_inputs.jsonl")
        shutil.copy2(SMOKE_FIXTURE, dest / "runtime_smoke_v1.json")
        if family == "qwen":
            for name in ("consumers_v3.py", "interpreter.py", "events.py"):
                shutil.copy2(ROOT / "src" / name, dest / name)
        runtime_lock = json.loads(RUNTIME_LOCK.read_text())
        runtime_model = runtime_lock["models"][family]
        spec = {
            "schema_version": 1,
            "experiment_id": "v5-cross-family-consumer",
            "model_family": family,
            "model_id": model["model_id"],
            "revision": model["revision"],
            "tokenizer_revision": model["revision"],
            "protocol_hash": PROTOCOL_HASH,
            "expected_protocol_hash": PROTOCOL_HASH,
            "prompt_shas": PROMPT_SHAS,
            "runtime_lock_id": runtime_lock["lock_id"],
            "runtime_lock_sha256": sha(RUNTIME_LOCK),
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_new_tokens": 256,
            "runtime": {
                "backend": runtime_model["backend"],
                "dtype": runtime_model["dtype"],
                "quantization": runtime_model.get("quantization", {"method": "none"}),
                "device_map": runtime_model["device_map"],
                "max_context_tokens": runtime_model["max_context_tokens"],
                "batch_size": runtime_model["batch_size"],
                "attention_backend": runtime_model["attention_backend"],
            },
            "source_manifest_sha256": sha(MODEL_INPUTS_MANIFEST),
        }
        (dest / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "family": family,
            "model_id": model["model_id"],
            "revision": model["revision"],
            "protocol_hash": PROTOCOL_HASH,
            "files": {
                p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(dest.iterdir()) if p.is_file()
            },
        }
        local_manifest = ROOT / f"versions/sem2act-v5/manifests/consumer_inputs_{family}.json"
        local_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"family": family, "path": str(dest), "files": manifest["files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
