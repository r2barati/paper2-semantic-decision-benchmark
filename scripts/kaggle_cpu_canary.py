"""Package and optionally submit one private non-lockbox Kaggle CPU canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "versions/sem2act-v5/compute/kaggle_cpu/canary_job.py"
FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
OWNER = "siavashsimin"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
ENGINE_SOURCE_DATASET = "siavashsimin/sem2act-v5-llama-cpp-b11194-source"
ENGINE_SOURCE_DATASET_VERSION = 2
ENGINE_ARCHIVE_SHA256 = "bc351a39f7a2c68f0e9827f86d0185c4ec7617be29493c4f8d349ecf41bbcbc6"
ENGINE_SOURCE_TREE_SHA256 = "1bbfb63c28543063eadf2d55b9e74f63866d4d7d3fbfe448bab4a2d6e08a84e8"
MODELS = {
    "qwen": {
        "kind": "consumer",
        "backend": "llama.cpp",
        "model_source": {
            "type": "official_kaggle_model",
            "source_ref": "qwen-lm/qwen-3/Transformers/8b-awq",
            "owner": "qwen-lm",
            "publisher": "QwenLM",
            "version_number": 1,
            "version_id": 391615,
            "model_type": "qwen3",
            "architecture": "Qwen3ForCausalLM",
            "content_sha256": "57f082c4a2ea54b2413677040731a10ec7dc1fbd43f44020958176c6215535b5",
        },
        "model_id": "Qwen/Qwen3-8B-AWQ",
        "revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
        "tokenizer_revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
        "gguf_filename": "qwen3-8b.Q4_K_M.gguf",
        "dtype": "ggml_q4_k_m",
        "bitsandbytes": "disabled",
        "device_map": "cpu",
        "max_context_tokens": 4096,
        "batch_size": 512,
        "ubatch_size": 512,
        "attention_backend": "llama.cpp_cpu",
    },
    "llama": {
        "kind": "consumer",
        "backend": "llama.cpp",
        "model_source": {
            "type": "official_kaggle_model",
            "source_ref": "metaresearch/llama-3.1/Transformers/8b-instruct",
            "owner": "metaresearch",
            "publisher": "Meta",
            "version_number": 2,
            "version_id": 104449,
        },
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "tokenizer_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "gguf_filename": "llama-3.1-8b-instruct.Q4_K_M.gguf",
        "dtype": "ggml_q4_k_m",
        "bitsandbytes": "disabled",
        "device_map": "cpu",
        "max_context_tokens": 4096,
        "batch_size": 512,
        "ubatch_size": 512,
        "attention_backend": "llama.cpp_cpu",
    },
    "mistral": {
        "kind": "consumer",
        "backend": "llama.cpp",
        "model_source": {
            "type": "verified_kaggle_dataset_assembly",
            "model_type": "mistral",
            "architecture": "MistralForCausalLM",
            "candidate_dataset": "phmcngc/snapx-mistral7b-instruct-v03-snapshot",
            "patch_dataset": "siavashsimin/sem2act-v5-mistral-tokenizer-patch",
            "patch_dataset_slug": "sem2act-v5-mistral-tokenizer-patch",
            "patch_filename": "tokenizer_config.json",
            "content_sha256": "805960a9bbd7a17f70da2170a56f9364ecfd10ddae6a7698e4e0d55cceffa0b0",
        },
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef",
        "tokenizer_revision": "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef",
        "gguf_filename": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        "dtype": "ggml_q4_k_m",
        "bitsandbytes": "disabled",
        "device_map": "cpu",
        "max_context_tokens": 4096,
        "batch_size": 512,
        "ubatch_size": 512,
        "attention_backend": "llama.cpp_cpu",
    },
    "reranker": {
        "kind": "reranker",
        "backend": "transformers_cpu",
        "model_id": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "tokenizer_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "dtype": "float32",
        "bitsandbytes": "disabled",
        "device": "cpu",
        "device_map": "cpu",
        "max_context_tokens": 1024,
        "batch_size": 1,
        "attention_backend": "eager",
    },
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(family: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "protocol_hash": PROTOCOL_HASH,
        "family": family,
        **MODELS[family],
        "quantization_type": "Q4_K_M" if MODELS[family]["kind"] == "consumer" else None,
        "engine_source_dataset": ENGINE_SOURCE_DATASET,
        "engine_source_archive_sha256": ENGINE_ARCHIVE_SHA256,
        "engine_source_tree_sha256": ENGINE_SOURCE_TREE_SHA256,
        "engine_source_dataset_version": ENGINE_SOURCE_DATASET_VERSION,
        "result_bearing_execution_started": False,
    }
    (directory / "canary_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    fixture = json.loads(FIXTURE.read_text())
    source = JOB.read_text()
    source = source.replace("EMBEDDED_CANARY_CONFIG = None", "EMBEDDED_CANARY_CONFIG = " + repr(config), 1)
    source = source.replace("EMBEDDED_RUNTIME_FIXTURE = None", "EMBEDDED_RUNTIME_FIXTURE = " + repr(fixture), 1)
    source = source.replace("EMBEDDED_RUNTIME_FIXTURE_SHA256 = None", "EMBEDDED_RUNTIME_FIXTURE_SHA256 = " + repr(sha(FIXTURE)), 1)
    if "EMBEDDED_CANARY_CONFIG = None" in source or "EMBEDDED_RUNTIME_FIXTURE = None" in source or "EMBEDDED_RUNTIME_FIXTURE_SHA256 = None" in source:
        raise RuntimeError("canary embedding placeholders were not fully replaced")
    (directory / "canary_job.py").write_text(source)
    shutil.copy2(FIXTURE, directory / "runtime_smoke_v1.json")
    slug = f"sem2act-v5-cpu-preflight-{family}"
    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": f"Sem2Act v5 CPU preflight {family}",
        "code_file": "canary_job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": ([ENGINE_SOURCE_DATASET] + (
            [MODELS[family]["model_source"]["candidate_dataset"], MODELS[family]["model_source"]["patch_dataset"]]
            if family == "mistral" else []
        )),
        "competition_sources": [],
        "kernel_sources": [],
        # Kaggle kernel metadata uses the versioned five-part attachment
        # reference; provenance manifests retain the four-part API reference.
        "model_sources": ([
            f"{MODELS[family]['model_source']['source_ref']}/{MODELS[family]['model_source']['version_number']}"
        ] if MODELS[family].get("model_source", {}).get("source_ref") else []),
    }
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "family": family,
        "slug": f"{OWNER}/{slug}",
        "files": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                  for path in sorted(directory.iterdir()) if path.is_file()},
        "fixture_sha256": sha(FIXTURE),
        "engine_source_dataset": ENGINE_SOURCE_DATASET,
        "engine_source_archive_sha256": ENGINE_ARCHIVE_SHA256,
        "engine_source_tree_sha256": ENGINE_SOURCE_TREE_SHA256,
        "engine_source_dataset_version": ENGINE_SOURCE_DATASET_VERSION,
        "protocol_hash": PROTOCOL_HASH,
        "result_bearing_execution_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=tuple(MODELS))
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--kaggle", default="/private/tmp/kaggle-cli-v2/bin/kaggle")
    args = parser.parse_args()
    if not args.push and args.package_dir is None:
        parser.error("use --push or --package-dir")
    if args.package_dir is None:
        package_dir = Path(tempfile.mkdtemp(prefix=f"sem2act-v5-cpu-{args.family}-"))
    else:
        package_dir = args.package_dir
        if package_dir.exists():
            shutil.rmtree(package_dir)
    manifest = package(args.family, package_dir)
    manifest["package_dir"] = str(package_dir)
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    if args.push:
        subprocess.run([args.kaggle, "kernels", "push", "-p", str(package_dir)], check=True)
        manifest["status"] = "submitted-non-result-canary"
    else:
        manifest["status"] = "packaged-non-result-canary"
    out = ROOT / "versions/sem2act-v5/manifests/execution_attempts" / f"cpu-canary-package-{args.family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
