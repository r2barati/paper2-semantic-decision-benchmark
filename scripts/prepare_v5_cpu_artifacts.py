"""Record or create the pinned official GGUF artifacts for CPU_BACKEND_V1.

Conversion is an explicit non-result operation.  The script refuses to mark
the artifact manifest complete unless the llama.cpp checkout, conversion
script, quantizer, model revisions, and all GGUF hashes are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "versions/sem2act-v5/manifests/cpu_model_artifacts.json"
ENGINE_COMMIT = "9f70b2cecd1a9a3f73ac525c47ca22a6ee9a7b69"
MODELS = {
    "qwen": ("Qwen/Qwen3-8B-AWQ", "4da05a8edb55c6046cce958586c33b61da07bb79", "qwen3-8b.Q4_K_M.gguf"),
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "llama-3.1-8b-instruct.Q4_K_M.gguf"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef", "mistral-7b-instruct-v0.3.Q4_K_M.gguf"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--mistral", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    engine_commit = git_head(args.engine_root)
    if engine_commit != ENGINE_COMMIT:
        raise SystemExit(f"llama.cpp checkout drift: expected {ENGINE_COMMIT}, got {engine_commit}")
    conversion = args.engine_root / "convert_hf_to_gguf.py"
    quantizer_candidates = [
        args.engine_root / "build/bin/llama-quantize",
        args.engine_root / "build/bin/quantize",
    ]
    quantizer = next((path for path in quantizer_candidates if path.exists()), None)
    if not conversion.exists() or quantizer is None:
        raise SystemExit("pinned conversion script or quantizer binary is missing")
    roots = {"qwen": args.qwen, "llama": args.llama, "mistral": args.mistral}
    models = {}
    for family, (model_id, revision, filename) in MODELS.items():
        path = args.artifact_dir / filename
        if not path.exists():
            raise SystemExit(f"missing converted GGUF for {family}: {path}")
        models[family] = {
            "model_id": model_id,
            "revision": revision,
            "tokenizer_revision": revision,
            "filename": filename,
            "path": str(path.resolve()),
            "sha256": sha(path),
            "source_model_dir": str(roots[family].resolve()),
        }
    manifest = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-cpu-model-artifacts-v1",
        "status": "complete-before-result-bearing-execution",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "engine_commit": ENGINE_COMMIT,
        "quantization_type": "Q4_K_M",
        "conversion_script": str(conversion.resolve()),
        "conversion_script_sha256": sha(conversion),
        "quantizer_binary": str(quantizer.resolve()),
        "quantizer_binary_sha256": sha(quantizer),
        "models": models,
        "conversion_commands": [
            "python convert_hf_to_gguf.py {MODEL_DIR} --outfile {MODEL}.f16.gguf --outtype f16",
            "llama-quantize {MODEL}.f16.gguf {MODEL}.Q4_K_M.gguf Q4_K_M",
        ],
        "result_bearing_execution_started": False,
    }
    if args.write:
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
