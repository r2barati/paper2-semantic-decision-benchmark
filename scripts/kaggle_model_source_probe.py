"""Package and optionally submit a source-only official Kaggle model probe."""
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
JOB = ROOT / "versions/sem2act-v5/compute/kaggle_cpu/source_probe_job.py"
OWNER = "siavashsimin"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
SOURCES = {
    "qwen": {
        "model_id": "Qwen/Qwen3-8B-AWQ",
        "revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
        "model_source": {"type": "official_kaggle_model", "source_ref": "qwen-lm/qwen-3/Transformers/8b-awq", "owner": "qwen-lm", "publisher": "QwenLM", "version_number": 1, "version_id": 391615, "model_type": "qwen3", "architecture": "Qwen3ForCausalLM"},
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "model_source": {"type": "official_kaggle_model", "source_ref": "metaresearch/llama-3.1/Transformers/8b-instruct", "owner": "metaresearch", "publisher": "Meta", "version_number": 2, "version_id": 104449, "model_type": "llama", "architecture": "LlamaForCausalLM"},
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
    config = {"schema_version": 1, "protocol_hash": PROTOCOL_HASH, "family": family, **SOURCES[family], "result_bearing_execution_started": False, "qrels_read": False}
    (directory / "source_probe_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    source = JOB.read_text().replace("EMBEDDED_SOURCE_CONFIG = None", "EMBEDDED_SOURCE_CONFIG = " + repr(config), 1)
    (directory / "source_probe_job.py").write_text(source)
    # Kaggle has two distinct identifiers here. The API/manifest provenance
    # reference is the four-part model-instance path; kernel metadata requires
    # the version number as a fifth path component.
    ref = SOURCES[family]["model_source"]["source_ref"]
    kernel_ref = f"{ref}/{SOURCES[family]['model_source']['version_number']}"
    metadata = {"id": f"{OWNER}/sem2act-v5-source-probe-{family}", "title": f"sem2act-v5-source-probe-{family}", "code_file": "source_probe_job.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": False, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [kernel_ref]}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"family": family, "slug": f"{OWNER}/sem2act-v5-source-probe-{family}", "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(directory.iterdir()) if p.is_file()}, "protocol_hash": PROTOCOL_HASH, "result_bearing_execution_started": False, "qrels_read": False, "source_ref": ref, "kernel_source_ref": kernel_ref}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=sorted(SOURCES))
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--kaggle", default="/usr/local/bin/kaggle")
    args = parser.parse_args()
    if not args.push and args.package_dir is None:
        parser.error("use --push or --package-dir")
    directory = args.package_dir or Path(tempfile.mkdtemp(prefix=f"sem2act-v5-source-probe-{args.family}-"))
    if directory.exists() and args.package_dir:
        shutil.rmtree(directory)
    manifest = package(args.family, directory)
    manifest.update({"package_dir": str(directory), "created_utc": datetime.now(timezone.utc).isoformat()})
    if args.push:
        subprocess.run([args.kaggle, "kernels", "push", "-p", str(directory)], check=True)
        manifest["status"] = "submitted-non-result-source-probe"
    else:
        manifest["status"] = "packaged-non-result-source-probe"
    out = ROOT / "versions/sem2act-v5/manifests/execution_attempts" / f"cpu-model-source-probe-{args.family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
