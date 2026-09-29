"""Package and submit the zero-inference exact Mistral source assembly probe."""
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
JOB = ROOT / "versions/sem2act-v5/compute/kaggle_cpu/mistral_source_assembly_probe_job.py"
OWNER = "siavashsimin"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
CANDIDATE_DATASET = "phmcngc/snapx-mistral7b-instruct-v03-snapshot"
PATCH_DATASET = "siavashsimin/sem2act-v5-mistral-tokenizer-patch"
EXPECTED = {
    "config.json": {"sha256": "affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99", "bytes": 601},
    "generation_config.json": {"sha256": "b4669f1b8f4185324bd9b12a69e85a1ad2289bc48111ef739cfaaea3bba6b0a9", "bytes": 116},
    "model-00001-of-00003.safetensors": {"sha256": "ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19", "bytes": 4949453792},
    "model-00002-of-00003.safetensors": {"sha256": "8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0", "bytes": 4999819336},
    "model-00003-of-00003.safetensors": {"sha256": "905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca", "bytes": 4546807800},
    "model.safetensors.index.json": {"sha256": "e489ba553b87cde188d921b1a8283c2e0b9d33d635b88147d96ff0fcd6250016", "bytes": 23950},
    "special_tokens_map.json": {"sha256": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0", "bytes": 414},
    "tokenizer.json": {"sha256": "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49", "bytes": 1961548},
    "tokenizer.model": {"sha256": "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89", "bytes": 587404},
    "tokenizer_config.json": {"sha256": "cf7d90917f89931e443b02253432a581abef86379533a5b962f7f2d3b7ac8d90", "bytes": 137799},
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--kaggle", default="/private/tmp/kaggle-cli-v2/bin/kaggle")
    args = parser.parse_args()
    if not args.push and args.package_dir is None:
        parser.error("use --push or --package-dir")
    package_dir = args.package_dir or Path(tempfile.mkdtemp(prefix="sem2act-v5-mistral-assembly-"))
    if args.package_dir and package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "protocol_hash": PROTOCOL_HASH,
        "family": "mistral",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef",
        "candidate_dataset": CANDIDATE_DATASET,
        "patch_dataset": PATCH_DATASET,
        "patch_dataset_slug": "sem2act-v5-mistral-tokenizer-patch",
        "expected_files": EXPECTED,
        "result_bearing_execution_started": False,
        "qrels_read": False,
    }
    (package_dir / "assembly_probe_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    source = JOB.read_text().replace("EMBEDDED_ASSEMBLY_CONFIG = None", "EMBEDDED_ASSEMBLY_CONFIG = " + repr(config), 1)
    (package_dir / "mistral_source_assembly_probe_job.py").write_text(source)
    metadata = {
        "id": f"{OWNER}/sem2act-v5-mistral-source-assembly-probe",
        "title": "Sem2Act v5 Mistral exact source assembly probe",
        "code_file": "mistral_source_assembly_probe_job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [CANDIDATE_DATASET, PATCH_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (package_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    manifest = {
        "family": "mistral",
        "slug": f"{OWNER}/sem2act-v5-mistral-source-assembly-probe",
        "candidate_dataset": CANDIDATE_DATASET,
        "patch_dataset": PATCH_DATASET,
        "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(package_dir.iterdir()) if p.is_file()},
        "protocol_hash": PROTOCOL_HASH,
        "result_bearing_execution_started": False,
        "qrels_read": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_dir": str(package_dir),
        "status": "packaged-non-result-source-probe",
    }
    if args.push:
        subprocess.run([args.kaggle, "kernels", "push", "-p", str(package_dir)], check=True)
        manifest["status"] = "submitted-non-result-source-probe"
    out = ROOT / "versions/sem2act-v5/manifests/execution_attempts/cpu-mistral-source-assembly-probe-v2.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
