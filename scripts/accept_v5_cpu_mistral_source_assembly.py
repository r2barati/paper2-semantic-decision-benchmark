"""Accept and freeze the verified zero-inference Mistral source assembly."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
ASSEMBLY_HASH = "805960a9bbd7a17f70da2170a56f9364ecfd10ddae6a7698e4e0d55cceffa0b0"
EXPECTED_SOURCE = {
    "type": "verified_kaggle_dataset_assembly",
    "model_type": "mistral",
    "architecture": "MistralForCausalLM",
    "candidate_dataset": "phmcngc/snapx-mistral7b-instruct-v03-snapshot",
    "patch_dataset": "siavashsimin/sem2act-v5-mistral-tokenizer-patch",
    "patch_dataset_slug": "sem2act-v5-mistral-tokenizer-patch",
    "patch_filename": "tokenizer_config.json",
    "content_sha256": ASSEMBLY_HASH,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    remote_path = args.probe_dir / "mistral_source_assembly_manifest.json"
    acceptance_path = args.probe_dir / "acceptance_manifest.json"
    if not remote_path.exists() or not acceptance_path.exists():
        raise SystemExit("completed source-assembly directory is required")
    remote = json.loads(remote_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("status") != "accepted-non-result-source-assembly":
        raise SystemExit("source assembly is not accepted")
    if remote.get("status") != "pass" or remote.get("source_only") is not True:
        raise SystemExit("source assembly did not pass")
    if remote.get("result_bearing_execution_started") is not False or remote.get("qrels_read") is not False or remote.get("inference_calls") != 0:
        raise SystemExit("source assembly is not zero-inference/qrel-free")
    if remote.get("required_files_match") is not True or remote.get("exact_file_set") is not True:
        raise SystemExit("source assembly file gate failed")
    snapshot = remote.get("source_model_snapshot") or {}
    if snapshot.get("files_sha256") != ASSEMBLY_HASH or snapshot.get("file_count") != 10:
        raise SystemExit("source assembly hash drift")

    lock_path = ROOT / "versions/sem2act-v5/manifests/cpu_runtime_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("protocol hash drift")
    lock["models"]["mistral"]["model_source"] = dict(EXPECTED_SOURCE)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    preflight_path = ROOT / "versions/sem2act-v5/manifests/cpu_model_source_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    model = preflight["models"]["mistral"]
    relative_acceptance = str(acceptance_path.relative_to(ROOT)) if acceptance_path.is_relative_to(ROOT) else str(acceptance_path)
    model["status"] = "source_assembly_passed"
    model["weights_downloaded"] = True
    model["source"] = dict(EXPECTED_SOURCE)
    model["source_snapshot_file_count"] = snapshot.get("file_count")
    model["source_probe_acceptance_manifest"] = relative_acceptance
    if relative_acceptance not in preflight.setdefault("source_probe_attempts", []):
        preflight["source_probe_attempts"].append(relative_acceptance)
    alternate = preflight.setdefault("alternate_official_sources", {}).setdefault("mistral", {})
    alternate["verified_mirror_assembly"] = relative_acceptance
    alternate["status"] = "verified_pinned_mirror_assembly"
    alternate["reason"] = "Public Kaggle mirror weights and required metadata matched the pinned HF revision; tokenizer_config.json was replaced by the exact pinned file and the assembled hash passed."
    preflight["last_source_handoff"] = {
        "family": "mistral",
        "content_sha256": ASSEMBLY_HASH,
        "accepted_manifest_sha256": sha(acceptance_path),
        "accepted_utc": datetime.now(timezone.utc).isoformat(),
    }
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

    handoff = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-source-handoff-mistral-assembly-v2",
        "status": "accepted-non-result-source-handoff",
        "accepted_utc": datetime.now(timezone.utc).isoformat(),
        "family": "mistral",
        "protocol_hash": PROTOCOL_HASH,
        "content_sha256": ASSEMBLY_HASH,
        "source_assembly": "candidate_weights_plus_pinned_tokenizer_config",
        "candidate_dataset": EXPECTED_SOURCE["candidate_dataset"],
        "patch_dataset": EXPECTED_SOURCE["patch_dataset"],
        "source_probe_manifest_sha256": sha(remote_path),
        "source_probe_acceptance_sha256": sha(acceptance_path),
        "result_bearing_execution_started": False,
        "qrels_read": False,
        "inference_calls": 0,
    }
    out = ROOT / "versions/sem2act-v5/manifests/execution_attempts/cpu-model-source-handoff-mistral-assembly-v2.json"
    out.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")
    print(json.dumps(handoff, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
