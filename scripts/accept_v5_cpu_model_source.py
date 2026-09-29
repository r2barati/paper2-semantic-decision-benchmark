"""Accept an already-verified official model source hash into CPU_BACKEND_V1."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
EXPECTED = {
    "qwen": {"source_ref": "qwen-lm/qwen-3/Transformers/8b-awq", "model_type": "qwen3", "architecture": "Qwen3ForCausalLM"},
    "llama": {"source_ref": "metaresearch/llama-3.1/Transformers/8b-instruct", "model_type": "llama", "architecture": "LlamaForCausalLM"},
}

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def update_yaml(path: Path, family: str, digest: str) -> None:
    lines = path.read_text().splitlines()
    ref = EXPECTED[family]["source_ref"]
    start = next((i for i, line in enumerate(lines) if line.strip() == f"source_ref: {ref}"), None)
    if start is None:
        raise SystemExit(f"source reference not found in {path}: {ref}")
    end = start + 1
    while end < len(lines) and (len(lines[end]) - len(lines[end].lstrip()) > len(lines[start]) - len(lines[start].lstrip()) or not lines[end].strip()):
        end += 1
    existing = next((i for i in range(start + 1, end) if lines[i].strip().startswith("content_sha256:")), None)
    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    line = indent + f"content_sha256: {digest}"
    if existing is not None:
        lines[existing] = line
    else:
        arch = next((i for i in range(start + 1, end) if lines[i].strip().startswith("architecture:")), None)
        lines.insert((arch + 1) if arch is not None else end, line)
    path.write_text("\n".join(lines) + "\n")

def update_packager(path: Path, family: str, digest: str) -> None:
    text = path.read_text()
    ref = EXPECTED[family]["source_ref"]
    pattern = re.compile(r'("source_ref": "' + re.escape(ref) + r'",.*?"architecture": "' + re.escape(EXPECTED[family]["architecture"]) + r'",)(\n)', re.S)
    replacement = r'\1\n            "content_sha256": "' + digest + r'",\2'
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"source reference block not found in {path}: {ref}")
    path.write_text(text)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=sorted(EXPECTED))
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    remote_path = args.probe_dir / "source_probe_manifest.json"
    acceptance_path = args.probe_dir / "acceptance_manifest.json"
    if not remote_path.exists() or not acceptance_path.exists():
        raise SystemExit("an accepted source-probe directory is required")
    remote = json.loads(remote_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("status") != "accepted-non-result-source-probe":
        raise SystemExit("source probe is not verifier-accepted")
    if remote.get("status") != "pass" or remote.get("source_only") is not True:
        raise SystemExit("source probe did not pass")
    if remote.get("result_bearing_execution_started") is not False or remote.get("qrels_read") is not False or remote.get("inference_calls") != 0:
        raise SystemExit("source probe is not zero-inference/qrel-free")
    source = remote.get("source") or {}
    for key, value in EXPECTED[args.family].items():
        if source.get(key) != value:
            raise SystemExit(f"source metadata mismatch for {key}")
    snapshot = remote.get("source_model_snapshot") or {}
    digest = snapshot.get("files_sha256")
    if not digest or not isinstance(digest, str):
        raise SystemExit("source content hash is missing")

    lock_path = ROOT / "versions/sem2act-v5/manifests/cpu_runtime_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("protocol hash drift")
    lock["models"][args.family]["model_source"]["content_sha256"] = digest
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    amendment_path = ROOT / "versions/sem2act-v5/amendments/cpu_backend_v1.yaml"
    update_yaml(amendment_path, args.family, digest)
    update_packager(ROOT / "scripts/kaggle_cpu_canary.py", args.family, digest)

    preflight_path = ROOT / "versions/sem2act-v5/manifests/cpu_model_source_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    model = preflight["models"][args.family]
    model["status"] = "source_probe_passed"
    model["weights_downloaded"] = True
    model["source"]["content_sha256"] = digest
    model["source_probe_acceptance_manifest"] = str((args.probe_dir / "acceptance_manifest.json").relative_to(ROOT)) if args.probe_dir.is_relative_to(ROOT) else str(args.probe_dir / "acceptance_manifest.json")
    model["source_snapshot_file_count"] = snapshot.get("file_count")
    preflight["last_source_handoff"] = {"family": args.family, "content_sha256": digest, "accepted_manifest_sha256": sha(acceptance_path), "accepted_utc": datetime.now(timezone.utc).isoformat()}
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

    handoff = {
        "schema_version": 1,
        "manifest_id": f"sem2act-v5-source-handoff-{args.family}-v1",
        "status": "accepted-non-result-source-handoff",
        "accepted_utc": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "protocol_hash": PROTOCOL_HASH,
        "content_sha256": digest,
        "source_ref": source["source_ref"],
        "source_probe_manifest_sha256": sha(remote_path),
        "source_probe_acceptance_sha256": sha(acceptance_path),
        "result_bearing_execution_started": False,
        "qrels_read": False,
    }
    out = ROOT / "versions/sem2act-v5/manifests/execution_attempts" / f"cpu-model-source-handoff-{args.family}.json"
    out.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")
    print(json.dumps(handoff, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
