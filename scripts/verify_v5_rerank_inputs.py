"""Verify a locally fetched qrel-free V5 reranker input bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json"
REQUIRED = {"corpus.jsonl", "queries.jsonl", "rerank_candidates.jsonl"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(MANIFEST.read_text())
    root = args.input_dir.expanduser().resolve()
    files = {path.name: path for path in root.iterdir() if path.is_file()}
    if set(files) != REQUIRED or any("qrel" in name.lower() for name in files):
        raise SystemExit("input bundle has missing, extra, or qrel-named files")
    actual = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(files.items())
    }
    if (record.get("status") != "remote-verified"
            or record.get("dataset") != "rezabarati2/sem2act-v5-rerank-inputs"
            or record.get("qrels_uploaded") is not False
            or record.get("remote_sha256") != actual
            or record.get("remote_hash_match") is not True):
        raise SystemExit("local input hashes do not match the verified staged dataset manifest")
    print(json.dumps({
        "status": "pass", "dataset": record["dataset"],
        "files": actual, "qrels_present": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
