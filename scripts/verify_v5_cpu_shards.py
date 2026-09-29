"""Verify exact, disjoint, exhaustive acceptance of CPU result shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_keys(path: Path, field: str) -> set[str]:
    if not path.exists():
        raise SystemExit(f"missing shard output: {path}")
    values = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = item[field]
        if key in values:
            raise SystemExit(f"duplicate output key: {key}")
        values.add(key)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("reranker", "qwen", "llama", "mistral"), required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.shard_root / "index.json"
    index = json.loads(index_path.read_text())
    if index.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("CPU shard index protocol hash drift")
    expected_all: set[str] = set()
    accepted_all: set[str] = set()
    shard_records = []
    for relative in index["shards"]:
        shard_path = ROOT / relative
        shard = json.loads(shard_path.read_text())
        if args.family == "reranker" and shard.get("stage") != "reranker":
            raise SystemExit(f"reranker shard stage mismatch: {shard_path}")
        if args.family != "reranker" and shard.get("family") != args.family:
            raise SystemExit(f"consumer shard family mismatch: {shard_path}")
        if args.family == "reranker":
            expected = set(shard["expected_query_keys"])
            output_key_file = args.results_root / shard_path.stem / "rankings.jsonl"
            field = "key"
        else:
            expected = set(shard["expected_call_keys"])
            output_key_file = args.results_root / shard_path.stem / "raw_index.jsonl"
            field = "key"
        if expected & expected_all:
            raise SystemExit(f"planned shard overlap in {shard_path}")
        expected_all.update(expected)
        run_manifest_path = args.results_root / shard_path.stem / "run_manifest.json"
        if not run_manifest_path.exists():
            raise SystemExit(f"missing run manifest: {run_manifest_path}")
        run_manifest = json.loads(run_manifest_path.read_text())
        if run_manifest.get("status") != "pass" or run_manifest.get("n_fail", 1) != 0:
            raise SystemExit(f"failed or incomplete shard: {run_manifest_path}")
        accepted = read_keys(output_key_file, field)
        if accepted != expected:
            raise SystemExit(f"accepted keyspace mismatch: {output_key_file}")
        if accepted & accepted_all:
            raise SystemExit(f"accepted shard overlap: {output_key_file}")
        accepted_all.update(accepted)
        shard_records.append({
            "shard_id": shard["manifest_id"],
            "planned_sha256": sha(shard_path),
            "run_manifest_sha256": sha(run_manifest_path),
            "accepted_keys": len(accepted),
        })
    if expected_all != accepted_all:
        raise SystemExit("accepted shards are not exhaustive")
    manifest = {
        "schema_version": 1,
        "manifest_id": f"sem2act-v5-cpu-{args.family}-merged-v1",
        "status": "pass",
        "protocol_hash": PROTOCOL_HASH,
        "family": args.family,
        "shard_count": len(shard_records),
        "expected_keys": len(expected_all),
        "accepted_keys": len(accepted_all),
        "exactly_disjoint": True,
        "exactly_exhaustive": True,
        "excluded_runs": [],
        "shards": shard_records,
        "result_bearing_execution_started": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
