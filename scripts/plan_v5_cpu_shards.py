"""Create deterministic, query-range CPU shard manifests for v5.

The planner reads only qrel-free evidence inputs.  It never loads qrels,
never executes a model, and emits immutable expected call/belief keyspaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_DIR = ROOT / "versions/sem2act-v5/manifests/cpu_shards"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"


def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    if any("qrel" in candidate.name.lower() for candidate in path.parent.iterdir() if candidate.is_file()):
        raise SystemExit("qrel-bearing file in input directory")
    data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any("qrel" in json.dumps(row, sort_keys=True).lower() for row in data):
        raise SystemExit("qrel token in evidence input")
    return data


def expected_systems(family: str) -> tuple[str, ...]:
    if family == "qwen":
        return ("bm25", "rerank", "oracle")
    if family in {"llama", "mistral"}:
        return ("rerank", "oracle")
    raise SystemExit(f"unknown family: {family}")


def call_keys(row: dict, family: str) -> list[str]:
    if row["system"] not in expected_systems(family):
        return []
    result = [f"{family}|{row['query_id']}|{row['system']}|C1|"]
    result.extend(f"{family}|{row['query_id']}|{row['system']}|C3|{doc['doc_id']}"
                 for doc in row["documents"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("qwen", "llama", "mistral"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--calls-per-shard", type=int, default=240)
    parser.add_argument("--output-root", type=Path, default=CPU_DIR)
    args = parser.parse_args()
    if args.calls_per_shard < 1:
        parser.error("--calls-per-shard must be positive")
    data = rows(args.inputs)
    by_query: dict[str, list[dict]] = {}
    for row in data:
        if len(row.get("documents", [])) != 3:
            raise SystemExit(f"{row.get('query_id')}: expected exactly three documents")
        by_query.setdefault(row["query_id"], []).append(row)
    if len(by_query) != 240:
        raise SystemExit(f"expected 240 query IDs, got {len(by_query)}")
    expected = {f"v5-lb-q{i:04d}" for i in range(1, 241)}
    if set(by_query) != expected:
        raise SystemExit("query keyspace is not exactly v5-lb-q0001..q0240")
    for qid, items in by_query.items():
        systems = {item["system"] for item in items}
        if systems != {"bm25", "rerank", "oracle"}:
            raise SystemExit(f"{qid}: source input systems drift: {sorted(systems)}")
        if len(items) != 3:
            raise SystemExit(f"{qid}: duplicate or missing source system rows")

    per_query = len(expected_systems(args.family)) * 4
    queries_per_shard = max(1, args.calls_per_shard // per_query)
    ordered_queries = sorted(by_query)
    n_shards = math.ceil(len(ordered_queries) / queries_per_shard)
    output_dir = args.output_root / args.family
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for shard_index in range(n_shards):
        selected = ordered_queries[shard_index * queries_per_shard:(shard_index + 1) * queries_per_shard]
        selected_rows = [row for qid in selected for row in sorted(by_query[qid], key=lambda item: item["system"])]
        expected_calls = [key for row in selected_rows for key in call_keys(row, args.family)]
        expected_beliefs = [
            f"{args.family}|{row['query_id']}|{row['system']}|{consumer}"
            for row in selected_rows if row["system"] in expected_systems(args.family)
            for consumer in ("C1", "C3")
        ]
        manifest = {
            "schema_version": 1,
            "manifest_id": f"sem2act-v5-cpu-{args.family}-shard-{shard_index:03d}",
            "status": "planned-before-result-bearing-execution",
            "protocol_hash": PROTOCOL_HASH,
            "family": args.family,
            "shard_index": shard_index,
            "query_ids": selected,
            "query_start": selected[0],
            "query_end": selected[-1],
            "expected_call_keys": expected_calls,
            "expected_belief_keys": expected_beliefs,
            "expected_calls": len(expected_calls),
            "expected_beliefs": len(expected_beliefs),
            "input_sha256": sha(args.inputs),
            "target_calls_per_shard": args.calls_per_shard,
            "resume_policy": "missing-key-only",
            "raw_content_addressed": True,
            "accepted_key_regeneration": False,
        }
        path = output_dir / f"shard-{shard_index:03d}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        shard_paths.append(path)
    index = {
        "schema_version": 1,
        "manifest_id": f"sem2act-v5-cpu-{args.family}-shards-v1",
        "status": "planned-before-result-bearing-execution",
        "protocol_hash": PROTOCOL_HASH,
        "family": args.family,
        "input_sha256": sha(args.inputs),
        "shard_count": len(shard_paths),
        "shards": [manifest_path(path) for path in shard_paths],
        "expected_total_calls": sum(json.loads(path.read_text())["expected_calls"] for path in shard_paths),
        "expected_total_beliefs": sum(json.loads(path.read_text())["expected_beliefs"] for path in shard_paths),
        "coverage": "exactly disjoint and exhaustive over the family-specific v5 keyspace",
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
