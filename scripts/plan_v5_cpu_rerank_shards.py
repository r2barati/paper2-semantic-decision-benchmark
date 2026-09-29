"""Plan exact query-range CPU reranker shards without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "versions/sem2act-v5/manifests/cpu_shards/reranker"
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--queries-per-shard", type=int, default=20)
    parser.add_argument("--resume-policy", choices=("missing-key-only", "one-shot-empty-output-only"),
                        default="missing-key-only")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    candidates = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    if len(queries) != 240 or len(candidates) != 240:
        raise SystemExit("reranker planner requires exactly 240 queries and candidate rows")
    qids = [row["_id"] for row in queries]
    if set(qids) != {f"v5-lb-q{i:04d}" for i in range(1, 241)} or len(set(qids)) != 240:
        raise SystemExit("reranker query keyspace drift")
    candidate_map = {row["query_id"]: row["doc_ids"] for row in candidates}
    if set(candidate_map) != set(qids) or any(len(ids) != 50 for ids in candidate_map.values()):
        raise SystemExit("reranker candidate keyspace or depth drift")
    if args.queries_per_shard < 1:
        parser.error("--queries-per-shard must be positive")
    ordered = sorted(qids)
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(math.ceil(len(ordered) / args.queries_per_shard)):
        selected = ordered[index * args.queries_per_shard:(index + 1) * args.queries_per_shard]
        expected = [f"reranker|{qid}" for qid in selected]
        manifest = {
            "schema_version": 1,
            "manifest_id": f"sem2act-v5-cpu-reranker-shard-{index:03d}",
            "status": "planned-before-result-bearing-execution",
            "protocol_hash": PROTOCOL_HASH,
            "stage": "reranker",
            "shard_index": index,
            "query_ids": selected,
            "expected_query_keys": expected,
            "expected_queries": len(selected),
            "candidate_depth": 50,
            "queries_sha256": sha(args.queries),
            "corpus_sha256": sha(args.corpus),
            "candidates_sha256": sha(args.candidates),
            "resume_policy": args.resume_policy,
        }
        path = args.output_root / f"shard-{index:03d}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    index = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-cpu-reranker-shards-v1",
        "status": "planned-before-result-bearing-execution",
        "protocol_hash": PROTOCOL_HASH,
        "stage": "reranker",
        "shards": [manifest_path(path) for path in paths],
        "shard_count": len(paths),
        "expected_total_queries": sum(json.loads(path.read_text())["expected_queries"] for path in paths),
        "coverage": "exactly disjoint and exhaustive over the 240-query reranker keyspace",
    }
    (args.output_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
