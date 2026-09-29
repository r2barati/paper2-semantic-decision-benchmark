"""Assemble qrel-free model-serving inputs after retrieval is frozen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RETRIEVAL = ROOT / "versions/sem2act-v5/runtime/retrieval_bundle"
OUT = ROOT / "versions/sem2act-v5/runtime/model_inputs"
SYSTEMS = ("bm25", "rerank", "oracle")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trec(path):
    out = {}
    for line in path.read_text().splitlines():
        qid, _, did, rank, *_ = line.split()
        out.setdefault(qid, []).append(did)
    return out


def main():
    queries = [json.loads(x) for x in (RETRIEVAL / "queries.jsonl").read_text().splitlines()]
    docs = {x["_id"]: x for x in map(json.loads, (RETRIEVAL / "corpus.jsonl").read_text().splitlines())}
    rankings = {
        "bm25": read_trec(RETRIEVAL / "bm25.trec"),
        "oracle": read_trec(RETRIEVAL / "oracle.trec"),
    }
    rerank_path = RETRIEVAL / "rerank.trec"
    if not rerank_path.exists():
        raise SystemExit("rerank.trec is required before model inputs are assembled")
    rankings["rerank"] = read_trec(rerank_path)
    rows = []
    for q in queries:
        qid = q["_id"]
        for system in SYSTEMS:
            top = rankings[system][qid][:3]
            rows.append({
                "query_id": qid,
                "query_text": q["text"],
                "system": system,
                "doc_ids": top,
                "documents": [{"doc_id": did, "text": docs[did]["text"]} for did in top],
                "metadata": q["metadata"],
            })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_inputs.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rows)
    )
    if any("qrel" in x.lower() for x in (OUT / "evidence_inputs.jsonl").read_text().splitlines()):
        raise SystemExit("qrel token leaked into model inputs")
    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-model-inputs",
        "qrels_read": False,
        "n_rows": len(rows),
        "rows_per_system": {s: sum(x["system"] == s for x in rows) for s in SYSTEMS},
        "files": {"evidence_inputs.jsonl": {
            "sha256": sha(OUT / "evidence_inputs.jsonl"),
            "bytes": (OUT / "evidence_inputs.jsonl").stat().st_size,
        }},
    }
    (OUT / "model_inputs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
