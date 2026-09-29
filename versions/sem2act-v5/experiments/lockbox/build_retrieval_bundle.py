"""Build qrel-free v5 retrieval inputs.

Oracle ranking is materialized from private qrels before model execution, but
the qrels themselves never enter the emitted bundle. Rerank candidates use the
frozen v3 retrieval recipe's BM25 candidate stage and the pinned Qwen reranker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LOCKBOX = ROOT / "versions/sem2act-v5/runtime/lockbox"
OUT = ROOT / "versions/sem2act-v5/runtime/retrieval_bundle"
RERANK_TOPN = 50


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        from collections import Counter
        import math
        self.texts = [tokenize(t) for t in texts]
        self.n = len(self.texts)
        self.lengths = [len(t) for t in self.texts]
        self.avg = sum(self.lengths) / self.n
        self.tf = [Counter(t) for t in self.texts]
        df = Counter()
        for t in self.texts:
            df.update(set(t))
        self.idf = {
            token: max(0.0, math.log(1 + (self.n - count + 0.5) / (count + 0.5)))
            for token, count in df.items()
        }
        self.k1, self.b = k1, b

    def score(self, query):
        out = np.zeros(self.n)
        query_terms = tokenize(query)
        for i, terms in enumerate(self.texts):
            total = 0.0
            for token in query_terms:
                f = self.tf[i].get(token, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * len(terms) / self.avg)
                total += self.idf.get(token, 0.0) * f * (self.k1 + 1) / denom
            out[i] = total
        return out


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    queries = [json.loads(x) for x in (LOCKBOX / "queries.jsonl").read_text().splitlines()]
    docs = [json.loads(x) for x in (LOCKBOX / "corpus.jsonl").read_text().splitlines()]
    doc_ids = [d["_id"] for d in docs]
    texts = [d["text"] for d in docs]
    qrels = {}
    with (LOCKBOX / "private/qrels.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    bm25 = BM25(texts)
    bm25_lines, oracle_lines, candidates = [], [], []
    for query in queries:
        qid = query["_id"]
        scores = bm25.score(query["text"])
        score_by_id = {did: float(scores[i]) for i, did in enumerate(doc_ids)}
        order = sorted(range(len(doc_ids)), key=lambda i: (-scores[i], doc_ids[i]))
        bm = [doc_ids[i] for i in order]
        oracle = sorted(doc_ids, key=lambda did: (-qrels.get(qid, {}).get(did, 0), did))
        for rank, did in enumerate(bm, start=1):
            bm25_lines.append(f"{qid} Q0 {did} {rank} {score_by_id[did]:.8f} v5-bm25")
        for rank, did in enumerate(oracle, start=1):
            oracle_lines.append(f"{qid} Q0 {did} {rank} {100000-rank} v5-oracle")
        candidates.append({"query_id": qid, "doc_ids": bm[:RERANK_TOPN]})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "queries.jsonl").write_text((LOCKBOX / "qrels_free/queries.jsonl").read_text())
    (OUT / "corpus.jsonl").write_text((LOCKBOX / "qrels_free/corpus.jsonl").read_text())
    (OUT / "bm25.trec").write_text("\n".join(bm25_lines) + "\n")
    (OUT / "oracle.trec").write_text("\n".join(oracle_lines) + "\n")
    (OUT / "rerank_candidates.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in candidates)
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-lockbox-retrieval",
        "qrels_read_during_build": True,
        "qrels_in_emitted_bundle": False,
        "rerank_model": "Qwen/Qwen3-Reranker-0.6B",
        "rerank_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "rerank_candidate_depth": RERANK_TOPN,
        "n_queries": len(queries),
        "files": {
            p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in (
                OUT / "queries.jsonl", OUT / "corpus.jsonl",
                OUT / "bm25.trec", OUT / "oracle.trec",
                OUT / "rerank_candidates.jsonl",
            )
        },
    }
    (OUT / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
