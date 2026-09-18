"""V3 Phase-1 prototype retrieval diagnostics (NO simulation/profit yet).

Ranks the frozen prototype corpus per query and reports STANDARD IR metrics
via ir_measures (cross-checked against src/retrieval.py hand-rolled nDCG).
Saves raw TREC runs with query-level scores for downstream analysis.

Systems: none / random / bm25 (in-repo, frozen) / dense-STANDIN (MiniLM,
pipeline validation only) / hybrid-RRF / rerank-STANDIN (MiniLM cross-encoder,
mechanics only) / oracle-relevant. Stand-ins are NEVER the frozen baselines;
see configs/v3/models.yaml.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus_v3 import build_proto_corpus
from src.retrieval import FrozenEmbeddings, rank as v2_rank

DATA = ROOT / "data" / "v3proto"
RUNS = ROOT / "runs" / "v3proto"
STANDIN_DENSE = "sentence-transformers/all-MiniLM-L6-v2"
STANDIN_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


class CorpusPool:
    """Adapter exposing the prototype corpus through the V2 rank() interface."""

    def __init__(self, docs, relevance, query):
        from src.warning_corpus import Document
        self.documents = [Document(doc_id=d.doc_id, text=d.text, kind=d.kind,
                                  regime=d.regime_described, entity=d.entity_node,
                                  window=d.window, source=d.source)
                          for d in docs]
        self.relevance = relevance
        self.query = query


def _load():
    build = build_proto_corpus()
    docs = list(build["docs"].values())
    queries = []
    with open(DATA / "queries.jsonl") as f:
        for line in f:
            queries.append(json.loads(line))
    qrels = defaultdict(dict)
    for split in ("dev", "test"):
        with open(DATA / "qrels" / f"{split}.tsv") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return docs, queries, qrels


def _rrf(rankings, k=RRF_K):
    score = defaultdict(float)
    for ranking in rankings:
        for i, did in enumerate(ranking, start=1):
            score[did] += 1.0 / (k + i)
    return sorted(score, key=lambda d: (-score[d], d))


def _dense_ranking(query_text, docs, model):
    from src.retrieval import _cosine_scores
    texts = [d.text for d in docs]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    q = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)[0]
    s = _cosine_scores(q, np.asarray(emb))
    ids = [d.doc_id for d in docs]
    return [ids[i] for i in np.argsort(-s, kind="stable")]


def run_diagnostics(seed=0, write=True):
    docs, queries, qrels = _load()
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(STANDIN_DENSE)
        print(f"stand-in dense: {STANDIN_DENSE} (PIPELINE VALIDATION ONLY)")
    except Exception as e:
        print(f"stand-in dense unavailable ({e}); dense/hybrid/rerank skipped")
        st_model = None

    rankings = {}
    for q in queries:
        qid, qtext = q["_id"], q["text"]
        rel = {d.doc_id: qrels[qid].get(d.doc_id, 0) for d in docs}
        pool = CorpusPool(docs, rel, qtext)
        r = {"random": v2_rank("random", pool, seed=seed),
             "bm25": v2_rank("bm25", pool, seed=seed),
             "oracle": v2_rank("oracle", pool, seed=seed)}
        if st_model is not None:
            dense = _dense_ranking(qtext, docs, st_model)
            r["dense-standin"] = dense
            r["hybrid-rrf"] = _rrf([r["bm25"], dense])
        rankings[qid] = (r, rel)

    # rerank-standin: cross-encoder over hybrid top-50 -> top-k (mechanics only)
    if st_model is not None:
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(STANDIN_RERANK)
            print(f"stand-in reranker: {STANDIN_RERANK} (MECHANICS ONLY)")
            for q in queries:
                qid = q["_id"]
                r, _ = rankings[qid]
                cand = r["hybrid-rrf"][:50]
                texts = [next(d.text for d in docs if d.doc_id == c) for c in cand]
                scores = ce.predict([(q["text"], t) for t in texts])
                order = np.argsort(-np.asarray(scores), kind="stable")
                top = [cand[int(i)] for i in order]
                rest = [d for d in r["hybrid-rrf"] if d not in top]
                r["rerank-standin"] = top + rest
        except Exception as e:
            print(f"stand-in reranker unavailable ({e}); rerank skipped")

    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        for sys_name in sorted({s for r, _ in rankings.values() for s in r}):
            with open(RUNS / f"{sys_name}.trec", "w") as f:
                for q in queries:
                    for i, did in enumerate(rankings[q["_id"]][0][sys_name], start=1):
                        f.write(f"{q['_id']} Q0 {did} {i} {1000 - i} {sys_name}\n")

    # metrics via ir_measures (graded qrels)
    import ir_measures
    from ir_measures import nDCG, Recall, RR, AP, Qrel, ScoredDoc
    qrels = [Qrel(query_id=qid, doc_id=did, relevance=g)
             for qid, rel in ((q["_id"], rankings[q["_id"]][1]) for q in queries)
             for did, g in rel.items()]
    runs = {}
    for sys_name in sorted({s for r, _ in rankings.values() for s in r}):
        runs[sys_name] = [ScoredDoc(query_id=q["_id"], doc_id=did, score=1000 - i)
                          for q in queries
                          for i, did in enumerate(rankings[q["_id"]][0][sys_name])]
    summary = {}
    for sys_name, run in runs.items():
        res = ir_measures.calc_aggregate([nDCG@10, Recall@10, Recall@20, RR, AP],
                                         qrels, run)
        summary[sys_name] = {str(k): float(v) for k, v in res.items()}
    # query-level nDCG@10 for downstream stats (per-run loop: the
    # multi-run dict form is not supported by the pytrec_eval provider here)
    qlevel = defaultdict(dict)
    for sys_name, run in runs.items():
        for row in ir_measures.iter_calc([nDCG@10], qrels, run):
            qlevel[row.query_id][sys_name] = float(row.value)
    out = {"aggregate": summary, "note": "stand-in dense/rerank validate pipeline only"}
    if write:
        (RUNS / "diagnostics.json").write_text(json.dumps(out, indent=2))
    return {"rankings": rankings, "summary": summary, "qlevel": dict(qlevel)}


if __name__ == "__main__":
    out = run_diagnostics()
    for s, m in sorted(out["summary"].items()):
        print(f"{s:16s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
