"""V3 main belief matrix: evidence sets x C0/C1/C3 x LLM x k (belief freeze BEFORE sim).

Evidence sets: none/random/bm25/dense/hybrid/rerank (trecs) + oracle-relevant,
oracle-factual + 7 interventions (arms.json). k in (3, 5). Consumers/prompts/tau
frozen (asserted). Every LLM I/O cached; schema-validated; explicit fail on
missing/invalid (never silent prior). Output: results/v3main/beliefs.parquet +
belief_manifest.json (SHA freeze).

Env: PAPER2_V3_CACHE_DIR (default results/v3main/beliefs_cache),
MODEL (default gpt-4o-2024-11-20), LLM_BASE_URL override for AWQ/vLLM.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.consumers_v3 import (assert_frozen_prompts, consume_c0, consume_c1,
                              consume_c3)

MODEL = os.environ.get("V3_MODEL", "gpt-4o-2024-11-20")
KS = (3, 5)
SYSTEM_TRECS = {"random": None, "bm25": "kaggle/inputs_main/bm25_full.trec",
                "dense": "runs/v3main/dense_full.trec",
                "hybrid": "runs/v3main/hybrid_k120_full.trec",
                "rerank": "runs/v3main/rerank_full.trec"}
OUT = ROOT / "results" / "v3main"


def _load_corpus():
    docs, queries = {}, []
    for line in open(ROOT / "data" / "v3" / "corpus.jsonl"):
        d = json.loads(line)
        docs[d["_id"]] = d
    for line in open(ROOT / "data" / "v3" / "queries.jsonl"):
        queries.append(json.loads(line))
    return docs, queries


def _load_rankings():
    rank = {}
    for sys_name, path in SYSTEM_TRECS.items():
        if path is None:  # random: seeded locally, deterministic
            continue
        with open(ROOT / path) as f:
            for line in f:
                qid, _, did, _, _, _ = line.split()
                rank.setdefault(sys_name, {}).setdefault(qid, []).append(did)
    return rank


def _random_rankings(docs, queries, seed=7):
    rng = np.random.default_rng(seed)
    ids = list(docs)
    return {q["_id"]: list(rng.permutation(ids)) for q in queries}


def _evidence_sets():
    docs, queries = _load_corpus()
    rank = _load_rankings()
    rank["random"] = _random_rankings(docs, queries)
    arms = json.loads((OUT / "arms.json").read_text()) if (OUT / "arms.json").exists() \
        else json.loads((ROOT / "runs" / "v3" / "arms.json").read_text())
    sets = {}  # (system, qid, k) -> [doc_ids]
    for sys_name, per_q in rank.items():
        for qid, ranking in per_q.items():
            for k in KS:
                sets[(sys_name, qid, k)] = ranking[:k]
    sets[("none", None, None)] = []
    for arm, per_q in arms.items():
        for qid, kd in per_q.items():
            if isinstance(kd, dict):
                for k in KS:
                    kk = str(k)
                    if kk in kd:
                        sets[(arm, qid, k)] = kd[kk]
            elif isinstance(kd, list):
                # interventions are defined on the k=3 base ONLY (frozen spec)
                sets[(arm, qid, 3)] = kd
    return docs, queries, sets


def _doc_view(docs, did):
    d = docs[did]
    return {"doc_id": did, "text": d["text"]}


def run_matrix(systems=None, consumers=("C0", "C1", "C3"), model=MODEL,
               limit_queries=None):
    assert_frozen_prompts()
    docs, queries, sets = _evidence_sets()
    qmeta = {q["_id"]: q["metadata"] for q in queries}
    qtext = {q["_id"]: q["text"] for q in queries}
    if systems is None:
        systems = sorted({s for s, _, _ in sets} - {"none"})
    if limit_queries:
        queries = [q for q in queries if q["_id"] in limit_queries]
    rows = []
    n_calls = n_hit = 0
    for q in queries:
        qid = q["_id"]
        for sys_name in systems:
            for k in KS:
                key = (sys_name, qid, k)
                if key not in sets:
                    continue
                top = [_doc_view(docs, did) for did in sets[key]]
                for consumer in consumers:
                    t0 = time.time()
                    if consumer == "C0":
                        b = consume_c0(top)
                    elif consumer == "C1":
                        b = consume_c1(top, qtext[qid], model)
                    else:
                        b = consume_c3(top, qtext[qid], model,
                                       qmeta[qid]["entity_node"])
                    rows.append({
                        "query_id": qid, "system": sys_name, "consumer": consumer,
                        "model": model if consumer != "C0" else "none", "k": k,
                        "doc_ids": [d["doc_id"] for d in top],
                        "p_normal": b.p_normal,
                        "p_supplier_delay": b.p_supplier_delay,
                        "p_demand_surge": b.p_demand_surge,
                        "abstain": bool(b.abstain), "confidence": b.confidence,
                        "latency_s": round(time.time() - t0, 3),
                        "split": qmeta[qid].get("split", ""),
                        "true_regime": qmeta[qid].get("true_regime", ""),
                    })
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"beliefs_{model.replace('/', '_')}.parquet", index=False)
    with open(OUT / f"beliefs_{model.replace('/', '_')}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"beliefs: {len(rows)} rows model={model} systems={len(systems)}")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--consumers", nargs="*", default=["C0", "C1", "C3"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--queries", nargs="*", default=None)
    a = ap.parse_args()
    run_matrix(systems=a.systems, consumers=list(a.consumers), model=a.model,
               limit_queries=set(a.queries) if a.queries else None)
