"""Phase 1B: rerank hybrid top-50 with real Qwen3-Reranker-0.6B (single-pair loop;
the baked modeling config lacks a pad token, so batch>1 is impossible).
Writes runs/v3proto/rerank-qwen3.trec + rankings json + ir_measures eval.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_transformers import CrossEncoder
from src.experiment_v3proto import _load, _rrf, CorpusPool
from src.retrieval import rank as v2_rank

ce = CrossEncoder('Qwen/Qwen3-Reranker-0.6B',
                  revision='e61197ed45024b0ed8a2d74b80b4d909f1255473',
                  trust_remote_code=True)
docs, queries, qrels = _load()
z = np.load(str(ROOT / 'runs/v3proto/qwen3emb_vectors.npz'))
dids = [str(x) for x in z['doc_ids']]
qid_list = [str(x) for x in z['qids']]
txt = {d.doc_id: d.text for d in docs}
out = {}
done = 0
for q in queries:
    qid = q['_id']
    rel = {d.doc_id: qrels[qid].get(d.doc_id, 0) for d in docs}
    pool = CorpusPool(docs, rel, q['text'])
    bm25 = v2_rank('bm25', pool)
    s = dict(zip(dids, (z['q_emb'][qid_list.index(qid)] @ z['doc_emb'].T)))
    dense = sorted(dids, key=lambda d: (-s[d], d))
    hyb = _rrf([bm25, dense])
    cand = hyb[:50]
    scores = [float(ce.predict([(q['text'], txt[c])])) for c in cand]
    top = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
    rest = [d for d in hyb if d not in top]
    out[qid] = top + rest
    done += 1
    print(f'{done}/20 {qid}', flush=True)

with open(ROOT / 'runs/v3proto/rerank-qwen3_rankings.json', 'w') as f:
    json.dump(out, f)
with open(ROOT / 'runs/v3proto/rerank-qwen3.trec', 'w') as f:
    for qid, r in out.items():
        for i, did in enumerate(r, start=1):
            f.write(f'{qid} Q0 {did} {i} {1000-i} rerank-qwen3\n')

import ir_measures
from ir_measures import nDCG, Recall, RR, AP, Qrel, ScoredDoc
qr = [Qrel(query_id=qid, doc_id=did, relevance=g)
      for qid, pool in ((q['_id'], None) for q in queries)
      for did, g in qrels[qid].items()]
run = [ScoredDoc(query_id=qid, doc_id=did, score=1000 - i)
       for qid, r in out.items() for i, did in enumerate(r)]
res = ir_measures.calc_aggregate([nDCG@10, Recall@10, Recall@20, RR, AP], qr, run)
print('RERANK-QWEN3-REAL:', {str(k): round(float(v), 3) for k, v in res.items()})
ql = {}
for row in ir_measures.iter_calc([nDCG@10], qr, run):
    ql[row.query_id] = round(float(row.value), 4)
json.dump(ql, open(ROOT / 'runs/v3proto/rerank-qwen3_qlevel.json', 'w'), indent=1)
print('DONE')
