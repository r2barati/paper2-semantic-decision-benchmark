"""Phase 1B consumer pilot on PRODUCTION rankings (NO simulation).

Dev queries x {bm25, dense-qwen3, hybrid-real, rerank-qwen3, oracle} x C0/C1/C3
with gpt-4o-2024-11-20. Qwen3-8B arm pending compute (same harness, base_url).
Compares D4 variation against the Phase-1 stand-in pilot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.consumers_v3 import consume_c0, consume_c1, consume_c3
from src.experiment_v3proto import _load

MODEL = "gpt-4o-2024-11-20"
SYSTEM_FILES = {"bm25": None, "dense-qwen3": "dense-qwen3.trec",
                "hybrid-real": "hybrid-real.trec", "rerank-qwen3": "rerank-qwen3.trec"}
K = 3
OUT = ROOT / "results" / "v3proto"


def _read_trec(path):
    rank = {}
    with open(path) as f:
        for line in f:
            qid, _, did, r, _, _ = line.split()
            rank.setdefault(qid, []).append(did)
    return rank


def main():
    docs, queries, qrels = _load()
    by_id = {d.doc_id: {"doc_id": d.doc_id, "text": d.text} for d in docs}
    # bm25 ranking recomputed deterministically
    from src.retrieval import rank as v2_rank
    from src.experiment_v3proto import CorpusPool
    dev = [q for q in queries if q["metadata"]["split"] == "dev"]
    trecs = {s: _read_trec(ROOT / "runs/v3proto" / f) for s, f in SYSTEM_FILES.items() if f}
    rows = []
    for q in dev:
        qid = q["_id"]
        rel = {d.doc_id: qrels[qid].get(d.doc_id, 0) for d in docs}
        trecs["bm25"] = {qid: v2_rank("bm25", CorpusPool(docs, rel, q["text"]))}
        # oracle: relevance-desc
        trecs["oracle"] = {qid: sorted(rel, key=lambda d: (-rel[d], d))}
        for sys_name, t in trecs.items():
            top = [by_id[did] for did in t[qid][:K]]
            for consumer, belief in [
                    ("C0", consume_c0(top)),
                    ("C1", consume_c1(top, q["text"], MODEL)),
                    ("C3", consume_c3(top, q["text"], MODEL, q["metadata"]["entity_node"])),
            ]:
                truth = q["metadata"]["true_regime"]
                p = {"normal": belief.p_normal, "supplier_delay": belief.p_supplier_delay,
                     "demand_surge": belief.p_demand_surge}
                pred = max(p, key=p.get)
                brier = sum((p[k] - (1.0 if k == truth else 0.0)) ** 2 for k in p)
                rows.append({"query_id": qid, "system": sys_name, "consumer": consumer,
                             "pred": pred, "correct": pred == truth,
                             "brier": round(brier, 4), "abstain": belief.abstain})
    with open(OUT / "pilot_beliefs_1b.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("== 1B D4 (production rankings, GPT-4o-2024-11-20) ==")
    for c in ("C0", "C1", "C3"):
        accs = []
        for s in ["bm25", "dense-qwen3", "hybrid-real", "rerank-qwen3", "oracle"]:
            sub = [r for r in rows if r["consumer"] == c and r["system"] == s]
            if not sub:
                print(f"  {c} x {s}: NO DATA (rerank pending?)")
                continue
            a = sum(r["correct"] for r in sub) / len(sub)
            accs.append(a)
            print(f"  {c} x {s:14s} acc={a:.2f} brier={sum(r['brier'] for r in sub)/len(sub):.3f}")
        print(f"  {c} range={max(accs)-min(accs):.2f}" if accs else "")
    sub3 = [r for r in rows if r["consumer"] == "C3"]
    print(f"  C3 abstention={sum(r['abstain'] for r in sub3)/len(sub3):.2f}")


if __name__ == "__main__":
    main()
