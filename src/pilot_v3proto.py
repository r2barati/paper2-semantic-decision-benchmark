"""V3 Phase-1 consumer pilot: dev queries x systems x C0/C1/C3 (NO simulation).

Reports D4 diagnostics: belief-accuracy variation across retrievers (C1),
C3 abstention rate. All LLM outputs cached under results/v3proto/llm_cache/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.consumers_v3 import consume_c0, consume_c1, consume_c3, freeze_prompts
from src.experiment_v3proto import _load, run_diagnostics

MODEL = "gpt-4o-2024-11-20"
SYSTEMS = ["bm25", "dense-standin", "hybrid-rrf", "oracle"]
K = 3
OUT = ROOT / "results" / "v3proto"


def main():
    freeze_prompts(ROOT / "configs" / "v3" / "prompts_freeze.json")
    docs, queries, qrels = _load()
    diag = run_diagnostics(write=False)
    dev = [q for q in queries if q["metadata"]["split"] == "dev"]
    by_id = {d.doc_id: {"doc_id": d.doc_id, "text": d.text} for d in docs}
    rows = []
    for q in dev:
        qid = q["_id"]
        ranking, _ = diag["rankings"][qid]
        for sys_name in SYSTEMS:
            top = [by_id[did] for did in ranking[sys_name][:K]]
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
                             "k": K, "model": MODEL if consumer != "C0" else "none",
                             "pred": pred, "correct": pred == truth, "brier": round(brier, 4),
                             "abstain": belief.abstain,
                             "confidence": round(belief.confidence, 3),
                             "nDCG@10": None})
                # attach retrieval quality
                rows[-1]["nDCG@10"] = round(diag["qlevel"].get(qid, {}).get(sys_name, float("nan")), 4)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pilot_beliefs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # D4 summary
    print("== D4: belief accuracy by (consumer, system) ==")
    for c in ("C0", "C1", "C3"):
        accs = []
        for s in SYSTEMS:
            sub = [r for r in rows if r["consumer"] == c and r["system"] == s]
            a = sum(r["correct"] for r in sub) / len(sub)
            accs.append(a)
            print(f"  {c} x {s:14s} acc={a:.2f} brier={sum(r['brier'] for r in sub)/len(sub):.3f}")
        print(f"  {c} range={max(accs)-min(accs):.2f}")
    sub3 = [r for r in rows if r["consumer"] == "C3"]
    print(f"  C3 abstention={sum(r['abstain'] for r in sub3)/len(sub3):.2f} "
          f"(C1 has no abstain path except empty evidence; mechanism checked separately)")


if __name__ == "__main__":
    main()
