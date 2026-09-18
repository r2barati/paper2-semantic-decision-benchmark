"""V3.1 complete judgments: grade EVERY (query, doc) pair by the deterministic rule.

No unjudged docs remain, so pooling variance is zero by construction. Test TEXT
untouched; test METRICS stay sealed until the main experiment (no system is scored
on test here -- this script writes labels only).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus_v3_full import build_full_corpus
from src.qrels_v31 import HEDGE_MARKERS, OPS_MARKERS, ROUTINE_MARKERS


def featurize(doc):
    t = doc.text
    m = re.search(r"\[node (\S+)", t)
    low = t.lower()
    return {"node": m.group(1) if m else None,
            "resolved": "resolved / closed" in t,
            "routine": doc.kind in ("offtopic", "shared") or
                       any(k in low for k in ROUTINE_MARKERS),
            "w42": "planning window w-42" in low,
            "ops": any(k in low for k in OPS_MARKERS),
            "hedge": any(k in low for k in HEDGE_MARKERS)}


def grade_feat(qnode, f):
    if f["node"] != qnode:
        return 0
    if f["resolved"]:
        return 1
    if f["routine"]:
        return 0
    if not f["w42"]:
        return 1 if f["ops"] else 0
    if not f["ops"]:
        return 0
    return 2 if f["hedge"] else 3


def main():
    b = build_full_corpus()
    feats = {dd: featurize(d) for dd, d in b["docs"].items()}
    out = ROOT / "data" / "v3" / "qrels_complete"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        n = 0
        with open(out / f"{split}.tsv", "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for q in b["queries"]:
                if q.split != split:
                    continue
                for dd, d in b["docs"].items():
                    g = grade_feat(q.entity_node, feats[dd])
                    if g:
                        f.write(f"{q.query_id}\t{dd}\t{g}\n")
                        n += 1
        print(split, "nonzero pairs:", n, flush=True)


if __name__ == "__main__":
    main()
