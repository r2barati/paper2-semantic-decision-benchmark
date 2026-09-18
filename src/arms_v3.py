"""V3 oracle + intervention arms (FROZEN definitions, main experiment).

OracleRelevant: top-k by grade desc (ties: doc_id).
OracleFactual: grade>=2 AND fresh AND entity_match AND stance != refute, grade-desc.
Interventions (on the OracleFactual k=3 base; prespecified set ONLY):
  drop-decisive, stale-swap, inject-contra, wrong-entity-swap, duplicate-top,
  order-reverse, irrelevant-inject.
Output: runs/v3/arms.json {arm: {qid: [doc_ids]}} for k in (3, 5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KS = (3, 5)


def load():
    ev = pd.read_parquet(ROOT / "data" / "v3" / "evidence_labels.parquet")
    return ev


def _top(rows, k):
    return rows.sort_values(["qrel", "doc_id"], ascending=[False, True])["doc_id"].tolist()[:k]


def build_arms():
    ev = load()
    arms = {}
    by_q = {qid: g for qid, g in ev.groupby("query_id")}
    arms["oracle-relevant"] = {}
    arms["oracle-factual"] = {}
    for qid, g in by_q.items():
        r = g[g["qrel"] >= 2]
        arms["oracle-relevant"][qid] = {k: _top(r, k) for k in KS}
        f = r[(r["fresh"]) & (r["entity_match"]) & (r["stance"] != "refute")]
        arms["oracle-factual"][qid] = {k: _top(f, k) for k in KS}
    base = {qid: arms["oracle-factual"][qid][3] for qid in by_q}
    pool = {qid: g for qid, g in by_q.items()}
    inter = {}
    for qid, g in pool.items():
        b = list(base[qid])
        stale = g[(g["fresh"] == False) & (g["entity_match"])]["doc_id"].tolist()
        contra = g[g["stance"] == "refute"]["doc_id"].tolist()
        wrong = g[(~g["entity_match"]) & (g["regime_described"].notna())]["doc_id"].tolist()
        routine = g[(g["doc_kind"] == "offtopic") | (g["doc_kind"] == "shared")]["doc_id"].tolist()
        iv = {}
        iv["drop-decisive"] = b[1:3] if len(b) >= 1 else b
        iv["stale-swap"] = ([stale[0]] + b[1:3])[:3] if stale and b else b
        iv["inject-contra"] = (b[:2] + contra[:1])[:3] if contra else b
        iv["wrong-entity-swap"] = ([wrong[0]] + b[1:3])[:3] if wrong and b else b
        iv["duplicate-top"] = ([b[0]] + b[:2])[:3] if b else b
        iv["order-reverse"] = list(reversed(b))
        iv["irrelevant-inject"] = (b[:2] + routine[:1])[:3] if routine else b
        inter[qid] = iv
    for name in ["drop-decisive", "stale-swap", "inject-contra", "wrong-entity-swap",
                 "duplicate-top", "order-reverse", "irrelevant-inject"]:
        arms[f"intervention-{name}"] = {qid: inter[qid][name] for qid in pool}
    return arms


def main():
    arms = build_arms()
    out = ROOT / "runs" / "v3" / "arms.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # JSON with int keys converted: {arm: {qid: {k: [...]}}}
    serial = {a: {qid: ({k: v[k] for k in v} if isinstance(v, dict) else v)
                  for qid, v in per_q.items()} for a, per_q in arms.items()}
    out.write_text(json.dumps(serial))
    print(f"arms: {sorted(arms)} n_queries: {len(arms['oracle-relevant'])}")


if __name__ == "__main__":
    main()
