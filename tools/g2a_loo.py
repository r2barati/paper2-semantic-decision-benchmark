"""G2A-0 LOO re-computation (zero LLM calls; lookup-only + deterministic rules).

Scope (DESIGN-G2A.md): rerank k=3 x {C0, C3-8B, C3-14B} x all 200 queries.
C3: per-doc judgments via recomputed cache keys (lookup-only; MISS = loud
failure, never a network call). C0: pure rule recompute (imported).
C1: excluded (needs new LLM calls -> 2A-1).

Gates:
 G1 full-set reproduction vs frozen parquet (1e-9 / abstain exact), 100%.
 G2 cache completeness (misses counted; expected 0; misses excluded).

Writes results/v3main/extension_g2a/loo_beliefs.parquet + g2a_gates.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.consumers_v3 import (ABSTAIN_TAU, FROZEN_C3_SHA, consume_c0,  # noqa: E402
                              prompt_sha, PROMPT_C3_DOC, assert_frozen_prompts)

assert_frozen_prompts()

OUT = ROOT / "results" / "v3main" / "extension_g2a"
OUT.mkdir(parents=True, exist_ok=True)

CACHE = {"Qwen/Qwen3-8B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_awq",
         "Qwen/Qwen3-14B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_qwen14"}
STRATA = [("C0", "none"), ("C3", "Qwen/Qwen3-8B-AWQ"), ("C3", "Qwen/Qwen3-14B-AWQ")]

assert prompt_sha(PROMPT_C3_DOC) == FROZEN_C3_SHA


def _parse_json(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _c3_key(model: str, user: str) -> Path:
    src = f"{model}||{FROZEN_C3_SHA}||{hashlib.sha256(user.encode()).hexdigest()[:16]}"
    return CACHE[model] / (hashlib.sha256(src.encode()).hexdigest()[:16] + ".json")


def _c3_judgment(model: str, own_node: str, text: str):
    """Lookup-only single-doc judgment. Raises FileNotFoundError on miss."""
    user = f"Operator's own node: {own_node}\n\nEvidence document:\n{text}"
    p = json.loads(_c3_key(model, user).read_text())
    assert p.get("prompt_sha") == FROZEN_C3_SHA, "prompt drift in cache entry"
    return _parse_json(p["raw"])


def _c3_aggregate(per_doc: list[tuple[str, dict]]):
    """Frozen C3 aggregation, byte-identical formula to consume_c3."""
    import math
    agg = {"normal": 0.0, "supplier_delay": 0.0, "demand_surge": 0.0}
    support = 0.0
    for _, e in per_doc:
        w = float(e["confidence"]) * (1.0 if e["entity_match"] else 0.2) * \
            (1.0 if e["fresh"] else 0.2)
        ev = e["event"]
        if ev in agg and e["stance"] == "support":
            agg[ev] += w
            support = max(support, w)
        elif ev in agg and e["stance"] == "refute":
            agg[ev] -= 0.5 * w
    if support < ABSTAIN_TAU:
        return {"p_normal": 1/3, "p_supplier_delay": 1/3, "p_demand_surge": 1/3,
                "abstain": True, "confidence": support}
    prior = {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30}
    scores = {k: agg[k] + 0.5 * prior[k] for k in agg}
    m = max(scores.values())
    ex = {k: math.exp(v - m) for k, v in scores.items()}
    tot = sum(ex.values())
    return {"p_normal": ex["normal"]/tot, "p_supplier_delay": ex["supplier_delay"]/tot,
            "p_demand_surge": ex["demand_surge"]/tot, "abstain": False,
            "confidence": min(1.0, support)}


def main():
    docs, qmeta = {}, {}
    for line in open(ROOT / "data" / "v3" / "corpus.jsonl"):
        d = json.loads(line)
        docs[d["_id"]] = d["text"]
    for line in open(ROOT / "data" / "v3" / "queries.jsonl"):
        q = json.loads(line)
        qmeta[q["_id"]] = q["metadata"]

    bel8 = pd.read_parquet(ROOT / "results/v3main/beliefs_Qwen_Qwen3-8B-AWQ.parquet")
    bel14 = pd.read_parquet(ROOT / "results/v3main/beliefs_Qwen_Qwen3-14B-AWQ.parquet")
    bel = pd.concat([bel8, bel14], ignore_index=True)
    base = bel[(bel["system"] == "rerank") & (bel["k"] == 3)]
    base = base[((base["consumer"] == "C0")) |
                ((base["consumer"] == "C3") &
                 (base["model"].isin(list(CACHE))))].copy()
    # Frozen-pipeline C0 twin dedupe (cf. tools/run_sim_v3main.py): every C0
    # key appears exactly twice (once per model file); twins must be identical.
    c0 = base[base["consumer"] == "C0"]
    grp = c0.groupby(["system", "query_id", "k"])
    assert (grp.size() == 2).all(), "every C0 key must appear exactly twice"
    scalar = [c for c in c0.columns if c not in ("doc_ids", "latency_s", "model")]
    assert bool((grp[scalar].nunique() == 1).all().all()), "C0 twin rows differ!"
    base = pd.concat([c0.sort_values(["system", "query_id", "k"]).drop_duplicates(
        ["system", "query_id", "k"], keep="first"),
        base[base["consumer"] != "C0"]], ignore_index=True)
    assert len(base) == 200 + 200 * 2, f"unexpected rerank base rows {len(base)}"

    rows, g1_bad, misses = [], [], 0
    for _, r in base.iterrows():
        qid, cons, model = r["query_id"], r["consumer"], r["model"]
        dids = list(r["doc_ids"])
        assert len(dids) == 3, f"{qid} expected 3 docs, got {len(dids)}"
        own = qmeta[qid]["entity_node"]
        subsets = {"full": dids, "drop0": dids[1:], "drop1": [dids[0], dids[2]],
                   "drop2": dids[:2]}
        for ab, sub in subsets.items():
            if cons == "C0":
                top = [{"doc_id": d, "text": docs[d]} for d in sub]
                b = consume_c0(top)
                out = {"p_normal": b.p_normal, "p_supplier_delay": b.p_supplier_delay,
                       "p_demand_surge": b.p_demand_surge, "abstain": b.abstain,
                       "confidence": b.confidence}
            else:
                try:
                    per = [(d, _c3_judgment(model, own, docs[d])) for d in sub]
                except FileNotFoundError as e:
                    misses += 1
                    if ab == "full":
                        g1_bad.append({"query_id": qid, "consumer": cons,
                                       "model": model, "error": f"cache miss full: {e}"})
                    continue
                out = _c3_aggregate(per)
            rows.append({"query_id": qid, "system": "rerank", "consumer": cons,
                         "model": model, "k": 3, "n_docs": len(sub),
                         "ablation": ab, "doc_ids": list(sub),
                         "split": r["split"], "true_regime": r["true_regime"], **out})
    # G1: compare full rows vs frozen (vectorized below)
    full = pd.DataFrame([x for x in rows if x["ablation"] == "full"])
    key = ["query_id", "consumer", "model"]
    mg = full.merge(base[key + ["p_normal", "p_supplier_delay", "p_demand_surge",
                                "abstain", "confidence"]],
                    on=key, suffixes=("", "_frozen"))
    assert len(mg) == len(full) == len(base), "full-row key mismatch"
    import numpy as np
    dmax = max((mg["p_normal"] - mg["p_normal_frozen"]).abs().max(),
               (mg["p_supplier_delay"] - mg["p_supplier_delay_frozen"]).abs().max(),
               (mg["p_demand_surge"] - mg["p_demand_surge_frozen"]).abs().max())
    ab_mismatch = int((mg["abstain"] != mg["abstain_frozen"]).sum())
    loo = pd.DataFrame([x for x in rows if x["ablation"] != "full"])
    loo.to_parquet(OUT / "loo_beliefs.parquet", index=False)
    gates = {"G1_max_abs_p_diff": float(dmax), "G1_abstain_mismatch": ab_mismatch,
             "G1_n": len(mg), "G1_pass": bool(dmax <= 1e-9 and ab_mismatch == 0),
             "G2_cache_misses": misses, "loo_rows": len(loo)}
    json.dump(gates, open(OUT / "g2a_gates.json", "w"), indent=1)
    print(json.dumps(gates, indent=1))
    if not gates["G1_pass"]:
        sys.exit(2)
    print(f"wrote {OUT/'loo_beliefs.parquet'} rows={len(loo)}")


if __name__ == "__main__":
    main()
