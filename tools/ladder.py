"""Workstream 2 ladder (frozen LADDER_SPEC.md; zero new LLM calls).

R1 nDCG@3 (ir_measures recompute, frozen trecs+qrels) -> R2 UDCG-inspired@3
(frozen C3 judgments, 8B primary + 14B sensitivity; lookup-only, misses
documented never filled) -> R3 Brier/accuracy (frozen semantic_all) ->
R4 rerank positional VoI (REUSED frozen verdicts, not recomputed) ->
R5 test dJ (frozen utility_by_query).
Rankings per stratum; adjacent-rung Kendall tau + pairwise sign agreement;
strong reversals (CI-excl-0 both rungs, opposite signs) vs weak
(point flips). Query bootstrap B=2000 (seeds 82000+).
Writes theory/ladder_tables.json + theory/ladder_verdict.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "theory"
TH = ROOT / "results" / "v3main" / "extension_g2a"

SYS = ["bm25", "dense", "hybrid", "rerank"]
TRECS = {"bm25": "kaggle/inputs_main/bm25_full.trec",
         "dense": "runs/v3main/dense_full.trec",
         "hybrid": "runs/v3main/hybrid_k120_full.trec",
         "rerank": "runs/v3main/rerank_full.trec"}
STRATA = [("C0", "none"), ("C1", "Qwen/Qwen3-8B-AWQ"), ("C1", "Qwen/Qwen3-14B-AWQ"),
          ("C3", "Qwen/Qwen3-8B-AWQ"), ("C3", "Qwen/Qwen3-14B-AWQ")]
B, SEED = 2000, 82000

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
test_ids = sorted(set(sp["test"]))
assert len(test_ids) == 160

# ---------------- R1 nDCG@3 ----------------
from ir_measures import ScoredDoc, Qrel, nDCG  # noqa: E402


def load_rank(sys_name):
    rank = {}
    with open(ROOT / TRECS[sys_name]) as f:
        for line in f:
            qid, _, did, _, s, _ = line.split()
            if qid in set(test_ids):
                rank.setdefault(qid, []).append(did)
    return rank


qrel = {}
with open(ROOT / "data" / "v3" / "qrels" / "test.tsv") as f:
    next(f)
    for line in f:
        q, d, g = line.split()
        if q in set(test_ids):
            qrel.setdefault(q, {})[d] = int(g)
import ir_measures  # noqa: E402
r1 = []
for sys_name in SYS:
    rk = load_rank(sys_name)
    for q in test_ids:
        qr = [Qrel(query_id=q, doc_id=d, relevance=g) for d, g in qrel.get(q, {}).items()]
        run = [ScoredDoc(query_id=q, doc_id=d, score=float(1000 - r))
               for r, d in enumerate(rk.get(q, [])[:3])]
        v = float(ir_measures.calc_aggregate([nDCG @ 3], qr, run)[nDCG @ 3]) if qr and run else 0.0
        r1.append({"system": sys_name, "query_id": q, "ndcg3": v})
R1 = pd.DataFrame(r1)
print("R1 done", flush=True)

# ---------------- R2 UDCG-inspired@3 ----------------
CACHE = {"Qwen/Qwen3-8B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_awq",
         "Qwen/Qwen3-14B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_qwen14"}
_docs, _qm = {}, {}
for line in open(ROOT / "data" / "v3" / "corpus.jsonl"):
    _d = json.loads(line)
    _docs[_d["_id"]] = _d["text"]
for line in open(ROOT / "data" / "v3" / "queries.jsonl"):
    _q = json.loads(line)
    if _q["_id"] in set(test_ids):
        _qm[_q["_id"]] = _q["metadata"]


def jug(model, own, text):
    user = f"Operator's own node: {own}\n\nEvidence document:\n{text}"
    ks = f"{model}||5966cadde90e8236||{hashlib.sha256(user.encode()).hexdigest()[:16]}"
    fn = hashlib.sha256(ks.encode()).hexdigest()[:16] + ".json"
    p = _CACHE[model] / fn
    if not p.exists():
        return None
    raw = json.loads(p.read_text())["raw"]
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def gain(e):
    if e["event"] not in ("normal", "supplier_delay", "demand_surge"):
        return 0.0
    w = float(e["confidence"]) * (1.0 if e["entity_match"] else 0.2) * \
        (1.0 if e["fresh"] else 0.2)
    if e["stance"] == "support":
        return w
    if e["stance"] == "refute":
        return -0.5 * w
    return 0.0


_CACHE = CACHE
r2rec, miss, idcg0 = [], 0, 0
for model in ("Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"):
    for sys_name in SYS:
        rk = load_rank(sys_name)
        for q in test_ids:
            us = []
            for did in rk.get(q, [])[:3]:
                e = jug(model, _qm[q]["entity_node"], _docs[did])
                if e is None:
                    miss += 1
                    us.append(0.0)
                else:
                    us.append(gain(e))
            dcg = sum(u / math.log2(r + 2) for r, u in enumerate(us))
            ideal = sorted(us, reverse=True)
            idcg = sum(u / math.log2(r + 2) for r, u in enumerate(ideal))
            if idcg <= 0:
                idcg0 += 1
                sc = 0.0
            else:
                sc = dcg / idcg
            r2rec.append({"system": sys_name, "query_id": q, "model": model,
                          "udcg3": float(sc)})
R2 = pd.DataFrame(r2rec)
R2P = R2[R2["model"] == "Qwen/Qwen3-8B-AWQ"].copy()
print(f"R2 done; lookup misses={miss} (documented, never filled); IDCG<=0 cases={idcg0}",
      flush=True)
sens = R2P.merge(R2[R2["model"] == "Qwen/Qwen3-14B-AWQ"],
                 on=["system", "query_id"], suffixes=("_8b", "_14b"))
from scipy.stats import spearmanr  # noqa: E402
rho14 = float(spearmanr(sens["udcg3_8b"], sens["udcg3_14b"])[0])
print(f"R2 8B-vs-14B annotator rho={rho14:.3f}", flush=True)

# ---------------- R3 / R5 frozen frames ----------------
sem = pd.read_parquet(ROOT / "results/v3main/semantic/semantic_all.parquet")
sem = sem[(sem["split"] == "test") & (sem["k"] == 3) &
          (sem["system"].isin(SYS)) & (sem["query_id"].isin(test_ids))].copy()
uq = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_query.parquet")
uq = uq[(uq["system"].isin(SYS)) & (uq["k"] == 3) &
        (uq["query_id"].isin(test_ids))].copy()

# ---------------- rankings + agreement ----------------
from scipy.stats import kendalltau  # noqa: E402
r1m = R1.groupby("system")["ndcg3"].mean().to_dict()
r2m = R2P.groupby("system")["udcg3"].mean().to_dict()


def rank_of(d):
    return {s: r for r, s in enumerate(sorted(d, key=lambda s: (-d[s], s)), start=1)}


out_rungs = {"R1_ndcg3": {s: round(r1m[s], 4) for s in SYS},
             "R1_rank": rank_of(r1m),
             "R2_udcg3": {s: round(r2m[s], 4) for s in SYS},
             "R2_rank": rank_of(r2m),
             "R2_annotator_rho_8b14b": round(rho14, 3),
             "R2_misses": miss, "R2_idcg0": idcg0}
pairs = list(combinations(SYS, 2))
rng = np.random.default_rng(SEED)


def boot_tau(df1, df2, key1="v1", key2="v2"):
    ts = []
    q = np.array(test_ids)
    a = df1.set_index("query_id")
    b = df2.set_index("query_id")
    for _ in range(B):
        qq = rng.choice(q, len(q), replace=True)
        m1 = a.loc[qq].groupby("system")["v"].mean()
        m2 = b.loc[qq].groupby("system")["v"].mean()
        r1 = [m1[s] for s in SYS]
        r2 = [m2[s] for s in SYS]
        ts.append(float(kendalltau(r1, r2)[0]))
    return ts


def pairdiff_ci(df, col="v"):
    a = df.set_index("query_id")
    res = {}
    q = np.array(test_ids)
    for x, y in pairs:
        # align by query order
        dx = a[a["system"] == x].loc[q, col].to_numpy()
        dy = a[a["system"] == y].loc[q, col].to_numpy()
        dd = dx - dy
        boots = np.array([rng.choice(dd, len(dd), replace=True).mean()
                          for _ in range(B)])
        res[f"{x}_vs_{y}"] = {"mean": round(float(dd.mean()), 4),
                              "CI95": [round(float(np.percentile(boots, 2.5)), 4),
                                       round(float(np.percentile(boots, 97.5)), 4)]}
    return res


R1f = R1.rename(columns={"ndcg3": "v"})
R2f = R2P.rename(columns={"udcg3": "v"})
tau12 = boot_tau(R1f[["query_id", "system", "v"]], R2f[["query_id", "system", "v"]])
d1 = pairdiff_ci(R1f)
d2 = pairdiff_ci(R2f)
ladder = {"rungs": out_rungs,
          "R1R2": {"tau_CI95": [round(float(np.percentile(tau12, 2.5)), 3),
                                round(float(np.percentile(tau12, 97.5)), 3)],
                   "diffs_R1": d1, "diffs_R2": d2}}
strong12, weak12 = [], []
for x, y in pairs:
    k = f"{x}_vs_{y}"
    s1 = np.sign(d1[k]["mean"]) if not (d1[k]["CI95"][0] <= 0 <= d1[k]["CI95"][1]) else 0
    s2 = np.sign(d2[k]["mean"]) if not (d2[k]["CI95"][0] <= 0 <= d2[k]["CI95"][1]) else 0
    if s1 != 0 and s2 != 0 and s1 != s2:
        strong12.append(k)
    elif np.sign(d1[k]["mean"]) != np.sign(d2[k]["mean"]) and d1[k]["mean"] != 0:
        weak12.append(k)
ladder["R1R2"]["strong_reversals"] = strong12
ladder["R1R2"]["weak_reversals"] = weak12

# per-stratum R2->R3, R3->R5, R1->R5
for (cons, model) in STRATA:
    tag = f"{cons}/{model.split('/')[-1]}"
    s3 = sem[(sem["consumer"] == cons) & (sem["model"] == model)]
    s5 = uq[(uq["consumer"] == cons) & (uq["model"] == model)]
    brier = s3.groupby("system")["brier"].mean().to_dict()
    dj = s5.groupby("system")["delta_vs_noinfo"].mean().to_dict()
    rb, rd = rank_of({s: -brier[s] for s in SYS}), rank_of(dj)  # lower Brier = better
    r2r = rank_of(r2m)
    entry = {"R2_rank": r2r, "R3_brier": {s: round(brier[s], 4) for s in SYS},
             "R3_rank": rb, "R5_dJ": {s: round(dj[s], 2) for s in SYS},
             "R5_rank": rd}
    # taus (point + CI) for R2R3, R3R5, R1R5 within stratum
    S3f = s3[["query_id", "system", "brier"]].rename(columns={"brier": "v"})
    S3f["v"] = -S3f["v"]
    S5f = s5[["query_id", "system", "delta_vs_noinfo"]].rename(
        columns={"delta_vs_noinfo": "v"})
    t23 = boot_tau(R2f[["query_id", "system", "v"]], S3f)
    t35 = boot_tau(S3f, S5f)
    t15 = boot_tau(R1f[["query_id", "system", "v"]], S5f)
    entry["tau_R2R3_CI95"] = [round(float(np.percentile(t23, 2.5)), 3),
                              round(float(np.percentile(t23, 97.5)), 3)]
    entry["tau_R3R5_CI95"] = [round(float(np.percentile(t35, 2.5)), 3),
                              round(float(np.percentile(t35, 97.5)), 3)]
    entry["tau_R1R5_CI95"] = [round(float(np.percentile(t15, 2.5)), 3),
                              round(float(np.percentile(t15, 97.5)), 3)]
    entry["diffs_R3"] = pairdiff_ci(S3f)
    entry["diffs_R5"] = pairdiff_ci(S5f)
    strong, weak = [], []
    d3, d5 = entry["diffs_R3"], entry["diffs_R5"]
    for x, y in pairs:
        k = f"{x}_vs_{y}"
        s3s = np.sign(d3[k]["mean"]) if not (d3[k]["CI95"][0] <= 0 <= d3[k]["CI95"][1]) else 0
        s5s = np.sign(d5[k]["mean"]) if not (d5[k]["CI95"][0] <= 0 <= d5[k]["CI95"][1]) else 0
        if s3s != 0 and s5s != 0 and s3s != s5s:
            strong.append(k)
        elif np.sign(d3[k]["mean"]) != np.sign(d5[k]["mean"]) and d3[k]["mean"] != 0:
            weak.append(k)
    entry["R3R5_strong_reversals"] = strong
    entry["R3R5_weak_reversals"] = weak
    ladder[tag] = entry

# R4 reuse (rerank positional VoI, frozen verdicts)
s4 = json.load(open(TH / "g2full_stage4.json"))
ladder["R4_reused"] = {
    "C1_cells": {k: {"mean": v["mean"], "CI95": v["CI95"]} for k, v in s4["claim_a"]["cells"].items()},
    "note": "rerank-only positional VoI; per-system VoI beyond rerank does not exist (spec limitation)"}

json.dump(ladder, open(OUT / "ladder_tables.json", "w"), indent=1)
print("ladder_tables written")
sv = {"R1R2_strong": strong12, "R1R2_weak": weak12}
for tag in [f"{c}/{m.split('/')[-1]}" for c, m in STRATA]:
    sv[tag + "_R3R5_strong"] = ladder[tag]["R3R5_strong_reversals"]
    sv[tag + "_R3R5_weak"] = ladder[tag]["R3R5_weak_reversals"]
json.dump(sv, open(OUT / "ladder_verdict.json", "w"), indent=1)
print(json.dumps(sv, indent=1))
