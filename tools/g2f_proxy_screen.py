"""G2-FULL Stage 2: dev-only deployable-proxy screening + freeze (NO test access).

Candidates (observables only): entropy, margin, maxProb, priorL1,
agree8B14B_L1, c1c3_L1, fd_sensitivity (controller finite-difference at
initial state, CPU-only). rankgap checked then dropped (zero variance:
rerank top-3 always ranks 1,2,3). Label inputs (nDCG/Brier/accuracy/
evidence_hit/true_regime) FORBIDDEN — asserted absent from feature frame.

Screen: cluster-bootstrapped Spearman(feature, per-query full-set dJ) on
dev-40 (clusters = 30 full-set groups, B=2000, seed 74000). Winner = best
|mean rho|. Threshold: deciles of dev feature distribution x orientations
(transmit iff f>=t / f<t); pick max dev pooled gated dJ (mean of 2 C1
stratum means); ties -> threshold closest to median, then '>='.
Freeze {feature, orientation, threshold} to g2full_proxy.json.

Asserts every query_id in dev_ids; refuses test IDs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_g2a"

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids = set(sp["dev"])
assert len(dev_ids) == 40

MODELS = ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"]
PRIOR = np.array([0.35, 0.35, 0.30])
B, SEED = 2000, 74000

bel8 = pd.read_parquet(ROOT / "results/v3main/beliefs_Qwen_Qwen3-8B-AWQ.parquet")
bel14 = pd.read_parquet(ROOT / "results/v3main/beliefs_Qwen_Qwen3-14B-AWQ.parquet")
bel = pd.concat([bel8, bel14], ignore_index=True)
c1 = bel[(bel["system"] == "rerank") & (bel["consumer"] == "C1") &
         (bel["k"] == 3) & (bel["query_id"].isin(dev_ids))].copy()
assert set(c1["query_id"]) == dev_ids and len(c1) == 80
c3 = bel[(bel["system"] == "rerank") & (bel["consumer"] == "C3") &
         (bel["k"] == 3) & (bel["query_id"].isin(dev_ids))].copy()

froz = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_query_devinclusive.parquet")
fq = froz[(froz["system"] == "rerank") & (froz["consumer"] == "C1") &
          (froz["k"] == 3) & (froz["query_id"].isin(dev_ids))]
dj = fq.set_index(["model", "query_id"])["delta_vs_noinfo"]

# duplicate-context clusters (dev): sha(query text + full doc IDs)
qs = {}
for line in open(ROOT / "kaggle" / "inputs_loo" / "queries_dev40.jsonl"):
    q = json.loads(line)
    qs[q["_id"]] = q["text"]
import hashlib
clus = {}
for qid in sorted(dev_ids):
    dids = sorted(c1[(c1["query_id"] == qid)]["doc_ids"].iloc[0])
    clus[qid] = hashlib.sha256((qs[qid] + "|" + ",".join(dids)).encode()).hexdigest()[:12]
cids = sorted(set(clus.values()))
assert len(cids) == 30, f"expected 30 dev clusters, got {len(cids)}"
cl_members = {c: [q for q in sorted(dev_ids) if clus[q] == c] for c in cids}


def _fd_sensitivity(p: np.ndarray) -> float:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.env import INITIAL_INVENTORY, CausalOptimizer, InventoryState
    from src.events import P5_DEMAND_MEAN, P5_HORIZON
    keys = ("normal", "supplier_delay", "demand_surge")

    def decide(prob):
        o = CausalOptimizer(seed=0, regime_probabilities=dict(zip(keys, prob)),
                            horizon=P5_HORIZON, initial_inventory=INITIAL_INVENTORY,
                            demand_mean=P5_DEMAND_MEAN)
        return o.decide(InventoryState(time=0, on_hand=float(INITIAL_INVENTORY)))

    base = decide(p)
    best = 0.0
    for i in range(3):
        for sgn in (1.0, -1.0):
            q = p.copy()
            q[i] = min(0.99, max(0.01, q[i] + sgn * 0.05))
            q = q / q.sum()
            best = max(best, abs(decide(q) - base))
    return float(best)


rows = []
for model in MODELS:
    m = c1[c1["model"] == model].set_index("query_id")
    m14 = bel[(bel["model"] == "Qwen/Qwen3-14B-AWQ")].set_index(
        ["query_id", "system", "consumer", "k"]) if model.endswith("8B-AWQ") else None
    o = bel[(bel["model"] == ("Qwen/Qwen3-14B-AWQ" if model.endswith("8B-AWQ")
                              else "Qwen/Qwen3-8B-AWQ"))].set_index(
        ["query_id", "system", "consumer", "k"])
    m3 = c3[c3["model"] == model].set_index("query_id")
    for qid in sorted(dev_ids):
        r = m.loc[qid]
        p = np.array([r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"]])
        ps = np.sort(p)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        po = o.loc[(qid, "rerank", "C1", 3)]
        q2 = np.array([po["p_normal"], po["p_supplier_delay"], po["p_demand_surge"]])
        r3 = m3.loc[qid]
        q3 = np.array([r3["p_normal"], r3["p_supplier_delay"], r3["p_demand_surge"]])
        rows.append({"model": model, "query_id": qid, "cluster": clus[qid],
                     "entropy": ent, "margin": float(ps[2] - ps[1]),
                     "maxProb": float(ps[2]),
                     "priorL1": float(np.abs(p - PRIOR).sum()),
                     "agreeXmodel_L1": float(np.abs(p - q2).sum()),
                     "c1c3_L1": float(np.abs(p - q3).sum()),
                     "fd_sens": _fd_sensitivity(p),
                     "dJ": float(dj.loc[(model, qid)])})
feat = pd.DataFrame(rows)
assert set(feat["query_id"]) <= dev_ids
FEATS = ["entropy", "margin", "maxProb", "priorL1", "agreeXmodel_L1",
         "c1c3_L1", "fd_sens"]

rng = np.random.default_rng(SEED)
screen = {}
for f in FEATS:
    rhos = []
    for _ in range(B):
        draw = rng.choice(cids, len(cids), replace=True)
        sub = feat[feat["cluster"].isin(draw)]
        r = spearmanr(sub[f].to_numpy(), sub["dJ"].to_numpy())[0]
        rhos.append(float(r))
    screen[f] = {"mean_rho": round(float(np.nanmean(rhos)), 3),
                 "CI95": [round(float(np.nanpercentile(rhos, 2.5)), 3),
                          round(float(np.nanpercentile(rhos, 97.5)), 3)]}
    print(f, screen[f])
winner = max(FEATS, key=lambda f: abs(screen[f]["mean_rho"]))
print("winner:", winner)

# threshold: deciles pooled over both models, both orientations
vals = np.sort(feat[winner].to_numpy())
grid = sorted(set(round(float(np.percentile(vals, q)), 6) for q in range(10, 100, 10)))
med = float(np.median(vals))
best = None
for orient in (">=", "<"):
    for t in grid:
        mask = feat[winner] >= t if orient == ">=" else feat[winner] < t
        g = feat[mask].groupby("model")["dJ"].mean()
        pooled = float(g.mean()) if len(g) else 0.0
        key = (-pooled, abs(t - med), 0 if orient == ">=" else 1)
        if best is None or key < best[0]:
            best = (key, {"orientation": orient, "threshold": t,
                          "dev_gated": round(pooled, 2)})
rule = {"feature": winner, **best[1]}
print("rule:", rule)
json.dump({"screen": screen, "grid": grid, "rule": rule,
           "n_dev_clusters": len(cids), "B": B, "seed": SEED, "frozen": True},
          open(OUT / "g2full_proxy.json", "w"), indent=1)
print("wrote", OUT / "g2full_proxy.json")
