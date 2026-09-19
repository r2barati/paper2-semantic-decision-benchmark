"""A1: frozen selectors on dev-40 -> freeze -> test selection regret.

Selectors (DESIGN.md): S-nDCG (argmax dev nDCG@10), S-Brier (min dev Brier),
S-J (argmax dev plain-mean delta_vs_noinfo), per stratum over
{bm25,dense,hybrid,rerank}. Test regret from frozen headline J (balanced).
H1b calibration: 500 dev half-split null. Writes extension_selection/.
Analysis-only. No new episodes. No frozen edits.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_selection"
OUT.mkdir(parents=True, exist_ok=True)
RETR = ["bm25", "dense", "hybrid", "rerank"]
STRATA = [("C0", "none"), ("C1", "Qwen/Qwen3-8B-AWQ"),
          ("C1", "Qwen/Qwen3-14B-AWQ"), ("C3", "Qwen/Qwen3-8B-AWQ"),
          ("C3", "Qwen/Qwen3-14B-AWQ")]
K = 3
rng = np.random.default_rng(70001)

dev_ids = set(json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))["dev"])

# --- dev metrics
ir = pd.read_parquet(ROOT / "results/v3main/semantic/ir_dev.parquet")
ndcg = ir[ir.system.isin(RETR)].groupby("system")["ndcg10"].mean()
sem = pd.read_parquet(ROOT / "results/v3main/semantic/semantic_all.parquet")
semd = sem[(sem.split == "dev") & (sem.system.isin(RETR)) & (sem.k == K)]
brier = semd.groupby(["system", "consumer", "model"])["brier"].mean()
uq = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_query_devinclusive.parquet")
uqd = uq[(uq.query_id.isin(dev_ids)) & (uq.system.isin(RETR)) & (uq.k == K)]
devJ = uqd.groupby(["system", "consumer", "model"])["delta_vs_noinfo"].mean()
print("dev nDCG:", ndcg.round(4).to_dict())

# --- test J per stratum (frozen headline, balanced)
ub = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_arm_balanced.parquet")
ub = ub[(ub.system.isin(RETR)) & (ub.k == K)]

sel, reg = {}, []
for cons, model in STRATA:
    s_ndcg = str(ndcg.idxmax())
    try:
        s_brier = str(brier.loc[(slice(None), cons, model)].idxmin())
    except KeyError:
        s_brier = None
    dj = devJ.loc[(slice(None), cons, model)]
    s_j = str(dj.idxmax()) if len(dj) else None
    tj = ub[(ub.consumer == cons) & (ub.model == model)].set_index("system")["J"]
    best = str(tj.idxmax())
    row = {"stratum": f"{cons}/{model.split('/')[-1]}",
           "S-nDCG": s_ndcg, "S-Brier": s_brier, "S-J": s_j,
           "test-best": best}
    for name, s in (("nDCG", s_ndcg), ("Brier", s_brier), ("J", s_j)):
        row[f"R_{name}"] = (round(float(tj.max() - tj[s]), 2) if s else None)
    sel[f"{cons}/{model}"] = row
    reg.append(row)
    print(row)

regd = pd.DataFrame(reg)
regd.to_csv(OUT / "A1_selections.csv", index=False)

# --- H1b calibration: 500 dev half-split nulls for pooled R_nDCG and R_Brier
devq = sorted(dev_ids)
null = {"nDCG": [], "Brier": []}
for b in range(500):
    half = set(rng.choice(devq, len(devq) // 2, replace=False))
    nd = ir[(ir.system.isin(RETR)) & (ir.query_id.isin(half))].groupby("system")["ndcg10"].mean()
    sn = str(nd.idxmax())
    bd = semd[semd.query_id.isin(half)].groupby(["system", "consumer", "model"])["brier"].mean()
    Rs = []
    for cons, model in STRATA:
        tj = ub[(ub.consumer == cons) & (ub.model == model)].set_index("system")["J"]
        Rs.append(float(tj.max() - tj[sn]))
        try:
            sb = str(bd.loc[(slice(None), cons, model)].idxmin())
            null["Brier"]  # ensure key
            Rs_b = float(tj.max() - tj[sb])
        except KeyError:
            Rs_b = float("nan")
        if b == 0:
            pass
    null["nDCG"].append(float(np.mean(Rs)))
# Brier null needs per-split collection; redo compactly below
nullB = []
for b in range(500):
    half = set(rng.choice(devq, len(devq) // 2, replace=False))
    bd = semd[semd.query_id.isin(half)].groupby(["system", "consumer", "model"])["brier"].mean()
    Rs = []
    for cons, model in STRATA:
        tj = ub[(ub.consumer == cons) & (ub.model == model)].set_index("system")["J"]
        try:
            sb = str(bd.loc[(slice(None), cons, model)].idxmin())
            Rs.append(float(tj.max() - tj[sb]))
        except KeyError:
            continue
    nullB.append(float(np.mean(Rs)))
null["Brier"] = nullB

obs_n = float(np.mean([r["R_nDCG"] for r in reg]))
obs_b = float(np.mean([r["R_Brier"] for r in reg if r["R_Brier"] is not None]))
h1b = {
    "obs_pooled_R_nDCG": round(obs_n, 2),
    "null95_nDCG": round(float(np.quantile(null["nDCG"], 0.95)), 2),
    "H1b_nDCG_holds": bool(obs_n > np.quantile(null["nDCG"], 0.95)),
    "obs_pooled_R_Brier": round(obs_b, 2),
    "null95_Brier": round(float(np.quantile(null["Brier"], 0.95)), 2),
    "H1b_Brier_holds": bool(obs_b > np.quantile(null["Brier"], 0.95)),
}
# Holm over the 2 H1b tests via empirical null p-values
p_n = float(np.mean(np.array(null["nDCG"]) >= obs_n))
p_b = float(np.mean(np.array(null["Brier"]) >= obs_b))
ps = sorted([("nDCG", p_n), ("Brier", p_b)], key=lambda x: x[1])
holm = {}
m = 2
for i, (name, p) in enumerate(ps):
    holm[name] = {"p_raw": round(p, 4), "p_holm": round(min(1.0, p * (m - i)), 4)}
h1b["holm"] = holm
json.dump({"selections": sel, "H1b": h1b}, open(OUT / "A1_H1b.json", "w"), indent=1)
print(json.dumps(h1b, indent=1))
