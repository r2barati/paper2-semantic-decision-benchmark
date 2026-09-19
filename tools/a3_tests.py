"""A3: H1a paired bootstrap + H2 permutation (locked rules in DESIGN.md).

H1a (PRIMARY): dR = mean_s[R(S-nDCG,s) - R(S-J,s)]; selections FROZEN from
A1 (rerank everywhere for nDCG; S-J per stratum); regret recomputed per
paired test-query bootstrap resample (plain-mean episode profits, B=5000,
seed 70000). Holds iff 95% CI lower > 0.
H2: system x consumer interaction F on test episode profits (4 retrievers x
C0/C1/C3, 8B primary, k=3, query blocks) vs within-query system-label
permutation null (5000 perms, seed 70001). 14B sensitivity (F+p, no gate).
Writes extension_selection/A3_verdict.json. No new episodes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_selection"
RETR = ["bm25", "dense", "hybrid", "rerank"]
STRATA = [("C0", "none"), ("C1", "Qwen/Qwen3-8B-AWQ"),
          ("C1", "Qwen/Qwen3-14B-AWQ"), ("C3", "Qwen/Qwen3-8B-AWQ"),
          ("C3", "Qwen/Qwen3-14B-AWQ")]
K = 3
rng = np.random.default_rng(70000)

A1 = json.load(open(OUT / "A1_H1b.json"))
SEL = A1["selections"]  # stratum -> {S-nDCG, S-J, ...}; keys 'C0/none' etc.
test_ids = set(json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))["test"])
ep = pd.read_parquet(ROOT / "results/v3main/sim/episodes.parquet")
ep = ep[(ep.system.isin(RETR)) & (ep.k == K) & (ep.query_id.isin(test_ids))]
tq = sorted(ep.query_id.unique())
print("test queries in episodes:", len(tq))

# --- H1a
B = 5000
dRs = np.empty(B)
sels = {}
for cons, model in STRATA:
    key = f"{cons}/{model}"
    sels[key] = (SEL[key]["S-nDCG"], SEL[key]["S-J"])
for b in range(B):
    qs = rng.choice(tq, len(tq), replace=True)
    d = ep[ep.query_id.isin(qs)]
    tot = 0.0
    for cons, model in STRATA:
        g = d[(d.consumer == cons) & (d.model == model)]
        m = g.groupby("system")["profit"].mean()
        sn, sj = sels[f"{cons}/{model}"]
        tot += (m.max() - m[sn]) - (m.max() - m[sj])
    dRs[b] = tot / len(STRATA)
lo, hi = np.quantile(dRs, [0.025, 0.975])
H1a = {"dR_point": round(float(np.mean(dRs)), 3),
       "CI": [round(float(lo), 3), round(float(hi), 3)],
       "holds": bool(lo > 0)}
print("H1a:", H1a)

# --- H2: two-way ANOVA interaction F with query blocks, 8B primary
def interaction_F(d):
    # d: DataFrame with system, consumer, query_id, profit
    d = d.copy()
    d["sys"] = pd.Categorical(d["system"]).codes
    d["con"] = pd.Categorical(d["consumer"]).codes
    d["qry"] = pd.Categorical(d["query_id"]).codes
    y = d["profit"].to_numpy()
    n = len(y)
    # full model: query + sys + con + sys*con via dummy LS
    import numpy.linalg as la
    Q = pd.get_dummies(d["qry"]).to_numpy()
    S = pd.get_dummies(d["sys"]).to_numpy()[:, 1:]
    C = pd.get_dummies(d["con"]).to_numpy()[:, 1:]
    SC = np.einsum("ij,ik->ijk", S, C).reshape(n, -1)
    X_full = np.column_stack([np.ones(n), Q[:, 1:], S, C, SC])
    X_red = np.column_stack([np.ones(n), Q[:, 1:], S, C])
    b_full, *_ = la.lstsq(X_full, y, rcond=None)
    b_red, *_ = la.lstsq(X_red, y, rcond=None)
    ss_full = float(((y - X_full @ b_full) ** 2).sum())
    ss_red = float(((y - X_red @ b_red) ** 2).sum())
    df_int = X_full.shape[1] - X_red.shape[1]
    df_res = n - X_full.shape[1]
    return ((ss_red - ss_full) / df_int) / (ss_full / df_res)

res8 = ep[(ep.consumer.isin(["C0", "C1", "C3"])) & (ep.model.isin(["none", "Qwen/Qwen3-8B-AWQ"]))]
# aggregate over seeds -> per-(query, system, consumer) means (faster, same design)
res8m = res8.groupby(["query_id", "system", "consumer"], as_index=False)["profit"].mean()
print("H2 design size:", res8m.shape)
print("H2 cells:", res8m.groupby(["system", "consumer"]).size().to_string())
F_obs = interaction_F(res8m)
rp = np.random.default_rng(70001)
Fp = np.empty(5000)
base = res8m.copy()
for b in range(5000):
    d = base.copy()
    # within-query permutation of system labels
    d["system"] = d.groupby("query_id", group_keys=False).apply(
        lambda g: pd.Series(rp.permutation(g["system"].to_numpy()), index=g.index)).values
    Fp[b] = interaction_F(d)
p_val = float((np.sum(Fp >= F_obs) + 1) / (len(Fp) + 1))
H2 = {"F_obs": round(float(F_obs), 3), "p_perm": round(p_val, 4),
      "holds": bool(p_val < 0.05)}
print("H2 (8B):", H2)

res14 = ep[(ep.consumer.isin(["C0", "C1", "C3"])) & (ep.model.isin(["none", "Qwen/Qwen3-14B-AWQ"]))]
res14m = res14.groupby(["query_id", "system", "consumer"], as_index=False)["profit"].mean()
F14 = interaction_F(res14m)
print("H2 sensitivity (14B): F =", round(float(F14), 3))

json.dump({"H1a": H1a, "H2_8B": H2, "H2_14B_F": round(float(F14), 3)},
          open(OUT / "A3_verdict.json", "w"), indent=1)
