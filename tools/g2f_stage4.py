"""G2-FULL Stage 4: SINGLE test-outcomes evaluation for claims (a)(b)(c).

All test-outcome reads in G2-FULL happen in this script, once. Inputs:
 fresh c1_loo_test_episodes.parquet + frozen episodes.parquet (NoInfo rung),
 frozen loo_episodes.parquet (2A-0 C0/C3 test LOO, reused), frozen test
 beliefs (proxy features), g2full_proxy.json (frozen rule), A1 selections +
 utility_by_arm_balanced (upstream baselines).

(a) per-position pooled C1 VoI, cluster bootstrap B=5000 seed 75000, Holm-6.
(b) within-query consumer-label permutation over {C0,C1-8B,C1-14B,C3}
    per-position VoI, 5000 perms seed 76000; regime table secondary.
(c) PRIMARY: frozen-proxy-gated pooled C1 dJ vs NoInfo CI>0 (cluster
    bootstrap) AND gate beats max{S-nDCG,S-Brier,S-J,always-transmit S-J}
    per-stratum baseline; per-stratum Holm secondaries.
Writes extension_g2a/g2full_stage4.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_g2a"

POS = ["drop0", "drop1", "drop2"]
MODELS = ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"]
B, SEED_B, SEED_P = 5000, 75000, 76000

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
test_ids = set(sp["test"])
assert len(test_ids) == 160

# --- clusters from test inputs (structure only; outcomes joined later) ---
_qs, _sets = {}, json.load(open(ROOT / "kaggle" / "inputs_loo_test" / "loo_sets_test.json"))["sets"]
for line in open(ROOT / "kaggle" / "inputs_loo_test" / "queries_loo.jsonl"):
    _q = json.loads(line)
    _qs[_q["_id"]] = _q["text"]
clus = {q: hashlib.sha256((_qs[q] + "|" + ",".join(sorted(_sets[q]["full"]))).encode()
                          ).hexdigest()[:12] for q in test_ids}
cids = sorted(set(clus.values()))
print(f"test clusters: {len(cids)}")
cl_members = {c: [q for q in sorted(test_ids) if clus[q] == c] for c in cids}

# --- per-query dJ: fresh C1 test replay vs frozen NoInfo test rung ---
ep = pd.read_parquet(OUT / "c1_loo_test_episodes.parquet")
froz = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_query.parquet")
assert set(ep["query_id"]) <= test_ids
ni_ep = pd.read_parquet(ROOT / "results/v3main/sim/episodes.parquet")
ni = ni_ep[(ni_ep["system"] == "noinfo") & (ni_ep["k"] == 3)].set_index(
    ["query_id", "seed"])["profit"]
dd = ep.merge(ni.rename("profit_ni"), on=["query_id", "seed"])
dd["dj"] = dd["profit"] - dd["profit_ni"]
g = dd.groupby(["model", "query_id", "ablation"])["dj"].mean().reset_index()
full_dj = g[g["ablation"] == "full"].set_index(["model", "query_id"])["dj"]
c1_voi_rows = []
for _, r in g[g["ablation"] != "full"].iterrows():
    c1_voi_rows.append({"model": r["model"], "query_id": r["query_id"],
                        "position": r["ablation"],
                        "voi": float(full_dj.loc[(r["model"], r["query_id"])] - r["dj"])})
c1voi = pd.DataFrame(c1_voi_rows)
assert len(c1voi) == 2 * 160 * 3

# --- 2A-0 frozen C0/C3 test VoI recomputed identically (reuse, same method) ---
ep0 = pd.read_parquet(OUT / "loo_episodes.parquet")
d0 = ep0.merge(ni.rename("profit_ni"), on=["query_id", "seed"])
d0["dj"] = d0["profit"] - d0["profit_ni"]
g0 = d0.groupby(["consumer", "model", "query_id", "ablation"])["dj"].mean().reset_index()
f0 = g0[g0["ablation"] == "full"].set_index(["consumer", "model", "query_id"])["dj"]
c03_rows = []
for _, r in g0[g0["ablation"] != "full"].iterrows():
    key = (r["consumer"], r["model"], r["query_id"])
    c03_rows.append({"consumer": r["consumer"], "model": r["model"],
                     "query_id": r["query_id"], "position": r["ablation"],
                     "voi": float(f0.loc[key] - r["dj"])})
c03 = pd.DataFrame(c03_rows)
c03 = c03[c03["query_id"].isin(test_ids)].copy()  # test only: 2A-0 replay covers dev+test
assert len(c03) == 3 * 160 * 3, f"C0/C3 test rows {len(c03)}"  # C0 + C3x2models

rng = np.random.default_rng(SEED_B)


def cboot_means(df: pd.DataFrame, col: str):
    """Cluster bootstrap of pooled mean; df must have 'query_id' + col."""
    qv = df.set_index("query_id")[col]
    boots = []
    for _ in range(B):
        draw = rng.choice(cids, len(cids), replace=True)
        qs_ = [q for c in draw for q in cl_members[c]]
        boots.append(float(qv.loc[qs_].mean()))
    return boots


# ================= (a) per-position pooled C1 VoI, Holm-6 =================
a_cells, a_p = {}, []
for model in MODELS:
    for p in POS:
        v = c1voi[(c1voi["model"] == model) & (c1voi["position"] == p)]
        qv = v.set_index("query_id")["voi"]
        obs = float(qv.mean())
        boots = np.array(cboot_means(v, "voi"))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        p_raw = float(min((boots <= 0).mean(), (boots >= 0).mean()) * 2)
        key = f"{model.split('/')[-1]}/{p}"
        a_cells[key] = {"mean": round(obs, 2), "CI95": [round(lo, 2), round(hi, 2)],
                        "p_raw": round(p_raw, 4)}
        a_p.append((key, p_raw))
a_p.sort(key=lambda x: x[1])
prev = 0.0
for rank, (key, p) in enumerate(a_p, start=1):
    adj = max(min(1.0, p * (len(a_p) - rank + 1)), prev)
    prev = adj
    a_cells[key]["p_holm"] = round(adj, 4)
    a_cells[key]["excl0"] = bool(a_cells[key]["CI95"][0] > 0 or
                                 a_cells[key]["CI95"][1] < 0)
a_success = any(c["excl0"] for c in a_cells.values())

# ================= (b) consumer-label permutation =================
CONS = [("C0", "none"), ("C1", "Qwen/Qwen3-8B-AWQ"),
        ("C1", "Qwen/Qwen3-14B-AWQ"), ("C3", "Qwen/Qwen3-8B-AWQ")]
c3m = "Qwen/Qwen3-8B-AWQ"  # C3 representative (8B primary, mirrors H2)
allv = pd.concat([
    c03[c03["consumer"] == "C0"].assign(cons="C0/none"),
    c1voi[c1voi["model"] == MODELS[0]].assign(cons="C1/8B"),
    c1voi[c1voi["model"] == MODELS[1]].assign(cons="C1/14B"),
    c03[(c03["consumer"] == "C3") & (c03["model"] == c3m)].assign(cons="C3/8B"),
])
piv = allv.pivot_table(index=["query_id", "position"], columns="cons",
                       values="voi").reset_index()
rp = np.random.default_rng(SEED_P)
b_res, regime = {}, {}
for p in POS:
    mat = piv[piv["position"] == p][["C0/none", "C1/8B", "C1/14B", "C3/8B"]].to_numpy()
    obs = float(mat.mean(axis=0).max() - mat.mean(axis=0).min())
    null = np.empty(B)
    for i in range(B):
        pm = np.array([rp.permutation(row) for row in mat])
        null[i] = pm.mean(axis=0).max() - pm.mean(axis=0).min()
    pval = float((null >= obs).mean())
    b_res[p] = {"obs_maxmin": round(obs, 2), "p_perm": round(pval, 4),
                "holds": bool(pval < 0.05)}
    regime[p] = {c: {"mean": round(float(mat[:, j].mean()), 2),
                     "CI95": [round(float(np.percentile(
                         [mat[rp.choice(len(mat), len(mat), replace=True), j].mean()
                          for _ in range(1000)], 2.5)), 2),
                              round(float(np.percentile(
                         [mat[rp.choice(len(mat), len(mat), replace=True), j].mean()
                          for _ in range(1000)], 97.5)), 2)]}
                 for j, c in enumerate(["C0/none", "C1/8B", "C1/14B", "C3/8B"])}
b_success = any(v["holds"] for v in b_res.values())

# ================= (c) frozen proxy gate vs upstream baselines =================
proxy = json.load(open(OUT / "g2full_proxy.json"))["rule"]
assert proxy["feature"] == "agreeXmodel_L1"
bel8 = pd.read_parquet(ROOT / "results/v3main/beliefs_Qwen_Qwen3-8B-AWQ.parquet")
bel14 = pd.read_parquet(ROOT / "results/v3main/beliefs_Qwen_Qwen3-14B-AWQ.parquet")
b8 = bel8[(bel8["system"] == "rerank") & (bel8["consumer"] == "C1") &
          (bel8["k"] == 3) & (bel8["query_id"].isin(test_ids))].set_index("query_id")
b14 = bel14[(bel14["system"] == "rerank") & (bel14["consumer"] == "C1") &
            (bel14["k"] == 3) & (bel14["query_id"].isin(test_ids))].set_index("query_id")
P = lambda r: np.array([r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"]])
feat = {q: float(np.abs(P(b8.loc[q]) - P(b14.loc[q])).sum()) for q in test_ids}
t = proxy["threshold"]
assert proxy["orientation"] == ">="
tx = {q: (feat[q] >= t) for q in test_ids}
print(f"proxy transmit rate test: {sum(tx.values())}/{len(tx)}")

fq = froz[(froz["system"] == "rerank") & (froz["consumer"] == "C1") &
          (froz["k"] == 3) & (froz["query_id"].isin(test_ids))]
fdj = fq.set_index(["model", "query_id"])["delta_vs_noinfo"]
gated_q = {(m, q): (float(fdj.loc[(m, q)]) if tx[q] else 0.0)
           for m in MODELS for q in test_ids}
# upstream baselines (A1 selections, frozen): S-nDCG/S-Brier/S-J per stratum
A1 = json.load(open(OUT / "A1_H1b.json"))["selections"] if (OUT / "A1_H1b.json").exists() \
    else json.load(open(ROOT / "results/v3main/extension_selection/A1_H1b.json"))["selections"]
base_arms = {}
for stratum, sel in A1.items():
    cons, mfull = stratum.split("/", 1)
    if cons != "C1":
        continue
    model = next(m for m in MODELS if m == mfull)
    base_arms[(cons, model)] = {n: sel[n] for n in ("S-nDCG", "S-Brier", "S-J")}
ub = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_arm_balanced.parquet")
ub = ub[(ub["k"] == 3)]
c_res = {}
diffs = []
for (cons, model) in [("C1", MODELS[0]), ("C1", MODELS[1])]:
    key = f"{cons}/{model.split('/')[-1]}"
    gq = np.array([gated_q[(model, q)] for q in sorted(test_ids)])
    obs = float(gq.mean())
    boots = np.array(cboot_means(
        pd.DataFrame({"query_id": sorted(test_ids), "g": gq}), "g"))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    # baselines: plain-mean test dJ of selected arms + always-transmit S-J
    sel = base_arms[(cons, model)]
    arms = list({sel["S-nDCG"], sel["S-Brier"], sel["S-J"]})
    bmeans = {}
    for a in arms:
        bmeans[a] = float(fq[(fq["model"] == model) &
                             (fq["system"] == a)].set_index("query_id")
                          .loc[sorted(test_ids)]["delta_vs_noinfo"].mean())
    best = max(bmeans.values())
    d = obs - best
    dboot = boots - np.array(cboot_means(
        pd.DataFrame({"query_id": sorted(test_ids),
                      "g": [float(fq[(fq['model'] == model) &
                                       (fq['system'] == max(bmeans, key=bmeans.get))]
                                  .set_index("query_id").loc[q]["delta_vs_noinfo"])
                            for q in sorted(test_ids)]}), "g"))
    dlo, dhi = float(np.percentile(dboot, 2.5)), float(np.percentile(dboot, 97.5))
    c_res[key] = {"gated_dJ": round(obs, 2), "CI95": [round(lo, 2), round(hi, 2)],
                  "baselines": {a: round(v, 2) for a, v in bmeans.items()},
                  "best_baseline": round(best, 2),
                  "diff_vs_best": round(d, 2),
                  "diff_CI95": [round(dlo, 2), round(dhi, 2)]}
    diffs.append((key, d, dlo, dhi))
# pooled: mean of stratum means, paired draws
pg = np.array([[gated_q[(m, q)] for q in sorted(test_ids)] for m in MODELS])
pobs = float(pg.mean())
rng2 = np.random.default_rng(SEED_B + 1)
pboots = []
for _ in range(B):
    draw = rng2.choice(cids, len(cids), replace=True)
    qs_ = [q for c in draw for q in cl_members[c]]
    idx = [sorted(test_ids).index(q) for q in qs_]
    pboots.append(float(pg[:, idx].mean()))
pboots = np.array(pboots)
plo, phi = float(np.percentile(pboots, 2.5)), float(np.percentile(pboots, 97.5))
c_primary = {"pooled_gated_dJ": round(pobs, 2), "CI95": [round(plo, 2), round(phi, 2)],
             "holds_gt0": bool(plo > 0)}
c_primary["holds_vs_best"] = bool(all(c_res[k]["diff_CI95"][0] > 0 for k in c_res))
c_primary["HOLDS"] = bool(c_primary["holds_gt0"] and c_primary["holds_vs_best"])

verdict = {"n_test_clusters": len(cids), "claim_a": {"cells": a_cells,
                                                    "SUCCESS": bool(a_success)},
           "claim_b": {"perm": b_res, "regime_table": regime,
                       "SUCCESS": bool(b_success)},
           "claim_c": {"primary": c_primary, "per_stratum": c_res,
                       "proxy": proxy,
                       "SUCCESS": bool(c_primary["HOLDS"])},
           "design": {"B": B, "seed_boot": SEED_B, "seed_perm": SEED_P,
                      "unit": "full-set cluster"}}
json.dump(verdict, open(OUT / "g2full_stage4.json", "w"), indent=1)
print(json.dumps({"a": a_success, "b": b_success, "c": c_primary}, indent=1))
