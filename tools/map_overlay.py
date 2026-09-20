"""§6 overlays + H1/H2/H3 verdict (CPU-only analysis; sweep tables frozen).

Loads theory/map_orders.parquet (state x node -> order). Classifies nodes
per state (DEAD/SENSITIVE/no-order-subset/BOUNDARY, tol 1e-6). Maps frozen
rerank-k=3 posteriors + test LOO pairs to nearest lattice nodes (assert
residual <= 0.03). Computes dead_frac / cross_rate / pos_frac with query
bootstraps (B=2000, seeds 81000/81001/81002). PRIMARY test-160, dev-40
consistency. Writes theory/map_verdict.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "theory"

STEP, N, TOL = 0.025, 40, 1e-6
NODES = [(i, j, N - i - j) for i in range(N + 1) for j in range(N - i + 1)]
NIDX = {n: k for k, n in enumerate(NODES)}
PRIOR_NI = NIDX[(14, 14, 12)]
B, S1, S2, S3 = 2000, 81000, 81001, 81002

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids, test_ids = set(sp["dev"]), set(sp["test"])

mo = pd.read_parquet(OUT / "map_orders.parquet")
assert len(mo) == 120 * 861, len(mo)
piv = mo.pivot_table(index=["true_regime", "seed", "t"], columns="node",
                     values="order")
assert piv.shape == (120, 861)
STATES = list(piv.index)

# lattice adjacency for boundary flags
NB = {}
for k, (i, j, kk) in enumerate(NODES):
    nb = []
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                       (0, 0, 1), (0, 0, -1)):
        q = (i + di, j + dj, kk + dk)
        if q in NIDX and sum(q) == N:
            nb.append(NIDX[q])
    NB[k] = nb

PR = piv[PRIOR_NI]  # prior-order per state (Series over STATES)


def nearest_node(p: np.ndarray):
    from itertools import product
    f = p * N
    best, bk = np.inf, None
    for bits in product((0, 1), repeat=3):
        g = np.floor(f).astype(int) + np.array(bits)
        if g.sum() != N or (g < 0).any():
            continue
        r = float(np.abs(p - g / N).sum())
        if r < best:
            best, bk = r, (int(g[0]), int(g[1]), int(g[2]))
    assert bk in NIDX, (p, bk)
    assert best <= 0.04, (p, best)  # lattice spacing bound (step 0.025)
    return NIDX[bk]


bel = pd.concat([pd.read_parquet(
    ROOT / f"results/v3main/beliefs_{m}.parquet")
    for m in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ")], ignore_index=True)
post = bel[(bel["system"] == "rerank") & (bel["k"] == 3)].copy()
_c0 = post[post["consumer"] == "C0"].sort_values(
    ["system", "query_id", "k"]).drop_duplicates(["system", "query_id", "k"])
post = pd.concat([_c0, post[post["consumer"] != "C0"]], ignore_index=True)

loo_c1 = pd.read_parquet(OUT.parent / "extension_g2a" / "c1_loo_test_beliefs.parquet")
loo_c3 = pd.read_parquet(OUT.parent / "extension_g2a" / "loo_beliefs.parquet")
loo_c3 = loo_c3[(loo_c3["system"] == "rerank") &
                (loo_c3["query_id"].isin(test_ids))].copy()

CONS = {"C0": ("C0", "none"), "C1-8B": ("C1", "Qwen/Qwen3-8B-AWQ"),
        "C1-14B": ("C1", "Qwen/Qwen3-14B-AWQ"),
        "C3-8B": ("C3", "Qwen/Qwen3-8B-AWQ"),
        "C3-14B": ("C3", "Qwen/Qwen3-14B-AWQ")}


def node_of(cons: str, model: str, qid: str) -> int:
    r = post[(post["consumer"] == cons) & (post["model"] == model) &
             (post["query_id"] == qid)]
    assert len(r) == 1, (cons, model, qid)
    r = r.iloc[0]
    return nearest_node(np.array([r["p_normal"], r["p_supplier_delay"],
                                  r["p_demand_surge"]]))


def loo_nodes(cons: str, model: str, qid: str):
    src = loo_c1 if cons == "C1" else loo_c3
    out = {}
    f = node_of(cons, model, qid)
    for pos in ("drop0", "drop1", "drop2"):
        r = src[(src["consumer"] == cons) & (src["model"] == model) &
                (src["query_id"] == qid) & (src["ablation"] == pos)]
        assert len(r) == 1, (cons, model, qid, pos)
        r = r.iloc[0]
        out[pos] = (f, nearest_node(np.array([r["p_normal"], r["p_supplier_delay"],
                                              r["p_demand_surge"]])))
    return out


dead, cross = {}, {}
PNODE = {}  # (cons, model, qid) -> lattice node, precomputed once
for label, (cons, model) in CONS.items():
    for q in sorted(test_ids) + sorted(dev_ids):
        PNODE[(cons, model, q)] = node_of(cons, model, q)
LNODE = {}  # (cons, model, qid, pos) -> (full_node, loo_node)
for label, (cons, model) in CONS.items():
    if cons in ("C1", "C3"):
        for q in sorted(test_ids):
            LNODE[(cons, model, q)] = loo_nodes(cons, model, q)
for label, (cons, model) in CONS.items():
    qids = sorted(test_ids)
    dn = [PNODE[(cons, model, q)] for q in qids]
    M = np.array([[abs(piv.loc[s, n] - PR.loc[s]) <= TOL for s in STATES]
                  for n in dn])
    dead[label] = M
    if cons in ("C1", "C3"):
        C = np.array([[int(any(abs(piv.loc[s, LNODE[(cons, model, q)][p][1]] -
                                       piv.loc[s, LNODE[(cons, model, q)][p][0]]) > TOL
                               for p in ("drop0", "drop1", "drop2")))
                       for s in STATES] for q in qids])
        Cp = {p: np.array([[int(abs(piv.loc[s, LNODE[(cons, model, q)][p][1]] -
                                       piv.loc[s, LNODE[(cons, model, q)][p][0]]) > TOL)
                            for s in STATES] for q in qids])
              for p in ("drop0", "drop1", "drop2")}
        cross[label] = (C, Cp)

rng = np.random.default_rng(S1)
qids = sorted(test_ids)


def boot_diff(A: np.ndarray, Bm: np.ndarray):
    d = []
    for _ in range(B):
        ix = rng.choice(len(qids), len(qids), replace=True)
        d.append(float(A[ix].mean() - Bm[ix].mean()))
    return np.array(d)


c3 = np.concatenate([dead["C3-8B"].ravel(), dead["C3-14B"].ravel()])
c1 = np.concatenate([dead["C1-8B"].ravel(), dead["C1-14B"].ravel()])
# paired by query: mean over states+models per query
c3q = np.array([np.concatenate([dead["C3-8B"][i], dead["C3-14B"][i]]).mean()
                for i in range(len(qids))])
c1q = np.array([np.concatenate([dead["C1-8B"][i], dead["C1-14B"][i]]).mean()
                for i in range(len(qids))])
d1 = boot_diff(c3q, c1q)
h1 = {"dead_C3": round(float(c3.mean()), 4), "dead_C1": round(float(c1.mean()), 4),
      "diff_CI95": [round(float(np.percentile(d1, 2.5)), 4),
                    round(float(np.percentile(d1, 97.5)), 4)]}
h1["HOLDS"] = bool(h1["diff_CI95"][0] > 0)

rng2 = np.random.default_rng(S2)
x1 = np.array([np.concatenate([cross["C1-8B"][0][i], cross["C1-14B"][0][i]]).mean()
               for i in range(len(qids))])
x3 = np.array([np.concatenate([cross["C3-8B"][0][i], cross["C3-14B"][0][i]]).mean()
               for i in range(len(qids))])
d2 = []
for _ in range(B):
    ix = rng2.choice(len(qids), len(qids), replace=True)
    d2.append(float(x1[ix].mean() - x3[ix].mean()))
d2 = np.array(d2)
h2 = {"cross_C1": round(float(np.concatenate(
    [cross["C1-8B"][0].ravel(), cross["C1-14B"][0].ravel()]).mean()), 4),
      "cross_C3": round(float(np.concatenate(
    [cross["C3-8B"][0].ravel(), cross["C3-14B"][0].ravel()]).mean()), 4),
      "diff_CI95": [round(float(np.percentile(d2, 2.5)), 4),
                    round(float(np.percentile(d2, 97.5)), 4)]}
h2["HOLDS"] = bool(h2["diff_CI95"][0] > 0)

rng3 = np.random.default_rng(S3)
c0n = [PNODE[("C0", "none", q)] for q in qids]
pm2 = np.array([[int(piv.loc[s, n] - PR.loc[s] > TOL)
                 for s in STATES] for n in c0n])
pf = float(pm2.mean())
b3 = []
for _ in range(B):
    ix = rng3.choice(len(qids), len(qids), replace=True)
    b3.append(float(pm2[ix].mean()))
b3 = np.array(b3)
h3 = {"pos_frac_C0": round(pf, 4),
      "CI95": [round(float(np.percentile(b3, 2.5)), 4),
               round(float(np.percentile(b3, 97.5)), 4)]}
h3["HOLDS"] = bool(h3["CI95"][0] > 0.5)

# dev-40 consistency (same stats, no gates)
dev = sorted(dev_ids)
dev_stats = {}
for label, (cons, model) in CONS.items():
    dn = [PNODE[(cons, model, q)] for q in dev]
    M = np.array([[abs(piv.loc[s, n] - PR.loc[s]) <= TOL for s in STATES]
                  for n in dn])
    dev_stats[label] = round(float(M.mean()), 4)

verdict = {"H1_dead": h1, "H2_cross": h2, "H3_C0pos": h3,
           "dev_dead_frac": dev_stats,
           "design": {"tol": TOL, "grid": 861, "states": 120, "B": B,
                      "unit": "query", "primary": "test-160"}}
json.dump(verdict, open(OUT / "map_verdict.json", "w"), indent=1)
print(json.dumps(verdict, indent=1))
