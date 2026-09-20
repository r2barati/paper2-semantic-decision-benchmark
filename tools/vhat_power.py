"""VHAT power/sensitivity calc (HISTORICAL labels only; decides bank N).

Population: 93 training clusters (dev 30 + test 63) with per-position pooled
VoI over all 5 strata (C0/C3 from loo_episodes, C1 dev/test from
c1_loo(_test)_episodes, all vs frozen NoInfo rung). No fresh-bank paths may
exist; asserted.

Simulation (DESIGN-VHAT §5): trials draw N clusters w/replacement, inject
true Delta, inner cluster-bootstrap CI (B=1000); power = exclusion rate over
2000 trials. Grid Delta {10,15,25,40} x N {60,90}. Rule: N=60 iff
power(60,25)>=0.8 else N=90 (recompute; cap 120).
Writes extension_g2a/vhat_power.json. Asserts it never reads fresh data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_g2a"

assert not (OUT / "vhat_bank").exists(), "fresh bank already exists -- power calc void"
assert not list(OUT.glob("*fresh*")), "fresh-bank artifacts present -- power calc void"

POS = ["drop0", "drop1", "drop2"]
sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids, test_ids = set(sp["dev"]), set(sp["test"])

froz = pd.read_parquet(ROOT / "results/v3main/sim/episodes.parquet")
ni = froz[(froz["system"] == "noinfo") & (froz["k"] == 3)].set_index(
    ["query_id", "seed"])["profit"]


def per_query_voi(ep_path: str) -> pd.DataFrame:
    ep = pd.read_parquet(OUT / ep_path)
    d = ep.merge(ni.rename("profit_ni"), on=["query_id", "seed"])
    d["dj"] = d["profit"] - d["profit_ni"]
    g = d.groupby(["consumer", "model", "query_id", "ablation"])["dj"].mean().reset_index()
    f = g[g["ablation"] == "full"].set_index(["consumer", "model", "query_id"])["dj"]
    rows = []
    for _, r in g[g["ablation"] != "full"].iterrows():
        key = (r["consumer"], r["model"], r["query_id"])
        rows.append({"consumer": r["consumer"], "query_id": r["query_id"],
                     "position": r["ablation"],
                     "voi": float(f.loc[key] - r["dj"])})
    return pd.DataFrame(rows)


voi = pd.concat([per_query_voi("loo_episodes.parquet"),
                 per_query_voi("c1_loo_episodes.parquet"),
                 per_query_voi("c1_loo_test_episodes.parquet")],
                ignore_index=True)
assert len(voi) == 3000, len(voi)

# cluster key = sha(query text + full doc IDs), dev and test inputs
_qs, _sets = {}, {}
for fn, key in (("kaggle/inputs_loo/queries_dev40.jsonl", "loo_sets.json"),
                ("kaggle/inputs_loo_test/queries_loo.jsonl", "loo_sets_test.json")):
    for line in open(ROOT / fn):
        _q = json.loads(line)
        _qs[_q["_id"]] = _q["text"]
    _s = json.load(open(ROOT / "kaggle" / ("inputs_loo" if "dev40" in fn else "inputs_loo_test") / key))
    _sets.update(_s["sets"])
clus = {q: hashlib.sha256((_qs[q] + "|" + ",".join(sorted(_sets[q]["full"]))).encode()
                          ).hexdigest()[:12] for q in set(voi["query_id"])}
n_clu = len(set(clus.values()))
print(f"training cluster population: {n_clu} (dev+test union; cross-split dups collapse 30+63)")
voi["cluster"] = voi["query_id"].map(clus)

# per (cluster, position) pooled VoI over strata
cp = voi.groupby(["cluster", "position"])["voi"].mean().reset_index()
clust = sorted(cp["cluster"].unique())

rng = np.random.default_rng(77000)
TRIALS, INNER = 2000, 1000
res = {}
for N in (60, 90):
    res[str(N)] = {}
    pool = {p: cp[cp["position"] == p].set_index("cluster")["voi"] for p in POS}
    for Delta in (10, 15, 25, 40):
        hits = 0
        for _ in range(TRIALS):
            draw = rng.choice(clust, N, replace=True)
            # pooled over positions (confirmation-style pooled estimand) + shift
            vals = np.array([pool[p].loc[list(draw)].mean() + Delta for p in POS])
            m = vals.mean()
            ib = np.array([np.mean(rng.choice(vals, len(vals), replace=True))
                           for _ in range(INNER)])
            lo, hi = np.percentile(ib, 2.5), np.percentile(ib, 97.5)
            if lo > 0 or hi < 0:
                hits += 1
        res[str(N)][str(Delta)] = round(hits / TRIALS, 3)
        print(f"N={N} Delta={Delta}: power={hits/TRIALS:.3f}", flush=True)

decision = 60 if res["60"]["25"] >= 0.8 else 90
out = {"power": res, "bank_N": decision, "rule": "N=60 iff power(60,25)>=0.8 else 90",
       "n_clusters_population": len(clust),
       "n_clusters_union": n_clu, "trials": TRIALS, "seed": 77000}
json.dump(out, open(OUT / "vhat_power.json", "w"), indent=1)
print("bank_N =", decision)
