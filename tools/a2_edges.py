"""A2: transfer-chain edge table + interface sign agreement.

Per (query, system, consumer, model) at k=3, test queries:
nDCG@10, evidence_hit, accuracy, brier, ADis/ADisRate (vs frozen NoInfo),
delta_vs_noinfo. Systems: 7 A0 systems. C0 twin replays deduped (keep first).
Sign agreement per interface: retrieval->belief (nDCG rank vs accuracy rank
across systems within query), belief->action (accuracy rank vs ADis rank),
action->utility (ADis rank vs delta rank). Rank transfer (Spearman) likewise.
Writes extension_selection/A2_edges.{parquet,csv}, A2_interfaces.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_selection"
SYSTEMS = ["bm25", "dense", "hybrid", "rerank", "random",
           "oracle-relevant", "oracle-factual"]
K = 3

test_ids = set(json.load(open(ROOT / "data/v3/splits/splits.json"))["test"])
ir = pd.read_parquet(ROOT / "results/v3main/semantic/ir_test.parquet")
ir = ir[(ir.system.isin(SYSTEMS)) & (ir.query_id.isin(test_ids))]
sem = pd.read_parquet(ROOT / "results/v3main/semantic/semantic_all.parquet")
sem = sem[(sem.split == "test") & (sem.system.isin(SYSTEMS)) & (sem.k == K)
          & (sem.query_id.isin(test_ids))]
uq = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_query.parquet")
uq = uq[(uq.system.isin(SYSTEMS)) & (uq.k == K) & (uq.query_id.isin(test_ids))]
act = pd.read_parquet(ROOT / "results/v3main/extension_actions/actions.parquet")
act = act.sort_values(["system", "consumer", "model", "query_id", "seed"]).drop_duplicates(
    ["system", "consumer", "model", "query_id", "seed"], keep="first")
ep = pd.read_parquet(ROOT / "results/v3main/sim/episodes.parquet")
noinfo = ep[(ep["system"] == "noinfo")].set_index(["query_id", "seed"])["profit"]

# per (arm, query) action vectors
ag = act.groupby(["system", "consumer", "model", "query_id"])
orders = ag["orders"].apply(list)
base = {}
for (q, s), g in ep[ep.system == "noinfo"].groupby(["query_id", "seed"]):
    base[(q, s)] = None  # placeholder; need order vectors, not profit
# NoInfo order vectors: replay not done for noinfo; derive from frozen? Not
# available -> use arm-vs-arm ADis? NO: DESIGN says vs frozen NoInfo rung.
# NoInfo = prior-belief CausalOptimizer run; its orders are deterministic
# given (query regime, seed). Reconstruct cheaply: rerun _run_p5_episode with
# no_info belief for test queries x 5 seeds (160*5=800 eps, ~10 min CPU).
print("edge inputs ready; computing NoInfo order vectors (800 eps)...")
import sys
sys.path.insert(0, str(ROOT))
from src.experiment_phase5 import _run_p5_episode
from src.events import Regime
from src.interpreter import no_info_regime_belief
qreg = dict(sem[["query_id", "true_regime"]].drop_duplicates().values)
ni_orders = {}
for qid in sorted(test_ids):
    for seed in [60000 + i for i in range(5)]:
        _, hist = _run_p5_episode(seed=seed, regime=Regime(qreg[qid]),
                                  sensor="noinfo+C0", controller="CausalOptimizer",
                                  regime_interp=no_info_regime_belief())
        ni_orders[(qid, seed)] = [float(s.order_quantity) for s in hist]
print("done NoInfo vectors")

rows = []
SEEDS5 = [60000 + i for i in range(5)]
adf = act.set_index(["system", "consumer", "model", "query_id", "seed"])["orders"]
for (sys_name, cons, model, qid) in adf.index.droplevel("seed").unique():
    try:
        o = np.array([adf.loc[(sys_name, cons, model, qid, s)] for s in SEEDS5],
                     dtype=float)
        nio = np.array([ni_orders[(qid, s)] for s in SEEDS5], dtype=float)
    except KeyError:
        continue
    d = np.abs(o - nio)
    adis = float(d.mean())
    adisrate = float((d > 0.5).mean())
    rows.append({"system": sys_name, "consumer": cons, "model": model,
                 "query_id": qid, "ADis": round(adis, 4),
                 "ADisRate": round(adisrate, 4)})
A = pd.DataFrame(rows)
A.to_parquet(OUT / "A2_actions.parquet", index=False)

edge = (ir.rename(columns={"ndcg10": "nDCG"})[["system", "query_id", "nDCG"]]
        .merge(sem[["system", "consumer", "model", "query_id", "evidence_hit",
                    "accuracy", "brier"]], on=["system", "query_id"], how="inner")
        .merge(uq[["system", "consumer", "model", "query_id", "delta_vs_noinfo"]]
               .rename(columns={"delta_vs_noinfo": "dJ"}),
               on=["system", "consumer", "model", "query_id"], how="inner")
        .merge(A, on=["system", "consumer", "model", "query_id"], how="inner"))
edge.to_parquet(OUT / "A2_edges.parquet", index=False)
print("edge rows:", len(edge))

# interface transfer: within-query pairwise-ordering agreement + Spearman
# ret->bel: nDCG vs accuracy; bel->act: accuracy vs ADis (better beliefs move
# actions further from NoInfo); act->util: ADis vs dJ.
from itertools import combinations

rec = {}
for (cons, model), g in edge.groupby(["consumer", "model"]):
    acc = {"ret-bel": [[], []], "bel-act": [[], []], "act-util": [[], []]}
    for qid, q in g.groupby("query_id"):
        if len(q) < 3:
            continue
        cols = {"ret-bel": ("nDCG", "accuracy"), "bel-act": ("accuracy", "ADis"),
                "act-util": ("ADis", "dJ")}
        for name, (a, b) in cols.items():
            x, y = q[a].to_numpy(), q[b].to_numpy()
            sp = spearmanr(x, y)[0]
            ags = []
            for i, j in combinations(range(len(q)), 2):
                sx = np.sign(x[i] - x[j])
                sy = np.sign(y[i] - y[j])
                ags.append(0.5 if (sx == 0 or sy == 0) else float(sx == sy))
            ag = float(np.mean(ags))
            acc[name][0].append(float(sp))
            acc[name][1].append(ag)
    rec[f"{cons}/{model}"] = {
        name: {"spearman": round(float(np.nanmean(v[0])), 3),
               "agree": round(float(np.nanmean(v[1])), 3),
               "n_queries": len(v[0])} for name, v in acc.items()}
print(json.dumps(rec, indent=1))
json.dump(rec, open(OUT / "A2_interfaces.json", "w"), indent=1)
