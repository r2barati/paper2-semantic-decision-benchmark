"""Cost/variance pilot for V3 sim eval (gated: requires frozen beliefs).

Measures single-episode + Hindsight MILP wall time and decomposes paired-
difference variance into query vs seed components on a small arm subset, to
choose n_seeds on evidence (seeds = env uncertainty only). Writes
results/v3main/sim_pilot/ (timing + variance summary only; pilot episodes
are NOT part of the main evaluation and are excluded from every J).
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sim_eval_v3main import require_frozen_beliefs, belief_row_to_interpretation

BELIEFS = ROOT / "results" / "v3main"
OUT = BELIEFS / "sim_pilot"
SEEDS = [60000 + i for i in range(8)]
ARMS = [
    ("bm25", "C0", "none", 3),
    ("rerank", "C1", "Qwen/Qwen3-8B-AWQ", 3),
    ("oracle-factual", "C3", "Qwen/Qwen3-8B-AWQ", 3),
    ("none-arm", "C0", "none", 3),  # NoInfo placeholder (no belief row)
]
N_QUERIES = 30


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.experiment_phase5 import _run_p5_episode
    from src.events import Regime
    from src.interpreter import RegimeInterpretation, no_info_regime_belief
    kind, sys_name, cons, model, k, qid, true_regime, probs, seed = item
    t0 = time.time()
    if kind == "hindsight":
        from src.env import HindsightOracle
        h = HindsightOracle(seed=seed, regime=Regime(true_regime))
        hist, _ = h.compute_optimal_trajectory()
        val = float(hist[-1].cumulative_profit)
    else:
        if kind == "noinfo":
            interp = no_info_regime_belief()
        elif kind == "perfect":
            interp = RegimeInterpretation(
                regime_probabilities={r: 1.0 if r == true_regime else 0.0
                                      for r in ("normal", "supplier_delay",
                                                "demand_surge")})
        else:
            interp = belief_row_to_interpretation({
                "p_normal": probs[0], "p_supplier_delay": probs[1],
                "p_demand_surge": probs[2], "abstain": False,
                "confidence": 1.0, "doc_ids": []})
        r, _ = _run_p5_episode(seed=seed, regime=Regime(true_regime),
                               sensor=f"{sys_name}+{cons}", controller="CausalOptimizer",
                               regime_interp=interp)
        val = float(r.total_profit)
    return {"system": sys_name, "consumer": cons, "model": model, "k": k,
            "query_id": qid, "true_regime": true_regime, "seed": seed,
            "kind": kind, "profit": val, "wall_s": round(time.time() - t0, 3)}


def main():
    require_frozen_beliefs(BELIEFS)
    OUT.mkdir(parents=True, exist_ok=True)
    bel = {m: pd.read_parquet(BELIEFS / f"beliefs_{m}.parquet")
           for m in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ")}
    queries = sorted(bel["Qwen_Qwen3-8B-AWQ"]["query_id"].unique())[:N_QUERIES]
    qreg = {}
    for m, df in bel.items():
        for _, r in df[["query_id", "true_regime"]].drop_duplicates().iterrows():
            qreg[r["query_id"]] = r["true_regime"]
    items = []
    for (sys_name, cons, model, k) in ARMS:
        for qid in queries:
            if sys_name == "none-arm":
                kind, probs = "noinfo", None
            else:
                src = bel["Qwen_Qwen3-8B-AWQ"] if model != "none" else bel["Qwen_Qwen3-8B-AWQ"]
                row = src[(src["system"] == sys_name) & (src["query_id"] == qid)
                          & (src["k"] == k) & (src["consumer"] == cons)]
                if not len(row):
                    continue
                row = row.iloc[0]
                kind, probs = "belief", (row["p_normal"], row["p_supplier_delay"],
                                         row["p_demand_surge"])
            for seed in SEEDS:
                items.append((kind, sys_name, cons, model, k, qid,
                              qreg[qid], probs, seed))
    # hindsight timing subset
    for qid in queries[:4]:
        items.append(("hindsight", "hindsight", "C0", "none", 3, qid,
                      qreg[qid], None, SEEDS[0]))
    print(f"pilot items: {len(items)}", flush=True)
    t0 = time.time()
    with Pool(7) as pool:
        rows = pool.map(_work, items, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "pilot_episodes.parquet", index=False)
    ep = df[df.kind != "hindsight"]
    summ = {
        "n_episodes": len(ep),
        "wall_min": round((time.time() - t0) / 60, 2),
        "mean_episode_s": round(float(ep["wall_s"].mean()), 3),
        "hindsight_s": sorted(df[df.kind == "hindsight"]["wall_s"].tolist()),
    }
    # variance decomposition of paired diffs (rerank/C1 - bm25/C0), per query:
    # seed-noise = mean over queries of within-query var across seeds;
    # query-signal = var across queries of query means.
    a = ep[(ep.system == "rerank") & (ep.consumer == "C1")].set_index(["query_id", "seed"])["profit"]
    b = ep[(ep.system == "bm25") & (ep.consumer == "C0")].set_index(["query_id", "seed"])["profit"]
    d = (a - b).reset_index()
    qm = d.groupby("query_id")["profit"]
    summ["paired_diff_query_var"] = round(float(qm.mean().var()), 2)
    summ["paired_diff_seed_var"] = round(float(qm.var().mean()), 2)
    (OUT / "pilot_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
