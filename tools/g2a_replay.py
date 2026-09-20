"""G2A-0 sim replay (CPU-only, frozen beliefs/controller/seeds, no LLM calls).

Replays 600 full-set rows (frozen rerank k=3 beliefs, for the G3 gate) + 1800
LOO rows (extension_g2a/loo_beliefs.parquet) through _run_p5_episode on paired
seeds 60000-60004. Records per-episode profit.

G3 GATE: replayed full-set profits must equal frozen episodes.parquet profits
within 1e-9 for every item; any mismatch -> STOP (exit 2), no verdict.

Writes results/v3main/extension_g2a/loo_episodes.parquet + g2a_verify.json.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sim_eval_v3main import require_frozen_beliefs, belief_row_to_interpretation  # noqa: E402

BELIEFS = ROOT / "results" / "v3main"
OUT = BELIEFS / "extension_g2a"
SEEDS = [60000 + i for i in range(5)]


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.experiment_phase5 import _run_p5_episode
    from src.events import Regime
    (sys_name, cons, model, qid, true_regime, probs, seed, ablation) = item
    interp = belief_row_to_interpretation({
        "p_normal": probs[0], "p_supplier_delay": probs[1],
        "p_demand_surge": probs[2], "abstain": False,
        "confidence": 1.0, "doc_ids": []})
    r, _ = _run_p5_episode(seed=seed, regime=Regime(true_regime),
                           sensor=f"{sys_name}+{cons}", controller="CausalOptimizer",
                           regime_interp=interp)
    return {"system": sys_name, "consumer": cons, "model": model, "k": 3,
            "query_id": qid, "true_regime": true_regime, "seed": seed,
            "ablation": ablation, "profit": float(r.total_profit)}


def main():
    require_frozen_beliefs(BELIEFS)
    OUT.mkdir(parents=True, exist_ok=True)
    loo = pd.read_parquet(OUT / "loo_beliefs.parquet")
    bel8 = pd.read_parquet(BELIEFS / "beliefs_Qwen_Qwen3-8B-AWQ.parquet")
    bel14 = pd.read_parquet(BELIEFS / "beliefs_Qwen_Qwen3-14B-AWQ.parquet")
    bel = pd.concat([bel8, bel14], ignore_index=True)
    full = bel[(bel["system"] == "rerank") & (bel["k"] == 3) &
               (((bel["consumer"] == "C0")) |
                ((bel["consumer"] == "C3") &
                 (bel["model"].isin(["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"]))))].copy()
    c0 = full[full["consumer"] == "C0"].sort_values(
        ["system", "query_id", "k"]).drop_duplicates(["system", "query_id", "k"])
    full = pd.concat([c0, full[full["consumer"] != "C0"]], ignore_index=True)
    full["ablation"] = "full"
    cols = ["system", "consumer", "model", "query_id", "true_regime",
            "p_normal", "p_supplier_delay", "p_demand_surge", "ablation"]
    items = []
    for df in (full[cols], loo[cols]):
        for _, r in df.iterrows():
            probs = (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"])
            for seed in SEEDS:
                items.append((r["system"], r["consumer"], r["model"],
                              r["query_id"], r["true_regime"], probs, seed,
                              r["ablation"]))
    print(f"replay items: {len(items)}", flush=True)
    t0 = time.time()
    done = []
    with Pool(7) as pool:
        for i, row in enumerate(pool.imap_unordered(_work, items, chunksize=8)):
            done.append(row)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    ep = pd.DataFrame(done)
    ep.to_parquet(OUT / "loo_episodes.parquet", index=False)
    # G3 gate: full-set replay vs frozen episodes.parquet
    froz = pd.read_parquet(BELIEFS / "sim" / "episodes.parquet")
    key = ["system", "consumer", "model", "k", "query_id", "seed"]
    mine = ep[ep["ablation"] == "full"]
    mg = mine.merge(froz[key + ["profit"]], on=key, suffixes=("", "_frozen"))
    assert len(mg) == len(mine), "replay/frozen key mismatch"
    diff = (mg["profit"] - mg["profit_frozen"]).abs()
    rep = {"n": len(mg), "max_abs_diff": float(diff.max()),
           "n_mismatch_gt_1e9": int((diff > 1e-9).sum())}
    print("G3 VERIFICATION:", json.dumps(rep), flush=True)
    json.dump(rep, open(OUT / "g2a_verify.json", "w"), indent=1)
    if rep["n_mismatch_gt_1e9"]:
        print("G3 GATE FAILED", flush=True)
        sys.exit(2)
    print("G3 GATE PASSED", flush=True)


if __name__ == "__main__":
    main()
