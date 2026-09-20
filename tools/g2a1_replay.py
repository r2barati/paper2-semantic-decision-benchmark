"""G2A-1 dev replay (CPU-only, frozen controller/seeds, dev queries only).

Replays 240 C1 LOO rows + 80 frozen full-set C1 rerank dev rows through
_run_p5_episode on paired seeds 60000-60004. Records per-episode profit AND
40-period order vectors (for ADis vs full).

G3-ANALOG GATE: replayed full-set profits must equal frozen dev episodes
within 1e-9 (all items); any mismatch -> STOP (exit 2).

Asserts every query_id in dev_ids (no test access).
Writes extension_g2a/c1_loo_episodes.parquet + g2a1_verify.json.
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

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids = set(sp["dev"])


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
    r, hist = _run_p5_episode(seed=seed, regime=Regime(true_regime),
                              sensor=f"{sys_name}+{cons}", controller="CausalOptimizer",
                              regime_interp=interp)
    return {"system": sys_name, "consumer": cons, "model": model, "k": 3,
            "query_id": qid, "true_regime": true_regime, "seed": seed,
            "ablation": ablation, "profit": float(r.total_profit),
            "orders": [float(s.order_quantity) for s in hist]}


def main():
    require_frozen_beliefs(BELIEFS)
    loo = pd.read_parquet(OUT / "c1_loo_beliefs.parquet")
    assert set(loo["query_id"]) <= dev_ids, "non-dev query in LOO rows!"
    assert (loo["split"] == "dev").all() and (loo["consumer"] == "C1").all()
    bel8 = pd.read_parquet(BELIEFS / "beliefs_Qwen_Qwen3-8B-AWQ.parquet")
    bel14 = pd.read_parquet(BELIEFS / "beliefs_Qwen_Qwen3-14B-AWQ.parquet")
    full = pd.concat([bel8, bel14], ignore_index=True)
    full = full[(full["system"] == "rerank") & (full["consumer"] == "C1") &
                (full["k"] == 3) & (full["query_id"].isin(dev_ids))].copy()
    assert len(full) == 80, f"expected 80 full C1 rerank dev rows, got {len(full)}"
    full["ablation"] = "full"
    qreg = dict(full[["query_id", "true_regime"]].drop_duplicates().values)
    assert len(qreg) == 40
    loo = loo.copy()
    loo["true_regime"] = loo["query_id"].map(qreg)
    assert loo["true_regime"].notna().all()
    cols = ["system", "consumer", "model", "query_id", "true_regime",
            "p_normal", "p_supplier_delay", "p_demand_surge", "ablation"]
    items = []
    for df in (full[cols], loo[["system", "consumer", "model", "query_id",
                                "true_regime", "p_normal", "p_supplier_delay",
                                "p_demand_surge", "ablation"]]):
        for _, r in df.iterrows():
            probs = (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"])
            for seed in SEEDS:
                items.append((r["system"], r["consumer"], r["model"],
                              r["query_id"], r["true_regime"], probs, seed,
                              r["ablation"]))
    print(f"replay items: {len(items)} (dev only)", flush=True)
    t0 = time.time()
    done = []
    with Pool(7) as pool:
        for row in pool.imap_unordered(_work, items, chunksize=8):
            done.append(row)
    ep = pd.DataFrame(done)
    froz = pd.read_parquet(BELIEFS / "sim" / "episodes.parquet")
    key = ["system", "consumer", "model", "k", "query_id", "seed"]
    mine = ep[ep["ablation"] == "full"].drop(columns=["orders"])
    mg = mine.merge(froz[key + ["profit"]], on=key, suffixes=("", "_frozen"))
    assert len(mg) == len(mine), "replay/frozen key mismatch"
    diff = (mg["profit"] - mg["profit_frozen"]).abs()
    rep = {"n": len(mg), "max_abs_diff": float(diff.max()),
           "n_mismatch_gt_1e9": int((diff > 1e-9).sum())}
    print("G3-ANALOG:", json.dumps(rep), flush=True)
    json.dump(rep, open(OUT / "g2a1_verify.json", "w"), indent=1)
    if rep["n_mismatch_gt_1e9"]:
        print("GATE FAILED", flush=True)
        sys.exit(2)
    ep.to_parquet(OUT / "c1_loo_episodes.parquet", index=False)
    print(f"GATE PASSED; wrote {OUT/'c1_loo_episodes.parquet'} rows={len(ep)} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
