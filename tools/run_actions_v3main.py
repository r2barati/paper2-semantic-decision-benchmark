"""A0 action-logging replay (extension, frozen beliefs/controller/seeds).

Replays the A2 subset (DESIGN.md) through _run_p5_episode and records
per-episode 40-period order vectors. VERIFICATION GATE: replayed profits
must equal frozen episodes.parquet profits within 1e-9 for every item.
Writes results/v3main/extension_actions/{actions,verify}.parquet/json.
CPU-only. No LLM calls. Does not touch frozen files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sim_eval_v3main import require_frozen_beliefs, belief_row_to_interpretation  # noqa: E402

BELIEFS = ROOT / "results" / "v3main"
OUT = BELIEFS / "extension_actions"
SEEDS = [60000 + i for i in range(5)]
SYSTEMS = ["bm25", "dense", "hybrid", "rerank", "random",
           "oracle-relevant", "oracle-factual"]
CONSUMERS = {"C0": ["none"], "C1": ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"],
             "C3": ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"]}
K = 3


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.experiment_phase5 import _run_p5_episode
    from src.events import Regime
    (sys_name, cons, model, qid, true_regime, probs, seed) = item
    t0 = time.time()
    interp = belief_row_to_interpretation({
        "p_normal": probs[0], "p_supplier_delay": probs[1],
        "p_demand_surge": probs[2], "abstain": False,
        "confidence": 1.0, "doc_ids": []})
    r, hist = _run_p5_episode(seed=seed, regime=Regime(true_regime),
                              sensor=f"{sys_name}+{cons}", controller="CausalOptimizer",
                              regime_interp=interp)
    orders = [float(s.order_quantity) for s in hist]
    assert len(orders) == 40, f"horizon drift: {len(orders)}"
    return {"system": sys_name, "consumer": cons, "model": model, "k": K,
            "query_id": qid, "seed": seed, "profit": float(r.total_profit),
            "orders": orders, "wall_s": round(time.time() - t0, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: replay only first N items (verification still runs)")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true",
                    help="merge per-shard outputs, run verification gate, exit")
    args = ap.parse_args()
    require_frozen_beliefs(BELIEFS)
    OUT.mkdir(parents=True, exist_ok=True)
    bel = pd.concat([pd.read_parquet(BELIEFS / f"beliefs_{m}.parquet")
                     for m in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ")],
                    ignore_index=True)
    test_ids = set(json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))["test"])
    sub = bel[(bel["system"].isin(SYSTEMS)) & (bel["k"] == K)
              & (bel["query_id"].isin(test_ids))]
    rows = []
    for _, r in sub.iterrows():
        if r["consumer"] not in CONSUMERS or r["model"] not in CONSUMERS[r["consumer"]]:
            continue
        rows.append(r)
    sub = pd.DataFrame(rows)
    items = []
    for _, r in sub.iterrows():
        probs = (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"])
        for seed in SEEDS:
            items.append((r["system"], r["consumer"], r["model"], r["query_id"],
                          r["true_regime"], probs, seed))
    if args.limit:
        items = items[:args.limit]
    if args.merge_only:
        return do_merge()
    tag = f"_s{args.shard}" if args.shard >= 0 else ""
    if args.shard >= 0:
        items = [it for i, it in enumerate(items) if i % args.nshards == args.shard]
        print(f"shard {args.shard}/{args.nshards}: {len(items)} items", flush=True)
    print(f"replay items: {len(items)}", flush=True)
    t0 = time.time()
    done = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_work, items, chunksize=8)):
            done.append(row)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    act = pd.DataFrame(done)
    act.to_parquet(OUT / f"actions{tag}.parquet", index=False)
    print(f"shard{tag} wrote {len(act)} rows in {time.time()-t0:.0f}s", flush=True)
    if args.shard < 0:
        return do_merge()
    return 0


def do_merge():
    import glob
    parts = sorted(glob.glob(str(OUT / "actions_s*.parquet")))
    assert parts, "no shard outputs to merge"
    act = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"merging {len(parts)} shards, {len(act)} rows", flush=True)
    # VERIFICATION GATE vs frozen episodes.parquet
    froz = pd.read_parquet(BELIEFS / "sim" / "episodes.parquet")
    key = ["system", "consumer", "model", "k", "query_id", "seed"]
    mg = act.merge(froz[key + ["profit"]], on=key, suffixes=("", "_frozen"))
    assert len(mg) == len(act), "replay/frozen key mismatch"
    diff = (mg["profit"] - mg["profit_frozen"]).abs()
    rep = {"n": len(mg), "max_abs_diff": float(diff.max()),
           "n_mismatch_gt_1e9": int((diff > 1e-9).sum())}
    print("VERIFICATION:", json.dumps(rep), flush=True)
    if rep["n_mismatch_gt_1e9"]:
        print("GATE FAILED — protocol divergence, see mismatches", flush=True)
        bad = mg[diff > 1e-9].head(10).to_dict("records")
        json.dump({"report": rep, "bad": bad}, open(OUT / "verify.json", "w"), indent=1)
        sys.exit(2)
    act.to_parquet(OUT / "actions.parquet", index=False)
    json.dump(rep, open(OUT / "verify.json", "w"), indent=1)
    print("GATE PASSED — actions written", flush=True)


if __name__ == "__main__":
    main()
