"""§6 empirical map: harvest + sweep (CPU-only, zero LLM calls).

harvest: 15 NoInfo closed-loop replays (3 regimes x seeds 60000-60004)
  mirroring _run_p5_episode with belief fixed at prior; snapshots at
  t in {0,5,10,12,15,20,25,30} (state copy + pipeline copy, live optimizer
  carrying true NoInfo history).
sweep: at each snapshot, 861-node simplex lattice (step 0.025) through
  set_regime_probabilities + decide(), pipeline snapshot/restore per node,
  prior restored after. Parallel over 15 trajectories on Pool(7).
GATE: harvested profits vs frozen noinfo episodes within 1e-9 (all 15
  trajectories), else STOP (exit 2).
Writes theory/map_states.parquet + theory/map_orders.parquet.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "v3main" / "theory"
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = [0, 5, 10, 12, 15, 20, 25, 30]
STEP = 0.025
N = int(round(1.0 / STEP))  # 40
NODES = [(i, j, N - i - j) for i in range(N + 1) for j in range(N - i + 1)]
assert len(NODES) == 861, len(NODES)
PRIOR = {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30}


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.env import (INITIAL_INVENTORY, CausalOptimizer, InventoryEnv,
                         PipelineSchedule)
    from src.events import P5_DEMAND_MEAN, P5_HORIZON, Regime
    from src.interpreter import no_info_regime_belief
    true_regime, seed = item
    env = InventoryEnv(seed=seed, regime=Regime(true_regime))
    env.reset()
    prior = no_info_regime_belief().normalized().regime_probabilities
    opt = CausalOptimizer(seed=seed, regime_probabilities=dict(prior),
                          horizon=P5_HORIZON, initial_inventory=INITIAL_INVENTORY,
                          demand_mean=P5_DEMAND_MEAN)
    states, hist = [], []
    for t in range(P5_HORIZON):
        if t in PERIODS:
            states.append({"true_regime": true_regime, "seed": seed, "t": t,
                           "on_hand": float(env._state.on_hand),
                           "pipeline": opt._pipeline.as_dict(),
                           "state": copy.deepcopy(env._state)})
        order = opt.decide(env._state)
        hist.append(env.step(order))
    profit = float(hist[-1].cumulative_profit)
    sweeps = []
    for s in states:
        snap = dict(s["pipeline"])
        st = s["state"]
        for ni, (i, j, k) in enumerate(NODES):
            opt._pipeline = PipelineSchedule(dict(snap))
            opt.set_regime_probabilities(
                {"normal": i / N, "supplier_delay": j / N, "demand_surge": k / N})
            sweeps.append({"true_regime": true_regime, "seed": seed, "t": s["t"],
                           "node": ni, "order": float(opt.decide(st))})
        opt._pipeline = PipelineSchedule(dict(snap))
        opt.set_regime_probabilities(dict(prior))
    return {"key": [true_regime, seed], "states": states,
            "sweeps": sweeps, "profit": profit}


def main():
    from src.events import Regime
    regs = [Regime.NORMAL.value, Regime.SUPPLIER_DELAY.value, Regime.DEMAND_SURGE.value]
    items = [(r, 60000 + i) for r in regs for i in range(5)]
    assert len(items) == 15
    t0 = time.time()
    S, W, P = [], [], {}
    with Pool(7) as pool:
        for out in pool.imap_unordered(_work, items):
            S.extend(out["states"])
            W.extend(out["sweeps"])
            P[tuple(out["key"])] = out["profit"]
    print(f"sweep done: {len(S)} states x {len(NODES)} nodes in {time.time()-t0:.0f}s",
          flush=True)
    assert len(S) == 120, len(S)
    st = pd.DataFrame([{k: v for k, v in s.items() if k not in ("state", "pipeline")}
                       for s in S])
    st.to_parquet(OUT / "map_states.parquet", index=False)
    pd.DataFrame(W).to_parquet(OUT / "map_orders.parquet", index=False)
    # GATE vs frozen noinfo episodes on (true_regime, seed)
    froz = pd.read_parquet(ROOT / "results/v3main/sim/episodes.parquet")
    ni = froz[(froz["system"] == "noinfo") & (froz["k"] == 3)]
    bad = []
    for (r, s) in items:
        ref = ni[(ni["true_regime"] == r) & (ni["seed"] == s)]["profit"]
        assert len(ref) >= 1, (r, s)
        d = abs(P[(r, s)] - float(ref.iloc[0]))
        if d > 1e-9:
            bad.append([r, s, d])
    rep = {"n": len(items), "max_abs_diff": max(
        abs(P[(r, s)] - float(ni[(ni["true_regime"] == r) & (ni["seed"] == s)]["profit"].iloc[0]))
        for (r, s) in items), "n_mismatch_gt_1e9": len(bad)}
    print("HARVEST GATE:", json.dumps(rep), flush=True)
    json.dump(rep, open(OUT / "map_verify.json", "w"), indent=1)
    if bad:
        print("GATE FAILED", flush=True)
        raise SystemExit(2)
    print("GATE PASSED", flush=True)


if __name__ == "__main__":
    main()
