"""V3 gym transfer (SECONDARY external validation; gated: frozen beliefs).

 gym-invmgmt==0.2.1 is a 2-class (surge/normal) world: V3 queries map
 demand_surge -> surge env, normal/supplier_delay -> normal env, and the
 3-class belief dict passes through (the adaptive controller reads only the
 normal/demand_surge keys; supplier_delay mass is ignored there). This is the
 same projection limitation recorded for Phase 8B — gym is corroboration,
 never the primary estimand.

 Run ONLY after the controlled analysis lands (protocol order):
  python3 tools/run_gym_v3main.py --phase episodes   # belief arms x queries x seeds
  python3 tools/run_gym_v3main.py --phase analysis   # J table, paired vs NoInfo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sim_eval_v3main import require_frozen_beliefs

BELIEFS = ROOT / "results" / "v3main"
OUT = BELIEFS / "gym"
SEEDS = [60000 + i for i in range(5)]
N_BOOT = 5000


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.gym_adapter import run_gym_episode
    from src.events import Regime
    (sys_name, cons, model, k, qid, true_regime, probs, seed) = item
    t0 = time.time()
    regime = Regime(true_regime)
    gym_regime = Regime.DEMAND_SURGE if regime == Regime.DEMAND_SURGE else Regime.NORMAL
    r, _ = run_gym_episode(seed=seed, regime=gym_regime,
                           sensor=f"{sys_name}+{cons}",
                           regime_probabilities={"normal": probs[0],
                                                 "supplier_delay": probs[1],
                                                 "demand_surge": probs[2]})
    return {"system": sys_name, "consumer": cons, "model": model, "k": k,
            "query_id": qid, "true_regime": true_regime,
            "gym_regime": gym_regime.value, "seed": seed,
            "reward": float(r.total_reward),
            "wall_s": round(time.time() - t0, 3)}


def load_beliefs_dedup():
    bel = pd.concat([pd.read_parquet(BELIEFS / f"beliefs_{m}.parquet")
                     for m in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ")],
                    ignore_index=True)
    keys = ["system", "query_id", "k"]
    c0 = bel[bel["consumer"] == "C0"].sort_values(keys).drop_duplicates(keys)
    bel = pd.concat([c0, bel[bel["consumer"] != "C0"]], ignore_index=True)
    bel["model"] = bel["model"].fillna("none")
    return bel


def phase_episodes():
    require_frozen_beliefs(BELIEFS)
    OUT.mkdir(parents=True, exist_ok=True)
    bel = load_beliefs_dedup()
    items = []
    for _, r in bel.iterrows():
        probs = (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"])
        for seed in SEEDS:
            items.append((r["system"], r["consumer"], r["model"], int(r["k"]),
                          r["query_id"], r["true_regime"], probs, seed))
    # NoInfo / PerfectBelief rungs (same seeds, paired)
    from src.interpreter import no_info_regime_belief
    ni = no_info_regime_belief().normalized().regime_probabilities
    qreg = dict(bel[["query_id", "true_regime"]].drop_duplicates().values)
    for qid, true_regime in sorted(qreg.items()):
        for tag, probs in (("NoInfo", (ni["normal"], ni["supplier_delay"], ni["demand_surge"])),
                           ("PerfectBelief", tuple(1.0 if r == true_regime else 0.0
                                                   for r in ("normal", "supplier_delay",
                                                             "demand_surge")))):
            for seed in SEEDS:
                items.append((tag, "C0", "none", 3, qid, true_regime, probs, seed))
    print(f"gym items: {len(items)}", flush=True)
    t0 = time.time()
    done, rows = 0, []
    with Pool(7) as pool:
        for row in pool.imap_unordered(_work, items, chunksize=32):
            rows.append(row)
            done += 1
            if done % 20000 == 0:
                print(f"  {done}/{len(items)} "
                      f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "gym_episodes.parquet", index=False)
    print(f"gym episodes: {len(df)} in {(time.time() - t0) / 60:.1f} min")


def phase_analysis():
    from src.metrics import weighted_benchmark_return, crossed_bootstrap_ci
    df = pd.read_parquet(OUT / "gym_episodes.parquet")

    def nest_J(sub):
        idx = {}
        for (reg, qid, seed), g in sub.groupby(["true_regime", "query_id", "seed"]):
            idx.setdefault(reg, {}).setdefault("main", {})[qid] = {
                int(seed): float(g["reward"].mean()) for seed in [g["seed"].iloc[0]]}
        # rebuild per-seed properly
        idx = {}
        for (reg, qid), gg in sub.groupby(["true_regime", "query_id"]):
            idx.setdefault(reg, {}).setdefault("main", {})[qid] = dict(
                zip(gg["seed"].astype(int), gg["reward"].astype(float)))
        return weighted_benchmark_return(idx)["value"]

    ref = df[df["system"] == "NoInfo"]
    J_noinfo = nest_J(ref)
    rows = []
    for vals, g in df[df["system"] != "NoInfo"].groupby(
            ["system", "consumer", "model", "k"]):
        arm = dict(zip(["system", "consumer", "model", "k"], vals))
        J = nest_J(g)
        a = g.set_index(["query_id", "seed"])["reward"]
        b = ref.set_index(["query_id", "seed"])["reward"]
        common = a.index.intersection(b.index)
        d = (a.loc[common] - b.loc[common]).reset_index().merge(
            g[["query_id", "true_regime"]].drop_duplicates(), on="query_id")
        nested = {}
        for reg, gg in d.groupby("true_regime"):
            for qid, ggg in gg.groupby("query_id"):
                nested.setdefault(reg, {}).setdefault("main", {})[qid] = dict(
                    zip(ggg["seed"].astype(int), ggg["reward"].astype(float)))
        ci = crossed_bootstrap_ci(nested, n_boot=N_BOOT, seed=42)
        rows.append({**arm, "J": J, "delta_vs_noinfo": J - J_noinfo,
                     "ci_lo": ci["ci_lower"], "ci_hi": ci["ci_upper"],
                     "p_value": ci["p_value"]})
    pd.DataFrame(rows).to_parquet(OUT / "gym_utility.parquet", index=False)
    print(f"gym arms: {len(rows)}; J_noinfo={J_noinfo:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["episodes", "analysis"], required=True)
    a = ap.parse_args()
    if a.phase == "episodes":
        phase_episodes()
    else:
        phase_analysis()


if __name__ == "__main__":
    main()
