"""Selective-abstention extension (LABELED EXTENSION, outside frozen matrix).

Reviewer-driven robustness (Round C): is harm just forced consumption? For
each (query, system, k, model): if the frozen C3 belief abstains, substitute
the NoInfo prior; else keep the C3 belief. Beliefs come ONLY from frozen
cache/parquets (zero LLM calls); episodes use the frozen controller/sim with
paired confirmation seeds. Answers whether abstention-fallback rescues
retrieval utility. Output: results/v3main/sim_selective/.
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sim_eval_v3main import require_frozen_beliefs, belief_row_to_interpretation

BELIEFS = ROOT / "results" / "v3main"
OUT = BELIEFS / "sim_selective"
SEEDS = [60000 + i for i in range(5)]
SYSTEMS = ["bm25", "dense", "hybrid", "rerank", "random"]
N_BOOT = 5000


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.experiment_phase5 import _run_p5_episode
    from src.events import Regime
    (sys_name, model, k, qid, true_regime, probs, seed) = item
    interp = belief_row_to_interpretation({
        "p_normal": probs[0], "p_supplier_delay": probs[1],
        "p_demand_surge": probs[2], "abstain": False,
        "confidence": 1.0, "doc_ids": []})
    r, _ = _run_p5_episode(seed=seed, regime=Regime(true_regime),
                           sensor=f"SelectiveC3+{sys_name}",
                           controller="CausalOptimizer", regime_interp=interp)
    return {"system": sys_name, "consumer": "SelectiveC3", "model": model,
            "k": k, "query_id": qid, "true_regime": true_regime,
            "seed": seed, "profit": float(r.total_profit)}


def main():
    require_frozen_beliefs(BELIEFS)
    OUT.mkdir(parents=True, exist_ok=True)
    prior = (1 / 3, 1 / 3, 1 / 3)
    items = []
    counts = {}
    for m in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ"):
        model = m.replace("Qwen_", "Qwen/")
        bel = pd.read_parquet(BELIEFS / f"beliefs_{m}.parquet")
        c3 = bel[(bel["consumer"] == "C3") & (bel["system"].isin(SYSTEMS))]
        qreg = dict(bel[["query_id", "true_regime"]].drop_duplicates().values)
        for _, r in c3.iterrows():
            probs = ((1 / 3, 1 / 3, 1 / 3) if bool(r["abstain"]) else
                     (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"]))
            counts[(m, bool(r["abstain"]))] = counts.get(
                (m, bool(r["abstain"])), 0) + 1
            for seed in SEEDS:
                items.append((r["system"], model, int(r["k"]), r["query_id"],
                              r["true_regime"], probs, seed))
    print(f"selective items: {len(items)}; abstain rows: "
          f"{sum(v for k, v in counts.items() if k[1])}", flush=True)
    t0 = time.time()
    done, rows = 0, []
    with Pool(7) as pool:
        for row in pool.imap_unordered(_work, items, chunksize=8):
            rows.append(row)
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(items)} "
                      f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    df = pd.DataFrame(rows)
    import json as _j
    _test = set(_j.loads(
        (ROOT / "data" / "v3" / "splits" / "splits.json").read_text())["test"])
    df = df[df["query_id"].isin(_test)].reset_index(drop=True)
    df.to_parquet(OUT / "selective_episodes.parquet", index=False)
    # J + paired CI vs NoInfo (balanced + empirical)
    sys.path.insert(0, str(ROOT / "tools"))
    from run_sim_v3main import nest_J, nest_diff, WEIGHT_VECTORS
    from src.metrics import crossed_bootstrap_ci
    ref = pd.read_parquet(BELIEFS / "sim" / "episodes.parquet")
    ref = ref[ref["rung"] == "NoInfo"]
    ref = ref[ref["query_id"].isin(set(df["query_id"]))]
    out_rows = []
    for wname, W in WEIGHT_VECTORS.items():
        Jn = nest_J(ref, W)
        for (s, m, k), g in df.groupby(["system", "model", "k"]):
            J = nest_J(g, W)
            ci = crossed_bootstrap_ci(nest_diff(g, ref), regime_weights=W,
                                      n_boot=N_BOOT, seed=42)
            out_rows.append({"system": s, "model": m, "k": k,
                             "weights": wname, "J": J,
                             "delta_vs_noinfo": J - Jn,
                             "ci_lo": ci["ci_lower"], "ci_hi": ci["ci_upper"],
                             "p_value": ci["p_value"]})
    res = pd.DataFrame(out_rows)
    res.to_parquet(OUT / "selective_utility.parquet", index=False)
    res.to_json(OUT / "selective_utility.json", orient="records", indent=1)
    print(res.round(1).to_string(index=False))
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
