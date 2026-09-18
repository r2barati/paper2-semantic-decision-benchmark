"""Controller-swap robustness episodes + analysis (LABELED EXTENSION).

Same frozen beliefs, same paired confirmation seeds (60000-60004), same
warning protocol (prior before t=12, belief after) — only the fixed policy
class changes (BeliefBaseStock instead of CausalOptimizer). Headline arms
only: bm25/rerank/oracle-factual x C0/C1/C3/8B + NoInfo/NoText_Tuned/
PerfectBelief rungs, k=3, test queries. Output:
results/v3main/sim_controllerB/.
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

from src.sim_eval_v3main import require_frozen_beliefs, belief_row_to_interpretation

BELIEFS = ROOT / "results" / "v3main"
OUT = BELIEFS / "sim_controllerB"
SEEDS = [60000 + i for i in range(5)]
N_BOOT = 5000
ARMS = [("bm25", "C0", "none"), ("bm25", "C1", "Qwen/Qwen3-8B-AWQ"),
        ("bm25", "C3", "Qwen/Qwen3-8B-AWQ"),
        ("rerank", "C0", "none"), ("rerank", "C1", "Qwen/Qwen3-8B-AWQ"),
        ("rerank", "C3", "Qwen/Qwen3-8B-AWQ"),
        ("oracle-factual", "C0", "none"),
        ("oracle-factual", "C1", "Qwen/Qwen3-8B-AWQ"),
        ("oracle-factual", "C3", "Qwen/Qwen3-8B-AWQ")]


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.env import InventoryEnv
    from src.events import Regime, P5_HORIZON, P5_WARNING_TIME
    from src.metrics import compute_episode_metrics
    from src.interpreter import no_info_regime_belief
    from src.controller_basestock import BeliefBaseStock
    (kind, sys_name, cons, model, qid, true_regime, probs, seed) = item
    t0 = time.time()
    env = InventoryEnv(seed=seed, regime=Regime(true_regime))
    env.reset()
    if kind == "noinfo":
        interp = no_info_regime_belief()
    elif kind == "perfect":
        from src.interpreter import RegimeInterpretation
        interp = RegimeInterpretation(
            regime_probabilities={r: 1.0 if r == true_regime else 0.0
                                  for r in ("normal", "supplier_delay",
                                            "demand_surge")})
    elif kind == "tuned":
        from src.interpreter import RegimeInterpretation
        interp = RegimeInterpretation(
            regime_probabilities={"normal": 0.0, "supplier_delay": 0.0,
                                  "demand_surge": 1.0})
    else:
        interp = belief_row_to_interpretation({
            "p_normal": probs[0], "p_supplier_delay": probs[1],
            "p_demand_surge": probs[2], "abstain": False,
            "confidence": 1.0, "doc_ids": []})
    prior = no_info_regime_belief().normalized().regime_probabilities
    belief = interp.normalized().regime_probabilities
    ctl = BeliefBaseStock(prior)
    history = []
    for t in range(P5_HORIZON):
        if t == P5_WARNING_TIME:
            ctl = BeliefBaseStock(belief)
        history.append(env.step(ctl.decide(env._state)))
    m = compute_episode_metrics(history, seed=seed,
                                condition=f"B+{sys_name}+{cons}")
    return {"system": sys_name, "consumer": cons, "model": model, "k": 3,
            "query_id": qid, "true_regime": true_regime, "seed": seed,
            "rung": kind, "profit": float(m.total_profit),
            "wall_s": round(time.time() - t0, 3)}


def main():
    require_frozen_beliefs(BELIEFS)
    OUT.mkdir(parents=True, exist_ok=True)
    bel = pd.read_parquet(BELIEFS / "beliefs_Qwen_Qwen3-8B-AWQ.parquet")
    test = set(json.loads(
        (ROOT / "data" / "v3" / "splits" / "splits.json").read_text())["test"])
    bel = bel[bel["query_id"].isin(test)]
    items = []
    for (sys_name, cons, model) in ARMS:
        sub = bel[(bel["system"] == sys_name) & (bel["consumer"] == cons)]
        if model == "none":
            sub = sub[sub["model"] == "none"]
        else:
            sub = sub[sub["model"] == model]
        for _, r in sub.iterrows():
            probs = (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"])
            for seed in SEEDS:
                items.append(("belief", sys_name, cons, model, r["query_id"],
                              r["true_regime"], probs, seed))
    qreg = dict(bel[["query_id", "true_regime"]].drop_duplicates().values)
    for qid, reg in sorted(qreg.items()):
        for kind, sys_name in (("noinfo", "NoInfo"), ("perfect", "PerfectBelief"),
                               ("tuned", "NoText_Tuned")):
            for seed in SEEDS:
                items.append((kind, sys_name, "C0", "none", qid, reg, None,
                              seed))
    print(f"controllerB items: {len(items)}", flush=True)
    t0 = time.time()
    done, rows = 0, []
    with Pool(7) as pool:
        for row in pool.imap_unordered(_work, items, chunksize=8):
            rows.append(row)
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(items)} "
                      f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "episodesB.parquet", index=False)
    # analysis: balanced J + paired CI vs NoInfo (same machinery)
    sys.path.insert(0, str(ROOT / "tools"))
    from run_sim_v3main import nest_J, nest_diff
    from src.metrics import crossed_bootstrap_ci
    ref = df[df["rung"] == "noinfo"]
    Jn = nest_J(ref)
    print(f"controllerB J_noinfo={Jn:.2f}", flush=True)
    out_rows = []
    for (s, c, m), g in df[df["rung"] == "belief"].groupby(
            ["system", "consumer", "model"]):
        J = nest_J(g)
        ci = crossed_bootstrap_ci(nest_diff(g, ref), n_boot=N_BOOT, seed=42)
        out_rows.append({"system": s, "consumer": c, "model": m, "J": J,
                         "delta_vs_noinfo": J - Jn,
                         "ci_lo": ci["ci_lower"], "ci_hi": ci["ci_upper"],
                         "p_value": ci["p_value"]})
    res = pd.DataFrame(out_rows)
    res.to_parquet(OUT / "utilityB.parquet", index=False)
    res.to_json(OUT / "utilityB.json", orient="records", indent=1)
    for _, r in res.iterrows():
        print(f"{r['system']:15s} {r['consumer']} d={r['delta_vs_noinfo']:+.1f} "
              f"[{r['ci_lo']:+.0f},{r['ci_hi']:+.0f}] p={r['p_value']:.4f}",
              flush=True)
    for rung in ("perfect", "tuned"):
        g = df[df["rung"] == rung]
        print(f"{rung}: J={nest_J(g):.1f}", flush=True)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
