"""NoText_Tuned extension (LABELED EXTENSION, not part of the frozen matrix).

Reviewer-gap closure (Reviewers C/E): the frozen matrix compares against the
benchmark prior (NoInfo). This tunes a CONSTANT no-text belief triple on DEV
queries only (splits.json dev, n=40) x tuning-only seeds (61000-61004,
disjoint from confirmation seeds 60000-60004), freezes the winner, and
evaluates it on TEST x confirmation seeds. The frozen confirmatory numbers
are untouched; extension results are reported separately.

Grid: 3-simplex step 0.1 (66 triples) x 40 dev queries x 5 seeds = 13,200 eps.
Test: 200 queries x 5 seeds = 1,000 eps.
Output: results/v3main/extension_notext_tuned.json (winner + tuning table).
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "v3main"
TUNE_SEEDS = [61000 + i for i in range(5)]
STEP = 0.1


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.experiment_phase5 import _run_p5_episode
    from src.events import Regime
    from src.interpreter import RegimeInterpretation
    (probs, qid, true_regime, seed) = item
    interp = RegimeInterpretation(
        regime_probabilities={"normal": probs[0],
                              "supplier_delay": probs[1],
                              "demand_surge": probs[2]})
    r, _ = _run_p5_episode(seed=seed, regime=Regime(true_regime),
                           sensor="NoText_Tuned", controller="CausalOptimizer",
                           regime_interp=interp)
    return {"p0": probs[0], "p1": probs[1], "p2": probs[2],
            "query_id": qid, "true_regime": true_regime, "seed": seed,
            "profit": float(r.total_profit)}


def grid():
    n = int(round(1.0 / STEP))
    out = []
    for i, j, k in itertools.product(range(n + 1), repeat=3):
        if i + j + k == n:
            out.append((i * STEP, j * STEP, k * STEP))
    return out


def main():
    splits = json.loads((ROOT / "data" / "v3" / "splits" / "splits.json").read_text())
    dev = splits["dev"]
    qreg = {}
    for line in open(ROOT / "data" / "v3" / "queries.jsonl"):
        import json as _j
        q = _j.loads(line)
        if q["_id"] in dev:
            qreg[q["_id"]] = q["metadata"]["true_regime"]
    assert len(qreg) == len(dev) == 40, (len(qreg), len(dev))
    triples = grid()
    print(f"grid: {len(triples)} triples x {len(qreg)} dev queries "
          f"x {len(TUNE_SEEDS)} seeds", flush=True)
    items = [(t, qid, reg, s) for t in triples for qid, reg in qreg.items()
             for s in TUNE_SEEDS]
    t0 = time.time()
    done, rows = 0, []
    with Pool(7) as pool:
        for row in pool.imap_unordered(_work, items, chunksize=16):
            rows.append(row)
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(items)} "
                      f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    # balanced J over regimes (uniform weights), dev only
    tab = df.groupby(["p0", "p1", "p2", "true_regime"])["profit"].mean().reset_index()
    J = tab.groupby(["p0", "p1", "p2"])["profit"].mean().reset_index()
    J = J.sort_values("profit", ascending=False).reset_index(drop=True)
    winner = J.iloc[0]
    res = {
        "grid_step": STEP, "n_triples": len(triples),
        "dev_queries": len(qreg), "tune_seeds": TUNE_SEEDS,
        "winner": {"p_normal": float(winner["p0"]),
                   "p_supplier_delay": float(winner["p1"]),
                   "p_demand_surge": float(winner["p2"]),
                   "dev_J": float(winner["profit"])},
        "tuning_table": J.to_dict(orient="records"),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    (OUT / "extension_notext_tuned.json").write_text(json.dumps(res, indent=1))
    print(f"winner: {res['winner']} ({res['elapsed_min']} min)")


if __name__ == "__main__":
    main()
