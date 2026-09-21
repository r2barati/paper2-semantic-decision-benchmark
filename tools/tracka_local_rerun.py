"""Local re-run of the Track A pilot cells at current HEAD (ground truth for the
cross-machine gate). Mirrors the Kaggle kernel cells exactly:
BeliefBaseStock x {(bm25,C1,8B),(rerank,C1,8B)} x k=3 x first-10 test queries
x seeds 60000-60004 = 100 episodes. Reuses run_controllerB_v3main._work.
Writes runs/v3main_tracka/pilot_local_rerun.parquet (untracked scratch).
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
sys.path.insert(0, str(ROOT / "tools"))

from run_controllerB_v3main import _work  # noqa: E402

BELIEFS = ROOT / "results" / "v3main"
OUT = ROOT / "runs" / "v3main_tracka"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [60000 + i for i in range(5)]


def main():
    spec = json.loads((ROOT / "kaggle" / "inputs_tracka" / "shard_pilot.json").read_text())
    assert spec["shard_id"] == "pilot-bstock-2x10", spec
    bel = pd.read_parquet(BELIEFS / "beliefs_Qwen_Qwen3-8B-AWQ.parquet")
    items = []
    for cell in spec["cells"]:
        sub = bel[(bel["system"] == cell["system"]) &
                  (bel["consumer"] == cell["consumer"]) &
                  (bel["model"] == cell["model"]) &
                  (bel["k"] == cell["k"]) &
                  (bel["query_id"].isin(spec["queries"]))]
        assert len(sub) == len(spec["queries"]), (cell, len(sub))
        for _, r in sub.iterrows():
            probs = (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"])
            for seed in spec["seeds"]:
                items.append(("belief", cell["system"], cell["consumer"], cell["model"],
                              r["query_id"], r["true_regime"], probs, seed))
    assert len(items) == 100, len(items)
    t0 = time.time()
    with Pool(7) as pool:
        rows = list(pool.imap_unordered(_work, items, chunksize=4))
    assert len(rows) == 100
    # _work returns k under key "k"? normalize to the Kaggle schema
    df = pd.DataFrame(rows)
    df["controller"] = "BeliefBaseStock"
    df.to_parquet(OUT / "pilot_local_rerun.parquet", index=False)
    print(f"local rerun: {len(df)} rows in {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
