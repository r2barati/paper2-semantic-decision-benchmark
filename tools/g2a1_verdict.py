"""G2A-1 verdict (dev-40 only; CPU analysis, no new episodes/calls).

Per (model, position) cell (6 cells): belief change (L1 prob shift vs full),
action change (ADis vs full order vectors), utility change (J-VoI =
dJ_full - dJ_loo vs frozen NoInfo rung). Paired query bootstrap B=5000.

Continuation gate (DESIGN-G2A1, binding): FIRE iff (a) per-position J-VoI CI
excludes 0 in >=1 cell AND (b) the same cell shows ADis-vs-full mean CI > 0.
Writes extension_g2a/g2a1_verdict.json. Dev only throughout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_g2a"

MODELS = ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"]
POS = ["drop0", "drop1", "drop2"]
B = 5000
SEED = 73000

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids = set(sp["dev"])

bel = pd.read_parquet(OUT / "c1_loo_beliefs.parquet")
assert set(bel["query_id"]) <= dev_ids
ep = pd.read_parquet(OUT / "c1_loo_episodes.parquet")
assert set(ep["query_id"]) <= dev_ids
froz = pd.read_parquet(ROOT / "results" / "v3main" / "sim" / "episodes.parquet")
full_froz = froz[(froz["system"] == "rerank") & (froz["consumer"] == "C1") &
                 (froz["k"] == 3) & (froz["query_id"].isin(dev_ids))]
ni = froz[(froz["system"] == "noinfo") & (froz["k"] == 3) &
          (froz["query_id"].isin(dev_ids))].set_index(["query_id", "seed"])["profit"]

# full-set beliefs per (model, query) from frozen parquet (dev slice)
bel8 = pd.read_parquet(ROOT / "results/v3main/beliefs_Qwen_Qwen3-8B-AWQ.parquet")
bel14 = pd.read_parquet(ROOT / "results/v3main/beliefs_Qwen_Qwen3-14B-AWQ.parquet")
full = pd.concat([bel8, bel14], ignore_index=True)
full = full[(full["system"] == "rerank") & (full["consumer"] == "C1") &
            (full["k"] == 3) & (full["query_id"].isin(dev_ids))]

rng = np.random.default_rng(SEED)
cells = {}
for model in MODELS:
    fm = full[full["model"] == model].set_index("query_id")
    lm = bel[bel["model"] == model]
    em = ep[ep["model"] == model]
    fm_ep = full_froz[full_froz["model"] == model]
    for pos in POS:
        per_q = []
        for qid in sorted(dev_ids):
            fr = fm.loc[qid]
            lr = lm[(lm["query_id"] == qid) & (lm["ablation"] == pos)].iloc[0]
            l1 = (abs(lr["p_normal"] - fr["p_normal"]) +
                  abs(lr["p_supplier_delay"] - fr["p_supplier_delay"]) +
                  abs(lr["p_demand_surge"] - fr["p_demand_surge"]))
            eo = em[(em["query_id"] == qid) & (em["ablation"] == pos)].set_index("seed")["orders"]
            fo = em[(em["query_id"] == qid) & (em["ablation"] == "full")].set_index("seed")["orders"] \
                if len(em[(em["query_id"] == qid) & (em["ablation"] == "full")]) else None
            assert fo is not None and len(fo) == 5 and len(eo) == 5, (qid, pos)
            adis = float(np.mean([np.abs(np.array(eo[s]) - np.array(fo[s])).mean()
                                  for s in eo.index]))
            d_f = float((fm_ep[fm_ep["query_id"] == qid].set_index("seed")["profit"] -
                         ni.loc[qid]).mean())
            d_l = float((em[(em["query_id"] == qid) & (em["ablation"] == pos)]
                         .set_index("seed")["profit"] - ni.loc[qid]).mean())
            per_q.append({"l1": float(l1), "adis": adis, "voi": d_f - d_l})
        pq = pd.DataFrame(per_q)
        cell = {}
        for col in ("l1", "adis", "voi"):
            v = pq[col].to_numpy()
            boots = np.array([v[rng.choice(len(v), len(v), replace=True)].mean()
                              for _ in range(B)])
            cell[col] = {"mean": round(float(v.mean()), 3),
                         "CI95": [round(float(np.percentile(boots, 2.5)), 3),
                                  round(float(np.percentile(boots, 97.5)), 3)]}
        cell["fire_a_voi_excl0"] = bool(cell["voi"]["CI95"][0] > 0 or
                                        cell["voi"]["CI95"][1] < 0)
        cell["fire_b_adis_gt0"] = bool(cell["adis"]["CI95"][0] > 0)
        key = f"{model.split('/')[-1]}/{pos}"
        cells[key] = cell
        print(key, {c: cells[key][c] for c in ("l1", "adis", "voi")})

fire = [k for k, c in cells.items()
        if c["fire_a_voi_excl0"] and c["fire_b_adis_gt0"]]
verdict = {"cells": cells,
           "FIRE": bool(fire), "fire_cells": fire,
           "design": {"B": B, "seed": SEED, "n_dev": 40, "test": "sealed, untouched"}}
json.dump(verdict, open(OUT / "g2a1_verdict.json", "w"), indent=1)
print("FIRE:", verdict["FIRE"], verdict["fire_cells"])
