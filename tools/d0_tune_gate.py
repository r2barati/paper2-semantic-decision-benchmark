"""D0 Gate-1 tuning (DEV-ONLY). Asserts it never loads test data.

Inputs (dev only):
  data/v3/splits/splits.json (dev ids)
  results/v3main/sim/utility_by_query_devinclusive.parquet (dev slice)
  results/v3main/semantic/semantic_all.parquet (split==dev, k==3)
Protocol (DESIGN-D0.md, frozen):
  per stratum: S-J = argmax dev plain-mean dJ over {bm25,dense,hybrid,rerank};
  G1-arm: transmit S-J iff dev mean > 0;
  G1-feat (Head-B primary): within S-J system, transmit q iff Brier_q < t,
    t in {0.10,0.20,0.30,0.40,0.50}, argmax dev gated mean, ties->smallest t,
    OFF if best <= 0.
Writes results/v3main/extension_d0/d0_tune.json. No test touch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_d0"
OUT.mkdir(parents=True, exist_ok=True)

RETR = ["bm25", "dense", "hybrid", "rerank"]
STRATA = [("C0", "none"), ("C1", "Qwen/Qwen3-8B-AWQ"),
          ("C1", "Qwen/Qwen3-14B-AWQ"), ("C3", "Qwen/Qwen3-8B-AWQ"),
          ("C3", "Qwen/Qwen3-14B-AWQ")]
K = 3
TGRID = [0.10, 0.20, 0.30, 0.40, 0.50]

# --- hard dev-only guard: refuse to even reference test-only files
FORBIDDEN = ["utility_by_query.parquet", "ir_test", "agg_semantic_test"]
for f in FORBIDDEN:
    assert f not in sys.argv, f"dev script must not take test file {f}"

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids = set(sp["dev"])
assert len(dev_ids) == 40, f"expected 40 dev ids, got {len(dev_ids)}"

uq = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_query_devinclusive.parquet")
uqd = uq[(uq["query_id"].isin(dev_ids)) & (uq["system"].isin(RETR)) & (uq["k"] == K)].copy()
assert len(uqd) == 5 * 4 * 40, f"unexpected dev uq rows {len(uqd)}"

sem = pd.read_parquet(ROOT / "results/v3main/semantic/semantic_all.parquet")
semd = sem[(sem["split"] == "dev") & (sem["system"].isin(RETR)) & (sem["k"] == K)].copy()
assert set(semd["query_id"].unique()) <= dev_ids, "dev sem leaked non-dev queries"
assert (semd["split"] == "dev").all()

tune: dict = {"strata": {}, "grid": TGRID, "tau_arm": 0, "k": K}
for cons, model in STRATA:
    key = f"{cons}/{model}"
    g = uqd[(uqd["consumer"] == cons) & (uqd["model"] == model)]
    devmean = g.groupby("system")["delta_vs_noinfo"].mean()
    sj = str(devmean.idxmax())
    sj_mean = float(devmean.max())
    arm_on = bool(sj_mean > 0)
    # Brier-gate within S-J system
    b = semd[(semd["consumer"] == cons) & (semd["model"] == model)
             & (semd["system"] == sj)][["query_id", "brier"]].drop_duplicates()
    d = g[g["system"] == sj][["query_id", "delta_vs_noinfo"]].drop_duplicates()
    m = d.merge(b, on="query_id", how="inner")
    assert len(m) == 40, f"{key}: expected 40 dev queries joined, got {len(m)}"
    best = {"t": None, "dev_gated": 0.0}
    cands = {}
    for t in TGRID:
        gated = m["delta_vs_noinfo"].where(m["brier"] < t, 0.0).mean()
        cands[str(t)] = round(float(gated), 4)
        if gated > best["dev_gated"] + 1e-12:
            best = {"t": t, "dev_gated": float(gated)}
    feat_on = best["t"] is not None and best["dev_gated"] > 0
    tune["strata"][key] = {
        "S-J": sj,
        "dev_means": {s: round(float(devmean[s]), 2) for s in devmean.index},
        "S-J_dev_mean": round(sj_mean, 2),
        "arm_on": arm_on,
        "arm_dev_gated": round(sj_mean if arm_on else 0.0, 2),
        "feat_t": best["t"] if feat_on else None,
        "feat_on": feat_on,
        "feat_dev_gated": round(best["dev_gated"], 2) if feat_on else 0.0,
        "feat_candidates": cands,
    }
    print(key, tune["strata"][key])

json.dump(tune, open(OUT / "d0_tune.json", "w"), indent=1)
print("wrote", OUT / "d0_tune.json (DEV ONLY, no test touched)")
