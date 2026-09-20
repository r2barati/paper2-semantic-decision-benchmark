"""D0 Gate-1 evaluation (SINGLE TEST TOUCH). Takes frozen d0_tune.json as input.

Inputs (test, loaded once here and only here in D0):
  data/v3/splits/splits.json (test ids)
  results/v3main/sim/utility_by_query.parquet (test-only per-query dJ)
  results/v3main/semantic/semantic_all.parquet (split==test, k==3, Brier)
  results/v3main/extension_d0/d0_tune.json (frozen dev decisions)
Protocol (DESIGN-D0.md): per stratum apply frozen feat_t (Head-B primary)
  and arm_on (secondary); pooled = mean of stratum means; paired test-query
  bootstrap B=5000 seed 71000. HOLDS iff 95% CI lower > 0.
Writes results/v3main/extension_d0/d0_verdict.json. No re-tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_d0"

RETR = ["bm25", "dense", "hybrid", "rerank"]
STRATA = [("C0", "none"), ("C1", "Qwen/Qwen3-8B-AWQ"),
          ("C1", "Qwen/Qwen3-14B-AWQ"), ("C3", "Qwen/Qwen3-8B-AWQ"),
          ("C3", "Qwen/Qwen3-14B-AWQ")]
K = 3
B = 5000
SEED = 71000

tune = json.load(open(OUT / "d0_tune.json"))
sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
test_ids = sorted(set(sp["test"]))
assert len(test_ids) == 160

# --- SINGLE TEST TOUCH (all test loads in D0 happen in this block) ---
uqt = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_query.parquet")
uqt = uqt[(uqt["system"].isin(RETR)) & (uqt["k"] == K)
          & (uqt["query_id"].isin(test_ids))].copy()
sem = pd.read_parquet(ROOT / "results/v3main/semantic/semantic_all.parquet")
semt = sem[(sem["split"] == "test") & (sem["system"].isin(RETR))
           & (sem["k"] == K) & (sem["query_id"].isin(test_ids))].copy()
assert len(uqt) == 5 * 4 * 160, f"unexpected test uq rows {len(uqt)}"
# --- end single touch; everything below is resampling of these frames ---

per_q: dict[str, pd.DataFrame] = {}
for cons, model in STRATA:
    key = f"{cons}/{model}"
    sj = tune["strata"][key]["S-J"]
    d = uqt[(uqt["consumer"] == cons) & (uqt["model"] == model)
            & (uqt["system"] == sj)][["query_id", "delta_vs_noinfo"]]
    b = semt[(semt["consumer"] == cons) & (semt["model"] == model)
             & (semt["system"] == sj)][["query_id", "brier"]].drop_duplicates()
    m = d.merge(b, on="query_id", how="inner")
    assert len(m) == 160, f"{key}: joined {len(m)} rows, expected 160"
    per_q[key] = m.set_index("query_id")

t_frozen = {k: tune["strata"][k]["feat_t"] for k, _ in
            [(f"{c}/{m}", None) for c, m in STRATA]}
arm_on = {k: tune["strata"][k]["arm_on"] for k, _ in
          [(f"{c}/{m}", None) for c, m in STRATA]}


def stratum_gated(key: str, qids) -> float:
    m = per_q[key].loc[qids]
    t = t_frozen[key]
    if t is None:
        return 0.0
    return float(m["delta_vs_noinfo"].where(m["brier"] < t, 0.0).mean())


def stratum_arm(key: str, qids) -> float:
    if not arm_on[key]:
        return 0.0
    return float(per_q[key].loc[qids]["delta_vs_noinfo"].mean())


rng = np.random.default_rng(SEED)
tids = np.array(test_ids)
boot_feat = np.empty(B)
boot_arm = np.empty(B)
for i in range(B):
    qs = rng.choice(tids, len(tids), replace=True)
    boot_feat[i] = float(np.mean([stratum_gated(k, qs) for k in per_q]))
    boot_arm[i] = float(np.mean([stratum_arm(k, qs) for k in per_q]))

obs_feat = float(np.mean([stratum_gated(k, tids) for k in per_q]))
obs_arm = float(np.mean([stratum_arm(k, tids) for k in per_q]))
ci_feat = [round(float(np.percentile(boot_feat, 2.5)), 2),
           round(float(np.percentile(boot_feat, 97.5)), 2)]
ci_arm = [round(float(np.percentile(boot_arm, 2.5)), 2),
          round(float(np.percentile(boot_arm, 97.5)), 2)]

# per-stratum secondaries with Holm over 5 (one-sided H0: gated <= 0)
secondaries = {}
pvals = []
for key in per_q:
    m = per_q[key]
    t = t_frozen[key]
    vals = m["delta_vs_noinfo"].where(m["brier"] < t, 0.0).to_numpy() \
        if t is not None else np.zeros(len(m))
    obs = float(vals.mean())
    # bootstrap within stratum using the same seed stream (independent draws)
    r2 = np.random.default_rng(SEED + hash(key) % 100000)
    idx = np.arange(len(vals))
    boots = np.array([vals[r2.choice(idx, len(idx), replace=True)].mean()
                      for _ in range(B)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    p = float((boots <= 0).mean())
    pvals.append((key, p))
    secondaries[key] = {"feat_t": t, "obs_gated_dJ": round(obs, 2),
                        "CI95": [round(lo, 2), round(hi, 2)], "p_onesided": p}
# Holm
pvals.sort(key=lambda x: x[1])
m_tests = len(pvals)
holm = {}
prev = 0.0
for rank, (key, p) in enumerate(pvals, start=1):
    adj = min(1.0, p * (m_tests - rank + 1))
    adj = max(adj, prev)
    prev = adj
    holm[key] = round(adj, 4)
for key in secondaries:
    secondaries[key]["p_holm"] = holm[key]

verdict = {
    "primary_HeadB_feat": {"obs_pooled_dJ": round(obs_feat, 2), "CI95": ci_feat,
                           "holds": bool(ci_feat[0] > 0)},
    "secondary_arm": {"obs_pooled_dJ": round(obs_arm, 2), "CI95": ci_arm,
                      "holds": bool(ci_arm[0] > 0)},
    "per_stratum": secondaries,
    "design": {"B": B, "seed": SEED, "k": K, "n_test": 160,
               "metric": "plain-mean per-query dJ vs NoInfo; pooled=mean of stratum means"},
}
json.dump(verdict, open(OUT / "d0_verdict.json", "w"), indent=1)
print(json.dumps(verdict, indent=1))
