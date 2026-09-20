"""G2A-0 verdict (CPU-only analysis; no new episodes, no LLM calls).

Inputs:
  results/v3main/extension_g2a/loo_episodes.parquet (fresh replay)
  frozen episodes.parquet (NoInfo rung), utility_by_query*.parquet (x-check),
  frozen utility_by_arm tables (intervention contrasts),
  data/v3/splits/splits.json (dev-40 / test-160).
Outputs results/v3main/extension_g2a/g2a_verdict.json:
  (1) LOO document-VoI per (stratum, drop-position): VoI = mean_q[dJ_full -
      dJ_loo], dJ = mean_seed(profit - profit_NoInfo), paired bootstrap
      B=5000 seed 72000, dev-40 section carries the pilot-signal decision,
      test-160 section is the once-only verdict;
  (2) frozen intervention contrasts (oracle-factual base) as precedent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main" / "extension_g2a"

STRATA = [("C0", "none"), ("C3", "Qwen/Qwen3-8B-AWQ"), ("C3", "Qwen/Qwen3-14B-AWQ")]
POS = ["drop0", "drop1", "drop2"]
B = 5000
SEED = 72000

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids, test_ids = set(sp["dev"]), set(sp["test"])

ep = pd.read_parquet(OUT / "loo_episodes.parquet")
froz = pd.read_parquet(ROOT / "results" / "v3main" / "sim" / "episodes.parquet")
ni = froz[(froz["system"] == "noinfo") & (froz["k"] == 3)].set_index(
    ["query_id", "seed"])["profit"]

d = ep.merge(ni.rename("profit_ni"), on=["query_id", "seed"])
d["dj"] = d["profit"] - d["profit_ni"]
# per (stratum, query, ablation) mean over seeds
g = d.groupby(["consumer", "model", "query_id", "ablation"])["dj"].mean().reset_index()
full = g[g["ablation"] == "full"].set_index(["consumer", "model", "query_id"])["dj"]
voi_rows = []
for _, r in g[g["ablation"] != "full"].iterrows():
    key = (r["consumer"], r["model"], r["query_id"])
    voi_rows.append({"consumer": r["consumer"], "model": r["model"],
                     "query_id": r["query_id"], "position": r["ablation"],
                     "voi": float(full.loc[key] - r["dj"]),
                     "split": "dev" if r["query_id"] in dev_ids else "test"})
voi = pd.DataFrame(voi_rows)
assert len(voi) == 3 * 200 * 3, f"unexpected voi rows {len(voi)}"

rng = np.random.default_rng(SEED)


def boot_ci(vals: np.ndarray):
    n = len(vals)
    boots = np.array([vals[rng.choice(n, n, replace=True)].mean() for _ in range(B)])
    return (round(float(vals.mean()), 2),
            [round(float(np.percentile(boots, 2.5)), 2),
             round(float(np.percentile(boots, 97.5)), 2)])


def section(df: pd.DataFrame):
    out = {}
    for (cons, model), gs in df.groupby(["consumer", "model"]):
        key = f"{cons}/{model.split('/')[-1]}"
        out[key] = {}
        for p in POS:
            v = gs[gs["position"] == p]["voi"].to_numpy()
            mean, ci = boot_ci(v)
            out[key][p] = {"mean": mean, "CI95": ci,
                           "excludes0": bool(ci[0] > 0 or ci[1] < 0),
                           "n": len(v)}
    return out


dev_sec = section(voi[voi["split"] == "dev"])
test_sec = section(voi[voi["split"] == "test"])

# Pilot-signal rule (dev-40, DESIGN-G2A §3): (i) any per-position CI excludes 0
# in >=1 stratum, or (ii) max-min per-position pooled VoI > pooled noise width.
sig_i = any(v["excludes0"] for s in dev_sec.values() for v in s.values())
pool_dev = voi[voi["split"] == "dev"].groupby("position")["voi"].mean()
pv = pool_dev.to_dict()
boots = []
for _ in range(B):
    samp = voi[voi["split"] == "dev"].groupby("position")["voi"].apply(
        lambda s: s.sample(frac=1.0, replace=True, random_state=rng).mean())
    boots.append(samp.max() - samp.min())
noise_w = float(np.percentile(boots, 95))
spread = float(max(pv.values()) - min(pv.values()))
pilot = {"rule_i_any_excludes0": bool(sig_i),
         "pooled_dev_means": {k: round(float(v), 2) for k, v in pv.items()},
         "spread": round(spread, 2), "noise95": round(noise_w, 2),
         "rule_ii_spread_gt_noise": bool(spread > noise_w)}
pilot["PROPOSE_2A1"] = bool(sig_i or spread > noise_w)

# Frozen intervention contrasts (precedent, oracle-factual base, k=3)
arms = ["oracle-factual", "intervention-drop-decisive", "intervention-stale-swap",
        "intervention-inject-contra", "intervention-wrong-entity-swap",
        "intervention-irrelevant-inject", "intervention-duplicate-top",
        "intervention-order-reverse"]
ub_dev = pd.read_parquet(
    ROOT / "results/v3main/sim/utility_by_arm_balanced_devinclusive.parquet")
ub_test = pd.read_parquet(ROOT / "results/v3main/sim/utility_by_arm_balanced.parquet")
interv = {}
for tag, ub in (("dev", ub_dev), ("test", ub_test)):
    u = ub[(ub["system"].isin(arms)) & (ub["k"] == 3)]
    for (cons, model), gs in u.groupby(["consumer", "model"]):
        key = f"{tag}/{cons}/{model.split('/')[-1]}"
        base = float(gs[gs["system"] == "oracle-factual"]["J"].iloc[0])
        interv[key] = {s: round(float(
            gs[gs["system"] == s]["J"].iloc[0]) - base, 2) for s in arms[1:]}
        interv[key]["oracle-factual_d_vs_noinfo"] = round(float(
            gs[gs["system"] == "oracle-factual"]["delta_vs_noinfo"].iloc[0]), 2)

verdict = {"loo_dev": dev_sec, "loo_test": test_sec, "pilot_signal": pilot,
           "interventions_vs_oraclefactual": interv,
           "design": {"B": B, "seed": SEED, "systems": ["rerank"], "k": 3,
                      "voi_def": "J(full)-J(minus-i) per query, mean over queries"}}
json.dump(verdict, open(OUT / "g2a_verdict.json", "w"), indent=1)
print(json.dumps({"pilot_signal": pilot}, indent=1))
print("test pooled means:",
      json.dumps({s: {p: test_sec[s][p]["mean"] for p in POS} for s in test_sec}))
