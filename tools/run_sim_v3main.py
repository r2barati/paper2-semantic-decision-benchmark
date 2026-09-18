"""V3 main sequential-decision evaluation (gated: requires frozen beliefs).

Phase 1 (--phase episodes): every frozen belief row x paired fresh seeds
(60000-60004, n=5; pilot variance ratio query:seed = 6.3:1 so n=10 buys
~1.5% — see results/v3main/sim_pilot/pilot_summary.json) through
belief -> frozen CausalOptimizer -> controlled InventoryEnv, plus NoInfo /
PerfectBelief / HindsightOracle rungs per query. Checkpoint:
results/v3main/sim/episodes.parquet.

Phase 2 (--phase analysis): J via weighted_benchmark_return (balanced,
regime->main->query->seed), paired effects vs NoInfo with crossed bootstrap
CIs, SIVR vs PerfectBelief with the signed guard, 4-gap decomposition,
query-level utility table, SIM_REPORT.md. No LLM calls anywhere here.

Usage:
  python3 tools/run_sim_v3main.py --phase episodes   # ~3h on 7 workers
  python3 tools/run_sim_v3main.py --phase analysis
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sim_eval_v3main import require_frozen_beliefs, belief_row_to_interpretation

BELIEFS = ROOT / "results" / "v3main"
OUT = BELIEFS / "sim"
SEEDS = [60000 + i for i in range(5)]
N_BOOT = 5000
RETRIEVAL_SYSTEMS = ["bm25", "dense", "hybrid", "rerank", "random"]


def rung_of(system, kind):
    if kind == "noinfo":
        return "NoInfo"
    if kind == "perfect":
        return "PerfectBelief"
    if kind == "hindsight":
        return "HindsightOracle"
    if system == "oracle-relevant":
        return "OracleRelevant"
    if system == "oracle-factual":
        return "OracleFactual"
    if system in RETRIEVAL_SYSTEMS:
        return "ActualRetrieval"
    return "intervention"


def _work(item):
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.experiment_phase5 import _run_p5_episode
    from src.events import Regime
    from src.interpreter import RegimeInterpretation, no_info_regime_belief
    (kind, sys_name, cons, model, k, qid, true_regime, probs, seed) = item
    t0 = time.time()
    if kind == "hindsight":
        from src.env import HindsightOracle
        h = HindsightOracle(seed=seed, regime=Regime(true_regime))
        hist, _ = h.compute_optimal_trajectory()
        val = float(hist[-1].cumulative_profit)
    else:
        if kind == "noinfo":
            interp = no_info_regime_belief()
        elif kind == "perfect":
            interp = RegimeInterpretation(
                regime_probabilities={r: 1.0 if r == true_regime else 0.0
                                      for r in ("normal", "supplier_delay",
                                                "demand_surge")})
        else:
            interp = belief_row_to_interpretation({
                "p_normal": probs[0], "p_supplier_delay": probs[1],
                "p_demand_surge": probs[2], "abstain": False,
                "confidence": 1.0, "doc_ids": []})
        r, _ = _run_p5_episode(seed=seed, regime=Regime(true_regime),
                               sensor=f"{sys_name}+{cons}", controller="CausalOptimizer",
                               regime_interp=interp)
        val = float(r.total_profit)
    return {"system": sys_name, "consumer": cons, "model": model, "k": k,
            "query_id": qid, "true_regime": true_regime, "seed": seed,
            "rung": rung_of(sys_name, kind), "profit": val,
            "wall_s": round(time.time() - t0, 3)}


def load_beliefs_dedup():
    bel = pd.concat([pd.read_parquet(BELIEFS / f"beliefs_{m}.parquet")
                     for m in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ")],
                    ignore_index=True)
    c0 = bel[bel["consumer"] == "C0"]
    keys = ["system", "query_id", "k"]
    grp = c0.groupby(keys)
    assert (grp.size() == 2).all(), "every C0 key must appear exactly twice"
    scalar = [c for c in c0.columns if c not in ("doc_ids", "latency_s", "model")]
    nun = grp[scalar].nunique()
    assert bool((nun == 1).all().all()), "C0 twin rows differ!"
    bel = pd.concat([c0.sort_values(keys).drop_duplicates(keys, keep="first"),
                     bel[bel["consumer"] != "C0"]], ignore_index=True)
    bel["model"] = bel["model"].fillna("none")
    return bel


def phase_episodes():
    require_frozen_beliefs(BELIEFS)
    OUT.mkdir(parents=True, exist_ok=True)
    bel = load_beliefs_dedup()
    qreg = dict(bel[["query_id", "true_regime"]].drop_duplicates().values)
    items = []
    for _, r in bel.iterrows():
        probs = (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"])
        for seed in SEEDS:
            items.append(("belief", r["system"], r["consumer"], r["model"],
                          int(r["k"]), r["query_id"], r["true_regime"],
                          probs, seed))
    for qid, true_regime in sorted(qreg.items()):
        for kind in ("noinfo", "perfect", "hindsight"):
            for seed in SEEDS:
                items.append((kind, kind, "C0", "none", 3, qid,
                              true_regime, None, seed))
    print(f"episode items: {len(items)}", flush=True)
    t0 = time.time()
    done, rows = 0, []
    with Pool(7) as pool:
        for row in pool.imap_unordered(_work, items, chunksize=8):
            rows.append(row)
            done += 1
            if done % 10000 == 0:
                print(f"  {done}/{len(items)} "
                      f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "episodes.parquet", index=False)
    print(f"episodes: {len(df)} rows in {(time.time() - t0) / 60:.1f} min")


def nest_J(df, regime_weights=None):
    """regime -> main -> query -> {seed: profit} for weighted_benchmark_return."""
    from src.metrics import weighted_benchmark_return
    idx = {}
    for (reg, qid, seed), g in df.groupby(["true_regime", "query_id", "seed"]):
        idx.setdefault(reg, {}).setdefault("main", {}).setdefault(
            qid, {})[seed] = float(g["profit"].mean())
    return weighted_benchmark_return(idx, regime_weights=regime_weights)["value"]


WEIGHT_VECTORS = {
    "balanced": None,
    "empirical": {"normal": 0.2, "supplier_delay": 0.4, "demand_surge": 0.4},
}


def qid_to_text():
    import hashlib
    import json as _j
    m = {}
    for line in open(ROOT / "data" / "v3" / "queries.jsonl"):
        q = _j.loads(line)
        m[q["_id"]] = hashlib.sha256(q["text"].encode()).hexdigest()[:16]
    return m


def cluster_diff_ci(diff_nesting, q2t, regime_weights=None, n_boot=5000,
                    seed=123):
    """Text-cluster bootstrap honesty check for one arm's paired diff.

    Resamples 64 text clusters (queries nested, multiplicity-weighted) with
    shared seed draws per regime — the same balanced-J_diff estimator, so any
    CI widening vs crossed_bootstrap_ci is pure clustering honesty, not a new
    estimand. Returns percentile CI + centred p (floor 1/(B+1)).
    """
    rng = np.random.default_rng(seed)
    w = regime_weights or {}
    regs = sorted(diff_nesting)
    W = {r: float(w.get(r, 1.0 / len(regs))) for r in regs}
    tot = sum(W.values())
    W = {r: W[r] / tot for r in regs}
    per_reg = {}
    for reg in regs:
        cells = diff_nesting[reg]["main"]
        qids = sorted(cells)
        seeds = sorted({s for q in qids for s in cells[q]})
        mat = np.array([[cells[q].get(s, np.nan) for s in seeds] for q in qids])
        texts = np.array([q2t[q] for q in qids])
        uniq = sorted(set(texts.tolist()))
        ti = np.array([uniq.index(t) for t in texts])
        per_reg[reg] = (mat, ti, seeds, uniq)
    stats = []
    for _ in range(n_boot):
        tot_m = 0.0
        for reg in regs:
            mat, ti, seeds, uniq = per_reg[reg]
            tdraw = rng.integers(0, len(uniq), size=len(uniq))
            sdraw = rng.integers(0, len(seeds), size=len(seeds))
            # multiplicity-weighted mean over resampled (text, seed) cells
            acc, wt = 0.0, 0
            for t in tdraw:
                rows = mat[ti == t][:, sdraw]
                acc += float(np.nansum(rows))
                wt += int(np.isfinite(rows).sum())
            tot_m += W[reg] * (acc / wt if wt else np.nan)
        stats.append(tot_m)
    stats = np.array(stats)
    # weighted point to match estimator
    point = 0.0
    for reg in regs:
        cells = diff_nesting[reg]["main"]
        vals = [v for q in cells.values() for v in q.values()]
        point += W[reg] * float(np.mean(vals))
    lo, hi = float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
    p = float((1 + np.sum(np.abs(stats - point) >= abs(point))) / (n_boot + 1))
    return {"cluster_ci_lo": lo, "cluster_ci_hi": hi, "cluster_p": p,
            "cluster_wider_by": float((hi - lo))}


def nest_diff(df_arm, df_ref):
    """Paired (query, seed) differences arm - ref, nested for crossed bootstrap."""
    a = df_arm.set_index(["query_id", "seed"])["profit"]
    b = df_ref.set_index(["query_id", "seed"])["profit"]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).reset_index()
    d = d.merge(df_arm[["query_id", "true_regime"]].drop_duplicates(),
                on="query_id")
    out = {}
    for reg, g in d.groupby("true_regime"):
        out.setdefault(reg, {}).setdefault("main", {})["q"] = dict(
            zip(g["seed"].astype(int), g["profit"].astype(float)))
    # variant must be query-level: rebuild per query
    out = {}
    for reg, g in d.groupby("true_regime"):
        for qid, gg in g.groupby("query_id"):
            out.setdefault(reg, {}).setdefault("main", {})[qid] = dict(
                zip(gg["seed"].astype(int), gg["profit"].astype(float)))
    return out


def _ana_work(task):
    """One (arm, weight-vector) analysis unit (picklable for Pool)."""
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from src.metrics import crossed_bootstrap_ci, signed_sivr
    import tools.run_sim_v3main as _M
    (arm, diff, J, J_noinfo, J_perfect, J_oraculum, W, q2t) = task
    ci = crossed_bootstrap_ci(diff, regime_weights=W, n_boot=N_BOOT, seed=42)
    cl = _M.cluster_diff_ci(diff, q2t, regime_weights=W, n_boot=N_BOOT)
    sv = signed_sivr(J, J_noinfo, J_perfect).as_dict()
    svF = signed_sivr(J, J_noinfo, J_oraculum).as_dict()
    return {**arm, "J": J, "delta_vs_noinfo": J - J_noinfo,
            "ci_lo": ci["ci_lower"], "ci_hi": ci["ci_upper"],
            "p_value": ci["p_value"], "p_floor": ci["p_value_floor"],
            "cluster_ci_lo": cl["cluster_ci_lo"],
            "cluster_ci_hi": cl["cluster_ci_hi"],
            "cluster_p": cl["cluster_p"],
            **{f"sivr_{k2}": v for k2, v in sv.items()},
            **{f"sivrF_{k2}": v for k2, v in svF.items()}}


def phase_analysis():
    from src.metrics import crossed_bootstrap_ci, signed_sivr
    import src.semantic_analysis_v3main as _S
    import json as _j
    df = pd.read_parquet(OUT / "episodes.parquet")
    # Headline estimands are TEST-ONLY (160 held-out queries). Dev queries
    # remain in episodes.parquet for coverage; dev-inclusive tables are
    # archived as *_devinclusive.* (review B-F2).
    _splits = _j.loads((ROOT / "data" / "v3" / "splits" / "splits.json").read_text())
    _test = set(_splits["test"])
    df = df[df["query_id"].isin(_test)].reset_index(drop=True)
    print(f"analysis queries: {df['query_id'].nunique()} (test-only)", flush=True)
    ref = df[df["rung"] == "NoInfo"]
    perf = df[df["rung"] == "PerfectBelief"]
    hind = df[df["rung"] == "HindsightOracle"]
    q2t = qid_to_text()
    arms = df[df["rung"].isin(
        ["ActualRetrieval", "OracleRelevant", "OracleFactual", "intervention"])]
    # Second SIVR reference: OracleFactual J matched on (consumer, model, k).
    J_oraclefactual = {}
    for wname, W in WEIGHT_VECTORS.items():
        J_oraclefactual[wname] = {}
        for vals, g in arms[arms["system"] == "oracle-factual"].groupby(
                ["consumer", "model", "k"]):
            J_oraclefactual[wname][tuple(vals)] = nest_J(g, W)
    for wname, W in WEIGHT_VECTORS.items():
        J_noinfo = nest_J(ref, W)
        J_perfect = nest_J(perf, W)
        J_hind = nest_J(hind, W)
        print(f"[{wname}] J_noinfo={J_noinfo:.2f} J_perfect={J_perfect:.2f} "
              f"J_hind={J_hind:.2f}", flush=True)
        keys = ["system", "consumer", "model", "k"]
        tasks = []
        for vals, g in arms.groupby(keys):
            arm = dict(zip(keys, vals))
            arm["n"] = len(g)
            tasks.append((arm, nest_diff(g, ref), nest_J(g, W), J_noinfo,
                          J_perfect, J_oraclefactual[wname][
                              (arm["consumer"], arm["model"], arm["k"])],
                          W, q2t))
        print(f"[{wname}] {len(tasks)} arms x boot parallel start", flush=True)
        from multiprocessing import Pool
        with Pool(7) as pool:
            rows = pool.map(_ana_work, tasks, chunksize=1)
        util = pd.DataFrame(rows)
        holm = _S.holm_correction(util["p_value"].to_numpy(dtype=float))
        holm.insert(0, "arm", util["system"] + "/" + util["consumer"] + "/"
                    + util["model"].str[-8:] + "/k" + util["k"].astype(str))
        util["p_holm"] = holm["p_holm"].to_numpy()
        util["reject_holm_05"] = holm["reject_holm_05"].to_numpy()
        util.to_parquet(OUT / f"utility_by_arm_{wname}.parquet", index=False)
        util.to_json(OUT / f"utility_by_arm_{wname}.json", orient="records",
                     indent=1)
        holm.to_parquet(OUT / f"holm_sim_arms_{wname}.parquet", index=False)
        print(f"[{wname}] arms: {len(util)} holm-reject: "
              f"{int(util['reject_holm_05'].sum())}", flush=True)
    import shutil
    shutil.copy(OUT / "utility_by_arm_balanced.parquet",
                OUT / "utility_by_arm.parquet")
    shutil.copy(OUT / "utility_by_arm_balanced.json",
                OUT / "utility_by_arm.json")
    # Regime-conditional exploratory CIs (balanced J only; seeds+queries pooled
    # within regime; NOT multiplicity-controlled; labels exploratory per review).
    import src.semantic_analysis_v3main as _Sx
    M8 = "Qwen/Qwen3-8B-AWQ"
    M14 = "Qwen/Qwen3-14B-AWQ"
    reg_rows = []
    EXP_ARMS = [("rerank", "C0", "none", 3), ("rerank", "C1", M8, 3),
                ("rerank", "C3", M8, 3), ("rerank", "C1", M14, 3),
                ("rerank", "C3", M14, 3), ("bm25", "C1", M8, 3),
                ("oracle-factual", "C3", M8, 3)]
    for (s, c, m, k) in EXP_ARMS:
        g = arms[(arms["system"] == s) & (arms["consumer"] == c)
                 & (arms["model"] == m) & (arms["k"] == k)]
        for reg, gg in g.groupby("true_regime"):
            rr = ref[ref["true_regime"] == reg]
            a = gg.set_index(["query_id", "seed"])["profit"]
            b = rr.set_index(["query_id", "seed"])["profit"]
            common = a.index.intersection(b.index)
            d = (a.loc[common] - b.loc[common]).to_numpy()
            rng = np.random.default_rng(11)
            boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
            reg_rows.append({"system": s, "consumer": c, "model": m, "k": k,
                             "regime": reg, "n": len(d),
                             "delta": float(d.mean()),
                             "ci_lo": float(np.percentile(boots, 2.5)),
                             "ci_hi": float(np.percentile(boots, 97.5)),
                             "exploratory": True})
    pd.DataFrame(reg_rows).to_parquet(OUT / "regime_exploratory.parquet",
                                      index=False)
    print(f"regime exploratory rows: {len(reg_rows)}", flush=True)

    # query-level utility table (mean over seeds, paired vs NoInfo)
    qutil = []
    ref_q = ref.groupby(["query_id", "seed"])["profit"].mean()
    for vals, g in arms.groupby(keys + ["query_id"]):
        d = dict(zip(keys + ["query_id"], vals))
        a = g.set_index("seed")["profit"]
        b = ref_q.loc[d["query_id"]]
        common = a.index.intersection(b.index)
        qutil.append({**d, "n_seeds": len(common),
                      "delta_vs_noinfo": float((a.loc[common] - b.loc[common]).mean())})
    qu = pd.DataFrame(qutil)
    qu.to_parquet(OUT / "utility_by_query.parquet", index=False)
    print(f"query utility rows: {len(qu)}")

    # 4-gap decomposition per weight vector (headline: rerank k=3;
    # per consumer/model + C0). Canonical four_gaps.* = balanced.
    def J_of(system, consumer, model, k, W):
        g = arms[(arms["system"] == system) & (arms["consumer"] == consumer)
                 & (arms["model"] == model) & (arms["k"] == k)]
        return nest_J(g, W)

    for wname, W in WEIGHT_VECTORS.items():
        Jn = nest_J(ref, W)
        Jp = nest_J(perf, W)
        Jh = nest_J(hind, W)
        gaps = []
        for model in ("none", "Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"):
            cons_list = ["C0"] if model == "none" else ["C1", "C3"]
            for cons in cons_list:
                j_ret = J_of("rerank", cons, model, 3, W)
                j_rel = J_of("oracle-relevant", cons, model, 3, W)
                j_fac = J_of("oracle-factual", cons, model, 3, W)
                gaps.append({"model": model, "consumer": cons, "k": 3,
                             "J_noinfo": Jn, "J_retrieval": j_ret,
                             "J_oracle_relevant": j_rel,
                             "J_oracle_factual": j_fac,
                             "J_perfect": Jp, "J_hindsight": Jh,
                             "retrieval_gap": j_ret - Jn,
                             "relevance_to_factual_gap": j_fac - j_rel,
                             "interpretation_gap": Jp - j_fac,
                             "downstream_control_gap": Jh - Jp})
        pd.DataFrame(gaps).to_parquet(OUT / f"four_gaps_{wname}.parquet",
                                      index=False)
        pd.DataFrame(gaps).to_json(OUT / f"four_gaps_{wname}.json",
                                   orient="records", indent=1)
        if wname == "balanced":
            pd.DataFrame(gaps).to_parquet(OUT / "four_gaps.parquet",
                                          index=False)
            pd.DataFrame(gaps).to_json(OUT / "four_gaps.json",
                                       orient="records", indent=1)
        for g in gaps:
            print({k: (round(v, 2) if isinstance(v, float) else v)
                   for k, v in g.items()})
    print("DONE — sim outputs under results/v3main/sim/")


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
