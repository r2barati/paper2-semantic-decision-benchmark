"""DEV pilot runner for EXP-P2-REASONING-AGENTIC-v1 (local CPU; DEV only).

Reads 60 cached GPU beliefs (smoke 30 + dev2 30), adds local NoInfo/RuleBased/
PerfectSemantic references, replays through the frozen controller+simulator on
DEV seeds 62100-62114, and writes all dev_pilot/ outputs. Diagnostic only:
no arm selection, no tuning, no performance claims.
"""

from __future__ import annotations

import json
import math
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.reasoning_agentic.evaluate import accuracy, ece_10bin, entropy  # noqa: E402
from experiments.reasoning_agentic.run_simulation import (  # noqa: E402
    pathwise_equal, run_episode_with_history)
from experiments.reasoning_agentic.stats import (  # noqa: E402
    HEADLINE, crossed_bootstrap, mechanism_cell, paired_stats)
from src.confirmation_templates import CONFIRMATION_TEMPLATES  # noqa: E402
from src.events import Regime  # noqa: E402
from src.experiment_phase5_5 import false_positive_cost  # noqa: E402
from src.interpreter import (  # noqa: E402
    no_info_regime_belief, perfect_semantic_regime_belief, rule_based_regime_extract)
from src.metrics import signed_sivr, standard_brier_score  # noqa: E402

DEV = ROOT / "results" / "reasoning_agentic" / "dev_pilot"
ORDER = ["normal", "supplier_delay", "demand_surge"]
REFS = ("NoInfo", "RuleBased", "PerfectSemantic")
LLM_ARMS = ("A0", "A1", "A2", "R0", "A3")
ALL_ARMS = LLM_ARMS + REFS


def load_beliefs() -> tuple[dict, list, list]:
    beliefs, raw_rows, call_logs = {}, [], []
    for run in ("reasoning_agentic", "reasoning_agentic_dev2"):
        d = ROOT / "runs" / run
        man = json.loads((d / "smoke_manifest.json").read_text())
        assert man["n_failures"] == 0, run
        call_logs += man["call_logs"]
        with zipfile.ZipFile(d / "cache_reasoning_smoke.zip") as z:
            for n in z.namelist():
                if n.endswith(".json"):
                    beliefs[n[:-5]] = json.loads(z.read(n))
    for k, v in beliefs.items():
        wid, arm = k.split("||")
        raw_rows.append({"key": k, "warning_id": wid, "arm": arm, "belief": v["belief"],
                         "raw": {kk: vv for kk, vv in v.items() if kk.startswith("raw")},
                         "evidence_ids": v.get("evidence_ids", []),
                         "evidence_sha16": v.get("evidence_sha16", [])})
    return beliefs, raw_rows, call_logs


def main() -> int:
    t0 = time.time()
    DEV.mkdir(parents=True, exist_ok=True)
    dev_ids = json.loads((DEV / "DEV_MANIFEST.json").read_text())["dev_warning_ids"]
    seeds = list(range(62100, 62115))
    by_id = {t["template_id"]: t for t in CONFIRMATION_TEMPLATES}
    assert set(dev_ids) <= set(by_id)
    reg_of = {i: by_id[i]["regime"].value for i in dev_ids}

    beliefs, raw_rows, call_logs = load_beliefs()
    assert all(f"{w}||{a}" in beliefs for w in dev_ids for a in LLM_ARMS), "belief coverage"

    # Reference beliefs (local, deterministic, no LLM).
    for w in dev_ids:
        b = no_info_regime_belief().normalized().regime_probabilities
        beliefs[f"{w}||NoInfo"] = {"belief": {"normal": b["normal"],
                                              "supplier_delay": b["supplier_delay"],
                                              "demand_surge": b["demand_surge"]}}
        rb = rule_based_regime_extract(by_id[w]["text"]).normalized().regime_probabilities
        beliefs[f"{w}||RuleBased"] = {"belief": {"normal": rb["normal"],
                                                 "supplier_delay": rb["supplier_delay"],
                                                 "demand_surge": rb["demand_surge"]}}
        pb = perfect_semantic_regime_belief(Regime(reg_of[w])).normalized().regime_probabilities
        beliefs[f"{w}||PerfectSemantic"] = {"belief": {"normal": pb.get("normal", 0.0),
                                                       "supplier_delay": pb.get("supplier_delay", 0.0),
                                                       "demand_surge": pb.get("demand_surge", 0.0)}}

    # ---- simulation: 12 x 15 x 8 = 1440 episodes ----
    ep_rows = []
    actions: dict[str, dict] = {}
    for w in dev_ids:
        reg = Regime(reg_of[w])
        for s in seeds:
            for a in ALL_ARMS:
                row, _ = run_episode_with_history(beliefs[f"{w}||{a}"]["belief"], reg, s)
                ep_rows.append({"warning_id": w, "regime": reg_of[w], "seed": s, "arm": a,
                                "total_profit": row["total_profit"], "fill_rate": row["fill_rate"]})
                actions.setdefault(w, {}).setdefault(a, {})[s] = row["actions"]
    print(f"sim done: {len(ep_rows)} episodes in {time.time()-t0:.0f}s", flush=True)

    import pandas as pd
    ep = pd.DataFrame(ep_rows)
    ep.to_parquet(DEV / "episode_results.parquet", index=False)

    # ---- semantic metrics (frozen Brier; NLL/acc/entropy) ----
    sem = []
    for w in dev_ids:
        for a in ALL_ARMS:
            p = beliefs[f"{w}||{a}"]["belief"]
            t = reg_of[w]
            sem.append({"warning_id": w, "regime": t, "arm": a,
                        "brier": standard_brier_score(p, t, class_order=ORDER),
                        "nll": -math.log(max(p[t], 1e-9)),
                        "accuracy": accuracy(p, t), "entropy": entropy(p),
                        "confidence": max(p.values())})
    sem_df = pd.DataFrame(sem)
    arm_ece = {a: ece_10bin(list(sem_df[sem_df["arm"] == a]["confidence"]),
                            list(sem_df[sem_df["arm"] == a]["accuracy"])) for a in ALL_ARMS}
    sem_df["arm_ece_10bin"] = sem_df["arm"].map(arm_ece)  # n=12/arm: diagnostic only
    sem_df.to_csv(DEV / "semantic_metrics.csv", index=False)

    # ---- operational metrics ----
    piv = ep.pivot_table(index=["warning_id", "seed"], columns="arm", values="total_profit")
    ops = []
    for a in ALL_ARMS:
        sub = piv.reset_index()
        d_ni = (sub[a] - sub["NoInfo"]).mean()
        d_a0 = (sub[a] - sub["A0"]).mean()
        s = signed_sivr(float(sub[a].mean()), float(sub["NoInfo"].mean()),
                        float(sub["PerfectSemantic"].mean()))
        ops.append({"arm": a, "mean_profit": float(sub[a].mean()),
                    "delta_vs_NoInfo": float(d_ni), "delta_vs_A0": float(d_a0),
                    "sivr": s.value, "sivr_status": s.status,
                    "win_frac_vs_NoInfo": float((sub[a] > sub["NoInfo"]).mean()),
                    "win_frac_vs_A0": float((sub[a] > sub["A0"]).mean())})
    # FP/FN + per-regime need regime column: merge
    ep_r = ep.merge(sem_df[["warning_id", "regime"]].drop_duplicates(), on="warning_id",
                    suffixes=("", "_y"), how="left")
    fpcost = {a: {"fp_n": 0, "fn_n": 0, "fp_cost": 0.0, "fn_cost": 0.0} for a in ALL_ARMS}
    for (w, s), g in ep_r.groupby(["warning_id", "seed"]):
        ni = g.set_index("arm")["total_profit"]["NoInfo"]
        for a in ALL_ARMS:
            p = beliefs[f"{w}||{a}"]["belief"]
            pred = max(p, key=p.get)
            c = false_positive_cost(float(ni), float(g.set_index("arm")["total_profit"][a]),
                                    reg_of[w], pred)
            if c["is_false_positive"]:
                fpcost[a]["fp_n"] += 1
                fpcost[a]["fp_cost"] += c["profit_loss"]
            if c["is_false_negative"]:
                fpcost[a]["fn_n"] += 1
                fpcost[a]["fn_cost"] += c["missed_protection"]
    for row in ops:
        row.update(fpcost[row["arm"]])
    ops_df = pd.DataFrame(ops)
    ops_df.to_csv(DEV / "operational_metrics.csv", index=False)
    per_reg = ep_r.groupby(["regime", "arm"])["total_profit"].mean().reset_index()
    per_reg.to_csv(DEV / "per_regime_metrics.csv", index=False)

    # ---- pairwise diagnostics (headline contrasts) ----
    import numpy as np
    sem_piv = sem_df.pivot_table(index="warning_id", columns="arm",
                                 values=["brier", "nll", "accuracy"])
    pair_rows = []
    for hi, lo in HEADLINE:
        db = np.array([sem_piv[("brier", hi)][w] - sem_piv[("brier", lo)][w] for w in dev_ids])
        dp = np.array([[piv.loc[(w, s)][hi] - piv.loc[(w, s)][lo] for s in seeds] for w in dev_ids])
        st_b = paired_stats(-db)  # improvement = brier decrease
        st_o = crossed_bootstrap(dp)
        st_of = paired_stats(dp.ravel())
        regs = {}
        for r in ("normal", "supplier_delay", "demand_surge"):
            ww = [w for w in dev_ids if reg_of[w] == r]
            dd = np.array([[piv.loc[(w, s)][hi] - piv.loc[(w, s)][lo] for s in seeds] for w in ww])
            regs[r] = float(dd.mean())
        pair_rows.append({"contrast": f"{hi}-{lo}", "dBrier_mean": float(db.mean()),
                          "dBrier_improve_mean": st_b["mean"], "dBrier_ci": [st_b["ci_lo"], st_b["ci_hi"]],
                          "dProfit_mean": st_o["mean"],
                          "dProfit_crossed_ci": [st_o["ci_lo"], st_o["ci_hi"]],
                          "dProfit_win_frac": st_of["win_frac"], "dProfit_cohens_d": st_of["cohens_d"],
                          **{f"dProfit_{r}": v for r, v in regs.items()}})
    pd.DataFrame(pair_rows).to_csv(DEV / "pairwise_diagnostics.csv", index=False)

    # ---- dead-zone analysis ----
    dz = []
    for hi, lo in HEADLINE:
        for w in dev_ids:
            db = (sem_piv[("brier", hi)][w] - sem_piv[("brier", lo)][w])
            for s in seeds:
                aa, ab = actions[w][hi][s], actions[w][lo][s]
                local_eq = abs(aa[12] - ab[12]) <= 1e-9
                path_eq = pathwise_equal(aa, ab)
                dp = piv.loc[(w, s)][hi] - piv.loc[(w, s)][lo]
                dz.append({"contrast": f"{hi}-{lo}", "warning_id": w, "seed": s,
                           "dBrier": float(db), "local_action_equal": bool(local_eq),
                           "pathwise_equal": bool(path_eq), "dProfit": float(dp),
                           "cell": mechanism_cell(float(db), not bool(path_eq), float(dp))})
    dz_df = pd.DataFrame(dz)
    dz_df.to_csv(DEV / "dead_zone_analysis.csv", index=False)
    viol = dz_df[dz_df["pathwise_equal"] & (dz_df["dProfit"].abs() > 1e-9)]
    gate_G = "PASS" if len(viol) == 0 else f"FAIL ({len(viol)})"

    # ---- agency chain (R0 vs A3) ----
    ag = []
    for w in dev_ids:
        a3 = beliefs[f"{w}||A3"]
        r0ids = set(beliefs[f"{w}||R0"].get("evidence_ids", []))
        struct = a3.get("struct", {})
        ev = a3.get("evidence_ids", [])
        d1, d2 = ev[:3], ev[3:]
        overlap = len(set(d1) & set(r0ids)) / 3 if r0ids else float("nan")
        q2 = bool(struct.get("query_2_used"))
        ev_changed = bool(set(d2) - set(d1)) if q2 else False
        bdiff = sum(abs(a3["belief"][k] - beliefs[f"{w}||R0"]["belief"][k]) for k in ORDER)
        for s in seeds:
            act_ch = not pathwise_equal(actions[w]["A3"][s], actions[w]["R0"][s])
            ag.append({"warning_id": w, "seed": s, "q2_used": q2,
                       "query_1": struct.get("query_1", ""), "query_2": struct.get("query_2", ""),
                       "r0_a3_topk_overlap": overlap, "q2_evidence_changed": ev_changed,
                       "belief_L1_vs_R0": float(bdiff), "action_changed": bool(act_ch),
                       "dProfit_vs_R0": float(piv.loc[(w, s)]["A3"] - piv.loc[(w, s)]["R0"])})
    pd.DataFrame(ag).to_csv(DEV / "agency_chain_analysis.csv", index=False)

    # ---- variance analysis ----
    va = []
    for hi, lo in HEADLINE:
        dp = np.array([[piv.loc[(w, s)][hi] - piv.loc[(w, s)][lo] for s in seeds] for w in dev_ids])
        wmean, smean = dp.mean(axis=1), dp.mean(axis=0)
        ib = np.mean([sum(abs(beliefs[f"{w}||{hi}"]["belief"][k]
                              - beliefs[f"{w}||{lo}"]["belief"][k]) for k in ORDER) < 1e-9
                      for w in dev_ids])
        ia = float((dz_df[dz_df["contrast"] == f"{hi}-{lo}"]["pathwise_equal"]).mean())
        va.append({"contrast": f"{hi}-{lo}", "warning_var": float(wmean.var(ddof=1)),
                   "seed_var": float(smean.var(ddof=1)),
                   "interaction_var": float((dp - wmean[:, None] - smean + dp.mean()).var()),
                   "identical_belief_frac": float(ib), "identical_action_frac": float(ia)})
    pd.DataFrame(va).to_csv(DEV / "variance_analysis.csv", index=False)

    # ---- compute accounting ----
    import pandas as _pd
    cl = _pd.DataFrame(call_logs)
    ca = cl.groupby("arm").agg(n_calls=("latency_s", "size"),
                               prompt_tokens=("prompt_tokens", "sum"),
                               completion_tokens=("completion_tokens", "sum"),
                               latency_p50=("latency_s", "median"),
                               latency_p95=("latency_s", lambda x: x.quantile(0.95)),
                               retrieval_calls=("retrieval_calls", "sum")).reset_index()
    ca["sim_episodes_replayed"] = len(dev_ids) * len(seeds)
    ca.to_csv(DEV / "compute_accounting.csv", index=False)

    # ---- beliefs / raw / retrieval parquet ----
    allrows = []
    for k, v in beliefs.items():
        wid, arm = k.split("||")
        if wid not in dev_ids:
            continue
        allrows.append({"warning_id": wid, "arm": arm, **{f"p_{r}": v["belief"][r] for r in ORDER}})
    _pd.DataFrame(allrows).to_parquet(DEV / "beliefs.parquet", index=False)
    with open(DEV / "raw_outputs.jsonl", "w") as f:
        for r in raw_rows:
            if r["warning_id"] in dev_ids:
                f.write(json.dumps(r) + "\n")
    _pd.DataFrame([{**r, "finish_reason": json.dumps(r.get("finish_reason"))} for r in call_logs]
                  ).to_parquet(DEV / "retrieval_logs.parquet", index=False)

    # ---- RUN_MANIFEST + report ----
    (DEV / "RUN_MANIFEST.json").write_text(json.dumps({
        "track": "EXP-P2-REASONING-AGENTIC-v1", "stage": "dev_pilot",
        "dev_warnings": len(dev_ids), "seeds": seeds, "arms": list(ALL_ARMS),
        "episodes": len(ep_rows), "llm_calls_new": 0,
        "llm_calls_reused": 84, "gate_G_pathwise_theorem": gate_G,
        "elapsed_s": round(time.time() - t0, 1)}, indent=2))
    rep = ["# DEV pilot report — EXP-P2-REASONING-AGENTIC-v1", "",
           "Diagnostic only: no arm selection, tuning, or performance claims.", "",
           "## Operational means", "", ops_df.to_string(index=False), "",
           "## Headline contrasts (diagnostic)", "",
           pd.DataFrame(pair_rows).to_string(index=False), "",
           "## Dead-zone cells", "", dz_df["cell"].value_counts().to_string(), "",
           f"Theorem gate G (pathwise-equal => |dProfit|<=1e-9): {gate_G}", "",
           "## Agency (A3 vs R0)", "",
           pd.DataFrame(ag).groupby("warning_id")[["q2_used", "action_changed"]].max().to_string(),
           "", "## Variance", "", pd.DataFrame(va).to_string(index=False)]
    (DEV / "DEV_PILOT_REPORT.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep), flush=True)
    return 0 if gate_G == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
