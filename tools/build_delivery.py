"""V3 delivery package builder (analysis outputs only; never touches beliefs).

Generates publication tables (MD), Holm-adjusted p-value tables, cost/latency
table, and ECIR figures (PNG+PDF) from frozen result parquets into
results/v3main/delivery/. Nothing here runs models, simulators, or edits
frozen artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.semantic_analysis_v3main as S

SEM = ROOT / "results" / "v3main" / "semantic"
SIM = ROOT / "results" / "v3main" / "sim"
GYM = ROOT / "results" / "v3main" / "gym"
OUT = ROOT / "results" / "v3main" / "delivery"
M8 = "Qwen/Qwen3-8B-AWQ"
M14 = "Qwen/Qwen3-14B-AWQ"


def md_table(df, path, floatfmt=".4f"):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / path, "w") as f:
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in df.columns) + " |\n")
        for _, r in df.iterrows():
            cells = []
            for v in r:
                if isinstance(v, float):
                    cells.append(f"{v:{floatfmt}}")
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"wrote {path} ({len(df)} rows)")


def p_str(p, floor=1 / 5001):
    if p <= floor + 1e-12:
        return "<0.0002"
    return f"{p:.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ir = pd.read_parquet(SEM / "ir_test_means.parquet")
    agg = pd.read_parquet(SEM / "agg_semantic_test.parquet")
    util = pd.read_parquet(SIM / "utility_by_arm.parquet")
    gaps = pd.read_parquet(SIM / "four_gaps.parquet")
    q1 = pd.read_parquet(SEM / "q1_corr.parquet")
    q2 = pd.read_parquet(SEM / "q2_stability.parquet")
    q3 = pd.read_parquet(SIM.parent / "semantic" / "q3_interaction.parquet")
    q5 = pd.read_parquet(SEM / "q5_c3c1.parquet")
    q6 = pd.read_parquet(SEM / "q6_xmodel.parquet")

    # 3. retrieval table
    md_table(ir.sort_values("ndcg10", ascending=False), "retrieval_table.md")
    # 4. semantic-quality table (k=3) + per-cell accuracy CIs
    a3 = agg[agg.k == 3].sort_values(["model", "consumer", "system"])
    md_table(a3, "semantic_table_k3.md")
    try:
        _ci = pd.read_parquet(SEM / "semantic_ci_k3.parquet")
        _am = a3.merge(_ci, on=["system", "consumer", "model"],
                       how="left", suffixes=("", "_ci"))
        md_table(_am, "semantic_table_k3_ci.md")
    except FileNotFoundError:
        pass
    # 5. Retriever x Consumer tables (semantic acc + sim delta)
    rx_sem = a3[a3.system.isin(["bm25", "dense", "hybrid", "rerank", "random",
                                "oracle-relevant", "oracle-factual"])][
        ["system", "consumer", "model", "accuracy", "brier"]]
    md_table(rx_sem, "retriever_consumer_semantic.md")
    u3 = util[util.k == 3][["system", "consumer", "model", "J",
                            "delta_vs_noinfo", "ci_lo", "ci_hi", "p_value",
                            "reject_holm_05", "cluster_ci_lo", "cluster_ci_hi"]]
    u3 = u3.copy()
    u3["p"] = u3["p_value"].map(p_str)
    u3["sig"] = u3["reject_holm_05"].map({True: "*", False: "ns"})
    md_table(u3.drop(columns=["p_value", "reject_holm_05"]),
             "retriever_consumer_utility.md", floatfmt=".2f")
    # stars = Holm-0.05 rejection (balanced); ns = Holm-n.s. (see B2 fix)
    ue = pd.read_parquet(SIM / "utility_by_arm_empirical.parquet")
    ue3 = ue[ue.k == 3][["system", "consumer", "model", "J",
                         "delta_vs_noinfo", "ci_lo", "ci_hi", "p_value",
                         "reject_holm_05"]]
    ue3 = ue3.copy()
    ue3["p"] = ue3["p_value"].map(p_str)
    ue3["sig"] = ue3["reject_holm_05"].map({True: "*", False: "ns"})
    md_table(ue3.drop(columns=["p_value", "reject_holm_05"]),
             "retriever_consumer_utility_empirical.md", floatfmt=".2f")
    reg = pd.read_parquet(SIM / "regime_exploratory.parquet")
    md_table(reg, "regime_exploratory.md", floatfmt=".1f")
    hl = pd.read_parquet(SEM / "headline_holm.parquet")
    md_table(hl, "headline_holm.md")
    # 6. sequential-utility full (both k) + 7. gaps
    md_table(util.sort_values(["k", "model", "consumer", "system"]),
             "utility_full.md", floatfmt=".2f")
    md_table(gaps, "oracle_gaps.md", floatfmt=".2f")
    ge = pd.read_parquet(SIM / "four_gaps_empirical.parquet")
    md_table(ge, "oracle_gaps_empirical.md", floatfmt=".2f")

    # 11a. Holm families (sim: saved by phase_analysis; q1/q2/q6 here)
    import shutil as _sh
    holm_frames = {}
    for wname in ("balanced", "empirical"):
        h = pd.read_parquet(SIM / f"holm_sim_arms_{wname}.parquet")
        holm_frames[f"sim_arms_{wname}"] = h
        _sh.copy(SIM / f"holm_sim_arms_{wname}.parquet",
                 OUT / f"holm_sim_arms_{wname}.parquet")
        md_table(h, f"holm_sim_arms_{wname}.md")
    for name, frame, col in (("q1_spearman", q1, "spearman_p_exact"),
                             ("q2_spearman", q2, "spearman_p"),
                             ("q6_spearman", q6, "spearman_p")):
        pv = frame[col].to_numpy(dtype=float) if frame[col].notna().all() \
            else frame[col].fillna(1.0).to_numpy(dtype=float)
        h = S.holm_correction(pv)
        h.insert(0, "row", frame.apply(
            lambda r: "/".join(str(r[c]) for c in frame.columns
                               if c in ("consumer", "model", "k", "system",
                                        "consumer_a", "consumer_b")), axis=1))
        holm_frames[name] = h
        h.to_parquet(OUT / f"holm_{name}.parquet", index=False)
        md_table(h, f"holm_{name}.md")
    print("holm balanced reject:",
          f"{int(holm_frames['sim_arms_balanced']['reject_holm_05'].sum())}/"
          f"{len(holm_frames['sim_arms_balanced'])}; empirical:",
          f"{int(holm_frames['sim_arms_empirical']['reject_holm_05'].sum())}/"
          f"{len(holm_frames['sim_arms_empirical'])}")

    # 10. cost/latency table (tokens from caches; compute from logs/manifests)
    tok = {"prompt_tokens": 0, "completion_tokens": 0, "files": 0}
    for d in ["beliefs_cache_awq", "beliefs_cache_qwen14"]:
        for p in (ROOT / "results" / "v3main" / d).glob("*.json"):
            u = json.loads(p.read_text()).get("usage", {}) or {}
            tok["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            tok["completion_tokens"] += int(u.get("completion_tokens") or 0)
            tok["files"] += 1
    ep = pd.read_parquet(SIM / "episodes.parquet")
    cost = pd.DataFrame([
        {"item": "LLM cache fills (2 models, frozen prompts)",
         "n": tok["files"], "prompt_tokens": tok["prompt_tokens"],
         "completion_tokens": tok["completion_tokens"],
         "usd_cost": 0.0, "note": "free Kaggle GPU tier; $0 API spend"},
        {"item": "controlled sim episodes", "n": len(ep),
         "prompt_tokens": 0, "completion_tokens": 0, "usd_cost": 0.0,
         "note": f"seeds 60000-60004; CPU-hours ≈ "
                 f"{ep['wall_s'].sum() / 3600:.1f} core-h (mp)"},
    ])
    md_table(cost, "cost_latency.md", floatfmt=".0f")

    # 12. figures (deterministic style; no timestamps)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    ir_m = ir.set_index("system")["ndcg10"].to_dict()
    a3r = agg[(agg.k == 3) & (agg.consumer == "C3") & (agg.model == M8)]
    amap = a3r.set_index("system")["accuracy"].to_dict()
    systems = ["bm25", "dense", "hybrid", "rerank"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter([ir_m[s] for s in systems], [amap[s] for s in systems], s=60)
    for s in systems:
        ax.annotate(s, (ir_m[s], amap[s]), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("retrieval nDCG@10 (test)")
    ax.set_ylabel("semantic accuracy C3/8B k=3 (test)")
    ax.set_title("Retrieval -> semantic transfer (frozen)")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_transfer.png")
    fig.savefig(figdir / "fig1_transfer.pdf")
    plt.close(fig)

    u3r = util[(util.k == 3) & (util.model == M8)]
    syslist = systems + ["random", "oracle-factual"]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(len(syslist))
    w = 0.22
    for j, cons in enumerate(["C0", "C1", "C3"]):
        sub = util[(util.k == 3) & (util.model.isin([M8, "none"]))
                   & (util.consumer == cons)
                   & (util.system.isin(syslist))]
        sub = sub.set_index("system").reindex(syslist)
        ax.bar(x + (j - 1) * w, sub["delta_vs_noinfo"].to_numpy(), w, label=cons,
               yerr=[(sub["delta_vs_noinfo"] - sub["ci_lo"]).to_numpy(),
                     (sub["ci_hi"] - sub["delta_vs_noinfo"]).to_numpy()],
               capsize=2)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(syslist[:-1] + ["oracle"], rotation=20)
    ax.set_ylabel("paired Δ profit vs NoInfo (95% CI)")
    ax.set_title("Retriever x Consumer utility (8B, k=3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig2_rxc.png")
    fig.savefig(figdir / "fig2_rxc.pdf")
    plt.close(fig)

    g = gaps[(gaps.model == M8) & (gaps.consumer == "C3")].iloc[0]
    stages = ["NoInfo", "rerank", "oracle", "perfect", "hindsight"]
    vals = [g["J_noinfo"], g["J_retrieval"], g["J_oracle_factual"],
            g["J_perfect"], g["J_hindsight"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.step(stages, vals, where="mid", linewidth=2)
    ax.scatter(stages, vals, s=40, zorder=3)
    ax.set_ylabel("balanced return J")
    ax.set_title("Oracle ladder waterfall (C3/8B, k=3)")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_ladder.png")
    fig.savefig(figdir / "fig3_ladder.pdf")
    plt.close(fig)

    q6r = q6[(q6.system == "rerank") & (q6.k == 3)]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.errorbar(q6r["consumer"], q6r["diff_b_minus_a"],
                yerr=[q6r["diff_b_minus_a"] - q6r["ci_lo"],
                      q6r["ci_hi"] - q6r["diff_b_minus_a"]],
                fmt="o", capsize=4)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("8B − 14B accuracy (95% CI)")
    ax.set_title("Cross-model parity (rerank, k=3)")
    fig.tight_layout()
    fig.savefig(figdir / "fig4_xmodel.png")
    fig.savefig(figdir / "fig4_xmodel.pdf")
    plt.close(fig)
    print("figures:", sorted(p.name for p in figdir.iterdir()))
    print("DONE — delivery under results/v3main/delivery/")


if __name__ == "__main__":
    main()
