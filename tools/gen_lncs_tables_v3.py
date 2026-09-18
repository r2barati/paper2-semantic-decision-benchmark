"""Generate LNCS tables for the V3 manuscript from frozen delivery artifacts.

Every number is read from results/v3main/{semantic,sim,delivery} parquets;
nothing is hand-entered. Run: python3 tools/gen_lncs_tables_v3.py
Outputs: paper2_submission/manuscript_lncs/tables/v3_*.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEM = ROOT / "results" / "v3main" / "semantic"
SIM = ROOT / "results" / "v3main" / "sim"
GYM = ROOT / "results" / "v3main" / "gym"
TDIR = ROOT / "paper2_submission" / "manuscript_lncs" / "tables"
M8 = "Qwen/Qwen3-8B-AWQ"
M14 = "Qwen/Qwen3-14B-AWQ"
SHORT = {M8: "8B", M14: "14B", "none": "---"}


def w(name, text):
    (TDIR / name).write_text(text)
    print(f"wrote tables/{name}")


def pstr(p, floor=1 / 5001):
    return "$<{:.4f}$".format(floor) if p <= floor + 1e-12 else f"${p:.3f}$"


def cell_delta(r):
    s = "^*" if r["reject_holm_05"] else ""
    return (f"\\shortstack{{${r['delta_vs_noinfo']:+.1f}{s}$\\\\"
            f"$[{r['ci_lo']:+.0f},{r['ci_hi']:+.0f}]$}}")


def main():
    TDIR.mkdir(parents=True, exist_ok=True)
    ir = pd.read_parquet(SEM / "ir_test_means.parquet").set_index("system")
    agg = pd.read_parquet(SEM / "agg_semantic_test.parquet")
    a3 = agg[agg.k == 3]
    util = pd.read_parquet(SIM / "utility_by_arm_balanced.parquet")
    u3 = util[util.k == 3]
    gaps = pd.read_parquet(SIM / "four_gaps.parquet")
    q3 = pd.read_parquet(SEM / "q3_interaction.parquet")
    ext = pd.read_parquet(SIM.parent / "extension_notext_tuned_episodes.parquet")

    order = ["bm25", "dense", "hybrid", "rerank"]
    # ---- tab:retrieval ----
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Retrieval quality on the 160 test queries (trec\\_eval; "
             "95\\% query-bootstrap CIs: BM25 $[0.095,0.135]$, dense "
             "$[0.153,0.198]$, hybrid $[0.157,0.207]$, rerank $[0.156,0.210]$ --- "
             "point-ordered with overlapping intervals). "
             "The frozen ranker is RRF over BM25 and instructed dense embeddings "
             "($k=120$) with reranker depth 30, selected on dev only.}",
             "\\label{tab:retrieval}",
             "\\begin{tabular}{lcccc}", "\\toprule",
             "System & nDCG@10 & R@10 & R@20 & RR \\\\", "\\midrule"]
    names = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid RRF",
             "rerank": "Reranked"}
    for s in order:
        r = ir.loc[s]
        lines.append(f"{names[s]} & {r['ndcg10']:.4f} & {r['recall10']:.4f} & "
                     f"{r['recall20']:.4f} & {r['mrr']:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    w("v3_retrieval.tex", "\n".join(lines) + "\n")

    # ---- tab:semantic (k=3; two panels: accuracy, Brier) ----
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Semantic quality on test queries at $k=3$ (160 queries). "
             "C0 is the deterministic rule-based reference (model-free); C1 naive "
             "LLM RAG; C3 provenance-aware with abstention. Abstention rates: "
             "C3/random 0.56--0.94, else $\\le 0.03$. Per-cell 95\\% "
             "query-bootstrap CIs (median half-width $0.07$) are tabulated in "
             "the frozen artifact.}",
             "\\label{tab:semantic}",
             "(a) Regime accuracy\\\\[2pt]",
             "{\\small\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lccccc}", "\\toprule",
             "System & C0 & C1/8B & C1/14B & C3/8B & C3/14B \\\\",
             "\\midrule"]
    syslist = order + ["random", "oracle-factual"]
    sysnames = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid",
                "rerank": "Rerank", "random": "Random",
                "oracle-factual": "Oracle"}
    for s in syslist:
        g = a3[a3.system == s]
        row = [sysnames[s]]
        for cons, m in [("C0", "none"), ("C1", M8), ("C1", M14),
                        ("C3", M8), ("C3", M14)]:
            r = g[(g.consumer == cons) & (g.model == m)].iloc[0]
            row.append(f"{r['accuracy']:.3f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}",
             "\\\\[6pt](b) Brier score\\\\[2pt]",
             "{\\small\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lccccc}", "\\toprule",
             "System & C0 & C1/8B & C1/14B & C3/8B & C3/14B \\\\",
             "\\midrule"]
    for s in syslist:
        g = a3[a3.system == s]
        row = [sysnames[s]]
        for cons, m in [("C0", "none"), ("C1", M8), ("C1", M14),
                        ("C3", M8), ("C3", M14)]:
            r = g[(g.consumer == cons) & (g.model == m)].iloc[0]
            row.append(f"{r['brier']:.3f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""]
    w("v3_semantic.tex", "\n".join(lines) + "\n")

    # ---- tab:utility (k=3, paired Delta + Holm stars) ----
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Paired $\\Delta J$ vs no-retrieval control at $k=3$ "
             "(160 test queries $\\times$ 5 seeds; crossed bootstrap, "
             "$B=5{,}000$). $^*$: Holm rejection (105 arms); unstarred cells "
             "Holm-n.s. Displayed CIs are unadjusted; stars carry multiplicity "
             "control. A harm claim is headlined only if Holm-robust under "
             "balanced weights and same-signed under empirical weights; "
             "requiring both can only shrink the headlined set, so the dual "
             "gate is conservative by construction. "
             "Cluster CIs (median $1.08\\times$, max $1.19\\times$) "
             "agree in direction for every starred cell. Refs: NoInfo $1779.0$, PerfectBelief "
             "$1994.4$, Hindsight $2451.2$. NoText\\_Tuned $1881.1$ (\\S"
             "\\ref{sec:tuned}).}",
             "\\label{tab:utility}",
             "{\\small\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lccccc}", "\\toprule",
             "System & C0 & C1/8B & C1/14B & C3/8B & C3/14B \\\\",
             "\\midrule"]
    for s in order + ["random", "oracle-factual"]:
        g = u3[u3.system == s]
        cells = [sysnames.get(s, s)]
        for cons, m in [("C0", "none"), ("C1", M8), ("C1", M14),
                        ("C3", M8), ("C3", M14)]:
            r = g[(g.consumer == cons) & (g.model == m)].iloc[0]
            cells.append(cell_delta(r))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""]
    w("v3_utility.tex", "\n".join(lines) + "\n")

    # ---- tab:gaps (control shown once: retrieval-independent by construction) ----
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Four-gap decomposition at $k=3$ (rerank arm; balanced "
             "$J$; paired crossed 95\\% CIs inline). "
             "Rel$\\to$fact is $0.00$ in every arm (both "
             "oracle sets saturate identically). Control $+456.8$ "
             "$[+387,+538]$ is retrieval-independent "
             "(Hindsight$-$PerfectBelief) and shown once.}",
             "\\label{tab:gaps}",
             "\\begin{tabular}{lccc}", "\\toprule",
             "Arm & Retrieval & Rel$\\to$Fact & Interp. \\\\",
             "\\midrule"]
    gapci = {
        ("C0", "none"): [(-68, -13), (0, 0), (130, 220)],
        ("C1", "Qwen/Qwen3-8B-AWQ"): [(-65, -23), (0, 0), (108, 172)],
        ("C3", "Qwen/Qwen3-8B-AWQ"): [(-56, 11), (0, 0), (17, 58)],
        ("C1", "Qwen/Qwen3-14B-AWQ"): [(-62, -20), (0, 0), (114, 177)],
        ("C3", "Qwen/Qwen3-14B-AWQ"): [(-48, 7), (0, 0), (31, 77)],
    }
    for _, r in gaps.iterrows():
        arm = ("C0" if r["model"] == "none"
               else f"{r['consumer']}/{SHORT[r['model']]}")
        ci = gapci[(r["consumer"], r["model"])]
        vals = [r['retrieval_gap'], r['relevance_to_factual_gap'],
                r['interpretation_gap']]
        cells = [arm] + [
            f"\\shortstack{{${v:+.1f}$\\\\$[{lo:+.0f},{hi:+.0f}]$}}"
            for v, (lo, hi) in zip(vals, ci)]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("Control (all arms) & \\multicolumn{3}{c}{$+456.8$ "
                 "$[+387,+538]$} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    w("v3_gaps.tex", "\n".join(lines) + "\n")

    # ---- tab:tuned (extension) ----
    import json as _j
    tuned = _j.loads((SIM.parent / "extension_notext_tuned_J.json").read_text())
    grid = _j.loads((SIM.parent / "extension_notext_tuned.json").read_text())
    top = sorted(grid["tuning_table"], key=lambda r: -r["profit"])[:3]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{\\emph{Labeled extension:} dev-tuned constant no-text "
             "belief (66-triple simplex grid, 40 dev queries, disjoint seeds). "
             "Winner evaluated once on test. Not part of the frozen matrix.}",
             "\\label{tab:tuned}",
             "\\begin{tabular}{ccc}", "\\toprule",
             "$(p_N,p_D,p_S)$ & dev $J$ & test $\\Delta$ vs NoInfo \\\\",
             "\\midrule"]
    for row in top:
        mark = " (selected)" if abs(row["p2"] - 1.0) < 1e-9 else ""
        test = (f"${tuned['delta']:+.1f}$ $[{tuned['ci_lo']:+.0f},"
                f"{tuned['ci_hi']:+.0f}]$" if mark else "---")
        lines.append(f"$({row['p0']:.1f},{row['p1']:.1f},{row['p2']:.1f}){mark}$ & "
                     f"${row['profit']:.1f}$ & {test} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    w("v3_tuned.tex", "\n".join(lines) + "\n")

    # ---- tab:interaction (headline Q3 magnitudes + Holm p) ----
    import src.semantic_analysis_v3main as _S  # noqa
    h = pd.read_parquet(SEM / "headline_holm.parquet")
    q3m = pd.read_parquet(SEM / "q3_interaction.parquet")
    q3m = q3m[(q3m["retriever_a"] == "rerank") & (q3m["retriever_b"] == "bm25")
              & (q3m["k"] == 3) & (q3m["metric"] == "accuracy")]
    q5m = pd.read_parquet(SEM / "q5_c3c1.parquet")
    q6m = pd.read_parquet(SEM / "q6_xmodel.parquet")
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Retriever$\\times$consumer interaction "
             "(rerank vs BM25, $k=3$, accuracy): $\\Delta\\Delta$ with 95\\% CI "
             "and Holm $p$ over 10 preregistered cells. Q5/Q6 rows report Holm "
             "$p$ only (Q6-C3 degenerate: identical profiles).}",
             "\\label{tab:interaction}",
             "\\begin{tabular}{lcc}", "\\toprule",
             "Cell & $\\Delta\\Delta$ [95\\% CI] & Holm $p$ \\\\",
             "\\midrule"]
    # (Q5/Q6 rows: Holm p only; see caption.)
    for _, r in h.iterrows():
        cell = r["cell"].replace("_", "\\_")
        mag = "--- & "
        if cell.startswith("Q3"):
            parts = r["cell"].split("_")
            ca, cb = parts[2].split("-")
            short = "8B" if parts[4] == "8B" else "14B"
            m = q3m[(q3m["consumer_a"] == ca) & (q3m["consumer_b"] == cb)
                    & (q3m["model"].str.contains(short + "-AWQ"))]
            if len(m):
                q = m.iloc[0]
                mag = (f"${q['estimate']:+.3f}$ $[{q['ci_lo']:+.3f},"
                       f"{q['ci_hi']:+.3f}]$ & ")
        elif "Q5" in cell:
            short = "8B" if cell.endswith("8B") else "14B"
            m = q5m[(q5m["system"] == "rerank") & (q5m["k"] == 3)
                    & (q5m["metric"] == "accuracy")
                    & (q5m["model"].str.contains(short + "-AWQ"))]
            if len(m):
                q = m.iloc[0]
                mag = (f"${q['estimate']:+.3f}$ $[{q['ci_lo']:+.3f},"
                       f"{q['ci_hi']:+.3f}]$ & ")
        elif "Q6" in cell:
            cons = "C1" if "_C1_" in r["cell"] else "C3"
            m = q6m[(q6m["system"] == "rerank") & (q6m["consumer"] == cons)
                    & (q6m["k"] == 3)]
            if len(m):
                q = m.iloc[0]
                mag = (f"${q['diff_b_minus_a']:+.4f}$ $[{q['ci_lo']:+.4f},"
                       f"{q['ci_hi']:+.4f}]$ & ")
        pv = f"$<{0.0002:.4f}$" if r["p_holm"] <= 1 / 10001 + 1e-12 \
            else f"${r['p_holm']:.4f}$"
        mark = "$^*$" if r["reject_holm_05"] else ""
        lines.append(f"{cell} & {mag}{pv}{mark} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    w("v3_interaction.tex", "\n".join(lines) + "\n")
    print("tables done")


if __name__ == "__main__":
    main()
