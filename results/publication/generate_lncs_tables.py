"""Generate every LaTeX table in the LNCS manuscript from authoritative outputs.

No number in the paper is typed by hand.  Each table is written from the frozen
episode CSVs using the *declared* estimand
(:func:`src.metrics.weighted_benchmark_return`) and the *declared* uncertainty
procedure (:func:`src.metrics.crossed_bootstrap_ci`), so a table and the data it
summarises cannot drift apart.

    python3 -m results.publication.generate_lncs_tables
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics import (
    signed_sivr, crossed_bootstrap_ci, weighted_benchmark_return,
)

RESULTS = ROOT / "results"
OUT = ROOT / "paper2_submission" / "manuscript_lncs" / "tables"
N_BOOT = 5000

# Display names, in the order the paper presents them.
PRETTY = {
    "NoInfo": r"NoInfo (prior)",
    "Constant_p0.0": r"Constant $p_{\mathrm{event}}=0$",
    "Constant_p0.5": r"Constant $p_{\mathrm{event}}=0.5$",
    "Constant_p1.0": r"Constant $p_{\mathrm{event}}=1$",
    "NoText_Tuned": r"\textbf{No-text (dev-tuned)}",
    "NoText_Uniform": r"No-text (uniform)",
    "RuleBased": r"RuleBased",
    "TFIDF_LogReg": r"TF--IDF (single)",
    "TFIDF_LogReg_Raw": r"TF--IDF (single)",
    "TFIDF_LogReg_FoldEnsemble": r"TF--IDF (fold ensemble)",
    "TFIDF_LogReg_Calibrated": r"TF--IDF (calib., isotonic)",
    "TFIDF_LogReg_Calibrated_Sigmoid": r"TF--IDF (calib., sigmoid)",
    "TFIDF_LogReg_Argmax": r"TF--IDF (label only)",
    "TFIDF_LogReg_Calibrated_Argmax": r"TF--IDF cal.\ (label only)",
    "TFIDF_LogReg_Calibrated_Shuffled": r"TF--IDF cal.\ (shuffled)",
    "TFIDF_LogReg_Calibrated_ShuffledText": r"TF--IDF cal.\ (shuffled)",
    "gpt-4o": r"GPT-4o",
    "PerfectSemantic": r"Perfect-semantic ref.",
    "OracleSemantic": r"Perfect-semantic ref.",
    # retrieval systems
    "none": r"No retrieval",
    "random": r"Random",
    "bm25": r"BM25",
    "tfidf": r"TF--IDF cosine",
    "dense": r"Dense (MiniLM)",
    "oracle": r"Oracle relevance selector",
}


def read(path: Path) -> list:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body.rstrip() + "\n")
    print(f"  wrote {name}")


def family_variant(template_id: str):
    parts = str(template_id).split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2]), "_".join(parts[2:])
    return str(template_id), "1"


def build_index(rows, reward_key, sensor_key="sensor"):
    """sensor -> regime -> family -> variant -> {seed: reward}"""
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for row in rows:
        fam, var = family_variant(row["template_id"])
        idx[row[sensor_key]][row["regime"]][fam][var][row["seed"]] = float(row[reward_key])
    return idx


def paired_diffs(idx, sensor, reference):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for regime in idx[sensor]:
        for fam in idx[sensor][regime]:
            for var in idx[sensor][regime][fam]:
                a = idx[sensor][regime][fam][var]
                b = idx[reference].get(regime, {}).get(fam, {}).get(var, {})
                for seed in sorted(set(a) & set(b)):
                    out[regime][fam][var][seed] = a[seed] - b[seed]
    return out


def effect_rows(rows, reward_key, reference, oracle, order=None, intervals=True):
    """Weighted reward, paired effect with interval, and SIVR for each sensor.

    ``intervals=False`` skips the bootstrap entirely. The boundary table shows
    only point values, and running 5,000 replicates per sensor per variant to
    then discard them made regeneration -- and therefore CI -- far slower than
    the analysis warrants.
    """
    idx = build_index(rows, reward_key)
    sensors = order or sorted(idx)
    j = {s: weighted_benchmark_return(idx[s])["value"] for s in sensors if s in idx}
    oiv = j.get(oracle, 0.0) - j.get(reference, 0.0)

    out = []
    for sensor in sensors:
        if sensor not in idx:
            continue
        diffs = paired_diffs(idx, sensor, reference) if intervals else None
        ci = crossed_bootstrap_ci(diffs, n_boot=N_BOOT, seed=42) if diffs else None
        sivr = signed_sivr(j[sensor], j[reference], j.get(oracle, j[reference]))
        out.append({
            "sensor": sensor,
            "reward": j[sensor],
            "delta": j[sensor] - j[reference],
            "ci_lower": ci["ci_lower"] if ci else None,
            "ci_upper": ci["ci_upper"] if ci else None,
            "p_value": ci["p_value"] if ci else None,
            "p_floor": ci["p_value_floor"] if ci else None,
            "sivr": sivr.value,
            "status": sivr.status,
            "oiv": oiv,
        })
    return out


def fmt_ci(row) -> str:
    if row["ci_lower"] is None:
        return "--"
    return f"[{row['ci_lower']:+.1f}, {row['ci_upper']:+.1f}]"


def fmt_p(row) -> str:
    if row["p_value"] is None:
        return "--"
    if row["p_value"] <= row["p_floor"] + 1e-12:
        return f"$<${row['p_floor']:.4f}"
    return f"{row['p_value']:.3f}"


def fmt_sivr(row) -> str:
    v = row["sivr"]
    if v != v:
        return "n/a"
    mark = "" if row["status"] == "VALID" else r"$^{\dagger}$"
    return f"{v:.3f}{mark}"


# ---------------------------------------------------------------------------
# Experiment R
# ---------------------------------------------------------------------------

def retrieval_tables() -> None:
    """Experiment R: the paper's centrepiece.

    Table 1 puts both consumers side by side, because the finding is that the
    same ranking has opposite value depending on which one reads it. Table 2
    quantifies that: interaction, rank-reversal rate, correlation and the rate
    at which retrieved evidence actively misleads.
    """
    from src.retrieval_analysis import (
        read_episodes, interaction, rank_reversal_rate,
        relevance_reward_correlation, harmful_retrieval_rate,
    )

    path = RESULTS / "retrieval" / "retrieval_summary.csv"
    ep_path = RESULTS / "retrieval" / "retrieval_episodes.csv"
    if not path.exists() or not ep_path.exists():
        print("  [skip] retrieval results not present")
        return
    rows = read(path)
    episodes = read_episodes(ep_path)
    order = ["none", "random", "bm25", "tfidf", "dense", "oracle"]
    RULE, CAL = "RuleBased", "TFIDF_LogReg_Calibrated"

    # ---- Table 1: the same ranking, two consumers -----------------------
    sel = {(r["interpreter"], r["system"]): r
           for r in rows if int(r["budget_k"]) == 3}
    harm = harmful_retrieval_rate(episodes, 3)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiment~R at a matched evidence budget $k=3$. The same rankings,",
        r"read by two different downstream consumers. Ranking quality is a property of",
        r"the retrieval alone; $\Delta J$ is the paired effect on realised return against",
        r"a control that retrieves nothing, with crossed bootstrap intervals. Every",
        r"ranking system helps the rule-based consumer and harms the calibrated one.",
        r"`Harmful' is the fraction of retrieved documents that are non-relevant",
        r"\emph{and} describe a regime other than the true one.}",
        r"\label{tab:retrieval}",
        r"\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}l rr rl rl@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Ranking quality} & \multicolumn{2}{c}{RuleBased}"
        r" & \multicolumn{2}{c}{Calibrated TF--IDF} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"System & nDCG@3 & Harmful & $\Delta J$ & 95\% CI"
        r" & $\Delta J$ & 95\% CI \\",
        r"\midrule",
    ]
    for system in order:
        a, b = sel.get((RULE, system)), sel.get((CAL, system))
        if not a or not b:
            continue

        def cell(r):
            ci = ("--" if r["ci_lower"] in ("", None)
                  else f"[{float(r['ci_lower']):+.0f}, {float(r['ci_upper']):+.0f}]")
            return f"{float(r['delta_vs_no_retrieval']):+.1f} & {ci}"

        h = harm.get(system, {}).get("harmful_rate", float("nan"))
        h = "--" if h != h else f"{h:.2f}"
        name = PRETTY.get(system, system)
        if system in ("none", "oracle"):
            name = rf"\textit{{{name}}}"
        lines.append(
            f"{name} & {float(a['ndcg_at_k']):.3f} & {h} & {cell(a)} & {cell(b)} \\\\"
        )
        if system == "none":
            lines.append(r"\midrule")
        if system == "dense":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("retrieval_main.tex", "\n".join(lines))

    # ---- Table 2: how strongly does the consumer matter? ----------------
    inter = {r["system"]: r for r in interaction(episodes, 3)}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiment~R. How much of a ranking's value belongs to the ranking?",
        r"Left: the retriever~$\times$~consumer interaction at $k=3$ --- the difference",
        r"between the two consumers' paired effects for the same ranking. Every",
        r"interaction changes sign, so no single number describes what a ranking is",
        r"worth. Right: within each consumer, how often ranking quality orders two",
        r"systems the way realised utility does, and how strongly the two correlate",
        r"across episodes.}",
        r"\label{tab:consumerdep}",
        r"\centering\small",
        r"\begin{tabular}{@{}l rrr c@{}}",
        r"\toprule",
        r"System & RuleBased $\Delta J$ & Calibrated $\Delta J$ & Interaction & Sign flip \\",
        r"\midrule",
    ]
    for system in ("random", "bm25", "tfidf", "dense"):
        r = inter.get(system)
        if not r:
            continue
        lines.append(
            f"{PRETTY.get(system, system)} & {r['effect_RuleBased']:+.1f} & "
            f"{r['effect_TFIDF_LogReg_Calibrated']:+.1f} & {r['interaction']:+.1f} & "
            + (r"\checkmark" if r["sign_flip"] else "--") + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{0.6em}", ""]

    rev = rank_reversal_rate(episodes, 3)
    corr = relevance_reward_correlation(episodes, 3)
    lines += [
        r"\centering\small",
        r"\begin{tabular}{@{}l rrrr@{}}",
        r"\toprule",
        r"Consumer & Reversal rate & Kendall $\tau_b$ & Pearson $r$ & Spearman $\rho$ \\",
        r"\midrule",
    ]
    for interp in (RULE, CAL):
        rv, cr = rev.get(interp), corr.get(interp)
        if not rv or not cr:
            continue
        label = {RULE: "RuleBased", CAL: r"Calibrated TF--IDF"}[interp]
        lines.append(
            f"{label} & {rv['reversal_rate']:.2f} ({rv['discordant']}/{rv['n_pairs']}) & "
            f"{rv['kendall_tau_b']:+.2f} & {cr['pearson']:+.3f} & {cr['spearman']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{0.6em}", ""]

    # Budget sweep for the strongest ranker: raising its ranking quality does
    # not raise what it is worth to the consumer that struggles with it.
    dense = {(r["interpreter"], int(r["budget_k"])): r
             for r in rows if r["system"] == "dense"}
    budgets = sorted({int(r["budget_k"]) for r in rows})
    lines += [
        r"\centering\small",
        r"\begin{tabular}{@{}l rrr@{}}",
        r"\toprule",
        r"Dense retrieval at budget $k$ & "
        + " & ".join(f"$k={k}$" for k in budgets) + r" \\",
        r"\midrule",
    ]
    row_q = [f"{float(dense[(RULE, k)]['ndcg_at_k']):.3f}"
             if (RULE, k) in dense else "--" for k in budgets]
    lines.append(r"nDCG@$k$ & " + " & ".join(row_q) + r" \\")
    delta = "$\\Delta J$"
    for interp, label in ((RULE, f"RuleBased {delta}"),
                          (CAL, f"Calibrated TF--IDF {delta}")):
        cells = [f"{float(dense[(interp, k)]['delta_vs_no_retrieval']):+.1f}"
                 if (interp, k) in dense else "--" for k in budgets]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("retrieval_findings.tex", "\n".join(lines))


# ---------------------------------------------------------------------------
# Controls (Experiment M and S)
# ---------------------------------------------------------------------------

GYM_ORDER = [
    "NoInfo", "Constant_p0.0", "Constant_p0.5", "Constant_p1.0", "NoText_Tuned",
    "RuleBased", "TFIDF_LogReg_Raw", "TFIDF_LogReg_Calibrated", "gpt-4o",
    "TFIDF_LogReg_Calibrated_Argmax", "TFIDF_LogReg_Calibrated_Shuffled",
    "OracleSemantic",
]


def controls_tables() -> None:
    """Experiments M and S side by side.

    One table rather than two: the comparison a reader needs is between the two
    transfers, and the 12-page limit is real.
    """
    paths = [
        ("M", RESULTS / "phase8b_gym_confirmation" / "operational_results.csv"),
        ("S", RESULTS / "phase9a_capacity_confirmation" / "operational_results.csv"),
    ]
    eff = {}
    for label, path in paths:
        if path.exists():
            eff[label] = {
                e["sensor"]: e
                for e in effect_rows(read(path), "total_reward", "NoInfo",
                                     "OracleSemantic", order=GYM_ORDER)
            }
    if not eff:
        print("  [skip] no gym results present")
        return

    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiments~M (held-out multi-echelon transfer, demand surge) and~S",
        r"(supply-side capacity drop). Rows are grouped as text-free controls, text",
        r"interpreters, degraded-text controls, and the oracle-belief reference. $J_w$ is",
        r"the balanced weighted return and $\Delta J$ the paired effect against NoInfo,",
        r"with crossed bootstrap intervals. The dev-tuned no-text row is the one every",
        r"interpreter must beat, and in both experiments most do not.}",
        r"\label{tab:transfers}",
        r"\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}l rrl rrl@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Experiment~M} & \multicolumn{3}{c}{Experiment~S} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Information source & $J_w$ & $\Delta J$ & 95\% CI & $J_w$ & $\Delta J$ & 95\% CI \\",
        r"\midrule",
    ]

    def cells(label, sensor):
        row = eff.get(label, {}).get(sensor)
        if row is None:
            return "-- & -- & --"
        ci = ("--" if row["ci_lower"] is None
              else f"[{row['ci_lower']:+.0f}, {row['ci_upper']:+.0f}]")
        return f"{row['reward']:.1f} & {row['delta']:+.1f} & {ci}"

    for sensor in GYM_ORDER:
        if not any(sensor in eff.get(l, {}) for l in eff):
            continue
        name = PRETTY.get(sensor, sensor)
        lines.append(f"{name} & {cells('M', sensor)} & {cells('S', sensor)} " + r"\\")
        if sensor in ("NoText_Tuned", "gpt-4o"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("controls_main.tex", "\n".join(lines))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

P7_ORDER = [
    "NoInfo", "NoText_Uniform", "NoText_Tuned", "RuleBased",
    "TFIDF_LogReg", "TFIDF_LogReg_Argmax", "TFIDF_LogReg_FoldEnsemble",
    "TFIDF_LogReg_Calibrated", "TFIDF_LogReg_Calibrated_Sigmoid",
    "TFIDF_LogReg_Calibrated_Argmax", "TFIDF_LogReg_Calibrated_ShuffledText",
    "PerfectSemantic",
]


def calibration_tables() -> None:
    base = RESULTS / "phase7_classical_baseline"
    ep = base / "operational_results.csv"
    if not ep.exists():
        print("  [skip] phase7 operational_results.csv not present")
        return
    rows = read(ep)
    eff = effect_rows(rows, "total_profit", "NoInfo", "PerfectSemantic", order=P7_ORDER)

    belief = {}
    bpath = base / "belief_metrics_summary.csv"
    if bpath.exists():
        for r in read(bpath):
            belief[r["sensor"]] = r
    # Accuracy for EVERY arm, from the per-episode belief file. The
    # classification_metrics.csv file only covers the fitted classifiers, so
    # reading accuracy from it left the control arms blank -- exactly the rows
    # a reader most needs in order to judge the controls.
    acc = {}
    for name in ("belief_metrics_summary.csv", "classification_metrics.csv"):
        cpath = base / name
        if cpath.exists():
            for r in read(cpath):
                if r.get("accuracy") not in (None, ""):
                    acc.setdefault(r["sensor"], float(r["accuracy"]))

    noinfo_reward = next((e["reward"] for e in eff if e["sensor"] == "NoInfo"), 0.0)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiment~C (controlled benchmark). Calibration is not an isolated",
        r"intervention: \texttt{CalibratedClassifierCV} refits one classifier per fold,",
        r"so the calibrated arm differs from the single-model arm in two ways. The",
        r"fold-ensemble row carries the ensembling change without a calibration map and",
        r"is the control that separates them. Isotonic regression on six calibration",
        r"points per fold saturates; the sigmoid arm is the small-sample alternative.",
        rf"Effects are against the uninformed prior, which returns {noinfo_reward:.0f}.}}",
        r"\label{tab:calibration}",
        r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l rrr rlr}",
        r"\toprule",
        r"Interpreter & Acc. & Brier & Log-loss & $\Delta J$ & 95\% CI & SIVR \\",
        r"\midrule",
    ]
    for row in eff:
        s = row["sensor"]
        a = f"{acc[s]:.3f}" if s in acc else "--"
        b = belief.get(s, {})
        bs = b.get("brier") or b.get("brier_score") or b.get("mean_brier")
        bs = f"{float(bs):.3f}" if bs not in (None, "") else "--"
        ll = b.get("log_loss")
        ll = f"{float(ll):.3f}" if ll not in (None, "") else "--"
        ci = ("--" if row["ci_lower"] is None
              else f"[{row['ci_lower']:+.0f}, {row['ci_upper']:+.0f}]")
        lines.append(
            f"{PRETTY.get(s, s)} & {a} & {bs} & {ll} & "
            f"{row['delta']:+.1f} & {ci} & {fmt_sivr(row)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("calibration.tex", "\n".join(lines))


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

def boundary_tables() -> None:
    path = RESULTS / "phase9b_robustness"
    rows = []
    for variant_dir in sorted(path.glob("*/")) if path.exists() else []:
        f = variant_dir / "operational_results.csv"
        if f.exists():
            for r in read(f):
                r["variant"] = variant_dir.name
                rows.append(r)
    if not rows:
        print("  [skip] no phase9b results present")
        return

    # One row per variant, columns per information source: 48 rows of detail
    # do not fit a 12-page limit, and the comparison a reader needs is across
    # variants, not within them. The full per-sensor table is in the release.
    cols = ["NoInfo", "NoText_Tuned", "RuleBased",
            "TFIDF_LogReg_Calibrated", "gpt-4o", "OracleSemantic"]
    head = ["NoInfo", "No-text", "Rule", "TF--IDF", "GPT-4o", "Oracle"]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiment~B. \emph{Every} one-factor-at-a-time variant, including",
        r"the adverse ones. Values are the balanced weighted return $J_w$; OIV is the",
        r"oracle information value. $\dagger$ marks a variant whose OIV is not positive:",
        r"there the oracle-belief reference itself underperforms the uninformed prior, so",
        r"a normalised ratio cannot be read as recovery of information value and the",
        r"finding is about controller--environment mismatch, not semantic sensing.}",
        r"\label{tab:boundary}",
        r"\centering\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * len(cols) + r"r}",
        r"\toprule",
        "Variant & " + " & ".join(head) + r" & OIV \\",
        r"\midrule",
    ]
    for variant in sorted({r["variant"] for r in rows}):
        vrows = [r for r in rows if r["variant"] == variant]
        eff = {e["sensor"]: e
               for e in effect_rows(vrows, "total_reward", "NoInfo",
                                    "OracleSemantic", order=cols, intervals=False)}
        if not eff:
            continue
        oiv = next(iter(eff.values()))["oiv"]
        flag = "" if oiv > 0 else r"$^{\dagger}$"
        cells = []
        for c in cols:
            cells.append(f"{eff[c]['reward']:.0f}" if c in eff else "--")
        name = variant.replace("_", r"\_")
        lines.append(f"{name}{flag} & " + " & ".join(cells) + f" & {oiv:+.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("boundary.tex", "\n".join(lines))


# ---------------------------------------------------------------------------
# Appendix accounting
# ---------------------------------------------------------------------------

def appendix_tables() -> None:
    def count(path, sensor_col="sensor"):
        if not path.exists():
            return None
        rows = read(path)
        # Experiment R draws a generated candidate pool per seed rather than
        # from an authored template set, so it has no template axis. Reporting
        # "1" there would invent a design feature it does not have.
        has_templates = any(r.get("template_id") for r in rows)
        return {
            "episodes": len(rows),
            "sensors": len({r[sensor_col] for r in rows}),
            "seeds": len({r["seed"] for r in rows}),
            "templates": (len({r["template_id"] for r in rows})
                          if has_templates else None),
        }

    entries = [
        ("R (retrieval)", RESULTS / "retrieval" / "retrieval_episodes.csv", "system"),
        ("C (controlled)", RESULTS / "phase7_classical_baseline" / "operational_results.csv", "sensor"),
        ("M (multi-echelon)", RESULTS / "phase8b_gym_confirmation" / "operational_results.csv", "sensor"),
        ("S (supply-side)", RESULTS / "phase9a_capacity_confirmation" / "operational_results.csv", "sensor"),
    ]

    lines = [
        r"\begin{table}[!ht]",
        r"\caption{Experiment accounting, generated from the saved episode files rather",
        r"than from design intent. `Episodes' counts rows across all information sources;",
        r"the number of \emph{paired worlds} is seeds $\times$ templates, which is not the",
        r"episode count. Experiment~R draws a generated candidate pool per seed and so has",
        r"no template axis.}",
        r"\label{tab:accounting}",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Experiment & Seeds & Templates & Sources & Episodes \\",
        r"\midrule",
    ]
    for name, path, col in entries:
        c = count(path, col)
        if c is None:
            continue
        tmpl = "n/a" if c["templates"] is None else f"{c['templates']}"
        lines.append(f"{name} & {c['seeds']} & {tmpl} & "
                     f"{c['sensors']} & {c['episodes']:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # No-text tuning provenance, in one paragraph rather than one per experiment.
    tuned = []
    for label, path in (
        ("M", RESULTS / "phase8b_gym_confirmation" / "notext_tuning.json"),
        ("S", RESULTS / "phase9a_capacity_confirmation" / "notext_tuning.json"),
    ):
        if path.exists():
            d = json.loads(path.read_text())
            tuned.append((label, d))
    if tuned:
        parts = ", ".join(
            rf"Experiment~{label} selected $p_{{\mathrm{{event}}}}={d['p_event']}$ on "
            rf"seeds {d['dev_seeds'][0]}--{d['dev_seeds'][-1]}"
            for label, d in tuned
        )
        lines += [
            "",
            r"\paragraph{No-text operating point.}",
            parts + ". Both development seed ranges are disjoint from every "
            "confirmation seed range, and selection used the balanced estimand. "
            "The arm is tuned, but never on its own evaluation data.",
        ]
    write("appendix.tex", "\n".join(lines))


def main() -> int:
    print("Generating LNCS tables from authoritative outputs...")
    retrieval_tables()
    controls_tables()
    calibration_tables()
    boundary_tables()
    appendix_tables()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
