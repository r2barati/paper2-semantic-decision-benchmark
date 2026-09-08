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
    "PerfectSemantic": r"Oracle belief",
    "OracleSemantic": r"Oracle belief",
    # retrieval systems
    "none": r"No retrieval",
    "random": r"Random",
    "bm25": r"BM25",
    "tfidf": r"TF--IDF cosine",
    "dense": r"Dense (MiniLM)",
    "oracle": r"Oracle selector",
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


def effect_rows(rows, reward_key, reference, oracle, order=None):
    """Weighted reward, paired effect with interval, and SIVR for each sensor."""
    idx = build_index(rows, reward_key)
    sensors = order or sorted(idx)
    j = {s: weighted_benchmark_return(idx[s])["value"] for s in sensors if s in idx}
    oiv = j.get(oracle, 0.0) - j.get(reference, 0.0)

    out = []
    for sensor in sensors:
        if sensor not in idx:
            continue
        diffs = paired_diffs(idx, sensor, reference)
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
    path = RESULTS / "retrieval" / "retrieval_summary.csv"
    if not path.exists():
        print("  [skip] retrieval_summary.csv not present")
        return
    rows = read(path)
    order = ["none", "random", "bm25", "tfidf", "dense", "oracle"]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiment~R. Retrieval quality and downstream decision utility on the",
        r"same episodes, at a matched evidence budget $k=3$, with the calibrated TF--IDF",
        r"interpreter and the receding-horizon controller held fixed. $\Delta J$ is the",
        r"paired effect against the no-retrieval control; intervals are crossed bootstrap",
        r"percentile intervals with shared seed draws.}",
        r"\label{tab:retrieval}",
        r"\centering\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & nDCG@3 & R@3 & MRR & $J_w$ & $\Delta J$ (95\% CI) \\",
        r"\midrule",
    ]
    interp = "TFIDF_LogReg_Calibrated"
    sel = {r["system"]: r for r in rows
           if r["interpreter"] == interp and int(r["budget_k"]) == 3}
    for system in order:
        r = sel.get(system)
        if not r:
            continue
        ci = ("--" if r["ci_lower"] in ("", None)
              else f"[{float(r['ci_lower']):+.1f}, {float(r['ci_upper']):+.1f}]")
        lines.append(
            f"{PRETTY.get(system, system)} & {float(r['ndcg_at_k']):.3f} & "
            f"{float(r['recall_at_k']):.3f} & {float(r['mrr']):.3f} & "
            f"{float(r['reward']):.1f} & {float(r['delta_vs_no_retrieval']):+.1f}~{ci} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("retrieval_main.tex", "\n".join(lines))

    # Budget sweep + rank agreement
    agree_path = RESULTS / "retrieval" / "rank_agreement.csv"
    agree = read(agree_path) if agree_path.exists() else []
    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiment~R. Do retrieval quality and decision utility rank the",
        r"ranking systems the same way? Kendall's $\tau_b$ between the nDCG@$k$ ordering",
        r"and the $J_w$ ordering over \{random, BM25, TF--IDF, dense\}; the no-retrieval",
        r"and oracle references are excluded because neither is a retrieval system.}",
        r"\label{tab:rankagreement}",
        r"\centering\small",
        r"\begin{tabular}{llrll}",
        r"\toprule",
        r"Interpreter & $k$ & $\tau_b$ & Best by nDCG & Best by reward \\",
        r"\midrule",
    ]
    for r in agree:
        tau = r["kendall_tau_b"]
        tau = "--" if tau in ("", None) else f"{float(tau):+.2f}"
        lines.append(
            f"{r['interpreter'].replace('_', ' ')} & {r['budget_k']} & {tau} & "
            f"{PRETTY.get(r['best_by_ndcg'], r['best_by_ndcg'])} & "
            f"{PRETTY.get(r['best_by_reward'], r['best_by_reward'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # Harm decomposition
    ep_path = RESULTS / "retrieval" / "retrieval_episodes.csv"
    if ep_path.exists():
        eps = [r for r in read(ep_path)
               if r["interpreter"] == interp and int(r["budget_k"]) == 3]
        by_sys = defaultdict(list)
        for r in eps:
            by_sys[r["system"]].append(r)
        lines += [
            "",
            r"\begin{table}[t]",
            r"\caption{Experiment~R. Why equal ranking quality need not mean equal",
            r"utility: how many of the $k=3$ retrieved documents actively mislead, that is,",
            r"are non-relevant \emph{and} describe a regime other than the true one.}",
            r"\label{tab:harm}",
            r"\centering\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"System & nDCG@3 & Misleading docs retrieved & Belief accuracy \\",
            r"\midrule",
        ]
        for system in order:
            rs = by_sys.get(system)
            if not rs:
                continue
            lines.append(
                f"{PRETTY.get(system, system)} & "
                f"{np.mean([float(x['ndcg_at_k']) for x in rs]):.3f} & "
                f"{np.mean([float(x['n_misleading_retrieved']) for x in rs]):.2f} & "
                f"{np.mean([1.0 if x['belief_correct'] == 'True' else 0.0 for x in rs]):.3f} \\\\"
            )
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


def _gym_table(rows, label, caption, brier=None) -> list:
    eff = effect_rows(rows, "total_reward", "NoInfo", "OracleSemantic", order=GYM_ORDER)
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering\small",
        r"\begin{tabular}{lrrlrr}",
        r"\toprule",
        r"Information source & $J_w$ & $\Delta J$ & 95\% CI & $p$ & SIVR \\",
        r"\midrule",
    ]
    for row in eff:
        name = PRETTY.get(row["sensor"], row["sensor"])
        if row["sensor"] in ("RuleBased", "TFIDF_LogReg_Raw",
                             "TFIDF_LogReg_Calibrated", "gpt-4o"):
            pass
        lines.append(
            f"{name} & {row['reward']:.1f} & {row['delta']:+.1f} & "
            f"{fmt_ci(row)} & {fmt_p(row)} & {fmt_sivr(row)} \\\\"
        )
        if row["sensor"] == "NoText_Tuned":
            lines.append(r"\midrule")
        if row["sensor"] == "gpt-4o":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def controls_tables() -> None:
    lines = []
    m_path = RESULTS / "phase8b_gym_confirmation" / "operational_results.csv"
    if m_path.exists():
        lines += _gym_table(
            read(m_path), "tab:expM",
            r"Experiment~M (held-out multi-echelon transfer, demand surge). "
            r"Rows are grouped as text-free controls, text interpreters, "
            r"degraded-text controls, and the oracle-belief reference. "
            r"$J_w$ is the balanced weighted return; $\Delta J$ is the paired "
            r"effect against NoInfo. A dev-tuned constant belief that reads no "
            r"text is the row every interpreter must beat.",
        )
    s_path = RESULTS / "phase9a_capacity_confirmation" / "operational_results.csv"
    if s_path.exists():
        lines += [""] + _gym_table(
            read(s_path), "tab:expS",
            r"Experiment~S (supply-side capacity-drop transfer). Same protocol, "
            r"a structurally different event mechanism.",
        )
    if lines:
        write("controls_main.tex", "\n".join(lines))
    else:
        print("  [skip] no gym results present")


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
                                    "OracleSemantic", order=cols)}
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
        return {
            "episodes": len(rows),
            "sensors": len({r[sensor_col] for r in rows}),
            "seeds": len({r["seed"] for r in rows}),
            "templates": len({r.get("template_id", "") for r in rows}),
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
        r"the number of \emph{paired worlds} is seeds $\times$ templates.}",
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
        lines.append(f"{name} & {c['seeds']} & {c['templates']} & "
                     f"{c['sensors']} & {c['episodes']:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # No-text tuning provenance
    for label, path in (
        ("M", RESULTS / "phase8b_gym_confirmation" / "notext_tuning.json"),
        ("S", RESULTS / "phase9a_capacity_confirmation" / "notext_tuning.json"),
    ):
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        lines += [
            "",
            rf"\paragraph{{No-text operating point (Experiment~{label}).}}",
            rf"Selected $p_{{\mathrm{{event}}}}={d['p_event']}$ on development seeds "
            rf"{d['dev_seeds'][0]}--{d['dev_seeds'][-1]}, which are disjoint from every "
            rf"confirmation seed range. Selection used the {d['selection_estimand']} "
            rf"estimand. The arm is tuned, but never on its own evaluation data.",
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
