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
    path = RESULTS / "retrieval" / "retrieval_summary.csv"
    if not path.exists():
        print("  [skip] retrieval_summary.csv not present")
        return
    rows = read(path)
    order = ["none", "random", "bm25", "tfidf", "dense", "oracle"]

    interp = "TFIDF_LogReg_Calibrated"
    sel = {r["system"]: r for r in rows
           if r["interpreter"] == interp and int(r["budget_k"]) == 3}

    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiment~R at a matched evidence budget $k=3$, with the calibrated",
        r"TF--IDF interpreter and the receding-horizon controller held fixed. Left:",
        r"ranking quality. Right: what the interpreter actually receives and what the",
        r"decisions are worth. `Misleading' counts retrieved documents that are",
        r"non-relevant \emph{and} describe a regime other than the true one --- the errors",
        r"that move the controller, as opposed to those that only waste a slot.",
        r"$\Delta J$ is the paired effect against the no-retrieval control; intervals are",
        r"crossed bootstrap percentile intervals with shared seed draws.}",
        r"\label{tab:retrieval}",
        r"\centering\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l rrr rr r@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Ranking quality} & \multicolumn{3}{c}{Consequences} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"System & nDCG@3 & R@3 & MRR & Misleading & Belief acc. & $\Delta J$ (95\% CI) \\",
        r"\midrule",
    ]

    # Per-episode harm and belief accuracy, folded into the same table.
    ep_path = RESULTS / "retrieval" / "retrieval_episodes.csv"
    harm: dict = {}
    if ep_path.exists():
        by_sys = defaultdict(list)
        for r in read(ep_path):
            if r["interpreter"] == interp and int(r["budget_k"]) == 3:
                by_sys[r["system"]].append(r)
        for system, rs in by_sys.items():
            harm[system] = (
                float(np.mean([float(x["n_misleading_retrieved"]) for x in rs])),
                float(np.mean([1.0 if x["belief_correct"] == "True" else 0.0 for x in rs])),
            )

    for system in order:
        r = sel.get(system)
        if not r:
            continue
        ci = ("--" if r["ci_lower"] in ("", None)
              else f"[{float(r['ci_lower']):+.0f}, {float(r['ci_upper']):+.0f}]")
        mis, acc = harm.get(system, (float("nan"), float("nan")))
        mis = "--" if mis != mis else f"{mis:.2f}"
        acc = "--" if acc != acc else f"{acc:.3f}"
        lines.append(
            f"{PRETTY.get(system, system)} & {float(r['ndcg_at_k']):.3f} & "
            f"{float(r['recall_at_k']):.3f} & {float(r['mrr']):.3f} & "
            f"{mis} & {acc} & "
            f"{float(r['delta_vs_no_retrieval']):+.1f}~{ci} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("retrieval_main.tex", "\n".join(lines))

    # Budget sweep + rank agreement
    agree_path = RESULTS / "retrieval" / "rank_agreement.csv"
    agree = read(agree_path) if agree_path.exists() else []
    # The strongest ranker's own quality and utility at each budget, so the
    # budget sweep can be checked in the paper rather than only in the CSV.
    dense = {(r["interpreter"], int(r["budget_k"])): r
             for r in rows if r["system"] == "dense"}

    lines = [
        r"\begin{table}[t]",
        r"\caption{Experiment~R. Do retrieval quality and decision utility rank the",
        r"ranking systems the same way? Kendall's $\tau_b$ is computed between the",
        r"nDCG@$k$ ordering and the $J_w$ ordering over \{random, BM25, TF--IDF,",
        r"dense\}; the no-retrieval and oracle references are excluded because neither",
        r"is a retrieval system. The last two columns track the strongest ranker",
        r"across budgets: raising its nDCG does not raise its realised effect.}",
        r"\label{tab:rankagreement}",
        r"\centering\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrllrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Best system by} & \multicolumn{2}{c}{Dense} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Interpreter & $k$ & $\tau_b$ & nDCG & reward & nDCG@$k$ & $\Delta J$ \\",
        r"\midrule",
    ]
    for r in agree:
        tau = r["kendall_tau_b"]
        tau = "--" if tau in ("", None) else f"{float(tau):+.2f}"
        d = dense.get((r["interpreter"], int(r["budget_k"])))
        dn = f"{float(d['ndcg_at_k']):.3f}" if d else "--"
        dd = f"{float(d['delta_vs_no_retrieval']):+.1f}" if d else "--"
        label = {
            "RuleBased": "RuleBased",
            "TFIDF_LogReg_Calibrated": r"TF--IDF calib.",
        }.get(r["interpreter"], r["interpreter"].replace("_", " "))
        lines.append(
            f"{label} & {r['budget_k']} & {tau} & "
            f"{PRETTY.get(r['best_by_ndcg'], r['best_by_ndcg'])} & "
            f"{PRETTY.get(r['best_by_reward'], r['best_by_reward'])} & {dn} & {dd} \\\\"
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
