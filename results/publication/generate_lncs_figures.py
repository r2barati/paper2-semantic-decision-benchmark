"""Generate the manuscript's figures from the stored episodes.

Like the tables, nothing here is hand-plotted: the figure reads the same frozen
episode file the analysis does, so the picture and the numbers cannot disagree.

    python3 -m results.publication.generate_lncs_figures
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = ROOT / "results"
OUT = ROOT / "paper2_submission" / "manuscript_lncs" / "figures"

# Categorical slots 1 and 2 of the validated reference palette, in fixed order.
# Checked computationally, not by eye: OKLCH lightness band PASS, chroma floor
# PASS, CVD separation dE=24.7 (target 8), normal-vision dE=33.6 (floor 15),
# contrast 4.30:1 and 3.12:1 against the surface (min 3).
SERIES = ["#2a78d6", "#eb6834"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a85"
SURFACE = "#fcfcfb"

RULE, CAL = "RuleBased", "TFIDF_LogReg_Calibrated"
LABELS = {RULE: "RuleBased consumer", CAL: "Calibrated TF-IDF consumer"}
SYSTEM_ORDER = ["random", "bm25", "tfidf", "dense", "oracle"]
SYSTEM_LABEL = {"random": "random", "bm25": "BM25", "tfidf": "TF-IDF",
                "dense": "dense", "oracle": "oracle"}


def read(path: Path) -> list:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def interaction_figure(budget: int = 3) -> Path:
    """The paper's thesis in one panel.

    x is a property of the retrieval alone; y is what that retrieval turned out
    to be worth. Two consumers, one zero line. The series sit on opposite sides
    of zero across the whole practical range and converge only at the oracle.
    """
    rows = [r for r in read(RESULTS / "retrieval" / "retrieval_summary.csv")
            if int(r["budget_k"]) == budget]
    by = {(r["interpreter"], r["system"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(5.1, 3.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Zero is the decision boundary: above it retrieval helped, below it
    # retrieval was worse than retrieving nothing at all.
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0, zorder=2)

    for slot, interp in enumerate((RULE, CAL)):
        xs, ys, lo, hi = [], [], [], []
        for system in SYSTEM_ORDER:
            r = by.get((interp, system))
            if not r:
                continue
            d = float(r["delta_vs_no_retrieval"])
            xs.append(float(r["ndcg_at_k"]))
            ys.append(d)
            lo.append(d - float(r["ci_lower"]))
            hi.append(float(r["ci_upper"]) - d)

        colour = SERIES[slot]
        # Marker shape carries identity alongside colour, so the figure survives
        # greyscale printing and colour-vision deficiency.
        marker = "o" if interp == RULE else "s"
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=colour,
                    elinewidth=1.2, capsize=2.5, alpha=0.75, zorder=3)
        ax.plot(xs, ys, color=colour, linewidth=2.0, marker=marker,
                markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=4, label=LABELS[interp])

    # Name the zero line where it cannot be mistaken for a data label.
    ax.annotate("retrieving nothing", xy=(1.06, 0), xytext=(0, 4),
                textcoords="offset points", fontsize=6.5,
                color=INK_SECONDARY, ha="right", va="bottom")

    # System names go on a top axis at their exact nDCG positions, staggered so
    # that BM25 and TF-IDF -- only 0.06 apart -- cannot collide.
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    positions, names = [], []
    for system in SYSTEM_ORDER:
        r = by.get((RULE, system))
        if r:
            positions.append(float(r["ndcg_at_k"]))
            names.append(SYSTEM_LABEL[system])
    top.set_xticks(positions)
    top.set_xticklabels(names, fontsize=6.5, color=INK_MUTED)
    top.tick_params(axis="x", length=2, colors=INK_MUTED, pad=1)
    for spine in ("top", "right", "left", "bottom"):
        top.spines[spine].set_visible(False)
    # Drop every second label a line down so close pairs never overlap.
    for i, label in enumerate(top.get_xticklabels()):
        if i % 2 == 1:
            label.set_y(label.get_position()[1] + 0.055)

    ax.set_xlabel(f"Ranking quality (nDCG@{budget})", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel(r"$\Delta J$ vs no retrieval", fontsize=8, color=INK_SECONDARY)
    ax.tick_params(labelsize=7.5, colors=INK_SECONDARY, length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.04, 1.08)

    leg = ax.legend(fontsize=7.5, frameon=False, loc="lower right",
                    handlelength=1.6, borderaxespad=0.3)
    for text in leg.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "interaction.pdf"
    fig.savefig(path, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(OUT / "interaction.png", format="png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    print("Generating LNCS figures from authoritative outputs...")
    path = interaction_figure()
    print(f"  wrote {path.relative_to(ROOT)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
