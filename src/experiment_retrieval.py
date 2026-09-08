"""Experiment R: does retrieval quality predict downstream decision utility?

This is the information-access experiment.  Everything downstream of retrieval
is held fixed -- the same interpreters and the same controller used by the rest
of the benchmark -- so any difference in operational value is attributable to
which evidence was selected.

Protocol per episode (seed x true regime):

1. Build the operator's candidate pool (:mod:`src.warning_corpus`): a couple of
   current reports about their own node, many structurally identical reports
   about *other* nodes, stale reports, and routine traffic.
2. Each retrieval system ranks the pool against one fixed operational query
   under a **matched evidence budget** of k documents.
3. The concatenated top-k text goes to a fixed interpreter, which returns a
   regime belief.
4. The fixed controller consumes the belief and the simulator returns reward.

Reported side by side: retrieval quality (nDCG@k, P@k, R@k, MRR) and downstream
utility (reward, paired effect vs the no-retrieval control, SIVR).  The
question the experiment answers is whether the two rank systems the same way.

Two references bracket the systems: ``none`` retrieves nothing (the
no-retrieval control) and ``oracle`` selects the genuinely relevant documents
(the evidence-selection ceiling).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.events import Regime, REGIME_WARNING_TEMPLATES
from src.classical_baseline import TFIDFLogReg, VARIANT_CALIBRATED
from src.interpreter import (
    no_info_regime_belief, rule_based_regime_extract,
)
from src.experiment_phase5 import _run_p5_episode
from src.metrics import (
    signed_sivr, crossed_bootstrap_ci, weighted_benchmark_return,
)
from src.retrieval import (
    RETRIEVAL_SYSTEMS, SYSTEM_NONE, SYSTEM_ORACLE, FrozenEmbeddings,
    rank, retrieval_metrics,
)
from src.warning_corpus import build_episode_corpus, OPERATIONAL_QUERY

RESULTS_DIR = ROOT / "results" / "retrieval"

# Confirmation seeds, disjoint from every other experiment's seed range.
RETRIEVAL_SEEDS = list(range(5000, 5030))
RETRIEVAL_REGIMES = [Regime.NORMAL, Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]

# Matched evidence budgets. Every system sees the same k.
BUDGETS = [1, 3, 5]

INTERPRETER_RULEBASED = "RuleBased"
INTERPRETER_TFIDF_CAL = "TFIDF_LogReg_Calibrated"
INTERPRETERS = [INTERPRETER_RULEBASED, INTERPRETER_TFIDF_CAL]

CONTROLLER = "CausalOptimizer"


def _save_csv(rows, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interpret(interpreter: str, text: str, tfidf_cal):
    """Fixed interpreters. Empty evidence yields the benchmark prior."""
    if not text.strip():
        return no_info_regime_belief()
    if interpreter == INTERPRETER_RULEBASED:
        return rule_based_regime_extract(text)
    if interpreter == INTERPRETER_TFIDF_CAL:
        return tfidf_cal.predict_regime(text).to_regime_interpretation()
    raise ValueError(f"unknown interpreter {interpreter!r}")


def run_retrieval_experiment(
    seeds=None, regimes=None, budgets=None, systems=None, interpreters=None,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    seeds = list(seeds or RETRIEVAL_SEEDS)
    regimes = list(regimes or RETRIEVAL_REGIMES)
    budgets = list(budgets or BUDGETS)
    systems = list(systems or RETRIEVAL_SYSTEMS)
    interpreters = list(interpreters or INTERPRETERS)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT R: RETRIEVAL QUALITY vs DOWNSTREAM DECISION UTILITY")
    print("=" * 70)
    print(f"seeds={len(seeds)} regimes={len(regimes)} systems={len(systems)} "
          f"budgets={budgets} interpreters={len(interpreters)}")
    print(f"query: {OPERATIONAL_QUERY[:88]}...")

    texts = [t["text"] for t in REGIME_WARNING_TEMPLATES]
    labels = [t["regime"].value for t in REGIME_WARNING_TEMPLATES]
    tfidf_cal = TFIDFLogReg(variant=VARIANT_CALIBRATED, seed=42).fit(texts, labels)

    embeddings = FrozenEmbeddings()
    rows = []
    total = len(seeds) * len(regimes) * len(systems) * len(budgets) * len(interpreters)
    done = 0

    for seed in seeds:
        for regime in regimes:
            corpus = build_episode_corpus(seed=seed, true_regime=regime)
            misleading = corpus.misleading_ids()

            rankings = {}
            for system in systems:
                rankings[system] = rank(system, corpus, seed=seed, embeddings=embeddings)

            for system in systems:
                ranked = rankings[system]
                for k in budgets:
                    top = ranked[:k]
                    evidence = "\n\n".join(corpus.doc(d).text for d in top)
                    quality = retrieval_metrics(ranked, corpus.relevance, k)
                    n_misleading = sum(1 for d in top if d in misleading)

                    for interpreter in interpreters:
                        belief = _interpret(interpreter, evidence, tfidf_cal)
                        result, _ = _run_p5_episode(
                            seed=seed, regime=regime,
                            sensor=f"{system}@{k}",
                            controller=CONTROLLER,
                            template_id=f"retr_{system}_k{k}",
                            regime_interp=belief,
                        )
                        probs = belief.normalized().regime_probabilities
                        rows.append({
                            "seed": seed,
                            "regime": regime.value,
                            "system": system,
                            "budget_k": k,
                            "interpreter": interpreter,
                            "reward": result.total_profit,
                            "fill_rate": result.fill_rate,
                            "precision_at_k": quality[f"precision_at_{k}"],
                            "recall_at_k": quality[f"recall_at_{k}"],
                            "ndcg_at_k": quality[f"ndcg_at_{k}"],
                            "mrr": quality["mrr"],
                            "n_retrieved": len(top),
                            "n_misleading_retrieved": n_misleading,
                            "belief_true_regime": probs.get(regime.value, 0.0),
                            "belief_argmax": belief.normalized().most_likely_regime,
                            "belief_correct": belief.normalized().most_likely_regime == regime.value,
                            "pool_size": len(corpus.documents),
                            "n_relevant_in_pool": corpus.n_relevant,
                        })
                        done += 1
                        if done % 200 == 0:
                            print(f"  [{done}/{total}] seed={seed} {system}@{k} "
                                  f"{interpreter} reward={result.total_profit:.1f}")

    _save_csv(rows, output_dir / "retrieval_episodes.csv")
    summary = summarize(rows, output_dir)
    print(f"\nOutputs saved to {output_dir}")
    return {"rows": rows, "summary": summary}


def _index(rows, key, budget, interpreter):
    """regime -> family -> variant -> {seed: value} for the weighted estimand.

    The retrieval study has one template family per regime (the pool is
    generated, not authored), so family and variant are constant and the
    weighting reduces to a balanced mean over regimes.
    """
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        if row["budget_k"] != budget or row["interpreter"] != interpreter:
            continue
        out[row["regime"]]["pool"]["generated"][row["seed"]] = float(row[key])
    return out


def summarize(rows, output_dir: Path) -> list:
    budgets = sorted({r["budget_k"] for r in rows})
    interpreters = sorted({r["interpreter"] for r in rows})
    systems = sorted({r["system"] for r in rows})

    summary = []
    for interpreter in interpreters:
        for k in budgets:
            subset = [r for r in rows if r["budget_k"] == k and r["interpreter"] == interpreter]
            by_system = {s: [r for r in subset if r["system"] == s] for s in systems}

            reward_of = {}
            for system in systems:
                idx = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                for row in by_system[system]:
                    idx[row["regime"]]["pool"]["generated"][row["seed"]] = row["reward"]
                reward_of[system] = weighted_benchmark_return(idx)["value"]

            base = reward_of.get(SYSTEM_NONE, 0.0)
            ceiling = reward_of.get(SYSTEM_ORACLE, base)

            for system in systems:
                srows = by_system[system]
                if not srows:
                    continue
                paired = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                none_by = {(r["seed"], r["regime"]): r["reward"]
                           for r in by_system.get(SYSTEM_NONE, [])}
                for row in srows:
                    ref = none_by.get((row["seed"], row["regime"]))
                    if ref is None:
                        continue
                    paired[row["regime"]]["pool"]["generated"][row["seed"]] = row["reward"] - ref

                ci = crossed_bootstrap_ci(paired, n_boot=2000, seed=42) if paired else None
                sivr = signed_sivr(reward_of[system], base, ceiling)

                summary.append({
                    "interpreter": interpreter,
                    "budget_k": k,
                    "system": system,
                    "reward": round(reward_of[system], 3),
                    "delta_vs_no_retrieval": round(reward_of[system] - base, 3),
                    "ci_lower": round(ci["ci_lower"], 3) if ci else None,
                    "ci_upper": round(ci["ci_upper"], 3) if ci else None,
                    "p_value": round(ci["p_value"], 5) if ci else None,
                    "p_value_floor": round(ci["p_value_floor"], 6) if ci else None,
                    "signed_sivr": round(sivr.value, 4) if sivr.value == sivr.value else None,
                    "sivr_status": sivr.status,
                    "ndcg_at_k": round(float(np.mean([r["ndcg_at_k"] for r in srows])), 4),
                    "precision_at_k": round(float(np.mean([r["precision_at_k"] for r in srows])), 4),
                    "recall_at_k": round(float(np.mean([r["recall_at_k"] for r in srows])), 4),
                    "mrr": round(float(np.mean([r["mrr"] for r in srows])), 4),
                    "mean_misleading_retrieved": round(
                        float(np.mean([r["n_misleading_retrieved"] for r in srows])), 4),
                    "belief_accuracy": round(
                        float(np.mean([1.0 if r["belief_correct"] else 0.0 for r in srows])), 4),
                    "n_episodes": len(srows),
                })

    _save_csv(summary, output_dir / "retrieval_summary.csv")
    _write_rank_agreement(summary, output_dir)
    return summary


def _write_rank_agreement(summary, output_dir: Path) -> None:
    """Does retrieval quality rank systems the way downstream utility does?

    Kendall's tau-b between the nDCG@k ordering and the reward ordering, over
    the ranking systems only (the two references bracket the range and are
    excluded, since `none` retrieves nothing and `oracle` is not a retrieval
    system).
    """
    from itertools import combinations

    lines = ["# Retrieval quality vs downstream utility: rank agreement", ""]
    rows_out = []
    grouped: dict = defaultdict(list)
    for row in summary:
        if row["system"] in (SYSTEM_NONE, SYSTEM_ORACLE):
            continue
        grouped[(row["interpreter"], row["budget_k"])].append(row)

    for (interpreter, k), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        conc = disc = 0
        for a, b in combinations(group, 2):
            dq = a["ndcg_at_k"] - b["ndcg_at_k"]
            du = a["reward"] - b["reward"]
            if dq == 0 or du == 0:
                continue
            if (dq > 0) == (du > 0):
                conc += 1
            else:
                disc += 1
        total = conc + disc
        tau = (conc - disc) / total if total else float("nan")
        rows_out.append({
            "interpreter": interpreter, "budget_k": k,
            "n_systems": len(group), "concordant": conc, "discordant": disc,
            "kendall_tau_b": round(tau, 4) if total else None,
            "best_by_ndcg": max(group, key=lambda r: r["ndcg_at_k"])["system"],
            "best_by_reward": max(group, key=lambda r: r["reward"])["system"],
        })

    _save_csv(rows_out, output_dir / "rank_agreement.csv")
    for row in rows_out:
        lines.append(
            f"- {row['interpreter']} @ k={row['budget_k']}: tau_b={row['kendall_tau_b']}, "
            f"best by nDCG = {row['best_by_ndcg']}, best by reward = {row['best_by_reward']}"
        )
    (output_dir / "rank_agreement.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_retrieval_experiment()
