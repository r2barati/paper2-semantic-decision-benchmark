"""Consumer-dependence analyses for Experiment R.

The central claim of the paper is that retrieval utility is a property of the
*pipeline*, not of the ranking: the same retriever helps one downstream consumer
and harms another. These are the statistics that make that claim measurable
rather than anecdotal, all computed from the per-episode records.

``interaction``
    The retriever x interpreter interaction. For each retriever, the paired
    effect against no-retrieval is computed separately per interpreter; the
    interaction is their difference. A large interaction with opposite signs is
    the phenomenon: relevance that transfers for one consumer and not another.
``rank_reversal_rate``
    Over all pairs of ranking systems, the fraction whose ordering by ranking
    quality disagrees with their ordering by realised utility. Kendall's tau
    summarises agreement; the reversal rate states the failure probability
    directly, which is what an evaluator inherits when they select a system on
    nDCG alone.
``relevance_reward_correlation``
    Episode-level Pearson and Spearman correlation between ranking quality and
    realised reward, per interpreter. This is deliberately computed within an
    interpreter: pooling across interpreters would average away the effect the
    paper is about.
``harmful_retrieval_rate``
    The fraction of retrieved documents that are non-relevant *and* describe a
    regime other than the true one, i.e. evidence that actively moves the
    controller rather than merely occupying a slot.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EPISODES = ROOT / "results" / "retrieval" / "retrieval_episodes.csv"

# References, not retrieval systems: `none` retrieves nothing and `oracle`
# selects by ground-truth relevance. Both bracket the range and are excluded
# from any statement about how *rankers* compare.
REFERENCE_SYSTEMS = ("none", "oracle")


def read_episodes(path: Path = DEFAULT_EPISODES) -> list:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _paired_by_episode(rows, system, reference="none"):
    """Reward differences against the reference, keyed by (seed, regime)."""
    ref = {(r["seed"], r["regime"]): float(r["reward"])
           for r in rows if r["system"] == reference}
    out = {}
    for r in rows:
        if r["system"] != system:
            continue
        key = (r["seed"], r["regime"])
        if key in ref:
            out[key] = float(r["reward"]) - ref[key]
    return out


def interaction(rows, budget: int, systems: Optional[Iterable[str]] = None,
                order: Optional[Iterable[str]] = None) -> list:
    """Retriever x interpreter interaction at one evidence budget.

    Reported as: the paired effect of each retriever under each interpreter,
    their difference (the interaction), and whether the two effects have
    opposite signs -- the case in which a ranking cannot be evaluated
    independently of its consumer.

    The interaction is ``effect[first] - effect[second]``. ``order`` fixes which
    is which; without it the two interpreters are taken in sorted order, so the
    sign would otherwise depend on how the consumers happen to be named. The
    unambiguous field is ``favours``: the consumer for which the ranking is
    worth more, which no renaming can change.
    """
    subset = [r for r in rows if int(r["budget_k"]) == budget]
    interpreters = sorted({r["interpreter"] for r in subset})
    if len(interpreters) != 2:
        raise ValueError(f"interaction needs exactly two interpreters, got {interpreters}")
    if order is not None:
        order = list(order)
        if sorted(order) != interpreters:
            raise ValueError(f"order {order} does not match interpreters {interpreters}")
        interpreters = order
    a, b = interpreters

    if systems is None:
        systems = sorted({r["system"] for r in subset} - set(REFERENCE_SYSTEMS))

    out = []
    for system in systems:
        eff = {}
        for interp in (a, b):
            rows_i = [r for r in subset if r["interpreter"] == interp]
            diffs = _paired_by_episode(rows_i, system)
            eff[interp] = float(np.mean(list(diffs.values()))) if diffs else float("nan")
        delta = eff[a] - eff[b]
        out.append({
            "system": system,
            "budget_k": budget,
            f"effect_{a}": eff[a],
            f"effect_{b}": eff[b],
            "interaction": delta,
            "interaction_magnitude": abs(delta),
            "favours": a if eff[a] >= eff[b] else b,
            "sign_flip": bool(eff[a] > 0) != bool(eff[b] > 0),
        })
    return out


def rank_reversal_rate(rows, budget: int, quality_key="ndcg_at_k") -> dict:
    """How often does ranking quality order two systems the wrong way?

    Computed over unordered pairs of ranking systems, separately per
    interpreter. `reversal_rate` is discordant / (concordant + discordant);
    Kendall's tau-b is `1 - 2 * reversal_rate` on the same pairs.
    """
    subset = [r for r in rows if int(r["budget_k"]) == budget]
    out = {}
    for interp in sorted({r["interpreter"] for r in subset}):
        rows_i = [r for r in subset if r["interpreter"] == interp]
        systems = sorted({r["system"] for r in rows_i} - set(REFERENCE_SYSTEMS))
        agg = {}
        for system in systems:
            srows = [r for r in rows_i if r["system"] == system]
            agg[system] = (
                float(np.mean([float(r[quality_key]) for r in srows])),
                float(np.mean([float(r["reward"]) for r in srows])),
            )
        conc = disc = ties = 0
        reversed_pairs = []
        for x, y in combinations(systems, 2):
            dq = agg[x][0] - agg[y][0]
            du = agg[x][1] - agg[y][1]
            if dq == 0 or du == 0:
                ties += 1
                continue
            if (dq > 0) == (du > 0):
                conc += 1
            else:
                disc += 1
                better = x if dq > 0 else y
                worse = y if dq > 0 else x
                reversed_pairs.append(f"{better}>{worse} by nDCG, reversed by reward")
        total = conc + disc
        out[interp] = {
            "n_systems": len(systems),
            "n_pairs": total + ties,
            "concordant": conc,
            "discordant": disc,
            "ties": ties,
            "reversal_rate": (disc / total) if total else float("nan"),
            "kendall_tau_b": ((conc - disc) / total) if total else float("nan"),
            "reversed_pairs": reversed_pairs,
        }
    return out


def relevance_reward_correlation(rows, budget: int, quality_key="ndcg_at_k") -> dict:
    """Episode-level correlation between ranking quality and realised reward.

    Within an interpreter, across all systems and episodes. Rewards are
    centred within (seed, regime) first, because episode difficulty varies far
    more than the retrieval effect does and would otherwise dominate the
    correlation.
    """
    subset = [r for r in rows if int(r["budget_k"]) == budget]
    out = {}
    for interp in sorted({r["interpreter"] for r in subset}):
        rows_i = [r for r in subset if r["interpreter"] == interp]
        by_episode = defaultdict(list)
        for r in rows_i:
            by_episode[(r["seed"], r["regime"])].append(r)

        q, w = [], []
        for _, group in by_episode.items():
            rewards = np.array([float(r["reward"]) for r in group])
            centred = rewards - rewards.mean()
            q.extend(float(r[quality_key]) for r in group)
            w.extend(centred.tolist())
        q_arr, w_arr = np.asarray(q), np.asarray(w)

        def spearman(x, y):
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            return float(np.corrcoef(rx, ry)[0, 1])

        out[interp] = {
            "n": int(q_arr.size),
            "pearson": float(np.corrcoef(q_arr, w_arr)[0, 1]),
            "spearman": spearman(q_arr, w_arr),
        }
    return out


def harmful_retrieval_rate(rows, budget: int) -> dict:
    """Fraction of retrieved documents that actively mislead, per system."""
    subset = [r for r in rows if int(r["budget_k"]) == budget]
    out = {}
    for system in sorted({r["system"] for r in subset}):
        srows = [r for r in subset if r["system"] == system]
        harmful = np.array([float(r["n_misleading_retrieved"]) for r in srows])
        retrieved = np.array([float(r["n_retrieved"]) for r in srows])
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(retrieved > 0, harmful / np.maximum(retrieved, 1), 0.0)
        out[system] = {
            "mean_harmful_docs": float(harmful.mean()),
            "harmful_rate": float(rate.mean()),
            "episodes_with_any_harmful": float((harmful > 0).mean()),
        }
    return out
