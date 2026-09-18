"""Query-level semantic-transfer analysis for the frozen V3 main experiment.

Code-only scaffolding: computes per-(system, query) IR metrics and
per-(system, consumer, model, k, query) semantic metrics, plus preregistered
Q1-Q7 helpers. Makes NO scientific claims; full LLM beliefs are not ready.

Conventions (frozen):
- IR via ``ir_measures`` 0.4.3, trec_eval definition (linear gains), per
  ``configs/v3/metrics.md``: nDCG@10, Recall@10, Recall@20, RR (no cutoff,
  matching ``results/v3main/retrieval_ir_draft.json``).
- Regimes: ("normal", "supplier_delay", "demand_surge").
- Brier score: mean squared error averaged over the 3 classes
  (``mean((p - onehot)^2)``); perfect beliefs -> 0.0, uniform -> 2/9.
- ECE: 10 equal-width bins over confidence, standard
  ``sum_b |acc_b - conf_b| * (n_b / n)``; computed at aggregate level
  (single-row ECE is degenerate), per-row rows carry correctness+confidence.
- Abstained rows: ``pred_regime == "abstain"``, accuracy 0.0; Brier/NLL use
  the reported (prior) probabilities.
- All paired statistics operate at query level with paired bootstrap CIs.

Sensitivity: every function is pure over DataFrames, so an alternative belief
cache (e.g. resolving the 24 first-wins divergences logged in
``results/v3main/beliefs_cache_awq_nondeterminism.json``) is supported by
re-running with an alternative beliefs frame; see :func:`with_alternative_cache`.
Use :func:`load_nondeterminism_report` to isolate the 15 disputed keys
(``same_parsed_belief is False``) with their later-variant payloads, rebuild an
alt frame flipping exactly those keys, and pass both frames to
:func:`sensitivity_max_delta` for max |delta| on effects/CIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REGIMES = ("normal", "supplier_delay", "demand_surge")
P_COLS = {
    "normal": "p_normal",
    "supplier_delay": "p_supplier_delay",
    "demand_surge": "p_demand_surge",
}
HIT_GRADE_THRESHOLD = 2  # Q4: top-k hit iff any doc has qrel grade >= 2
N_ECE_BINS = 10
ABSTAIN_LABEL = "abstain"

# Q6 framing per configs/v3/MODEL_AMENDMENT_PHI4.md (defaults only; callers
# may pass any model labels present in their beliefs frame).
# Phi-4 was rejected at the probe gate (NOTES.md); the frozen frontier anchor
# is Qwen3-14B-AWQ, so Q6 defaults are now the within-family scale check.
MODEL_PHI4 = "stelterlab/phi-4-AWQ"
MODEL_QWEN3_8B = "Qwen/Qwen3-8B-AWQ"
MODEL_QWEN3_14B = "Qwen/Qwen3-14B-AWQ"

ROOT = Path(__file__).resolve().parent.parent

# Canonical trec locations for the four retrieval systems of the main run.
MAIN_TRECS = {
    "bm25": "kaggle/inputs_main/bm25_full.trec",
    "dense": "runs/v3main/dense_full.trec",
    "hybrid": "runs/v3main/hybrid_k120_full.trec",
    "rerank": "runs/v3main/rerank_full.trec",
}
QRELS_TEST = "data/v3/qrels/test.tsv"
QRELS_DEV = "data/v3/qrels/dev.tsv"
QUERIES_JSONL = "data/v3/queries.jsonl"
SPLITS_JSON = "data/v3/splits/splits.json"
# AWQ merge-collision log (NOTES.md 17:50 entry): 1285 content-shared keys,
# 24 diverged raw / 15 with cross-session parsed-belief differences, first-wins
# merge. Sensitivity must flip those 15 to the later variant and re-run.
NONDETERMINISM_JSON = "results/v3main/beliefs_cache_awq_nondeterminism.json"


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_qrels_tsv(path) -> dict[str, dict[str, int]]:
    """Load a 3-col ``query-id corpus-id score`` qrels TSV (with header)."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        header = f.readline().split()
        has_header = header[0].lower().startswith("query")
        if not has_header:  # pragma: no cover - all committed qrels have headers
            _ingest_qrel_line(qrels, header)
        for line in f:
            parts = line.split()
            if parts:
                _ingest_qrel_line(qrels, parts)
    return qrels


def _ingest_qrel_line(qrels, parts) -> None:
    qid, did, grade = parts[0], parts[1], int(parts[2])
    qrels.setdefault(qid, {})[did] = grade


def load_trec(path) -> dict[str, list[str]]:
    """Load a TREC run file into ``{qid: [doc_id in rank order]}``."""
    run: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, did = parts[0], parts[2]
            run.setdefault(qid, []).append(did)
    return run


def load_beliefs(path) -> pd.DataFrame:
    """Load a beliefs parquet/JSONL file (row schema: src/beliefs_v3main.py)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path, lines=True)


def load_query_meta(queries_jsonl=QUERIES_JSONL) -> dict[str, dict]:
    """Map query_id -> metadata dict (true_regime, entity_node, split)."""
    p = Path(queries_jsonl)
    if not p.is_absolute():
        p = ROOT / p
    meta = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                meta[q["_id"]] = dict(q.get("metadata", {}))
    return meta


def load_splits(splits_json=SPLITS_JSON) -> dict:
    p = Path(splits_json)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text())


def with_alternative_cache(
    df: pd.DataFrame,
    alt_df: pd.DataFrame,
    keys=("query_id", "system", "consumer", "model", "k"),
) -> pd.DataFrame:
    """Return ``df`` with rows replaced by ``alt_df`` rows on matching keys.

    Sensitivity hook for the AWQ first-wins divergences: build ``alt_df``
    from an alternative cache resolution, then re-run every analysis on the
    returned frame. Rows in ``alt_df`` whose keys are absent from ``df``
    are appended.
    """
    keys = list(keys)
    alt_indexed = alt_df.set_index(keys)
    out = df.set_index(keys)
    alt_indexed = alt_indexed.reindex(columns=out.columns)
    out.update(alt_indexed)
    new_keys = alt_indexed.index.difference(out.index)
    if len(new_keys):
        out = pd.concat([out, alt_indexed.loc[new_keys]])
    return out.reset_index()


# ----------------------------------------------------------------------------
# Nondeterminism sensitivity (NOTES.md 17:50 merge collision)
# ----------------------------------------------------------------------------

def parse_cache_raw(raw: str) -> dict:
    """Parse one ``kept_raw``/``later_raw`` payload from the collision log."""
    return json.loads(raw)


def load_nondeterminism_report(
    path=NONDETERMINISM_JSON,
    root: Path = ROOT,
) -> dict:
    """Parse ``beliefs_cache_awq_nondeterminism.json`` (schema-first helper).

    Real-file schema: top-level ``{"policy", "n_diverged_files", "diverged"}``
    where each diverged entry holds ``file`` (content-addressed cache key),
    ``kept_shard`` (``"earlier"`` under first-wins), ``later_shard``,
    ``same_parsed_belief`` (bool), ``kept_raw``/``later_raw`` (JSON strings;
    C1-shape ``normal``/``supplier_delay``/``demand_surge`` + unused
    estimated_* fields, or C3-shape ``entity_match``/``event``/``fresh``/
    ``stance``/``confidence``).

    Returns ``{"policy", "n_diverged_files", "diverged", "disputed",
    "n_disputed", "n_same_belief"}``; each entry is enriched with
    ``kept_parsed``/``later_parsed`` dicts. ``disputed`` (``same_parsed_belief
    is False``; 15 rows in the frozen log) is the sensitivity set: re-resolve
    exactly these keys to ``later_raw`` and re-run.
    """
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    report = json.loads(p.read_text())
    diverged = []
    for entry in report.get("diverged", []):
        enriched = dict(entry)
        enriched["kept_parsed"] = parse_cache_raw(entry["kept_raw"])
        enriched["later_parsed"] = parse_cache_raw(entry["later_raw"])
        diverged.append(enriched)
    disputed = [e for e in diverged if not e.get("same_parsed_belief", True)]
    return {
        "policy": report.get("policy", ""),
        "n_diverged_files": report.get("n_diverged_files", len(diverged)),
        "diverged": diverged,
        "disputed": disputed,
        "n_disputed": len(disputed),
        "n_same_belief": len(diverged) - len(disputed),
    }


def _effect_triplet(result) -> tuple:
    """Extract (estimate, ci_lo, ci_hi) from a comparison result."""
    if isinstance(result, dict):
        return (
            float(result.get("estimate", float("nan"))),
            float(result.get("ci_lo", float("nan"))),
            float(result.get("ci_hi", float("nan"))),
        )
    return float(result), float("nan"), float("nan")


def sensitivity_max_delta(
    base_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    comparisons: dict,
) -> dict:
    """Re-run paired comparisons on base vs alt frames; report max |delta|.

    ``comparisons`` maps ``name -> fn(df)`` where each ``fn`` runs one key
    paired comparison (e.g. a ``paired_contrast`` / ``q3_...`` / ``q5_...``
    lambda) and returns an estimate dict (``estimate`` + ``ci_lo``/``ci_hi``)
    or a bare float. ``alt_df`` is typically built with
    :func:`with_alternative_cache` after re-resolving the 15 disputed keys
    from :func:`load_nondeterminism_report` to their later variant.

    Row join note: belief rows carry no cache-key column, so the
    cache-file -> row mapping must be supplied by the caller when building
    ``alt_df`` (explicit per-row overrides); this runner stays pure over
    the two frames so the flip itself is fully unit-testable.

    Returns ``{"per_comparison": DataFrame, "max_abs_effect_delta": float,
    "max_abs_ci_delta": float}`` with deltas defined alt-minus-base.
    """
    if not comparisons:
        raise ValueError("comparisons must be non-empty")
    rows = []
    for name, fn in comparisons.items():
        b_est, b_lo, b_hi = _effect_triplet(fn(base_df))
        a_est, a_lo, a_hi = _effect_triplet(fn(alt_df))
        rows.append({
            "comparison": name,
            "base_estimate": b_est,
            "alt_estimate": a_est,
            "effect_delta": float(a_est - b_est),
            "abs_effect_delta": float(abs(a_est - b_est)),
            "base_ci_lo": b_lo,
            "base_ci_hi": b_hi,
            "alt_ci_lo": a_lo,
            "alt_ci_hi": a_hi,
            "ci_lo_delta": float(a_lo - b_lo),
            "ci_hi_delta": float(a_hi - b_hi),
        })
    per = pd.DataFrame(rows)
    max_eff = float(per["abs_effect_delta"].max())
    ci_vals = per[["ci_lo_delta", "ci_hi_delta"]].abs().to_numpy(dtype=float)
    max_ci = float(np.nanmax(ci_vals)) if np.isfinite(ci_vals).any() else float("nan")
    return {
        "per_comparison": per,
        "max_abs_effect_delta": max_eff,
        "max_abs_ci_delta": max_ci,
    }


# ----------------------------------------------------------------------------
# IR: per-(system, query), trec_eval definition via ir_measures
# ----------------------------------------------------------------------------

def ir_per_query(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    query_ids=None,
) -> pd.DataFrame:
    """nDCG@10, Recall@10/20, RR per query via ir_measures (trec_eval defn).

    Missing queries in ``run`` score 0.0 on every metric.
    """
    import ir_measures
    from ir_measures import Qrel, ScoredDoc, nDCG, R, RR
    from ir_measures import iter_calc

    measures = [nDCG @ 10, R @ 10, R @ 20, RR]
    qrel_objs, run_objs = [], []
    qids = query_ids if query_ids is not None else sorted(set(qrels) | set(run))
    for qid in qids:
        for did, grade in qrels.get(qid, {}).items():
            qrel_objs.append(Qrel(qid, did, int(grade), "0"))
        for rank, did in enumerate(run.get(qid, [])):
            # Scores need only preserve rank order.
            run_objs.append(ScoredDoc(qid, did, float(len(run.get(qid, [])) - rank)))
    per_q: dict[str, dict[str, float]] = {qid: {} for qid in qids}
    for m in iter_calc(measures, qrel_objs, run_objs):
        per_q[m.query_id][str(m.measure)] = float(m.value)
    rows = [
        {
            "query_id": qid,
            "ndcg10": per_q[qid].get("nDCG@10", 0.0),
            "recall10": per_q[qid].get("R@10", 0.0),
            "recall20": per_q[qid].get("R@20", 0.0),
            "mrr": per_q[qid].get("RR", 0.0),
        }
        for qid in qids
    ]
    return pd.DataFrame(rows)


def ir_per_system(
    trec_paths: dict[str, str] | None = None,
    qrels_path: str = QRELS_TEST,
    query_ids=None,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Stacked per-(system, query) IR frame for the given trecs + qrels."""
    trec_paths = trec_paths if trec_paths is not None else dict(MAIN_TRECS)
    qp = Path(qrels_path)
    if not qp.is_absolute():
        qp = root / qp
    qrels = load_qrels_tsv(qp)
    frames = []
    for system, rel in trec_paths.items():
        tp = Path(rel)
        if not tp.is_absolute():
            tp = root / tp
        per_q = ir_per_query(qrels, load_trec(tp), query_ids=query_ids)
        per_q.insert(0, "system", system)
        frames.append(per_q)
    return pd.concat(frames, ignore_index=True)


def ir_aggregate(ir_df: pd.DataFrame) -> pd.DataFrame:
    """Mean IR metrics per system (macro-average over queries)."""
    return (
        ir_df.groupby("system")[["ndcg10", "recall10", "recall20", "mrr"]]
        .mean()
        .reset_index()
    )


# ----------------------------------------------------------------------------
# Semantic: per-(system, consumer, model, k, query)
# ----------------------------------------------------------------------------

def query_semantic_metrics(
    p_normal: float,
    p_supplier_delay: float,
    p_demand_surge: float,
    true_regime: str,
    abstain: bool = False,
    confidence: float = 0.0,
    eps: float = 1e-12,
) -> dict:
    """Regime accuracy, Brier, NLL for one belief row vs ``true_regime``."""
    if true_regime not in REGIMES:
        raise ValueError(f"unknown true_regime: {true_regime!r}")
    probs = np.array(
        [float(p_normal), float(p_supplier_delay), float(p_demand_surge)],
        dtype=float,
    )
    if np.any(probs < 0) or abs(float(probs.sum()) - 1.0) > 1e-6:
        raise ValueError(f"probabilities must be in [0,1] and sum to 1: {probs}")
    onehot = np.array([1.0 if r == true_regime else 0.0 for r in REGIMES])
    p_true = float(probs[REGIMES.index(true_regime)])
    if abstain:
        pred_regime: str = ABSTAIN_LABEL
        accuracy = 0.0
    else:
        pred_regime = REGIMES[int(np.argmax(probs))]
        accuracy = 1.0 if pred_regime == true_regime else 0.0
    brier = float(np.mean((probs - onehot) ** 2))
    nll = float(-np.log(max(p_true, eps)))
    return {
        "pred_regime": pred_regime,
        "accuracy": accuracy,
        "brier": brier,
        "nll": nll,
        "p_true": p_true,
        "abstain": bool(abstain),
        "confidence": float(confidence),
    }


def semantic_per_query(
    beliefs_df: pd.DataFrame,
    query_meta: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Per-row semantic metrics, preserving (system, consumer, model, k).

    Uses the ``true_regime`` column when present, else joins ``query_meta``
    (or loads ``data/v3/queries.jsonl``).
    """
    df = beliefs_df.copy()
    if "true_regime" not in df.columns or df["true_regime"].isna().any():
        meta = query_meta if query_meta is not None else load_query_meta()
        df["true_regime"] = df["query_id"].map(
            lambda q: meta.get(q, {}).get("true_regime", "")
        )
    records = [
        query_semantic_metrics(
            r["p_normal"],
            r["p_supplier_delay"],
            r["p_demand_surge"],
            r["true_regime"],
            abstain=bool(r.get("abstain", False)),
            confidence=float(r.get("confidence", 0.0)),
        )
        for _, r in df.iterrows()
    ]
    met = pd.DataFrame(records)
    keep = [c for c in ("query_id", "system", "consumer", "model", "k",
                        "true_regime", "split") if c in df.columns]
    return pd.concat([df[keep].reset_index(drop=True), met], axis=1)


def ece_10bin(
    confidence: np.ndarray | pd.Series,
    correct: np.ndarray | pd.Series,
    n_bins: int = N_ECE_BINS,
) -> float:
    """Expected calibration error over ``n_bins`` equal-width confidence bins."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.shape != corr.shape:
        raise ValueError("confidence and correct must have the same shape")
    n = len(conf)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        n_b = int(in_bin.sum())
        if n_b:
            ece += abs(float(corr[in_bin].mean()) - float(conf[in_bin].mean())) * n_b / n
    return float(ece)


def aggregate_semantic(
    sem_df: pd.DataFrame,
    group_keys=("system", "consumer", "model", "k"),
    n_bins: int = N_ECE_BINS,
) -> pd.DataFrame:
    """Mean accuracy/Brier/NLL, abstention rate, mean confidence, ECE per group."""
    rows = []
    for keys, g in sem_df.groupby(list(group_keys)):
        keys = (keys,) if len(group_keys) == 1 else tuple(keys)
        rows.append({
            **dict(zip(group_keys, keys)),
            "n": len(g),
            "accuracy": float(g["accuracy"].mean()),
            "brier": float(g["brier"].mean()),
            "nll": float(g["nll"].mean()),
            "abstention_rate": float(g["abstain"].mean()),
            "confidence": float(g["confidence"].mean()),
            "ece": ece_10bin(g["confidence"].to_numpy(),
                             g["accuracy"].to_numpy(), n_bins=n_bins),
        })
    cols = list(group_keys) + ["n", "accuracy", "brier", "nll",
                               "abstention_rate", "confidence", "ece"]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------
# Generic paired statistics (query level)
# ----------------------------------------------------------------------------

def paired_bootstrap_ci(
    x,
    y=None,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for the mean of paired differences ``x - y`` (or ``x``).

    Resamples query indices with replacement; deterministic given ``seed``.
    """
    x = np.asarray(x, dtype=float)
    diffs = x - np.asarray(y, dtype=float) if y is not None else x
    if diffs.shape[0] == 0:
        raise ValueError("empty input")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "estimate": float(np.mean(diffs)),
        "n": int(len(diffs)),
        "ci_lo": float(np.percentile(means, lo_q)),
        "ci_hi": float(np.percentile(means, hi_q)),
        "ci_level": float(ci),
        "n_boot": int(n_boot),
    }


def bootstrap_paired_p(x, y=None, n_boot=10000, seed=0) -> float:
    """Centred bootstrap test-inversion p for mean(x - y) == 0.

    (1 + #{|b* - point| >= |point|}) / (n_boot + 1); bounded below by
    1/(n_boot+1), never exactly zero. Same resampling as paired_bootstrap_ci
    would use at this seed, so CIs and p-values are mutually consistent.
    """
    x = np.asarray(x, dtype=float)
    diffs = x - np.asarray(y, dtype=float) if y is not None else x
    if diffs.shape[0] == 0:
        raise ValueError("empty input")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    point = float(np.mean(diffs))
    return float((1 + np.sum(np.abs(means - point) >= abs(point))) / (n_boot + 1))


def unpaired_diff_ci(
    a,
    b,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for ``mean(a) - mean(b)`` (independent groups, e.g. Q4)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("empty group")
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    dist = ma - mb
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "estimate": float(np.mean(a) - np.mean(b)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "ci_lo": float(np.percentile(dist, lo_q)),
        "ci_hi": float(np.percentile(dist, hi_q)),
        "ci_level": float(ci),
        "n_boot": int(n_boot),
    }


def holm_correction(p_values) -> pd.DataFrame:
    """Holm step-down adjustment. Returns p, adj. p, reject (alpha=0.05)."""
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adj[idx] = min(running_max, 1.0)
    out = pd.DataFrame({"p_value": p, "p_holm": adj,
                        "reject_holm_05": adj < 0.05})
    return out


def rank_corr(a, b) -> dict:
    """Spearman rho + Kendall tau (with two-sided p-values) for two vectors.

    Asymptotic p-values from scipy break down at tiny n (spearmanr returns
    p=0.0 for perfect n=4 correlation); ``spearman_p_exact`` /
    ``kendall_p_exact`` give exact permutation p-values for n<=8, else None.
    """
    import itertools
    import math
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rho, p_rho = stats.spearmanr(a, b)
    tau, p_tau = stats.kendalltau(a, b)
    out = {"n": int(len(a)), "spearman_rho": float(rho),
           "spearman_p": float(p_rho), "kendall_tau": float(tau),
           "kendall_p": float(p_tau), "spearman_p_exact": None,
           "kendall_p_exact": None}
    n = len(a)
    if 3 <= n <= 8 and np.isfinite(rho):
        base = list(range(n))
        ra = stats.rankdata(a)
        count_r = count_t = 0
        total = math.factorial(n)
        for perm in itertools.permutations(base):
            rp, _ = stats.spearmanr(ra, stats.rankdata(list(perm)))
            if abs(float(rp)) >= abs(float(rho)) - 1e-12:
                count_r += 1
            tp, _ = stats.kendalltau(ra, list(perm))
            if abs(float(tp)) >= abs(float(tau)) - 1e-12:
                count_t += 1
        out["spearman_p_exact"] = count_r / total
        out["kendall_p_exact"] = count_t / total
    return out


# ----------------------------------------------------------------------------
# Q1-Q7 scaffolding (all pure over tidy frames; query-level pairing)
# ----------------------------------------------------------------------------

def _system_means(df, metric, extra_keys=()) -> pd.DataFrame:
    keys = ["system", *extra_keys]
    return df.groupby(keys)[metric].mean().reset_index()


def q1_retrieval_semantic_corr(
    ir_df: pd.DataFrame,
    sem_df: pd.DataFrame,
    consumer: str,
    model=None,
    k=None,
    ir_metric: str = "ndcg10",
    sem_metric: str = "accuracy",
    query_ids=None,
) -> dict:
    """Q1: rank correlation of system orderings (retrieval vs semantic).

    Correlates per-system means across systems within one consumer slice
    (optionally fixing model/k and a query subset). Small-n by design
    (one point per retrieval system); see ``n`` in the output.
    """
    g = sem_df[sem_df["consumer"] == consumer]
    if model is not None:
        g = g[g["model"] == model]
    if k is not None:
        g = g[g["k"] == k]
    use_q = query_ids
    ir = ir_df if use_q is None else ir_df[ir_df["query_id"].isin(use_q)]
    gg = g if use_q is None else g[g["query_id"].isin(use_q)]
    ir_m = ir.groupby("system")[ir_metric].mean()
    sem_m = gg.groupby("system")[sem_metric].mean()
    systems = sorted(set(ir_m.index) & set(sem_m.index))
    out = rank_corr([ir_m[s] for s in systems], [sem_m[s] for s in systems])
    out.update({"consumer": consumer, "model": model, "k": k,
                "ir_metric": ir_metric, "sem_metric": sem_metric,
                "systems": systems})
    return out


def q2_rank_stability(
    sem_df: pd.DataFrame,
    consumers=("C0", "C1", "C3"),
    model=None,
    k=None,
    metric: str = "accuracy",
    query_ids=None,
) -> pd.DataFrame:
    """Q2: rank correlations of system orderings between consumer pairs."""
    sub = sem_df[sem_df["consumer"].isin(consumers)]
    if model is not None:
        sub = sub[sub["model"] == model]
    if k is not None:
        sub = sub[sub["k"] == k]
    if query_ids is not None:
        sub = sub[sub["query_id"].isin(query_ids)]
    means = sub.groupby(["consumer", "system"])[metric].mean().unstack("consumer")
    rows = []
    consumers = [c for c in consumers if c in means.columns]
    for i in range(len(consumers)):
        for j in range(i + 1, len(consumers)):
            a, b = consumers[i], consumers[j]
            paired = means[[a, b]].dropna()
            rc = rank_corr(paired[a].to_numpy(), paired[b].to_numpy())
            rc.update({"consumer_a": a, "consumer_b": b,
                       "metric": metric,
                       "systems": list(paired.index)})
            rows.append(rc)
    return pd.DataFrame(rows)


def paired_contrast(
    sem_df: pd.DataFrame,
    fixed: dict,
    vary_key: str,
    level_a,
    level_b,
    metric: str = "accuracy",
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired query-level contrast ``level_b - level_a`` of ``vary_key``.

    ``fixed`` pins the other slice keys (e.g. consumer/model/k/system);
    pairing is on ``query_id`` (inner join, queries present in both arms).
    """
    sub = sem_df
    for col, val in fixed.items():
        sub = sub[sub[col] == val]
    a = sub[sub[vary_key] == level_a].set_index("query_id")[metric]
    b = sub[sub[vary_key] == level_b].set_index("query_id")[metric]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("no overlapping queries for paired contrast")
    out = paired_bootstrap_ci(b.loc[common].to_numpy(),
                              a.loc[common].to_numpy(),
                              n_boot=n_boot, ci=ci, seed=seed)
    out.update({"vary_key": vary_key, "level_a": level_a,
                "level_b": level_b, "metric": metric, "fixed": dict(fixed)})
    return out


def q3_retriever_consumer_interaction(
    sem_df: pd.DataFrame,
    retriever_a: str,
    retriever_b: str,
    consumer_a: str = "C0",
    consumer_b: str = "C1",
    model=None,
    k=None,
    metric: str = "accuracy",
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Q3: Retriever x Consumer interaction at query level.

    Interaction = (B_consB - A_consB) - (B_consA - A_consA) per query,
    mean + paired bootstrap CI over queries present in all four arms.
    """
    sub = sem_df
    if model is not None:
        sub = sub[sub["model"] == model]
    if k is not None:
        sub = sub[sub["k"] == k]
    arms = {}
    for sys_name in (retriever_a, retriever_b):
        for cons in (consumer_a, consumer_b):
            s = sub[(sub["system"] == sys_name) & (sub["consumer"] == cons)]
            arms[(sys_name, cons)] = s.set_index("query_id")[metric]
    common = None
    for s in arms.values():
        common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) == 0:
        raise ValueError("no overlapping queries across the four arms")
    per_q = ((arms[(retriever_b, consumer_b)].loc[common]
              - arms[(retriever_a, consumer_b)].loc[common])
             - (arms[(retriever_b, consumer_a)].loc[common]
                - arms[(retriever_a, consumer_a)].loc[common]))
    out = paired_bootstrap_ci(per_q.to_numpy(),
                              n_boot=n_boot, ci=ci, seed=seed)
    out.update({"retriever_a": retriever_a, "retriever_b": retriever_b,
                "consumer_a": consumer_a, "consumer_b": consumer_b,
                "metric": metric})
    return out


def add_hit_indicator(
    beliefs_df: pd.DataFrame,
    qrels: dict[str, dict[str, int]],
    k_col: str = "k",
    grade_threshold: int = HIT_GRADE_THRESHOLD,
) -> pd.DataFrame:
    """Add top-k evidence-hit indicator (any retrieved doc grade>=threshold).

    Uses the row's own ``doc_ids`` truncated to the row's ``k``.
    """
    df = beliefs_df.copy()

    def _hit(row) -> int:
        raw_docs = row.get("doc_ids", None)
        docs = list(raw_docs) if raw_docs is not None else []
        docs = docs[: int(row.get(k_col, 0) or 0)]
        grades = qrels.get(row["query_id"], {})
        return int(any(grades.get(d, 0) >= grade_threshold for d in docs))

    df["evidence_hit"] = [ _hit(r) for _, r in df.iterrows()]
    return df


def q4_hit_vs_miss(
    sem_df: pd.DataFrame,
    metric: str = "accuracy",
    hit_col: str = "evidence_hit",
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Q4: evidence-hit vs miss contrast (independent groups, bootstrap CI)."""
    hit = sem_df[sem_df[hit_col] == 1][metric].to_numpy(dtype=float)
    miss = sem_df[sem_df[hit_col] == 0][metric].to_numpy(dtype=float)
    out = unpaired_diff_ci(hit, miss, n_boot=n_boot, ci=ci, seed=seed)
    out.update({"metric": metric, "comparison": "hit-minus-miss"})
    return out


def q5_c3_vs_c1_gain(
    sem_df: pd.DataFrame,
    system=None,
    model=None,
    k=None,
    metric: str = "accuracy",
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Q5: C3-vs-C1 paired gain per query (C3 minus C1)."""
    fixed: dict = {}
    if system is not None:
        fixed["system"] = system
    if model is not None:
        fixed["model"] = model
    if k is not None:
        fixed["k"] = k
    return paired_contrast(sem_df, fixed, "consumer", "C1", "C3",
                           metric=metric, n_boot=n_boot, ci=ci, seed=seed)


def q6_cross_model(
    sem_df: pd.DataFrame,
    model_a: str = MODEL_QWEN3_14B,
    model_b: str = MODEL_QWEN3_8B,
    metric: str = "accuracy",
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Q6: cross-model replication (Qwen3-14B x Qwen3-8B scale check).

    Per (system, consumer, k): mean metric per model, paired diff
    (model_b - model_a) with bootstrap CI, and query-level correlation
    of the two models' scores.
    """
    rows = []
    for (system, consumer, k), g in sem_df.groupby(["system", "consumer", "k"]):
        a = g[g["model"] == model_a].set_index("query_id")[metric]
        b = g[g["model"] == model_b].set_index("query_id")[metric]
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        av, bv = a.loc[common].to_numpy(dtype=float), b.loc[common].to_numpy(dtype=float)
        ci_out = paired_bootstrap_ci(bv, av, n_boot=n_boot, ci=ci, seed=seed)
        rc = rank_corr(av, bv)
        rows.append({
            "system": system, "consumer": consumer, "k": k,
            "model_a": model_a, "model_b": model_b,
            "mean_a": float(np.mean(av)), "mean_b": float(np.mean(bv)),
            "diff_b_minus_a": ci_out["estimate"],
            "ci_lo": ci_out["ci_lo"], "ci_hi": ci_out["ci_hi"],
            "spearman_rho": rc["spearman_rho"], "spearman_p": rc["spearman_p"],
            "kendall_tau": rc["kendall_tau"], "kendall_p": rc["kendall_p"],
            "n": ci_out["n"],
        })
    cols = ["system", "consumer", "k", "model_a", "model_b", "mean_a",
            "mean_b", "diff_b_minus_a", "ci_lo", "ci_hi",
            "spearman_rho", "spearman_p", "kendall_tau", "kendall_p", "n"]
    return pd.DataFrame(rows, columns=cols)


def add_split_flags(
    sem_df: pd.DataFrame,
    splits: dict | None = None,
    query_meta: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Add ``split`` (dev/test) and ``held_out_entity`` flags.

    ``held_out_entity`` marks queries whose ``entity_node`` is in
    ``splits["held_out_entities"]``. Entity nodes come from the beliefs
    frame when present, else from ``queries.jsonl`` metadata.
    """
    splits = splits if splits is not None else load_splits()
    meta = query_meta if query_meta is not None else load_query_meta()
    df = sem_df.copy()
    held = set(splits.get("held_out_entities", []))
    if "entity_node" not in df.columns:
        df["entity_node"] = df["query_id"].map(
            lambda q: meta.get(q, {}).get("entity_node", "")
        )
    if "split" not in df.columns:
        dev = set(splits.get("dev", []))
        test = set(splits.get("test", []))
        df["split"] = df["query_id"].map(
            lambda q: "dev" if q in dev else ("test" if q in test else "")
        )
    df["held_out_entity"] = df["entity_node"].isin(held)
    return df


def q7_heldout(
    sem_df: pd.DataFrame,
    group_keys=("system", "consumer", "model", "k"),
    metrics=("accuracy", "brier", "nll"),
) -> pd.DataFrame:
    """Q7: means over dev/test x held-out-entity subsets (no tuning decisions).

    Requires ``split`` and ``held_out_entity`` columns (see
    :func:`add_split_flags`). Reports descriptive means + n only.
    """
    for col in ("split", "held_out_entity"):
        if col not in sem_df.columns:
            raise ValueError(f"q7_heldout needs column {col!r}; run add_split_flags first")
    keys = [*group_keys, "split", "held_out_entity"]
    agg = sem_df.groupby(keys)[list(metrics)].mean().reset_index()
    n = sem_df.groupby(keys).size().reset_index(name="n")
    return agg.merge(n, on=keys)
