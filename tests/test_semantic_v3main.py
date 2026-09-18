"""Tests for src/semantic_analysis_v3main.py (frozen V3 main scaffolding).

All fixtures are hand-constructed with KNOWN answers. No frozen artifacts
are read; no simulator runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.semantic_analysis_v3main import (
    HIT_GRADE_THRESHOLD,
    REGIMES,
    add_hit_indicator,
    add_split_flags,
    aggregate_semantic,
    ece_10bin,
    holm_correction,
    ir_per_query,
    load_nondeterminism_report,
    paired_bootstrap_ci,
    paired_contrast,
    parse_cache_raw,
    q1_retrieval_semantic_corr,
    q2_rank_stability,
    q3_retriever_consumer_interaction,
    q4_hit_vs_miss,
    q5_c3_vs_c1_gain,
    q6_cross_model,
    q7_heldout,
    query_semantic_metrics,
    rank_corr,
    semantic_per_query,
    sensitivity_max_delta,
    unpaired_diff_ci,
    with_alternative_cache,
)

P3 = 1.0 / 3.0


# ----------------------------------------------------------------------------
# query_semantic_metrics: exact values
# ----------------------------------------------------------------------------

def test_perfect_beliefs_exact():
    m = query_semantic_metrics(0.0, 1.0, 0.0, "supplier_delay",
                               abstain=False, confidence=0.9)
    assert m["pred_regime"] == "supplier_delay"
    assert m["accuracy"] == 1.0
    assert m["brier"] == 0.0
    assert m["nll"] == 0.0
    assert m["p_true"] == 1.0
    assert m["abstain"] is False
    assert m["confidence"] == 0.9


def test_uniform_abstain_exact():
    m = query_semantic_metrics(P3, P3, P3, "normal",
                               abstain=True, confidence=0.0)
    assert m["pred_regime"] == "abstain"
    assert m["accuracy"] == 0.0
    # mean over 3 classes: ((1/3-1)^2 + 2*(1/3)^2)/3 = (6/9)/3 = 2/9
    assert abs(m["brier"] - 2.0 / 9.0) < 1e-12
    assert abs(m["nll"] - np.log(3.0)) < 1e-12
    assert abs(m["p_true"] - P3) < 1e-12


def test_wrong_confident_exact():
    m = query_semantic_metrics(0.7, 0.2, 0.1, "demand_surge",
                               abstain=False, confidence=0.7)
    assert m["pred_regime"] == "normal"
    assert m["accuracy"] == 0.0
    # ((0.7)^2 + (0.2)^2 + (0.9)^2)/3 = (0.49+0.04+0.81)/3 = 1.34/3
    assert abs(m["brier"] - 1.34 / 3.0) < 1e-12
    assert abs(m["nll"] - (-np.log(0.1))) < 1e-12


# ----------------------------------------------------------------------------
# ECE: known answers
# ----------------------------------------------------------------------------

def test_ece_perfect_calibration_is_zero():
    conf = np.array([1.0, 1.0, 0.0, 0.0])
    corr = np.array([1.0, 1.0, 0.0, 0.0])
    assert ece_10bin(conf, corr) == 0.0


def test_ece_fully_miscalibrated_is_half():
    conf = np.ones(10)
    corr = np.array([1.0] * 5 + [0.0] * 5)
    assert abs(ece_10bin(conf, corr) - 0.5) < 1e-12


def test_ece_single_bin_gap():
    # all mass in the top bin: |0.25 - 0.95| = 0.7
    conf = np.array([0.95, 0.95, 0.95, 0.95])
    corr = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(ece_10bin(conf, corr) - 0.7) < 1e-12


# ----------------------------------------------------------------------------
# IR: perfect vs reversed rankings with known answers
# ----------------------------------------------------------------------------

def _tiny_qrels():
    return {"q1": {"d1": 3, "d2": 0}, "q2": {"d3": 2, "d4": 1}}


def test_ir_perfect_ranking_is_one():
    run = {"q1": ["d1", "d2"], "q2": ["d3", "d4"]}
    df = ir_per_query(_tiny_qrels(), run)
    assert set(df["query_id"]) == {"q1", "q2"}
    assert (df["ndcg10"] == 1.0).all()
    assert (df["recall10"] == 1.0).all()
    assert (df["recall20"] == 1.0).all()
    assert (df["mrr"] == 1.0).all()


def test_ir_reversed_ranking_known_values():
    # q1: relevant d1 (grade 3) at rank 2 -> RR 1/2, nDCG 1/log2(3)
    run = {"q1": ["d2", "d1"], "q2": ["d3", "d4"]}
    df = ir_per_query(_tiny_qrels(), run).set_index("query_id")
    assert df.loc["q1", "mrr"] == 0.5
    assert abs(df.loc["q1", "ndcg10"] - 1.0 / np.log2(3.0)) < 1e-9
    assert df.loc["q1", "recall10"] == 1.0  # still retrieved within 10
    assert df.loc["q2", "ndcg10"] == 1.0


def test_ir_missing_query_scores_zero():
    df = ir_per_query(_tiny_qrels(), {"q1": ["d1"]}, query_ids=["q1", "q9"])
    row = df.set_index("query_id").loc["q9"]
    assert (row[["ndcg10", "recall10", "recall20", "mrr"]] == 0.0).all()


# ----------------------------------------------------------------------------
# semantic_per_query + aggregate on tiny belief frames
# ----------------------------------------------------------------------------

def _tiny_beliefs():
    return pd.DataFrame([
        {"query_id": "q1", "system": "dense", "consumer": "C0", "model": "none",
         "k": 3, "true_regime": "normal", "split": "test",
         "p_normal": 1.0, "p_supplier_delay": 0.0, "p_demand_surge": 0.0,
         "abstain": False, "confidence": 1.0,
         "doc_ids": ["d1", "d2", "d3"]},
        {"query_id": "q2", "system": "dense", "consumer": "C0", "model": "none",
         "k": 3, "true_regime": "normal", "split": "test",
         "p_normal": P3, "p_supplier_delay": P3, "p_demand_surge": P3,
         "abstain": True, "confidence": 0.0,
         "doc_ids": ["d9", "d8", "d7"]},
    ])


def test_semantic_per_query_exact():
    sem = semantic_per_query(_tiny_beliefs())
    assert list(sem["accuracy"]) == [1.0, 0.0]
    assert list(sem["brier"])[0] == 0.0
    assert abs(sem["brier"].iloc[1] - 2.0 / 9.0) < 1e-12
    assert list(sem["pred_regime"]) == ["normal", "abstain"]
    assert set(REGIMES) == {"normal", "supplier_delay", "demand_surge"}


def test_aggregate_semantic_known_means():
    sem = semantic_per_query(_tiny_beliefs())
    agg = aggregate_semantic(sem)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["n"] == 2
    assert row["accuracy"] == 0.5
    assert abs(row["brier"] - (0.0 + 2.0 / 9.0) / 2.0) < 1e-12
    assert row["abstention_rate"] == 0.5


# ----------------------------------------------------------------------------
# Bootstrap CIs: degenerate exact cases
# ----------------------------------------------------------------------------

def test_paired_bootstrap_identical_is_zero_width():
    x = np.array([1.0, 0.0, 1.0, 0.0])
    out = paired_bootstrap_ci(x, x, n_boot=500, seed=0)
    assert out["estimate"] == 0.0
    assert out["ci_lo"] == 0.0 and out["ci_hi"] == 0.0


def test_paired_bootstrap_constant_diff_exact():
    out = paired_bootstrap_ci(np.ones(6), np.zeros(6), n_boot=500, seed=1)
    assert out["estimate"] == 1.0
    assert out["ci_lo"] == 1.0 and out["ci_hi"] == 1.0


def test_unpaired_diff_ci_known_estimate():
    out = unpaired_diff_ci([1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                           n_boot=200, seed=0)
    assert out["estimate"] == 1.0
    assert out["ci_lo"] <= 1.0 <= out["ci_hi"]


# ----------------------------------------------------------------------------
# Holm + rank correlations: known answers
# ----------------------------------------------------------------------------

def test_holm_known_adjustment():
    out = holm_correction([0.01, 0.02, 0.03])
    # ordered: 3*.01=.03, 2*.02=.04, max(.04, 1*.03)=.04
    assert list(out["p_holm"].round(12)) == [0.03, 0.04, 0.04]
    assert out["reject_holm_05"].all()


def test_holm_nonreject():
    out = holm_correction([0.5, 0.6])
    assert not out["reject_holm_05"].any()
    assert (out["p_holm"] <= 1.0).all()


def test_rank_corr_perfect():
    rc = rank_corr([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert rc["spearman_rho"] == 1.0
    assert rc["kendall_tau"] == 1.0
    assert rc["n"] == 4


def test_rank_corr_reversed():
    rc = rank_corr([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert rc["spearman_rho"] == -1.0
    assert rc["kendall_tau"] == -1.0


# ----------------------------------------------------------------------------
# Q-scaffolding on hand-built frames
# ----------------------------------------------------------------------------

def _q_frame():
    # 3 queries x 2 systems x 2 consumers; C1 gains over C0 on 'good' only.
    rows = []
    for qi in ("q1", "q2", "q3"):
        for sys_name, good in (("good", [1.0, 1.0, 0.0]),
                               ("bad", [0.0, 0.0, 1.0])):
            for cons in ("C0", "C1"):
                acc_c0 = good[["q1", "q2", "q3"].index(qi)]
                acc = acc_c0 if cons == "C0" else 1.0  # C1 perfect everywhere
                p = [1.0, 0.0, 0.0] if acc == 1.0 else [0.0, 1.0, 0.0]
                rows.append({
                    "query_id": qi, "system": sys_name, "consumer": cons,
                    "model": "m", "k": 3, "true_regime": "normal",
                    "split": "test", "entity_node": "S1",
                    "p_normal": p[0], "p_supplier_delay": p[1],
                    "p_demand_surge": p[2], "abstain": False,
                    "confidence": 1.0 if acc == 1.0 else 0.0,
                    "doc_ids": ["d1"],
                })
    return semantic_per_query(pd.DataFrame(rows))


def test_paired_contrast_c1_minus_c0():
    sem = _q_frame()
    out = paired_contrast(sem, {"system": "good", "model": "m", "k": 3},
                          "consumer", "C0", "C1", n_boot=500, seed=0)
    # good/C0 accuracies: 1,1,0 ; good/C1: 1,1,1 -> diff mean 1/3
    assert abs(out["estimate"] - 1.0 / 3.0) < 1e-12
    assert out["n"] == 3


def test_q5_c3_vs_c1_smoke():
    sem = _q_frame()
    sem["consumer"] = sem["consumer"].replace({"C0": "C3"})
    out = q5_c3_vs_c1_gain(sem, system="bad", model="m", k=3,
                           n_boot=200, seed=0)
    assert out["level_a"] == "C1" and out["level_b"] == "C3"


def test_q3_interaction_zero_when_no_difference():
    sem = _q_frame()
    # consumer C0==C1 on 'bad' arm? No: build symmetric frame -> interaction 0.
    sub = sem[sem["system"] == "good"].copy()
    sub["system"] = sub["system"].replace({"good": "good2"})
    sym = pd.concat([sem[sem["system"] == "good"], sub], ignore_index=True)
    out = q3_retriever_consumer_interaction(
        sym, "good", "good2", "C0", "C1", n_boot=200, seed=0)
    assert out["estimate"] == 0.0


def test_q1_corr_perfect_rank_agreement():
    ir = pd.DataFrame([
        {"query_id": "q1", "system": "good", "ndcg10": 0.9,
         "recall10": 1.0, "recall20": 1.0, "mrr": 1.0},
        {"query_id": "q1", "system": "bad", "ndcg10": 0.1,
         "recall10": 0.0, "recall20": 0.0, "mrr": 0.0},
    ])
    sem = pd.DataFrame([
        {"query_id": "q1", "system": "good", "consumer": "C0",
         "model": "m", "k": 3, "accuracy": 1.0},
        {"query_id": "q1", "system": "bad", "consumer": "C0",
         "model": "m", "k": 3, "accuracy": 0.0},
    ])
    out = q1_retrieval_semantic_corr(ir, sem, consumer="C0")
    assert abs(out["spearman_rho"] - 1.0) < 1e-12
    assert abs(out["kendall_tau"] - 1.0) < 1e-12


def test_q2_stability_identical_orderings():
    sem = pd.DataFrame([
        {"query_id": "q1", "system": s, "consumer": c,
         "model": "m", "k": 3, "accuracy": a}
        for s, a in (("good", 1.0), ("bad", 0.0))
        for c in ("C0", "C1")
    ])
    tab = q2_rank_stability(sem, consumers=("C0", "C1"))
    assert len(tab) == 1
    assert abs(tab.iloc[0]["spearman_rho"] - 1.0) < 1e-12


def test_q4_hit_vs_miss_known_direction():
    sem = pd.DataFrame({
        "query_id": ["q1", "q2", "q3", "q4"],
        "evidence_hit": [1, 1, 0, 0],
        "accuracy": [1.0, 1.0, 0.0, 0.0],
    })
    out = q4_hit_vs_miss(sem, n_boot=200, seed=0)
    assert out["estimate"] == 1.0


def test_add_hit_indicator_threshold():
    df = pd.DataFrame([
        {"query_id": "q1", "k": 2, "doc_ids": ["d1", "d9"]},
        {"query_id": "q1", "k": 1, "doc_ids": ["d9", "d1"]},
        {"query_id": "q2", "k": 2, "doc_ids": ["d9"]},
    ])
    qrels = {"q1": {"d1": HIT_GRADE_THRESHOLD, "d9": 1}}
    out = add_hit_indicator(df, qrels)
    assert list(out["evidence_hit"]) == [1, 0, 0]


def test_q6_cross_model_identical_models():
    sem = pd.DataFrame([
        {"query_id": q, "system": "dense", "consumer": "C1",
         "model": m, "k": 3, "accuracy": float(i % 2)}
        for i, q in enumerate(["q1", "q2", "q3", "q4"])
        for m in ("mA", "mB")
    ])
    tab = q6_cross_model(sem, model_a="mA", model_b="mB",
                         n_boot=200, seed=0)
    assert len(tab) == 1
    row = tab.iloc[0]
    assert row["diff_b_minus_a"] == 0.0
    assert row["spearman_rho"] == 1.0


def test_q7_heldout_grouping():
    sem = pd.DataFrame([
        {"query_id": "q1", "system": "s", "consumer": "C0", "model": "m",
         "k": 3, "split": "test", "held_out_entity": True,
         "accuracy": 1.0, "brier": 0.0, "nll": 0.0},
        {"query_id": "q2", "system": "s", "consumer": "C0", "model": "m",
         "k": 3, "split": "dev", "held_out_entity": False,
         "accuracy": 0.0, "brier": 0.5, "nll": 1.0},
    ])
    tab = q7_heldout(sem)
    assert len(tab) == 2
    assert set(tab["n"]) == {1}


def test_add_split_flags_from_splits_dict():
    sem = pd.DataFrame([{"query_id": "q1"}, {"query_id": "q2"}])
    meta = {"q1": {"entity_node": "H1"}, "q2": {"entity_node": "S9"}}
    splits = {"dev": ["q2"], "test": ["q1"], "held_out_entities": ["H1"]}
    out = add_split_flags(sem, splits=splits, query_meta=meta)
    assert list(out["split"]) == ["test", "dev"]
    assert list(out["held_out_entity"]) == [True, False]


def test_with_alternative_cache_replaces_rows():
    df = pd.DataFrame([
        {"query_id": "q1", "system": "s", "consumer": "C1", "model": "m",
         "k": 3, "p_normal": 1.0},
        {"query_id": "q2", "system": "s", "consumer": "C1", "model": "m",
         "k": 3, "p_normal": 0.0},
    ])
    alt = pd.DataFrame([
        {"query_id": "q2", "system": "s", "consumer": "C1", "model": "m",
         "k": 3, "p_normal": 0.5},
    ])
    out = with_alternative_cache(df, alt)
    assert len(out) == 2
    assert out.set_index("query_id").loc["q2", "p_normal"] == 0.5
    assert out.set_index("query_id").loc["q1", "p_normal"] == 1.0


# ----------------------------------------------------------------------------
# Nondeterminism sensitivity: report parsing + 2-key flip with known answers
# ----------------------------------------------------------------------------

def _mini_nondeterminism_report():
    c1_kept = ('{"normal": 0.4, "supplier_delay": 0.5, "demand_surge": 0.1, '
               '"estimated_lt_increase": 3}')
    c1_later = ('{"normal": 0.3, "supplier_delay": 0.6, "demand_surge": 0.1, '
                '"estimated_lt_increase": 3}')
    c3_kept = ('{"entity_match": true, "event": "supplier_delay", '
               '"fresh": true, "stance": "support", "confidence": 0.9}')
    c3_later = ('{"entity_match": true, "event": "supplier_delay", '
                '"fresh": true, "stance": "support", "confidence": 0.95}')
    return {
        "policy": "first-wins in numeric shard-start order",
        "n_diverged_files": 3,
        "diverged": [
            {"file": "aaaa.json", "kept_shard": "earlier",
             "later_shard": "cache_shard_50_100.zip",
             "same_parsed_belief": False,
             "kept_raw": c1_kept, "later_raw": c1_later},
            {"file": "bbbb.json", "kept_shard": "earlier",
             "later_shard": "cache_shard_50_100.zip",
             "same_parsed_belief": False,
             "kept_raw": c3_kept, "later_raw": c3_later},
            {"file": "cccc.json", "kept_shard": "earlier",
             "later_shard": "cache_shard_50_100.zip",
             "same_parsed_belief": True,
             "kept_raw": c1_kept, "later_raw": c1_kept},
        ],
    }


def test_parse_cache_raw_roundtrip():
    d = parse_cache_raw('{"normal": 0.3, "supplier_delay": 0.6, "demand_surge": 0.1}')
    assert d == {"normal": 0.3, "supplier_delay": 0.6, "demand_surge": 0.1}


def test_load_nondeterminism_report_filters_disputed(tmp_path):
    import json

    p = tmp_path / "nondet.json"
    p.write_text(json.dumps(_mini_nondeterminism_report()))
    rep = load_nondeterminism_report(p)
    assert rep["policy"].startswith("first-wins")
    assert rep["n_diverged_files"] == 3
    assert rep["n_disputed"] == 2
    assert rep["n_same_belief"] == 1
    assert {e["file"] for e in rep["disputed"]} == {"aaaa.json", "bbbb.json"}
    # later variant payloads are parsed (C1 regime shape + C3 per-doc shape)
    by_file = {e["file"]: e for e in rep["disputed"]}
    assert by_file["aaaa.json"]["later_parsed"]["supplier_delay"] == 0.6
    assert by_file["bbbb.json"]["later_parsed"]["confidence"] == 0.95


def _sensitivity_beliefs(right=("q1", "q2", "q3", "q4"), wrong=()):
    # C1 wrong on q3/q4; C3 perfect everywhere in base. Flipping the 2
    # disputed keys (q3,q4 of C3) to wrong gives alt with zero C3-C1 gap.
    rows = []
    for qi in ("q1", "q2", "q3", "q4"):
        for cons in ("C1", "C3"):
            correct = (qi in right) if cons == "C3" else (qi not in ("q3", "q4"))
            if cons == "C3" and qi in wrong:
                correct = False
            p = [1.0, 0.0, 0.0] if correct else [0.0, 1.0, 0.0]
            rows.append({
                "query_id": qi, "system": "s", "consumer": cons,
                "model": "m", "k": 3, "true_regime": "normal",
                "split": "test", "p_normal": p[0],
                "p_supplier_delay": p[1], "p_demand_surge": p[2],
                "abstain": False, "confidence": 1.0, "doc_ids": ["d1"],
            })
    return pd.DataFrame(rows)


def _c3_vs_c1(df):
    return paired_contrast(df, {"system": "s", "model": "m", "k": 3},
                           "consumer", "C1", "C3", n_boot=500, seed=0)


def test_sensitivity_two_key_flip_exact():
    base_beliefs = _sensitivity_beliefs()
    # Later-variant resolution flips exactly 2 keys: q3,q4 of C3 to wrong.
    flip = pd.DataFrame([
        {"query_id": qi, "system": "s", "consumer": "C3", "model": "m",
         "k": 3, "p_normal": 0.0, "p_supplier_delay": 1.0,
         "p_demand_surge": 0.0}
        for qi in ("q3", "q4")
    ])
    alt_beliefs = with_alternative_cache(base_beliefs, flip)
    base = semantic_per_query(base_beliefs)
    alt = semantic_per_query(alt_beliefs)
    # base: C1 acc [1,1,0,0], C3 [1,1,1,1] -> gap 0.5; alt C3 [1,1,0,0] -> 0.0
    assert abs(_c3_vs_c1(base)["estimate"] - 0.5) < 1e-12
    assert abs(_c3_vs_c1(alt)["estimate"] - 0.0) < 1e-12
    out = sensitivity_max_delta(base, alt, {"c3_minus_c1": _c3_vs_c1})
    row = out["per_comparison"].iloc[0]
    assert row["comparison"] == "c3_minus_c1"
    assert abs(row["base_estimate"] - 0.5) < 1e-12
    assert abs(row["alt_estimate"] - 0.0) < 1e-12
    assert abs(row["effect_delta"] - (-0.5)) < 1e-12
    assert abs(out["max_abs_effect_delta"] - 0.5) < 1e-12


def test_sensitivity_noop_flip_is_zero():
    base = semantic_per_query(_sensitivity_beliefs())
    out = sensitivity_max_delta(base, base.copy(), {"c3_minus_c1": _c3_vs_c1})
    assert out["max_abs_effect_delta"] == 0.0
    assert out["per_comparison"]["ci_lo_delta"].notna().all()
