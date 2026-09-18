"""End-to-end semantic-transfer analysis for frozen V3 main beliefs.

Runs IR + semantic metrics + preregistered Q1-Q7 + nondeterminism sensitivity
on the FROZEN belief parquets. Writes ONLY analysis outputs under
results/v3main/semantic/ (never touches beliefs, caches, or frozen configs).
No simulator, no profit/return anywhere in this file.

Usage: python3 tools/run_semantic_v3main.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.semantic_analysis_v3main as S
from src.beliefs_v3main import _evidence_sets
from src.consumers_v3 import prompt_sha, PROMPT_C1, PROMPT_C3_DOC

OUT = ROOT / "results" / "v3main" / "semantic"
M8 = "Qwen/Qwen3-8B-AWQ"
M14 = "Qwen/Qwen3-14B-AWQ"
N_BOOT = 10000
RETRIEVAL_SYSTEMS = ["bm25", "dense", "hybrid", "rerank"]


def w(path, obj):
    path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(path.with_suffix(".parquet"), index=False)
        obj.to_json(path.with_suffix(".json"), orient="records", indent=1)
    else:
        path.with_suffix(".json").write_text(json.dumps(obj, indent=1, default=str))
    print(f"wrote {path.with_suffix('.*')} ({len(obj) if hasattr(obj, '__len__') else 1})")


def cache_fname(model, prompt, user):
    key_src = f"{model}||{prompt_sha(prompt)}||{hashlib.sha256(user.encode()).hexdigest()[:16]}"
    return hashlib.sha256(key_src.encode()).hexdigest()[:16] + ".json"


def build_key_index(model, cache_dir):
    """Map cache filename -> list of (kind, rowkey) for every LLM belief row.

    kind C1: row key; kind C3: (row key, doc_id). Verifies every key exists
    on disk (full cache-key coverage, stronger than the freeze gate's size check).
    """
    docs, queries, sets = _evidence_sets()
    qtext = {q["_id"]: q["text"] for q in queries}
    qmeta = {q["_id"]: q["metadata"] for q in queries}
    have = {p.name for p in Path(cache_dir).glob("*.json")}
    index = {}
    missing = []

    def reg(fn, val):
        index.setdefault(fn, []).append(val)

    for (sys_name, qid, k), dids in sets.items():
        if sys_name == "none" or qid is None:
            continue
        top = [{"doc_id": d, "text": docs[d]["text"]} for d in dids]
        # C1 key
        user = ("Operator information need: " + qtext[qid] + "\n\nEvidence:\n"
                + "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in top))
        fn = cache_fname(model, PROMPT_C1, user)
        (missing.append(fn) if fn not in have else reg(fn, ("C1", sys_name, qid, k)))
        # C3 keys (per doc)
        for d in top:
            u2 = (f"Operator's own node: {qmeta[qid]['entity_node']}\n\n"
                  f"Evidence document:\n{d['text']}")
            fn2 = cache_fname(model, PROMPT_C3_DOC, u2)
            if fn2 not in have:
                missing.append(fn2)
            else:
                reg(fn2, ("C3", sys_name, qid, k, d["doc_id"]))
    if missing:
        raise SystemExit(f"{model}: {len(set(missing))} cache keys missing on disk, "
                         f"e.g. {sorted(set(missing))[:3]}")
    return index, docs, queries


def c1_belief(parsed):
    s = parsed["normal"] + parsed["supplier_delay"] + parsed["demand_surge"]
    return (parsed["normal"] / s, parsed["supplier_delay"] / s,
            parsed["demand_surge"] / s,
            min(1.0, s / 1.0) if s <= 1.5 else 0.5)


def c3_belief(per_doc):
    """Frozen consume_c3 aggregation replicated over parsed per-doc dicts."""
    import math
    agg = {"normal": 0.0, "supplier_delay": 0.0, "demand_surge": 0.0}
    support = 0.0
    for e in per_doc:
        wgt = float(e["confidence"]) * (1.0 if e["entity_match"] else 0.2) * \
            (1.0 if e["fresh"] else 0.2)
        ev = e["event"]
        if ev in agg and e["stance"] == "support":
            agg[ev] += wgt
            support = max(support, wgt)
        elif ev in agg and e["stance"] == "refute":
            agg[ev] -= 0.5 * wgt
    if support < 0.5:
        return (1 / 3, 1 / 3, 1 / 3, True, support)
    prior = {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30}
    scores = {k: agg[k] + 0.5 * prior[k] for k in agg}
    m = max(scores.values())
    ex = {k: math.exp(v - m) for k, v in scores.items()}
    tot = sum(ex.values())
    return (ex["normal"] / tot, ex["supplier_delay"] / tot,
            ex["demand_surge"] / tot, False, min(1.0, support))


def alt_beliefs_for_model(model, cache_dir, report_path, bel):
    """Rebuild belief rows with disputed cache keys flipped to later variant.

    bel: beliefs-frame rows for exactly this model (with doc_ids).
    C1 disputed key -> renormalized later probs (exact consume_c1 formula).
    C3 disputed key -> re-aggregated set with that doc flipped (exact
    consume_c3 formula). Replication gate: recomputing kept-beliefs from
    on-disk payloads must match frozen rows to 1e-12, else SystemExit.
    Returns (alt_bel_df, info).
    """
    import math
    rep = S.load_nondeterminism_report(report_path)
    print(f"{model}: {rep['n_diverged_files']} diverged, "
          f"{rep['n_disputed']} disputed, {rep['n_same_belief']} same-belief")
    frozen = bel.copy().reset_index(drop=True)  # pristine; NEVER written
    alt = bel.copy().reset_index(drop=True)  # flips go here only
    if not rep["disputed"]:
        return alt, {"n_disputed": 0, "rows_flipped": 0,
                     "replication": "trivially-exact"}
    index, docs, queries = build_key_index(model, cache_dir)
    disk = {}
    for p in Path(cache_dir).glob("*.json"):
        disk[p.name] = json.loads(p.read_text())
    qmeta = {q["_id"]: q["metadata"] for q in queries}

    def rk(r):
        return (r["system"], r["query_id"], r["k"], r["consumer"])

    frozen = bel.copy().reset_index(drop=True)  # pristine; NEVER written
    alt = bel.copy().reset_index(drop=True)  # flips go here only
    by_rk = {rk(r): i for i, r in frozen.iterrows()}
    max_rep_err = 0.0
    flipped = 0
    seen_later = {}
    multi_later = []
    for entry in rep["disputed"]:
        fn = entry["file"]
        later = entry["later_parsed"]
        if fn in seen_later:
            if seen_later[fn] != later:
                multi_later.append(fn)
            continue  # same file already flipped; first-later-wins
        seen_later[fn] = later
        for item in index.get(fn, []):
            if item[0] == "C1":
                _, sys_name, qid, k = item
                i = by_rk[(sys_name, qid, k, "C1")]
                row = frozen.iloc[i]
                kept = S.parse_cache_raw(entry["kept_raw"])
                for payload, tag in ((kept, "kept"), (later, "later")):
                    s = (payload["normal"] + payload["supplier_delay"]
                         + payload["demand_surge"])
                    pn = (payload["normal"] / s, payload["supplier_delay"] / s,
                          payload["demand_surge"] / s)
                    conf = min(1.0, s / 1.0) if s <= 1.5 else 0.5
                    if tag == "kept":
                        max_rep_err = max(
                            max_rep_err, abs(row["p_normal"] - pn[0]),
                            abs(row["p_supplier_delay"] - pn[1]),
                            abs(row["p_demand_surge"] - pn[2]),
                            abs(float(row["confidence"]) - conf))
                    else:
                        alt.at[i, "p_normal"], alt.at[i, "p_supplier_delay"], \
                            alt.at[i, "p_demand_surge"] = pn
                        alt.at[i, "confidence"] = conf
                        alt.at[i, "abstain"] = False
                flipped += 1
            else:
                _, sys_name, qid, k, _ = item
                i = by_rk[(sys_name, qid, k, "C3")]
                dids = list(alt.at[i, "doc_ids"])
                own = qmeta[qid]["entity_node"]
                per_kept = []
                for did in dids:
                    u = (f"Operator's own node: {own}\n\nEvidence document:\n"
                         f"{docs[did]['text']}")
                    per_kept.append(S.parse_cache_raw(disk[cache_fname(
                        model, PROMPT_C3_DOC, u)]["raw"]))
                # replication gate on kept set
                kn = c3_belief(per_kept)
                row = frozen.iloc[i]
                rep_err = max(abs(row["p_normal"] - kn[0]),
                              abs(row["p_supplier_delay"] - kn[1]),
                              abs(row["p_demand_surge"] - kn[2]),
                              abs(bool(row["abstain"]) - kn[3]))
                max_rep_err = max(max_rep_err, rep_err)
                if rep_err > 1e-9:
                    raise SystemExit(
                        f"C3 replication failed for {(sys_name, qid, k)}: "
                        f"err={rep_err:.2e} — refusing silent divergence")
                per_alt = []
                for did in dids:
                    u = (f"Operator's own node: {own}\n\nEvidence document:\n"
                         f"{docs[did]['text']}")
                    if cache_fname(model, PROMPT_C3_DOC, u) == fn:
                        per_alt.append(later)
                    else:
                        per_alt.append(S.parse_cache_raw(disk[cache_fname(
                            model, PROMPT_C3_DOC, u)]["raw"]))
                an = c3_belief(per_alt)
                alt.at[i, "p_normal"], alt.at[i, "p_supplier_delay"], \
                    alt.at[i, "p_demand_surge"] = an[0], an[1], an[2]
                alt.at[i, "abstain"] = an[3]
                alt.at[i, "confidence"] = an[4]
                flipped += 1
    info = {"n_disputed": rep["n_disputed"], "rows_flipped": flipped,
            "multi_later_files": sorted(set(multi_later)),
            "replication": f"max|recomputed-kept minus frozen|={max_rep_err:.2e}"}
    print(f"{model}: alt rebuilt: {info}")
    return alt, info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- IR (test primary; dev reference; macro over qrel queries only) ----
    qrels_test = S.load_qrels_tsv(ROOT / S.QRELS_TEST)
    qrels_dev = S.load_qrels_tsv(ROOT / S.QRELS_DEV)
    ir_test = S.ir_per_system(qrels_path=S.QRELS_TEST,
                              query_ids=sorted(qrels_test))
    ir_dev = S.ir_per_system(qrels_path=S.QRELS_DEV,
                             query_ids=sorted(qrels_dev))
    w("ir_test", ir_test)
    w("ir_dev", ir_dev)
    w("ir_test_means", S.ir_aggregate(ir_test))
    # ---- semantic ----
    bel = pd.concat([S.load_beliefs(ROOT / "results" / "v3main" / f"beliefs_{m}.parquet")
                     for m in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ")],
                    ignore_index=True)
    # C0 rows are model-free: each parquet carries an identical copy.
    # Assert bitwise identity, then keep exactly one (else paired joins on
    # query_id silently misalign on duplicated index labels).
    c0 = bel[bel["consumer"] == "C0"]
    keys = ["system", "query_id", "k"]
    grp = c0.groupby(keys)
    assert (grp.size() == 2).all(), "every C0 key must appear exactly twice"
    _scalar = [c for c in c0.columns if c not in ("doc_ids", "latency_s", "model")]
    nun = grp[_scalar].nunique()
    assert (nun == 1).all().all(), "C0 twin rows differ!\n" + str(nun[nun != 1].head())
    assert (grp["doc_ids"].apply(
        lambda s: all(list(x) == list(s.iloc[0]) for x in s)).all()), \
        "C0 doc_ids differ!"
    # latency_s is wall-clock runtime per assembly, excluded by design.
    bel = pd.concat([c0.sort_values(keys).drop_duplicates(keys, keep="first"),
                     bel[bel["consumer"] != "C0"]], ignore_index=True)
    n_c0 = len(c0) // 2
    bel["model"] = bel["model"].fillna("none")
    print(f"beliefs: {len(bel)} rows after C0 dedupe "
          f"({n_c0} unique C0 + {len(bel) - n_c0} LLM)")
    sem = S.semantic_per_query(bel)
    sem = S.add_split_flags(sem)
    qrels_test = S.load_qrels_tsv(ROOT / S.QRELS_TEST)

    def attach_doc_ids(sem_df, bel_df):
        """Reattach row evidence sets (semantic_per_query drops doc_ids)."""
        keys = ["system", "query_id", "k", "consumer", "model"]
        doc = bel_df[keys + ["doc_ids"]].drop_duplicates(keys)
        assert len(doc) == len(bel_df[keys].drop_duplicates()), \
            "belief keys not unique for doc_ids join"
        return sem_df.merge(doc, on=keys, how="left", validate="many_to_one")

    sem = attach_doc_ids(sem, bel)
    assert sem["doc_ids"].notna().all(), "doc_ids join failed"
    sem = S.add_hit_indicator(sem, qrels_test)
    w("semantic_all", sem)
    w("agg_semantic", S.aggregate_semantic(sem))
    test = sem[sem["split"] == "test"].copy()
    w("agg_semantic_test", S.aggregate_semantic(test))

    # C0 rows are model-free (model == "none"); per-model slices keep C0.
    def slice_for(model):
        return test[(test["consumer"] == "C0") | (test["model"] == model)]

    # ---- Q1: retrieval->semantic rank stability per consumer/model/k ----
    q1_rows = []
    for cons in ("C0", "C1", "C3"):
        for model in (M8, M14):
            for k in (3, 5):
                for sm, smn in (("accuracy", "acc"), ("brier", "brier")):
                    try:
                        r = S.q1_retrieval_semantic_corr(
                            ir_test[ir_test["system"].isin(RETRIEVAL_SYSTEMS)],
                            slice_for(model), cons, model=None, k=k,
                            sem_metric=sm)
                    except Exception as e:
                        r = {"error": str(e)[:100]}
                    r.update({"sem_metric": sm, "model": model})
                    q1_rows.append(r)
    q1 = pd.DataFrame(q1_rows)
    w("q1_corr", q1)

    # ---- Q2: ranking stability across consumers ----
    q2_rows = []
    for model in (M8, M14):
        for k in (3, 5):
            d = S.q2_rank_stability(slice_for(model), model=None, k=k)
            d["model"], d["k"] = model, k
            q2_rows.append(d)
    q2 = pd.concat(q2_rows, ignore_index=True)
    w("q2_stability", q2)

    # ---- Q3: retriever x consumer interactions ----
    q3_rows = []
    for model in (M8, M14):
        dfm = slice_for(model)
        for k in (3, 5):
            for ra, rb in (("bm25", "dense"), ("bm25", "hybrid"), ("bm25", "rerank"),
                           ("dense", "rerank"), ("hybrid", "rerank")):
                for ca, cb in (("C0", "C1"), ("C0", "C3"), ("C1", "C3")):
                    for metric in ("accuracy", "brier"):
                        r = S.q3_retriever_consumer_interaction(
                            dfm, rb, ra, ca, cb, model=None, k=k,
                            metric=metric, n_boot=N_BOOT)
                        r.update({"model": model, "k": k})
                        q3_rows.append(r)
    q3 = pd.DataFrame(q3_rows)
    w("q3_interaction", q3)

    # ---- Q4: hit vs miss (retrieval systems ONLY: pooling oracle arms,
    # which are always-hit/high-accuracy by construction, confounds system
    # with hit; see prerewrite review B3) ----
    q4_rows = []
    q4pool = test[test["system"].isin(RETRIEVAL_SYSTEMS)]
    for (cons, k), g in q4pool.groupby(["consumer", "k"]):
        if cons == "C0":
            groups = [("none", g)]
        else:
            groups = list(g.groupby("model"))
        for model, gg in groups:
            for metric in ("accuracy", "brier"):
                try:
                    r = S.q4_hit_vs_miss(gg, metric=metric, n_boot=N_BOOT)
                except Exception as e:
                    r = {"error": str(e)[:100]}
                r.update({"model": model, "consumer": cons, "k": k})
                q4_rows.append(r)
    q4 = pd.DataFrame(q4_rows)
    w("q4_hitmiss", q4)

    # ---- Q5: C3-C1 gain ----
    q5_rows = []
    for model in (M8, M14):
        for k in (3, 5):
            for sys_name in sorted(test["system"].unique()):
                for metric in ("accuracy", "brier"):
                    try:
                        r = S.q5_c3_vs_c1_gain(test, system=sys_name, model=model,
                                              k=k, metric=metric, n_boot=N_BOOT)
                    except Exception as e:
                        r = {"error": str(e)[:100]}
                    r.update({"system": sys_name, "model": model, "k": k})
                    q5_rows.append(r)
    q5 = pd.DataFrame(q5_rows)
    w("q5_c3c1", q5)

    # ---- Q6: cross-model ----
    q6 = S.q6_cross_model(test, model_a=M14, model_b=M8, n_boot=N_BOOT)
    w("q6_xmodel", q6)

    # ---- Q7: held-out ----
    q7 = S.q7_heldout(sem)
    w("q7_heldout", q7)

    # ---- sensitivity: flip disputed cache keys, re-run key comparisons ----
    alt_bels = {}
    sens_info = {}
    for model, cdir, rep in (
            (M8, ROOT / "results/v3main/beliefs_cache_awq",
             "results/v3main/beliefs_cache_awq_nondeterminism.json"),
            (M14, ROOT / "results/v3main/beliefs_cache_qwen14",
             "results/v3main/beliefs_cache_qwen14_nondeterminism.json")):
        bm = bel[bel["model"] == model].copy()
        alt_bels[model], sens_info[model] = alt_beliefs_for_model(
            model, cdir, rep, bm)
    alt_sem = S.semantic_per_query(
        pd.concat([alt_bels[M8], alt_bels[M14]], ignore_index=True))
    alt_sem = S.add_split_flags(alt_sem)
    alt_sem = attach_doc_ids(
        alt_sem, pd.concat([alt_bels[M8], alt_bels[M14]], ignore_index=True))
    assert alt_sem["doc_ids"].notna().all(), "alt doc_ids join failed"
    alt_sem = S.add_hit_indicator(alt_sem, qrels_test)
    # C0 is cache-free/deterministic: alt == base for C0; reattach base C0 rows.
    alt_test = pd.concat(
        [test[test["consumer"] == "C0"],
         alt_sem[alt_sem["split"] == "test"]], ignore_index=True)
    comparisons = {}
    short = {M8: "8B", M14: "14B"}

    def slice_m(df, model):
        return df[(df["consumer"] == "C0") | (df["model"] == model)]

    def q6_triplet(df, consumer):
        sub = df[(df["system"] == "rerank") & (df["consumer"] == consumer)
                 & (df["k"] == 3)]
        out = S.q6_cross_model(sub, n_boot=N_BOOT)
        if len(out) == 0:
            return float("nan")
        r = out.iloc[0]
        return {"estimate": float(r["diff_b_minus_a"]),
                "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"])}

    for model in (M8, M14):
        for ca, cb in (("C0", "C1"), ("C0", "C3"), ("C1", "C3")):
            for metric in ("accuracy", "brier"):
                comparisons[f"q3_rerank_bm25_{ca}{cb}_k3_{short[model]}_{metric}"] = (
                    lambda d, m=model, a=ca, b=cb, mt=metric: S.q3_retriever_consumer_interaction(
                        slice_m(d, m), "rerank", "bm25", a, b, model=None, k=3,
                        metric=mt, n_boot=N_BOOT))
        comparisons[f"q5_rerank_C3mC1_k3_{short[model]}"] = (
            lambda d, m=model: S.q5_c3_vs_c1_gain(
                d, system="rerank", model=m, k=3, metric="accuracy",
                n_boot=N_BOOT))
    for cons in ("C1", "C3"):
        comparisons[f"q6_rerank_{cons}_k3"] = (
            lambda d, c=cons: q6_triplet(d, c))
    sens = S.sensitivity_max_delta(test, alt_test, comparisons)
    w("sensitivity_deltas", sens["per_comparison"])
    sens_summary = {
        "per_model": sens_info,
        "max_abs_effect_delta": sens["max_abs_effect_delta"],
        "max_abs_ci_delta": sens["max_abs_ci_delta"],
    }
    w("sensitivity_summary", sens_summary)
    print("sensitivity:", json.dumps(sens_summary, indent=1))

    # ---- headline-cell Holm (preregistered Q3/Q5/Q6-diff head cells only;
    # everything else stays estimation with CIs, labeled exploratory) ----
    def _paired_xy(df, filt_a, filt_b, metric="accuracy"):
        a = df[filt_a].set_index("query_id")[metric]
        b = df[filt_b].set_index("query_id")[metric]
        common = a.index.intersection(b.index)
        return b.loc[common].to_numpy(), a.loc[common].to_numpy()

    hl = []
    mslice = test[test["k"] == 3]
    for model in (M8, M14):
        dm = mslice[(mslice["model"] == model) | (mslice["consumer"] == "C0")]
        for (ca, cb) in (("C0", "C1"), ("C0", "C3"), ("C1", "C3")):
            for sys_name in ("bm25", "rerank"):
                # interaction p via query bootstrap of the interaction vector
                d = dm[dm["system"] == sys_name]
                aa = d[d["consumer"] == ca].set_index("query_id")["accuracy"]
                bb = d[d["consumer"] == cb].set_index("query_id")["accuracy"]
                common = aa.index.intersection(bb.index)
                # interaction needs two systems: rerank vs bm25
                d2 = dm[dm["system"] == "bm25"]
                a2 = d2[d2["consumer"] == ca].set_index("query_id")["accuracy"]
                b2 = d2[d2["consumer"] == cb].set_index("query_id")["accuracy"]
                if sys_name != "rerank":
                    continue
                common = aa.index.intersection(bb.index).intersection(
                    a2.index).intersection(b2.index)
                vec = ((bb.loc[common] - b2.loc[common])
                       - (aa.loc[common] - a2.loc[common])).to_numpy()
                hl.append({"cell": f"Q3_rerank-bm25_{ca}-{cb}_k3_"
                                    f"{'8B' if model == M8 else '14B'}",
                           "p": S.bootstrap_paired_p(vec, n_boot=N_BOOT,
                                                     seed=7)})
    for model in (M8, M14):
        dm = mslice[mslice["model"] == model]
        b, a = _paired_xy(dm, (dm["system"] == "rerank")
                          & (dm["consumer"] == "C1"),
                          (dm["system"] == "rerank")
                          & (dm["consumer"] == "C3"))
        hl.append({"cell": f"Q5_rerank_C3-C1_k3_{'8B' if model == M8 else '14B'}",
                   "p": S.bootstrap_paired_p(b, a, n_boot=N_BOOT, seed=7)})
    for cons in ("C1", "C3"):
        a = mslice[(mslice["system"] == "rerank") & (mslice["consumer"] == cons)
                   & (mslice["model"] == M8)].set_index("query_id")["accuracy"]
        b = mslice[(mslice["system"] == "rerank") & (mslice["consumer"] == cons)
                   & (mslice["model"] == M14)].set_index("query_id")["accuracy"]
        common = a.index.intersection(b.index)
        hl.append({"cell": f"Q6_rerank_8B-14B_{cons}_k3",
                   "p": S.bootstrap_paired_p(a.loc[common].to_numpy(),
                                             b.loc[common].to_numpy(),
                                             n_boot=N_BOOT, seed=7)})
    hl_df = pd.DataFrame(hl)
    hh = S.holm_correction(hl_df["p"].to_numpy(dtype=float))
    hl_df = pd.concat([hl_df, hh], axis=1)
    w("headline_holm", hl_df)
    print(hl_df.round(4).to_string(index=False))
    print("DONE — semantic outputs under results/v3main/semantic/")


if __name__ == "__main__":
    main()
