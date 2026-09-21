"""VHAT2 Phase 3 (Track B): factored consumer-conditioned Operational-VoI.

Frozen form (DESIGN-VHAT2 §B1; SOLE licensed response to STEP0_AUTOPSY):
  Vhat2(e,b,c,pos) = dBhat(e,c,pos) x Shat(b,c | dBhat).
Pooled GBM trunk, SAME frozen CFG as Vhat (max_iter=200, lr=0.05, depth=3,
l2=1.0, random_state=77000) — NO per-stratum fits. MANDATORY consumer
one-hot (5 strata) + position one-hot (3) in BOTH heads.

Shat CONDITIONS on dBhat output via TWO-STAGE fit with REFIT predictions
(documented choice): dBhat is fit first; Shat's extra feature on a fit row
is the dBhat prediction from the head fit on that SAME fit set
(in-sample/refit). OOF rows use their fold-test dBhat predictions. Refit
was chosen so that no prediction consumed in Shat training ever saw a row
outside its own fit clusters (zero cross-cluster leakage through the
conditioning feature). No nested inner-CV variant was run.

Feature budget = DESIGN-VHAT §10 unchanged: own-belief + embeddings/ranks +
controller-FD (+ native C3 per-doc judgments for C3 rows).
Cross-model/consumer + label-dependent inputs FORBIDDEN (asserted by exact
feature allowlist). No fresh-bank access (asserted absent: no vhat2_bank,
no *fresh* artifacts).

Corrected conformal chain (DESIGN-VHAT2 §B2, THE validity object):
exchangeable cluster -> S_c = max_{i in c}(VHAT2_i - V_i) [OVERPREDICTION
residual: LCB fails exactly when Vhat-V exceeds Q] ->
Q = finite-sample ceil((n+1)(1-alpha))/n quantile (alpha=0.1 frozen) ->
LCB_i = Vhat2_i - Q. Calibration = SAME frozen 20% cluster split as Vhat
(seed-77001 reproduction; identical 13 calibration IDs asserted via
cluster-universe equality with frozen vhat_oof.parquet + deterministic
reseed). Stratum residuals/quantiles are diagnostics ONLY, never
thresholds.

Pipeline: 5-fold cluster-grouped CV (frozen 77000 construction reused) ->
OOF predictions (gate i,iii); final fit on frozen 80% -> max-score Q on
calibration (gate ii). Single-minimum-drop rule (DESIGN-VHAT §11 honored).
Outputs vhat2_oof.parquet, vhat2_heads.pkl, vhat2_model.json,
vhat2_gate.json (BUILD_BANK_PROPOSAL verdict: PROPOSES untouched-bank
confirmation on pass; never confirmatory itself).

Seeds: 77000/77001 = frozen reproductions (CV folds, calibration split);
80xxx = new family for new randomness (bootstraps 80000/81000).
Development-only evidence. STOP rules bind: no re-tuning, no architecture
changes after numbers.

FIREWALL: reads Step-0-era inputs + historical LOO labels + frozen
beliefs/caches only. Never reads Track A (no extension_tracka files, no
Agentick results). Never writes vhat_* files.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold  # noqa: F401 (doc: grouped CV family)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "v3main" / "extension_g2a"

assert not (OUT / "vhat2_bank").exists() and not list(OUT.glob("*fresh*")), \
    "fresh-bank artifacts present -- Track B training void"
import sklearn
assert sklearn.__version__ == "1.6.1", sklearn.__version__

POS = ["drop0", "drop1", "drop2"]
PRIOR = np.array([0.35, 0.35, 0.30])
PCOLS = ("p_normal", "p_supplier_delay", "p_demand_surge")
CFG = dict(max_iter=200, learning_rate=0.05, max_depth=3,
           l2_regularization=1.0, random_state=77000)
assert CFG == {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3,
               "l2_regularization": 1.0, "random_state": 77000}, CFG
SEEDS = {"cv_shuffle": 77000, "calib": 77001, "boot_i": 80000,
         "boot_iii": 81000}

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids, test_ids = set(sp["dev"]), set(sp["test"])

froz = pd.read_parquet(ROOT / "results/v3main/sim/episodes.parquet")
ni = froz[(froz["system"] == "noinfo") & (froz["k"] == 3)].set_index(
    ["query_id", "seed"])["profit"]

FULLB = pd.concat([pd.read_parquet(
    ROOT / f"results/v3main/beliefs_{m}.parquet")
    for m in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ")], ignore_index=True)
_c0 = FULLB[FULLB["consumer"] == "C0"].sort_values(
    ["system", "query_id", "k"]).drop_duplicates(["system", "query_id", "k"])
FULLB = pd.concat([_c0, FULLB[FULLB["consumer"] != "C0"]], ignore_index=True)


def loo_labels(ep_path: str, bel_loo_path: str) -> pd.DataFrame:
    ep = pd.read_parquet(OUT / ep_path)
    d = ep.merge(ni.rename("profit_ni"), on=["query_id", "seed"])
    d["dj"] = d["profit"] - d["profit_ni"]
    g = d.groupby(["consumer", "model", "query_id", "ablation"])["dj"].mean().reset_index()
    f = g[g["ablation"] == "full"].set_index(["consumer", "model", "query_id"])["dj"]
    bloo = pd.read_parquet(OUT / bel_loo_path)
    bloo = bloo[bloo["ablation"].isin(POS)]
    rows = []
    for _, r in g[g["ablation"] != "full"].iterrows():
        key = (r["consumer"], r["model"], r["query_id"])
        lr = bloo[(bloo["consumer"] == r["consumer"]) & (bloo["model"] == r["model"]) &
                  (bloo["query_id"] == r["query_id"]) & (bloo["ablation"] == r["ablation"])]
        fr = FULLB[(FULLB["consumer"] == r["consumer"]) & (FULLB["model"] == r["model"]) &
                   (FULLB["query_id"] == r["query_id"]) & (FULLB["system"] == "rerank") &
                   (FULLB["k"] == 3)]
        assert len(lr) == 1 and len(fr) == 1, key
        fr = fr.iloc[0]
        lp = np.array([lr.iloc[0][c] for c in PCOLS])
        fp = np.array([fr[c] for c in PCOLS])
        fids = list(fr["doc_ids"])
        lids = list(lr.iloc[0]["doc_ids"])
        drop = [x for x in fids if x not in lids]
        assert len(drop) == 1, (key, fids, lids)
        rows.append({"consumer": r["consumer"], "model": r["model"],
                     "query_id": r["query_id"], "position": r["ablation"],
                     "rank_dropped": fids.index(drop[0]), "doc_dropped": drop[0],
                     "doc_ids": lids, "full_ids": fids,
                     "voi": float(f.loc[key] - r["dj"]),
                     "l1": float(np.abs(lp - fp).sum()),
                     "fp": fp, "lp": lp,
                     "full_conf": float(fr["confidence"]),
                     "full_abstain": bool(fr["abstain"])})
    return pd.DataFrame(rows)


print("loading labels...", flush=True)
lab = pd.concat([
    loo_labels("loo_episodes.parquet", "loo_beliefs.parquet"),
    loo_labels("c1_loo_episodes.parquet", "c1_loo_beliefs.parquet"),
    loo_labels("c1_loo_test_episodes.parquet", "c1_loo_test_beliefs.parquet"),
], ignore_index=True)
print(f"label rows: {len(lab)} (expect 3000)")
assert len(lab) == 3000, len(lab)
assert (lab["rank_dropped"] == lab["position"].str[4:].astype(int)).all()

# ---------------- strata (consumer one-hot, 5) + positions ----------------
lab["stratum"] = lab.apply(
    lambda r: f"{r['consumer']}/{r['model'].split('/')[-1] if r['model'] != 'none' else 'none'}",
    axis=1)
STRATA = ["C0/none", "C1/Qwen3-8B-AWQ", "C1/Qwen3-14B-AWQ",
          "C3/Qwen3-8B-AWQ", "C3/Qwen3-14B-AWQ"]
assert sorted(lab["stratum"].unique().tolist()) == sorted(STRATA), \
    sorted(lab["stratum"].unique().tolist())
assert sorted(lab["position"].unique().tolist()) == POS, \
    sorted(lab["position"].unique().tolist())
SCOLS = ["s_C0", "s_C1_8B", "s_C1_14B", "s_C3_8B", "s_C3_14B"]
POSCOLS = ["p_drop0", "p_drop1", "p_drop2"]

# ---------------- clusters ----------------
_qs, _sets = {}, {}
for fn, sf, k in (("kaggle/inputs_loo/queries_dev40.jsonl", "loo_sets.json", "inputs_loo"),
                  ("kaggle/inputs_loo_test/queries_loo.jsonl", "loo_sets_test.json",
                   "inputs_loo_test")):
    for line in open(ROOT / fn):
        _q = json.loads(line)
        _qs[_q["_id"]] = _q["text"]
    _s = json.load(open(ROOT / "kaggle" / k / sf))
    _sets.update(_s["sets"])
lab["cluster"] = lab["query_id"].map(
    lambda q: hashlib.sha256((_qs[q] + "|" + ",".join(sorted(_sets[q]["full"]))).encode()
                             ).hexdigest()[:12])
n_clu = lab["cluster"].nunique()
print(f"clusters: {n_clu}")
assert n_clu == 64, n_clu

# ---------------- features (deployability budget §10, unchanged) ----------------
Z = np.load(ROOT / "runs/v3main/qwen3emb_main_vectors.npz")
DIDX = {d: i for i, d in enumerate(Z["doc_ids"])}
QIDX = {q: i for i, q in enumerate(Z["qids"])}
DE, QE = Z["doc_emb"].astype(float), Z["q_emb"].astype(float)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


_qmeta = {}
for line in open(ROOT / "data" / "v3" / "queries.jsonl"):
    _q = json.loads(line)
    _qmeta[_q["_id"]] = _q["metadata"]
_docs = {}
for line in open(ROOT / "data" / "v3" / "corpus.jsonl"):
    _d = json.loads(line)
    _docs[_d["_id"]] = _d["text"]
_C3CACHE = {"Qwen/Qwen3-8B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_awq",
            "Qwen/Qwen3-14B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_qwen14"}


def _c3_judgment(model: str, own: str, text: str):
    import re
    user = f"Operator's own node: {own}\n\nEvidence document:\n{text}"
    ks = f"{model}||5966cadde90e8236||{hashlib.sha256(user.encode()).hexdigest()[:16]}"
    fn = hashlib.sha256(ks.encode()).hexdigest()[:16] + ".json"
    p = json.loads((_C3CACHE[model] / fn).read_text())
    raw = re.sub(r"<think>.*?</think>", "", p["raw"], flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _fd(p: np.ndarray) -> float:
    from src.env import (INITIAL_INVENTORY, CausalOptimizer, InventoryState)
    from src.events import P5_DEMAND_MEAN, P5_HORIZON
    keys = ("normal", "supplier_delay", "demand_surge")

    def decide(prob):
        o = CausalOptimizer(seed=0, regime_probabilities=dict(zip(keys, prob)),
                            horizon=P5_HORIZON, initial_inventory=INITIAL_INVENTORY,
                            demand_mean=P5_DEMAND_MEAN)
        return o.decide(InventoryState(time=0, on_hand=float(INITIAL_INVENTORY)))

    base = decide(p)
    best = 0.0
    for i in range(3):
        for sgn in (1.0, -1.0):
            q = p.copy()
            q[i] = min(0.99, max(0.01, q[i] + sgn * 0.05))
            q = q / q.sum()
            best = max(best, abs(decide(q) - base))
    return float(best)


EFEATS = ["rank_dropped", "cos_q_drop", "cos_q_rem", "cos_drop_rem",
          "nrm_drop", "c3w_drop", "c3w_rem"]
BFEATS = ["p0", "p1", "p2", "entropy", "margin", "maxProb", "priorL1",
          "full_conf", "fd_sens"]
DBFEATS = EFEATS + SCOLS + POSCOLS          # dBhat head (15)
SBFEATS = BFEATS + SCOLS + POSCOLS + ["dbhat"]  # Shat head (18, conditioned)
FORBIDDEN = {"true_regime", "ndcg", "brier", "accuracy", "evidence_hit",
             "agreeXmodel", "c1c3", "p_true", "split"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    F = []
    for _, r in df.iterrows():
        p, qid = r["fp"], r["query_id"]
        ps = np.sort(p)
        qe, de = QE[QIDX[qid]], DE[DIDX[r["doc_dropped"]]]
        rem = DE[[DIDX[d] for d in r["doc_ids"]]].mean(axis=0)
        c3w_drop = c3w_rem = 0.0
        if r["consumer"] == "C3":
            own = _qmeta[qid]["entity_node"]
            jd = _c3_judgment(r["model"], own, _docs[r["doc_dropped"]])
            c3w_drop = float(jd["confidence"]) * (1.0 if jd["entity_match"] else 0.2) * \
                (1.0 if jd["fresh"] else 0.2)
            ws = []
            for d in r["doc_ids"]:
                je = _c3_judgment(r["model"], own, _docs[d])
                ws.append(float(je["confidence"]) *
                          (1.0 if je["entity_match"] else 0.2) *
                          (1.0 if je["fresh"] else 0.2))
            c3w_rem = float(np.mean(ws))
        row = {"rank_dropped": r["rank_dropped"],
               "cos_q_drop": cos(qe, de), "cos_q_rem": cos(qe, rem),
               "cos_drop_rem": cos(de, rem),
               "nrm_drop": float(np.linalg.norm(de)),
               "c3w_drop": c3w_drop, "c3w_rem": c3w_rem,
               "p0": p[0], "p1": p[1], "p2": p[2],
               "entropy": float(-(p * np.log(p + 1e-12)).sum()),
               "margin": float(ps[2] - ps[1]), "maxProb": float(ps[2]),
               "priorL1": float(np.abs(p - PRIOR).sum()),
               "full_conf": r["full_conf"], "fd_sens": _fd(p)}
        for s, c in zip(STRATA, SCOLS):
            row[c] = (r["stratum"] == s)
        for pp, c in zip(POS, POSCOLS):
            row[c] = (r["position"] == pp)
        F.append(row)
    F = pd.DataFrame(F)
    # Deployability-budget audit: exact allowlist + forbidden exclusion
    assert set(F.columns) == set(EFEATS + BFEATS + SCOLS + POSCOLS), \
        set(F.columns) ^ set(EFEATS + BFEATS + SCOLS + POSCOLS)
    assert not set(F) & FORBIDDEN
    assert "dbhat" not in F.columns  # conditioning added only per-fit, two-stage
    return F


print("building features...", flush=True)
X = build_features(lab)
y_l1, y_voi = lab["l1"].to_numpy(), lab["voi"].to_numpy()
XD = X[DBFEATS].copy()
XSB = X[[c for c in SBFEATS if c != "dbhat"]].copy()
groups = lab["cluster"].to_numpy()

# ---------------- frozen cluster universe check (read-only, historical) ----------------
_vhat_oof = pd.read_parquet(OUT / "vhat_oof.parquet")
assert set(groups.tolist()) == set(_vhat_oof["cluster"].tolist()), \
    "cluster universe differs from frozen Vhat -- split reproduction void"
assert len(_vhat_oof) == 3000
del _vhat_oof

# ---------------- 5-fold cluster CV -> OOF (two-stage, refit conditioning) ----------------
rng = np.random.default_rng(SEEDS["cv_shuffle"])
order = rng.permutation(sorted(set(groups)))
fold_of = {c: i % 5 for i, c in enumerate(order)}
folds = np.array([fold_of[c] for c in groups])
oof_db = np.empty(len(lab))
oof_s = np.empty(len(lab))
for k in range(5):
    tr, te = folds != k, folds == k
    h1 = HistGradientBoostingRegressor(**CFG).fit(XD.to_numpy()[tr], y_l1[tr])
    oof_db[te] = h1.predict(XD.to_numpy()[te])
    dbhat_tr = h1.predict(XD.to_numpy()[tr])  # REFIT predictions, same fit rows
    XS_tr = np.column_stack([XSB.to_numpy()[tr], dbhat_tr])
    h2 = HistGradientBoostingRegressor(**CFG).fit(
        XS_tr, y_voi[tr], sample_weight=y_l1[tr] + 0.01)
    XS_te = np.column_stack([XSB.to_numpy()[te], oof_db[te]])
    oof_s[te] = h2.predict(XS_te)
oof_v = oof_db * oof_s
pd.DataFrame({"cluster": groups, "consumer": lab["consumer"].to_numpy(),
              "model": lab["model"].to_numpy(), "query_id": lab["query_id"].to_numpy(),
              "position": lab["position"].to_numpy(),
              "stratum": lab["stratum"].to_numpy(),
              "voi": y_voi, "l1": y_l1,
              "oof_dbhat": oof_db, "oof_shat": oof_s, "oof_v": oof_v}).to_parquet(
    OUT / "vhat2_oof.parquet", index=False)
print("OOF done")

# ---------------- final fit on SAME frozen 80% + calibration on frozen 20% ----------------
rng2 = np.random.default_rng(SEEDS["calib"])
cal = set(rng2.choice(sorted(set(groups)), size=int(round(0.2 * n_clu)), replace=False))
is_cal = np.array([c in cal for c in groups])
print(f"calibration clusters: {len(cal)} / {n_clu}")
assert len(cal) == 13, len(cal)
print("calibration IDs:", sorted(cal), flush=True)
H1 = HistGradientBoostingRegressor(**CFG).fit(XD.to_numpy()[~is_cal], y_l1[~is_cal])
_db80 = H1.predict(XD.to_numpy()[~is_cal])  # REFIT predictions, same fit rows
H2 = HistGradientBoostingRegressor(**CFG).fit(
    np.column_stack([XSB.to_numpy()[~is_cal], _db80]), y_voi[~is_cal],
    sample_weight=y_l1[~is_cal] + 0.01)
with open(OUT / "vhat2_heads.pkl", "wb") as f:
    pickle.dump({"dBhat": H1, "Shat": H2, "sklearn": sklearn.__version__,
                 "dbfeats": DBFEATS, "sbfeats": SBFEATS,
                 "shat_conditioning": "refit dBhat predictions on same fit rows"}, f)


def finite_quantile(scores: list[float], alpha: float) -> tuple[float, int, int]:
    """Finite-sample one-sided quantile: k=ceil((n+1)(1-alpha))/n level."""
    n = len(scores)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    assert 1 <= k <= n, (k, n)
    return float(sorted(scores)[k - 1]), k, n


def predict_vhat2(H1, H2, XDm: np.ndarray, XSBm: np.ndarray) -> np.ndarray:
    db = H1.predict(XDm)
    return db * H2.predict(np.column_stack([XSBm, db]))


# Corrected chain: S_c = max_rows(Vhat2 - V) per calibration cluster
Scal = []
for c in sorted(cal):
    m = groups == c
    Scal.append(float((predict_vhat2(H1, H2, XD.to_numpy()[m], XSB.to_numpy()[m])
                       - y_voi[m]).max()))
Q, Qk, Qn = finite_quantile(Scal, 0.1)  # alpha=0.1 frozen
print(f"calibration: n_clu={len(Scal)} Q(one-sided max-score, k={Qk}/{Qn})={Q:.3f}")

json.dump({"dbfeats": DBFEATS, "sbfeats": SBFEATS, "cfg": CFG,
           "alpha": 0.1, "Q": Q, "Q_rank": [Qk, Qn],
           "shat_conditioning": "refit dBhat predictions on same fit rows "
           "(two-stage; CV OOF rows use fold-test dBhat predictions)",
           "score": "S_c = max_rows(Vhat2 - V) per cluster (overprediction)",
           "n_fit_clusters": int(n_clu - len(cal)),
           "n_cal_clusters": len(cal), "calibration_ids": sorted(cal),
           "sklearn": sklearn.__version__,
           "fit": "frozen 80pct clusters (seed-77001 reproduction)",
           "calibration": "frozen 20pct, never fit"},
          open(OUT / "vhat2_model.json", "w"), indent=1)
print("model frozen")

# ---------------- B2 development/feasibility gate (OOF + calibration) ----------------
oof = pd.read_parquet(OUT / "vhat2_oof.parquet")
assert (oof["voi"].to_numpy() == y_voi).all() and (oof["cluster"].to_numpy() == groups).all()

# (i) cluster-bootstrapped Spearman, pooled + per-stratum point-positive
rngi = np.random.default_rng(SEEDS["boot_i"])
uc = sorted(set(groups))
rhos = []
for _ in range(2000):
    draw = rngi.choice(uc, len(uc), replace=True)
    sub = oof[oof["cluster"].isin(draw)]
    rhos.append(float(spearmanr(sub["oof_v"].to_numpy(), sub["voi"].to_numpy())[0]))
rhos = np.array(rhos)
ci_i = [round(float(np.percentile(rhos, 2.5)), 3), round(float(np.percentile(rhos, 97.5)), 3)]
per_stratum = {}
for s in STRATA:
    ss = oof[oof["stratum"] == s]
    per_stratum[s] = round(float(spearmanr(ss["oof_v"].to_numpy(),
                                           ss["voi"].to_numpy())[0]), 3)
n_pos = sum(v > 0 for v in per_stratum.values())
pass_i = bool(ci_i[0] > 0) and n_pos >= 4
print(f"(i) pooled rho CI={ci_i} per-stratum={per_stratum} (+={n_pos}/5) PASS={pass_i}",
      flush=True)

# (ii) coverage on frozen calibration clusters with FINAL predictor (no upper band)
is_cal = np.array([c in cal for c in groups])
lcb_cal = predict_vhat2(H1, H2, XD.to_numpy()[is_cal], XSB.to_numpy()[is_cal]) - Q
cov = float((y_voi[is_cal] >= lcb_cal).mean())
pass_ii = bool(cov >= 0.85)
print(f"(ii) calibration coverage={cov:.4f} (n_rows={is_cal.sum()}) PASS={pass_ii}",
      flush=True)

# (iii) OOF single-minimum-drop gating vs always-transmit (§11).
# LCB uses TRUE out-of-fold predictions (oof_v); QOOF is the finite-sample
# corrected quantile of OOF cluster-MAX overprediction scores
# (k=ceil(65*0.9)/64 -> 59/64), mirroring the frozen corrected chain.
Soof = []
for c in sorted(set(groups)):
    m = groups == c
    Soof.append(float((oof_v[m] - y_voi[m]).max()))
QOOF, QOk, QOn = finite_quantile(Soof, 0.1)
lcb_oof = oof_v - QOOF
print(f"(iii) QOOF(max-score, k={QOk}/{QOn})={QOOF:.3f}", flush=True)
# per-query gated dJ: need dJ_full + dJ per position
djtab = {}
for ep_path in ("loo_episodes.parquet", "c1_loo_episodes.parquet", "c1_loo_test_episodes.parquet"):
    ep = pd.read_parquet(OUT / ep_path)
    dd = ep.merge(ni.rename("profit_ni"), on=["query_id", "seed"])
    dd["dj"] = dd["profit"] - dd["profit_ni"]
    for (cons, model, qid, ab), v in dd.groupby(["consumer", "model", "query_id", "ablation"])["dj"].mean().items():
        djtab[(cons, model, qid, ab)] = float(v)
idx = pd.DataFrame({"cluster": groups, "consumer": lab["consumer"].to_numpy(),
                    "model": lab["model"].to_numpy(), "query_id": lab["query_id"].to_numpy(),
                    "position": lab["position"].to_numpy(),
                    "stratum": lab["stratum"].to_numpy()})
idx["lcb"] = lcb_oof
gated_rows = []
for (cons, model, qid), gq in idx.groupby(["consumer", "model", "query_id"]):
    b = gq.loc[gq["lcb"].idxmin()]
    drop = b["position"] if b["lcb"] <= 0 else None
    ab = drop if drop else "full"
    gated_rows.append({"cluster": b["cluster"], "consumer": cons, "model": model,
                       "query_id": qid, "position": b["position"],
                       "stratum": b["stratum"], "dropped": bool(drop is not None),
                       "gated_dj": djtab[(cons, model, qid, ab)],
                       "full_dj": djtab[(cons, model, qid, "full")]})
G = pd.DataFrame(gated_rows)
G["diff"] = G["gated_dj"] - G["full_dj"]
assert sorted(G["stratum"].unique().tolist()) == sorted(STRATA), \
    sorted(G["stratum"].unique().tolist())
# selectivity = drop fraction: queries where min-LCB position was dropped
n_q = len(G)
n_drop = int(G["dropped"].sum())
selectivity = n_drop / n_q
pass_sel = bool(0.05 <= selectivity <= 0.95)
rng3 = np.random.default_rng(SEEDS["boot_iii"])
uc3 = sorted(set(G["cluster"].to_numpy()))
boots = []
for _ in range(5000):
    draw = rng3.choice(uc3, len(uc3), replace=True)
    boots.append(float(G[G["cluster"].isin(draw)].groupby(
        ["stratum", "position"])["diff"].mean().mean()))
boots = np.array(boots)
ci_iii = [round(float(np.percentile(boots, 2.5)), 3),
          round(float(np.percentile(boots, 97.5)), 3)]
pass_iii_main = bool(ci_iii[0] > 0)
strat_detail = {s: round(float(G[G["stratum"] == s]["diff"].mean()), 2) for s in STRATA}
pos_detail = {p: round(float(G[G["position"] == p]["diff"].mean()), 2) for p in POS}
pass_iii = bool(pass_sel and pass_iii_main)
print(f"(iii) selectivity={selectivity:.3f} (drops={n_drop}/{n_q}) pooled diff CI={ci_iii} "
      f"PASS={pass_iii}", flush=True)
print(f"     strata_detail={strat_detail}", flush=True)
print(f"     position_detail={pos_detail}", flush=True)

gate = {"i": {"CI": ci_i, "per_stratum": per_stratum,
              "strata_positive": int(n_pos), "PASS": bool(pass_i)},
        "ii": {"coverage": round(cov, 4), "threshold": 0.85,
               "Q_rank": [Qk, Qn], "PASS": bool(pass_ii)},
        "iii": {"selectivity": round(selectivity, 4),
                "selectivity_band": [0.05, 0.95],
                "QOOF": round(QOOF, 4), "QOOF_rank": [QOk, QOn],
                "CI": ci_iii, "strata_detail": strat_detail,
                "position_detail": pos_detail, "PASS": bool(pass_iii)},
        "BUILD_BANK_PROPOSAL": bool(pass_i and pass_ii and pass_iii),
        "claim_ceiling": "development-only evidence; a pass PROPOSES "
                         "untouched-bank confirmation, never confirmatory"}
json.dump(gate, open(OUT / "vhat2_gate.json", "w"), indent=1)
print("GATE:", json.dumps(gate), flush=True)
