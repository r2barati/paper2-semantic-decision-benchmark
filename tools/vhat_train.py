"""VHAT Phase 3: train consumer-conditioned Operational-VoI on 3000 labels.

Form (DESIGN-VHAT §1,§10): Vhat(e,b,c) = dBhat(e,c) x Shat(b,c).
 dBhat: evidence+consumer features -> L1 belief shift.
 Shat:  belief+consumer features -> VoI, sample_weight = L1+0.01.
Heads: HistGradientBoostingRegressor(max_iter=200, lr=0.05, depth=3,
 l2=1.0, random_state=77000) — FIXED, no selection.
Deployability budget (§10): precomputed embeddings/ranks/texts/positions +
 OWN full-set belief (+ native C3 per-doc judgments for C3 rows) + analytic
 controller-FD. Cross-model/consumer + label-dependent inputs FORBIDDEN
 (asserted). No fresh-bank access (asserted absent).

Pipeline: 5-fold cluster-grouped CV -> OOF predictions (gate i,iii);
final fit on 80% clusters -> Q0.9 of cluster-mean residuals on frozen 20%
calibration split (alpha=0.1) -> gate (ii). Outputs vhat_oof.parquet,
vhat_heads.pkl, vhat_model.json, vhat_gate.json (BUILD-BANK decision).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "v3main" / "extension_g2a"

assert not (OUT / "vhat_bank").exists() and not list(OUT.glob("*fresh*")), \
    "fresh-bank artifacts present -- training void"
import sklearn
assert sklearn.__version__ == "1.6.1", sklearn.__version__

POS = ["drop0", "drop1", "drop2"]
PRIOR = np.array([0.35, 0.35, 0.30])
PCOLS = ("p_normal", "p_supplier_delay", "p_demand_surge")
CFG = dict(max_iter=200, learning_rate=0.05, max_depth=3,
           l2_regularization=1.0, random_state=77000)
SEEDS = {"cv_shuffle": 77000, "calib": 77001, "boot_i": 78000, "boot_iii": 79000}

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

# ---------------- features (deployability budget §10) ----------------
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
CFEATS = ["c_C0", "c_C1", "c_C38", "c_C314"]


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
        F.append({"rank_dropped": r["rank_dropped"],
                  "cos_q_drop": cos(qe, de), "cos_q_rem": cos(qe, rem),
                  "cos_drop_rem": cos(de, rem),
                  "nrm_drop": float(np.linalg.norm(de)),
                  "c3w_drop": c3w_drop, "c3w_rem": c3w_rem,
                  "p0": p[0], "p1": p[1], "p2": p[2],
                  "entropy": float(-(p * np.log(p + 1e-12)).sum()),
                  "margin": float(ps[2] - ps[1]), "maxProb": float(ps[2]),
                  "priorL1": float(np.abs(p - PRIOR).sum()),
                  "full_conf": r["full_conf"], "fd_sens": _fd(p),
                  "c_C0": r["consumer"] == "C0",
                  "c_C1": r["consumer"] == "C1",
                  "c_C38": (r["consumer"] == "C3") and r["model"].endswith("8B-AWQ"),
                  "c_C314": (r["consumer"] == "C3") and r["model"].endswith("14B-AWQ")})
    F = pd.DataFrame(F)
    # FORBIDDEN-input audit: no labels, no cross-model/consumer leakage
    assert not set(F) & {"true_regime", "ndcg", "brier", "accuracy", "evidence_hit",
                         "agreeXmodel", "c1c3", "p_true", "split"}
    return F


print("building features...", flush=True)
X = build_features(lab)
y_l1, y_voi = lab["l1"].to_numpy(), lab["voi"].to_numpy()
XE = X[EFEATS + CFEATS].copy()
XB = X[BFEATS + CFEATS].copy()
groups = lab["cluster"].to_numpy()

# ---------------- 5-fold cluster CV -> OOF ----------------
rng = np.random.default_rng(SEEDS["cv_shuffle"])
order = rng.permutation(sorted(set(groups)))
fold_of = {c: i % 5 for i, c in enumerate(order)}
folds = np.array([fold_of[c] for c in groups])
oof_db = np.empty(len(lab))
oof_s = np.empty(len(lab))
for k in range(5):
    tr, te = folds != k, folds == k
    h1 = HistGradientBoostingRegressor(**CFG).fit(XE.to_numpy()[tr], y_l1[tr])
    oof_db[te] = h1.predict(XE.to_numpy()[te])
    h2 = HistGradientBoostingRegressor(**CFG).fit(
        XB.to_numpy()[tr], y_voi[tr],
        sample_weight=y_l1[tr] + 0.01)
    oof_s[te] = h2.predict(XB.to_numpy()[te])
oof_v = oof_db * oof_s
pd.DataFrame({"cluster": groups, "consumer": lab["consumer"].to_numpy(),
              "model": lab["model"].to_numpy(), "query_id": lab["query_id"].to_numpy(),
              "position": lab["position"].to_numpy(),
              "voi": y_voi, "l1": y_l1, "oof_v": oof_v}).to_parquet(
    OUT / "vhat_oof.parquet", index=False)
print("OOF done")

# ---------------- final fit on 80% + calibration on frozen 20% ----------------
rng2 = np.random.default_rng(SEEDS["calib"])
cal = set(rng2.choice(sorted(set(groups)), size=int(round(0.2 * n_clu)), replace=False))
is_cal = np.array([c in cal for c in groups])
print(f"calibration clusters: {len(cal)} / {n_clu}")
H1 = HistGradientBoostingRegressor(**CFG).fit(XE.to_numpy()[~is_cal], y_l1[~is_cal])
H2 = HistGradientBoostingRegressor(**CFG).fit(
    XB.to_numpy()[~is_cal], y_voi[~is_cal],
    sample_weight=y_l1[~is_cal] + 0.01)
with open(OUT / "vhat_heads.pkl", "wb") as f:
    pickle.dump({"dBhat": H1, "Shat": H2, "sklearn": sklearn.__version__,
                 "efeats": EFEATS + CFEATS, "bfeats": BFEATS + CFEATS}, f)
resid = []
for c in sorted(cal):
    m = groups == c
    resid.append(float((y_voi[m] - H1.predict(XE.to_numpy()[m]) *
                        H2.predict(XB.to_numpy()[m])).mean()))
Q = float(np.quantile(resid, 0.9))  # alpha=0.1 frozen
print(f"calibration: n_clu={len(resid)} Q0.9={Q:.3f}")

json.dump({"efeats": EFEATS + CFEATS, "bfeats": BFEATS + CFEATS, "cfg": CFG,
           "alpha": 0.1, "Q": Q, "n_fit_clusters": int(n_clu - len(cal)),
           "n_cal_clusters": len(cal), "sklearn": sklearn.__version__,
           "fit": "80pct clusters", "calibration": "frozen 20pct, never fit"},
          open(OUT / "vhat_model.json", "w"), indent=1)
print("model frozen")

# ---------------- historical feasibility gate (§9; OOF + calibration) ----------------
oof = pd.read_parquet(OUT / "vhat_oof.parquet")
assert (oof["voi"].to_numpy() == y_voi).all() and (oof["cluster"].to_numpy() == groups).all()

# (i) cluster-bootstrapped Spearman, pooled + per-consumer stability
rngi = np.random.default_rng(SEEDS["boot_i"])
uc = sorted(set(groups))
rhos = []
for _ in range(2000):
    draw = rngi.choice(uc, len(uc), replace=True)
    sub = oof[oof["cluster"].isin(draw)]
    rhos.append(float(spearmanr(sub["oof_v"].to_numpy(), sub["voi"].to_numpy())[0]))
rhos = np.array(rhos)
ci_i = [round(float(np.percentile(rhos, 2.5)), 3), round(float(np.percentile(rhos, 97.5)), 3)]
per_cons = {}
for (c, m) in sorted(set(zip(oof["consumer"].to_numpy(), oof["model"].to_numpy()))):
    s = oof[(oof["consumer"] == c) & (oof["model"] == m)]
    r = float(spearmanr(s["oof_v"].to_numpy(), s["voi"].to_numpy())[0])
    per_cons[f"{c}/{m.split('/')[-1] if m != 'none' else 'none'}"] = round(r, 3)
pass_i = bool(ci_i[0] > 0) and sum(v > 0 for v in per_cons.values()) >= 4
print(f"(i) pooled rho CI={ci_i} per-consumer={per_cons} PASS={pass_i}", flush=True)

# (ii) coverage on frozen calibration split with FINAL predictor
is_cal = np.array([c in cal for c in groups])
lcb_cal = (H1.predict(XE.to_numpy()[is_cal]) * H2.predict(XB.to_numpy()[is_cal])) - Q
cov = float((y_voi[is_cal] >= lcb_cal).mean())
pass_ii = bool(0.85 <= cov <= 0.95)
print(f"(ii) calibration coverage={cov:.3f} (n={is_cal.sum()}) PASS={pass_ii}", flush=True)

# (iii) OOF single-minimum-drop gating vs always-transmit (§11).
# LCB uses TRUE out-of-fold predictions (oof_v); QOOF is the 0.9-quantile of
# OOF cluster-mean residuals. (A prior revision mistakenly used final-fit
# predictions here; fixed 2026-09-20 after an independent replicate
# disagreed. The frozen rule always meant OOF quantities.)
qo = []
for c in sorted(set(groups)):
    m = groups == c
    qo.append(float((y_voi[m] - oof_v[m]).mean()))
QOOF = float(np.quantile(qo, 0.9))
lcb_oof = oof_v - QOOF
# per-query gated dJ: need dJ_full + dJ per position
djtab = {}
for ep_path in ("loo_episodes.parquet", "c1_loo_episodes.parquet", "c1_loo_test_episodes.parquet"):
    ep = pd.read_parquet(OUT / ep_path)
    dd = ep.merge(ni.rename("profit_ni"), on=["query_id", "seed"])
    dd["dj"] = dd["profit"] - dd["profit_ni"]
    for (cons, model, qid, ab), v in dd.groupby(["consumer", "model", "query_id", "ablation"])["dj"].mean().items():
        djtab[(cons, model, qid, ab)] = float(v)
strat_of = {}
gated_rows = []
idx = pd.DataFrame({"cluster": groups, "consumer": lab["consumer"].to_numpy(),
                    "model": lab["model"].to_numpy(), "query_id": lab["query_id"].to_numpy(),
                    "position": lab["position"].to_numpy()})
idx["lcb"] = lcb_oof
for (cons, model, qid), gq in idx.groupby(["consumer", "model", "query_id"]):
    b = gq.loc[gq["lcb"].idxmin()]
    drop = b["position"] if b["lcb"] <= 0 else None
    ab = drop if drop else "full"
    gated_rows.append({"cluster": b["cluster"], "consumer": cons, "model": model,
                       "query_id": qid, "position": b["position"],
                       "gated_dj": djtab[(cons, model, qid, ab)],
                       "full_dj": djtab[(cons, model, qid, "full")]})
G = pd.DataFrame(gated_rows)
G["diff"] = G["gated_dj"] - G["full_dj"]
G["stratum"] = G["consumer"] + "/" + G["model"].str.split("/").str[-1]
stra = sorted(G["stratum"].unique())
assert len(stra) == 5, stra
rng3 = np.random.default_rng(SEEDS["boot_iii"])
uc3 = sorted(set(G["cluster"].to_numpy()))
pm = G.groupby(["consumer", "position"])["diff"].mean()
boots = []
for _ in range(5000):
    draw = rng3.choice(uc3, len(uc3), replace=True)
    boots.append(float(G[G["cluster"].isin(draw)].groupby(
        ["stratum", "position"])["diff"].mean().mean()))
boots = np.array(boots)
ci_iii = [round(float(np.percentile(boots, 2.5)), 3),
          round(float(np.percentile(boots, 97.5)), 3)]
pass_iii_main = bool(ci_iii[0] > 0)
strat_pos = sum(float(G[G["stratum"] == c]["diff"].mean()) > 0 for c in stra)
pos_pos = sum(float(G[G["position"] == p]["diff"].mean()) > 0 for p in POS)
strat_detail = {c: round(float(G[G["stratum"] == c]["diff"].mean()), 2) for c in stra}
pos_detail = {p: round(float(G[G["position"] == p]["diff"].mean()), 2) for p in POS}
pass_iii = bool(pass_iii_main and strat_pos >= 4 and pos_pos >= 2)
print(f"(iii) pooled diff CI={ci_iii} strata+={strat_pos}/5 pos+={pos_pos}/3 PASS={pass_iii}",
      flush=True)

gate = {"i": {"CI": ci_i, "per_consumer": per_cons, "PASS": bool(pass_i)},
        "ii": {"coverage": round(cov, 4), "band": [0.85, 0.95],
               "PASS": bool(pass_ii)},
        "iii": {"CI": ci_iii, "strata_positive": int(strat_pos),
                "positions_positive": int(pos_pos),
                "strata_detail": strat_detail, "position_detail": pos_detail,
                "PASS": bool(pass_iii)},
        "BUILD_BANK": bool(pass_i and pass_ii and pass_iii)}
json.dump(gate, open(OUT / "vhat_gate.json", "w"), indent=1)
print("GATE:", json.dumps(gate), flush=True)
