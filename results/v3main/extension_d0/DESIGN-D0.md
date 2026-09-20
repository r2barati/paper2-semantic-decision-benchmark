# D0 — FROZEN DESIGN (2026-09-20, pre-execution; revised per owner corrections)

Scope: Gate-1 belief-VoI ONLY. Analysis-only + frozen-episode reuse.
No new LLM calls. No frozen-matrix edits. No document-VoI. No Agentick.
Outputs: `results/v3main/extension_d0/` + `tools/d0_*.py` + LEDGER entry.

## 1. Primary quantity (utility space)

```
VoI(e) = E[J | do(e available)] − E[J | do(e unavailable)]
```

* `J`: per-(arm,query) `delta_vs_noinfo = mean_seed(profit − profit_NoInfo)`
  from `utility_by_query*.parquet`; headline balanced `J` from
  `utility_by_arm_balanced.parquet` reported as secondary reference only.
* `do(unavailable)` for Gate-1 = frozen **NoInfo** belief rung
  (locked per owner answer). No minus-e context (that is Gate-2 territory).
* **Decomposition is diagnostic only**: `evidence→belief` (accuracy/Brier),
  `belief→action` (ADis/ADisRate from A0 replay), `action→J` (paired profit
  diffs). **ADis is a mediator/criticality measure, never the VoI definition.**

## 2. Belief-VoI vs document-VoI (validity boundary, frozen)

* **Belief-VoI (Gate-1, valid on existing artifacts):** contrasts of *frozen
  beliefs*, e.g. `J(b_system,q) − J(b_NoInfo,q)` per query, or gated
  transmit-vs-NoInfo policies built from them. Estimates
  `belief→action→utility` value only. It does NOT estimate the causal value
  of any single passage: the frozen LLM row already incorporated its full
  context, so replaying the same belief minus-e would be invalid.
* **Document-VoI (Gate-2, NOT authorized):** `do(doc available/unavailable)`
  needs passage-ablation beliefs (systematic LOO/swap-in C1/C3 re-evaluation).
  Existing `intervention-*` arms are precedent that such beliefs are the right
  currency, not a substitute. Gate-2 opens only if Gate-1 passes AND LLM
  budget is authorized. No Gate-1 result may be worded as document-causal.

## 3. Heads (one scorer, two carriers; Head-B primary)

* **Head-B belief gate (PRIMARY, attacks the A2 bel→act bottleneck):**
  per-belief transmit/fall-back rule. Criticality:
  ```
  C(b) = E[J(π(b′)) − J(π(b)) | b′ = posterior, b = NoInfo prior]
  ```
  (locked per owner answer). Operationalized per §4: transmit posterior iff
  expected action-value effect is positive, else fall back to NoInfo (ΔJ = 0).
* **Head-A VoI reranker (secondary):** rescoring/filtering top-k sets by
  expected `ΔJ`. In Gate-1 it appears only as the *acquisition* stage the
  gate refines (S-J-selected system per stratum); no new ranking is learned.
* Pre-registered mechanistic read: reranker lifts nDCG but gate lifts J →
  bottleneck was transmission, not acquisition.

## 4. Gate-1 protocol (frozen)

Strata (5): `(C0,none)`, `(C1,8B)`, `(C1,14B)`, `(C3,8B)`, `(C3,14B)`.
Selectable systems: `{bm25,dense,hybrid,rerank}` (random/oracle excluded, as
in Stage-A). `k=3` headline only.

* **S-J acquisition (dev):** per stratum, `S-J` = argmax dev plain-mean ΔJ
  over selectable systems (`utility_by_query_devinclusive`, dev-40).
* **G1-arm (secondary):** transmit `S-J` iff its dev mean ΔJ > `tau_arm = 0`
  (fixed, no tuning); else all-NoInfo for the stratum.
* **G1-feat / Head-B (PRIMARY):** within the stratum's `S-J` system, transmit
  query `q` iff `Brier_q < t`, else NoInfo. Feature = per-belief **Brier**
  from `semantic_all` (dev slice for tuning; varies in all strata, unlike
  confidence which is constant 1.0 for C0/C1, and unlike `evidence_hit`
  which is degenerate 0 on dev). Threshold grid (frozen):
  `t ∈ {0.10, 0.20, 0.30, 0.40, 0.50}`. Pick per stratum the `t` maximizing
  dev mean gated ΔJ; ties → smallest `t`; if best dev gated ΔJ ≤ 0, gate OFF
  (all-NoInfo). Brier-gate is the Head-B carrier; arm-gate is reported for
  comparison (no rescue).
* **Discipline (owner addition, binding):** ALL model/threshold/scorer
  choices (S-J systems, per-stratum `t`, ON/OFF) are made on **dev-40** by
  `tools/d0_tune_gate.py`, which asserts it never loads test data. **Test-160
  is touched once only**, by `tools/d0_eval_gate.py` taking the frozen tune
  output as input. Any test reloading, re-tuning, or grid expansion after the
  single evaluation is a protocol violation.

## 5. Gate-1 verdict (frozen)

* Primary: pooled mean gated ΔJ (Head-B) vs NoInfo over the 5 strata
  (mean of stratum means on test-160). Uncertainty: paired test-query
  bootstrap, resample 160 query IDs w/replacement, same set across strata,
  `B=5000`, seed `71000`, plain-mean metric. **HOLDS iff two-sided 95% CI
  lower bound > 0** (strictly positive test ΔJ).
* Secondaries (no rescue): per-stratum gated ΔJ + CIs (Holm over 5),
  G1-arm pooled/per-stratum, headline balanced-J reference, dev gated ΔJ
  for transfer comparison.
* **Decision rule:** if primary HOLDS → Gate-2 (document-VoI) becomes
  scientifically justified and may be proposed with LLM budget. If not →
  STOP: no document-VoI, no Agentick; D0 collapses to a methods-note and V3
  stands unchanged.

## 6. Non-goals

No new LLM calls; no new sim episodes beyond reuse of frozen
`episodes.parquet`/`utility_by_query` (bootstrap is resampling, not simulation);
no frozen-matrix edits; no Agentick/ORAgentBench; no manuscript prose.
Test-160 single touch is the only test-data access in D0.
