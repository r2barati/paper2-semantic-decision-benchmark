# G2-FULL — FROZEN DESIGN (2026-09-20, pre-execution)

Question: is document-level operational value consumer-dependent, and can a
deployable (label-free) proxy predict/gate it to improve held-out J?
Positioning (binding): "document-level operational value is
consumer-dependent; the tested consumers exhibit qualitatively different
value regimes." Architecture-isolation language ("determines") is NOT used.

## 1. Scope (frozen)

Test-160 × rerank k=3 × C1 × {8B,14B} × LOO{drop0,drop1,drop2} = 960 new C1
calls. No C3 calls, no k=5, no system expansion, no swap-ins. C0/C3 test LOO
VoI is REUSED frozen from 2A-0 (`g2a_verdict.json`), never recomputed.

## 2. Cluster unit (frozen; the 90-not-120 safeguard)

Cluster key = `sha256(query text + full doc IDs)`. All inference resamples
**full-set clusters** with replacement (each draw contributes member queries
× positions; members share C1 contexts but differ in regime/seeds). Dev: 30
clusters. Test: count derived from test inputs at build (outcomes untouched).
Bootstrap `B=5000` throughout; dev seed `74000`, test seed `75000`,
permutation seed `76000`.

## 3. Proxy discipline (frozen)

`Ĉ(b)` uses ONLY observables available at inference (entropy, margin,
maxProb, prior-movement L1, 8B-vs-14B disagreement, C1-vs-C3 disagreement,
rank-gap margin, controller finite-difference sensitivity at initial state).
Label-dependent inputs (nDCG, Brier, accuracy, evidence_hit, true_regime)
are FORBIDDEN as proxy inputs. Rule form: screen candidates on dev by
cluster-bootstrapped correlation with dev VoI → freeze ONE feature + ONE
shared threshold (drop doc iff predicted harmful) maximizing dev gated ΔJ.
Frozen to `g2full_proxy.json` + committed BEFORE test kernels are pushed;
the 960 calls cannot inform it. A failed deployable proxy is recorded as the
oracle-to-deployable gap, never re-tuned.

## 4. Claim tests (frozen; single Stage-4 test touch for all three)

* **(a) Removal causally changes J:** per-position pooled C1 VoI
  (`J(full)−J(minus-i)`, cluster bootstrap), Holm over 6 cells (3 pos × 2
  models). Success: ≥1 CI excludes 0.
* **(b) Consumer-dependent:** permutation test shuffling consumer labels
  within query over {C0, C1-8B, C1-14B, C3} per-position VoI (C0/C3 reused
  frozen); test statistic = max−min consumer pooled VoI; exact p from 5000
  perms. Success: p<0.05 in ≥1 position. Regime table (CIs side-by-side) is
  secondary description.
* **(c) Deployable proxy predicts + improves (PRIMARY):** frozen-proxy-gated
  pooled ΔJ vs NoInfo with CI excluding 0 (cluster bootstrap), AND the gate
  outperforms the best ungated upstream-selection baseline available at test.
  Baseline set (frozen, from A1 selections): per stratum, max test J among
  {S-nDCG-selected arm, S-Brier-selected arm, S-J-selected arm,
  always-transmit S-J} — reported individually, gate must beat the max. This
  blocks the "weak-transmission-policy" review. Per-stratum Holm secondaries.

## 5. Gates before test spend

G1 full-set reproduction (new test contexts vs frozen test beliefs, 1e-9);
G2 cache completeness (misses excluded, never new calls); G3-analog replay
profit match vs frozen test episodes (1e-9). Any gate fails → STOP, no
Stage-4 evaluation.

## 6. Decision rule

(c) holds → Agentick may be proposed. (a) holds but (c) fails → the
oracle-to-deployable gap is the finding; no Agentick. (a) fails → stop;
consumer-architecture story stands on C0/C3/C1-dev. All outcomes committed
including fail.

## 7. Non-goals

No C3 calls, no k=5, no system expansion, no 2D work, no manuscript prose.
Test outcomes are read ONCE, in Stage 4, after proxy + baselines are frozen.
