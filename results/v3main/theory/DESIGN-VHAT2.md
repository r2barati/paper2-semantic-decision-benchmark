# DESIGN-VHAT2 — FROZEN (2026-09-20; "cluster-conformal Vhat2 repair")

Track B = ONE diagnosis-driven method attempt on historical labels only.
No fresh LLM calls, no fresh bank unless B2 passes. "Mondrian" appears here
only as considered-and-rejected (n=13 granularity: 7.7%/rank, SE ~=8.3pp,
5x estimation risk, same clusters span all strata).

## B1. Model (frozen form; each piece cites STEP0_AUTOPSY)

Vhat2(e,b,c,pos) = dBhat(e,c,pos) x Shat(b,c | dBhat). Pooled GBM trunk
(same frozen CFG; NO independent per-stratum fits — 480-row overfit risk),
with MANDATORY consumer one-hot (5) + position one-hot (3) in BOTH heads
(F5: pooled-position eta2~=0 with flipping stratum peaks; one-hot alone
only shifts intercepts). Shat CONDITIONS on dBhat output (not post-hoc
product): sensitivity matters only when belief shifts (F5: L1->|VoI|
0.77 overall; signed mapping flips by stratum). C3 per-doc weights stay;
zero-inflation acknowledged (F2: 35% exact-0; L1==0 rows predicted mean
3.04 bias noted, not patched post-hoc).
Feature budget = DESIGN-VHAT §10 unchanged (own-belief + embeddings/ranks +
controller-FD; cross-model/consumer + labels forbidden).

## B2. Corrected cluster conformal chain (frozen; THE validity object)

Exchangeable cluster -> S_c = max_{i in c}(VHAT2_i - V_i) [OVERPREDICTION
residual: LCB fails exactly when Vhat-V exceeds Q] -> Q_{1-alpha}^{conf}
(finite-sample ceil((n+1)(1-alpha))/n quantile, alpha=0.1 frozen) ->
LCB_i = Vhat2_i - Q, giving simultaneous one-sided coverage within a new
cluster, marginal over clusters. Cluster-disjoint calibration kept
(frozen 20% split); exchangeability unit never row-level. Stratum
residuals/quantiles are diagnostics ONLY, never thresholds.

## B3. B2 development/feasibility gate (frozen; NOT confirmatory)

(i) pooled OOF Spearman CI excludes 0 AND >=4/5 strata point-positive;
(ii) coverage >= 0.85 (nominal 0.90; NO upper band — conservative coverage
is not failure) on calibration clusters;
(iii) efficiency + utility: LCB selectivity (drop fraction) in
[0.05, 0.95] AND pooled gated-vs-always-transmit CI excludes 0 positive
(strata/positions secondaries reported).
B2-pass authorizes PROPOSING (not spending) the untouched-bank
confirmation; any positive B2 MUST be described as development-only
evidence for Vhat2. B2-fail -> STOP, gap stands (likely strata: coverage
under heteroscedasticity even with max-scores, or selectivity collapse).

## Non-goals / firewall

No fresh data in Track B (asserted absent, as before). No architecture
changes after seeing B2 numbers. Track B results may not alter Track A
masks/hypotheses. V3 + all prior verdicts stand untouched. Only an
untouched bank + held-out win can ever make Vhat2 confirmatory.
