# Experiment B: Paper-1 Gymnasium Confirmation (Phase 8B)

## Purpose

Test whether the semantic-value phenomenon transfers to a richer multi-echelon supply chain environment with different dynamics, topology, and controller architecture.

## Environment

Paper-1 `gym-invmgmt` v0.1.0 (commit `a745fd5`), Serial-v0 scenario. Multi-echelon supply chain: RM(4) → Factory(3, C=100) → Dist(2) → Retail(1) → Market(0). Lead times L=[0,4,4]. DemandSurge: base_mu=10.0, shock_mag=2.0, shock_time=15, 30-period horizon.

## Regime Space

2-class: Normal (no disruption) and DemandSurge (demand doubling at t=15).
NoInfo uses P(Normal)=0.70, P(DemandSurge)=0.30, but the 12/12 evaluation
template set makes the primary benchmark estimand balanced 50/50. Both are
reported.

## Controller

BeliefAdaptiveController: base-stock order-up-to policy where effective_mu = P(Normal) × base_mu + P(DemandSurge) × base_mu × surge_multiplier. Fixed across all conditions.

## Held-Out Templates

24 templates (12 surge + 12 normal) from CONFIRMATION_TEMPLATES, genuinely different wording from the 18 training templates. All 3 ambiguity levels represented.

## Design Controls

- **Disjoint seeds:** 3100–3129 (not used in any prior experiment)
- **Held-out language:** Templates never seen during TF-IDF training
- **Frozen config:** No retuning based on Phase-8B results
- **Corrected fill rate:** Uses retail sales (env.S), not replenishment orders (env.R)
- **Hierarchical bootstrap:** Regime-stratified family × variant × paired-seed resampling
- **No leakage:** Controller reads only current-period inventory state

## Results (Headline; corrected)

| Sensor | AggSIVR | Mean Reward | Fill Rate | Belief Acc | Hier. 95% CI |
|--------|--------:|------------:|----------:|-----------:|:-------------|
| NoInfo | 0.000 | 467.7 | 0.858 | 0.500 | — |
| RuleBased | 1.198 | 568.7 | 0.947 | 0.792 | [96.7, 104.2] sig (Δ reward) |
| TFIDF_Raw | 0.575 | 516.2 | 0.901 | 0.917 | [36.1, 58.2] sig (Δ reward) |
| TFIDF_Calibrated | 0.841 | 538.6 | 0.911 | 0.958 | [29.9, 96.1] sig (Δ reward) |
| gpt-4o | 0.797 | 534.9 | 0.906 | 0.958 | [33.9, 84.5] sig (Δ reward) |
| OracleSemantic | 1.000 | 552.0 | 0.920 | 1.000 | — |

The table is the balanced benchmark estimate. The separate deployment-prior
table applies 70/30 weights. Raw reward deltas are primary; SIVR is secondary.
`OracleSemantic` is perfect semantic belief through the same fixed controller,
not an optimal policy or reward upper bound.

## Interpretation

The phenomenon survives operational-complexity transfer:
1. Semantic information has measurable operational value in a multi-echelon supply chain
2. Calibration matters: TFIDF_Calibrated (0.841) outperforms TFIDF_Raw (0.575)
3. Classification accuracy does not determine operational value ranking
4. gpt-4o and TFIDF_Calibrated have similar belief accuracy (0.958) but different SIVR (0.797 vs 0.841), suggesting different failure modes interact differently with the controller
