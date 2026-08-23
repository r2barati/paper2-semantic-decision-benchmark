# Corrected Results Summary

The corrected analysis treats balanced performance over the deliberately
constructed evaluation set as the primary benchmark estimand. Expected value
under the declared deployment prior is secondary. All reward comparisons are
paired and use the regime-stratified family/variant/seed bootstrap; simultaneous
sensor comparisons in Phase 8B and Phase 9A use Holm adjustment. Phase 9B is
descriptive robustness/boundary evidence.

## Defensible findings

- In Phase 7, calibrated TF-IDF has lower conventional multiclass Brier score
  than raw TF-IDF (0.240 vs 0.552) and lower log-loss (0.397 vs 0.932), while
  raw TF-IDF has higher accuracy (0.889 vs 0.861). This calibration result
  remains supported.
- In balanced Phase 8B, all four non-reference sensors have positive raw reward
  deltas versus NoInfo: approximately 101.0 (RuleBased), 48.5 (raw TF-IDF),
  70.8 (calibrated TF-IDF), and 67.1 (gpt-4o). The held-out
  linguistic/operational-complexity result remains, but it is a realized effect
  under the fixed controller.
- The corrected Phase 8B SIVRs are 1.198, 0.575, 0.841, and 0.797 for
  RuleBased, raw TF-IDF, calibrated TF-IDF, and gpt-4o, respectively, under
  the balanced estimand. These are secondary normalizations; RuleBased
  exceeding one is permitted because OracleSemantic is not an upper bound.
- Phase 9A establishes cross-event operational transfer. It does not by itself
  establish clean linguistic generalization because the original evaluation
  includes TF-IDF training templates. The test-template-only analysis is
  reported separately when supported by saved episodes.
- Phase 9B establishes robustness and boundary behavior of the fixed
  belief-to-control mapping. `long_lead` and `lost_sales` are not automatically
  semantic-sensing failures: the audit separately reports oracle-vs-NoInfo
  reward, sensor deltas, and signed OIV status.

## Interpretation boundary

The evidence supports the narrower claim that semantic classification accuracy
alone does not determine operational decision value. Realized value depends
jointly on belief quality, probability calibration, the downstream
belief-to-control mapping, and operational system dynamics. It does not
establish LLM superiority, universal semantic information value,
controller-independent value, industrial deployment effectiveness, arbitrary
event/topology generalization, or universal SIVR robustness.

Corrected tables are generated from frozen episode CSVs by
`results/publication/generate_corrected_tables.py`; historical Phase 7/8/9
reports remain unchanged for provenance.
