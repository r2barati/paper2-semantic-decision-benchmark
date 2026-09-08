# Validity disclosures

Regenerated 7 September 2026. Each item states a threat and what the release
does about it. Items marked **open** are not resolved.

## Resolved by construction

| Threat | How it is closed |
|---|---|
| An uninformed controller sees hidden truth | `CausalOptimizer` requires an explicit `knows_true_disruption`; a behavioural test drives the planner over a fixed state trajectory under two hidden disruptions and asserts identical NoInfo orders |
| Planner and simulator disagree about arrivals | one shared `PipelineSchedule`; impulse tests for L = 1..5 plus a planner/simulator agreement test |
| Semantic sensors act before the warning exists | `warning_time` is honoured; a test asserts NoInfo and oracle orders are identical before period 12 |
| The oracle reference is not actually optimal | the MILP objective is checked against the simulated return (gap 0.0 over 30 seeds and all regimes) |
| A missing artifact silently becomes a 50/50 prior | cache is consulted before credentials; a miss raises `MissingFrozenArtifact` |
| An unknown sensor name silently becomes NoInfo | every dispatch raises `ValueError` |
| A stale checkpoint is used | `tools/verify_release.py` checks artifact completeness and replays stored beliefs |

## Disclosed limitations

| Limitation | Status |
|---|---|
| In-sample two-class beliefs from the calibrated interpreter are oracle-equivalent | **open by design.** This is why the confirmation experiments use held-out templates; a test records the fact rather than asserting it away |
| Isotonic calibration saturates on six-point calibration sets | **open.** Reported, with a sigmoid arm as the small-sample alternative |
| Three of six boundary variants have negative oracle information value | **open.** Marked; no normalised ratio is reported for them |
| Experiment R has one template family per regime | **open.** Pools are generated per seed, so that experiment carries no language-axis variance |
| Relevance is defined by construction, not by assessors | **open.** Stated in the limitations section |
| Family-level inference rests on three designed categories | **open.** Intervals are reported as conditional on the fixed template set |
| LLM provenance | **partially open.** Model labels are reconstructed from cache keys; 50 of 213 entries are labelled `unknown`. Complete provider responses, model snapshots and collection timestamps are not preserved |
| Dense retrieval is one general-purpose encoder | **open.** No claim is made about the ceiling of dense retrieval on this task |
| Evidence is consumed by concatenation | **open.** Belief-level pooling is not evaluated |
| Author declarations, conflicts, concurrent submission, AI use | **open.** Human facts; not certifiable from this repository |

## Superseded disclosures

The previous version of this file did not disclose the historical NoInfo
information leak, the optimiser's first-period sales bound, the arrival
off-by-one, the unused `warning_time` parameter, or the absence of text-free
controls, because none of them had been found. They are listed in
`AUDIT_RESPONSE_2026.md` with the repair and its check.

---

<!-- previous content retained below -->

# Validity disclosures

| Issue | Status | Manuscript treatment |
|---|---|---|
| Phase 8A linguistic leakage/diagnostic concerns | RESOLVED by scope | omit from primary evidence; describe as exploratory history only if needed |
| Phase 8B held-out template leakage | RESOLVED | present as primary held-out result |
| Phase 9A TF-IDF train-template inclusion | DISCLOSE | operational cross-event transfer, not clean linguistic OOD |
| Phase 9B negative SIVR variants | DISCLOSE | controller/environment mismatch and negative oracle reference |
| NoInfo prior mismatch | RESOLVED/DISCLOSE | report corrected balanced/prior estimands and code-executed priors |
| Brier implementation correction | RESOLVED | use corrected conventional multiclass metric |
| GPT-4o Phase 9A fallback bug | RESOLVED/DISCLOSE | post-fix audit says published value unchanged; do not hide history |
| fixed controller | DISCLOSE | value is controller-dependent |
| synthetic text/inventory | DISCLOSE | no production or universal claim |
| any fatal validity defect | NONE FOUND | no experiment reopening |
