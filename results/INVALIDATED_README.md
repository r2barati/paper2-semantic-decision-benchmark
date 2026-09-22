# INVALIDATED results — do not cite

- v1.0 tables with information-boundary leak (`CausalOptimizer` saw true
  disruption), optimizer LP that could not sell stock, arrival timing `t+L+1`,
  ignored `warning_time` (8 periods unearned notice), no text-free controls.
  Source: `REPRODUCIBILITY.md:10-15`, `src/experiment_phase5.py:114-123`,
  `paper2_submission/SUBMISSION_READINESS.md:22-66`.
- Absolute-denominator SIVRs and Holm p-values reported as exactly `0.0`
  (`results/publication/SCIENTIFIC_LEDGER_CORRECTED.md:19-22`,
  `results/final_audit/STATISTICAL_AUDIT.md` B=10000 formula). Correct rule:
  `SCIENTIFIC_LEDGER.md:8-9` B=5000 → min p `1/(B+1)=0.000200`, never zero;
  denominator signed with `ZERO_OR_NEAR_ZERO` / `NEGATIVE_ORACLE` guards
  (`docs/METRICS.md`, `src/metrics.py:signed_sivr`).
- S test-only violations: all-template S Rule (`1447.02 SIVR 1.006`) includes
  9 train templates — cannot support linguistic generalisation
  (`results/correction_audit/PHASE9_REINTERPRETATION.md:5-9`). Use test-only
  subset as exploratory only.
- Phase 9B ratios without dagger: `long_lead`/`lost_sales`/`short_lead` have
  negative OIV (`boundary.tex` †, `corrected_phase9b_boundary.md`
  `NEGATIVE_ORACLE_REFERENCE_VALUE`) — ratios uninterpretable; cite as
  controller/environment mismatch with dagger + signed guard.
- `results/v3main/sim_controllerB/utilityB.json` means: `delivery/LEDGER.md:442-448`
  duplication (4296/7200 key-groups differ ≤621, mixed reruns). Numbers unusable
  until rebuilt; qualitative pairing-dependence is hypothesis only.
- G2FULL/D0 `p_raw/p_holm/p_perm = 0.0` entries (`g2full_stage4.json:51-52`,
  `d0_verdict.json:26,40,54,68`) violate the B5000 floor; CIs remain the valid
  object.
- `FINAL_TEST_AUDIT.md` 268/268 (2026-08-23) and `CLEAN_REPRODUCTION_AUDIT.md`
  (commit `856a176`) are superseded history (`SUBMISSION_READINESS.md:102-107`);
  current count is `pytest` output, clean-clone at HEAD not demonstrated
  (dirty tree + untracked `runs/`, `artifacts/`, `v1/`, `v2/`, `legacy/`).
