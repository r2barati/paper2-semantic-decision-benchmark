# Claim-source matrix

| Manuscript claim | Source | Location | Status |
|---|---|---|---|
| raw TF-IDF has higher accuracy but calibrated TF-IDF has better calibration and SIVR | `results/final_audit/STATISTICAL_AUDIT.md` | sections 4 and 7 | supported |
| semantic sensors have positive Phase-8B reward deltas | `results/correction_audit/raw_reward_effects.csv` | Phase-8B rows | supported |
| Phase 8B is held-out | `results/final_audit/FINAL_PAPER_READINESS_AUDIT.md` | leakage/phase sections | supported |
| supply-side transfer exists | `results/PHASE9_REPORT.md` | Phase 9A | supported with template caveat |
| negative Phase-9B SIVR indicates controller mismatch | `results/correction_audit/PHASE9_REINTERPRETATION.md` | boundary section | supported |
| stronger controller mediates value | `results/PHASE4_FINAL_REPORT.md` | historical diagnostic | bounded/supporting |
| all 268 tests pass | `FINAL_TEST_AUDIT.md` | summary | supported |

Unsupported numerical claims: **0 in the planned manuscript**. Unsupported
novelty claims: **0**, provided the paper uses the bounded language in
`NOVELTY_LEDGER.md`.
