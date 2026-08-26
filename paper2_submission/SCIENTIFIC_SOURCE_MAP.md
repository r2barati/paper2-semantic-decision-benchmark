# Scientific source map

All claims below are limited to frozen artifacts at commit
`0be5181926d6ab1def43ef499f8ed57ee7e31b3c` unless explicitly marked
historical. Numerical claims are not regenerated in this production pass.

| Claim | Authoritative artifact | File/row | Statistic | Allowed wording |
|---|---|---|---|---|
| calibrated beliefs can have greater operational value despite lower accuracy | corrected statistical audit | `results/final_audit/STATISTICAL_AUDIT.md`, section 4 | raw accuracy .889 vs calibrated .861; Brier .552 vs .240; log-loss .932 vs .397; Phase-7 SIVR .250 vs .648 | “In this benchmark, calibration improved downstream value despite a small accuracy decrease.” |
| semantic sensing creates value under held-out operational-complexity transfer | frozen ledger/results | `SCIENTIFIC_LEDGER.md`; `results/correction_audit/raw_reward_effects.csv`; Phase-8B rows | RuleBased +101.0 [96.7,104.2], raw TF-IDF +48.5 [36.1,58.2], calibrated TF-IDF +70.8 [29.9,96.1], GPT-4o +67.1 [33.9,84.5] vs NoInfo | “Positive paired reward effects were observed under the fixed controller.” |
| Phase 8B is the primary linguistic generalization result | final audit | `results/final_audit/FINAL_PAPER_READINESS_AUDIT.md`, leakage and phase sections | 30 seeds, 24 held-out templates, 4,320 episodes | “The primary held-out result is Phase 8B.” |
| supply-side transfer is supported, with linguistic caveat | Phase 9 report and correction audit | `results/PHASE9_REPORT.md`; `results/correction_audit/PHASE9_REINTERPRETATION.md` | 30 seeds, 16 templates, capacity-drop shock; RuleBased SIVR .823 | “The framework transfers operationally to a supply-side shock; TF-IDF linguistic transfer is not claimed from the all-template analysis.” |
| robustness boundary is controller/environment mismatch | Phase 9B audit | `results/correction_audit/phase9b_boundary_analysis.csv`; `PHASE9_REINTERPRETATION.md` | long-lead and lost-sales Oracle OIV negative | “Negative SIVR is a boundary diagnostic, not evidence of semantic failure.” |
| stronger control can suppress semantic value | historical Phase 4 report | `results/PHASE4_FINAL_REPORT.md` | heuristic SIVR range .709–1.049; CausalOptimizer SIVR 0 for all sensors | “A historical controlled diagnostic suggests controller capability mediates realized semantic value.” |
| all headline numbers are reproducible offline | final test audit | `FINAL_TEST_AUDIT.md` | 268/268 tests pass; no API-dependent tests | “The frozen package replays offline without live API access.” |

## Frozen-source policy

Phase 8A remains exploratory because of the documented linguistic leakage,
fill-rate audit, and TF-IDF saturation concerns. Phase 8B is primary. Phase 9A
is cross-event operational transfer with a training-template caveat. Phase 9B
is descriptive robustness/boundary analysis. No new numbers are introduced.
