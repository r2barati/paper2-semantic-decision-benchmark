# Scientific Ledger (corrected, generated)

Every value below is read from `results/correction_audit`; no experimental value is hardcoded.

## Phase 7 belief quality

| Sensor | Accuracy | Standard Brier | Log-loss |
|---|---:|---:|---:|
| NoInfo | 0.333333 | 0.668333 | 1.101206 |
| PerfectSemantic | 1.000000 | 0.000000 | 0.000000 |
| RuleBased | 0.833333 | 0.317269 | 1.406550 |
| TFIDF_LogReg | 0.888889 | 0.552044 | 0.932421 |
| TFIDF_LogReg_Calibrated | 0.861111 | 0.239561 | 0.397220 |

## Primary paired reward effects

| Phase | Sensor | Δ vs NoInfo | 95% CI | Holm p |
|---|---|---:|---|---:|
| Phase 8B | RuleBased | 100.974847 | [96.725374, 104.155100] | 0.0 |
| Phase 8B | TFIDF_LogReg_Calibrated | 70.845353 | [29.943359, 96.144889] | 0.0 |
| Phase 8B | TFIDF_LogReg_Raw | 48.483723 | [36.063617, 58.238798] | 0.0 |
| Phase 8B | gpt-4o | 67.120741 | [33.874778, 84.512063] | 0.0 |
| Phase 9A | RuleBased | 113.406893 | [47.314993, 178.263857] | 0.0 |
| Phase 9A | TFIDF_LogReg_Calibrated | 26.534482 | [-122.371759, 119.826764] | 0.592 |
| Phase 9A | TFIDF_LogReg_Raw | 29.218210 | [-1.368139, 61.601957] | 0.23399999999999999 |
| Phase 9A | gpt-4o | 38.941596 | [-34.071459, 84.573951] | 0.592 |
