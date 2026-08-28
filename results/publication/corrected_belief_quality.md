# Corrected belief-quality table

Brier is conventional multiclass sum_k (p_k-y_k)^2.

| Phase | Sensor | Accuracy | Brier | Log-loss |
|---|---|---:|---:|---:|
| Phase 5.5 | LLM:gpt-3.5-turbo | 0.389 | 0.569 | 0.785 |
| Phase 5.5 | LLM:gpt-4o | 0.944 | 0.088 | 0.186 |
| Phase 5.5 | LLM:gpt-4o-mini | 0.722 | 0.315 | 0.522 |
| Phase 5.5 | NoInfo | 0.333 | 0.668 | 1.101 |
| Phase 5.5 | PerfectSemantic | 1.000 | 0.000 | 0.000 |
| Phase 5.5 | RuleBased | 1.000 | 0.241 | 0.418 |
| Phase 6 | LLM:gpt-3.5-turbo | 0.361 | 0.720 | 1.005 |
| Phase 6 | LLM:gpt-4o | 0.917 | 0.116 | 0.196 |
| Phase 6 | LLM:gpt-4o-mini | 0.722 | 0.321 | 0.534 |
| Phase 6 | NoInfo | 0.333 | 0.668 | 1.101 |
| Phase 6 | PerfectSemantic | 1.000 | 0.000 | 0.000 |
| Phase 6 | RuleBased | 0.833 | 0.317 | 1.407 |
| Phase 7 | NoInfo | 0.333 | 0.668 | 1.101 |
| Phase 7 | PerfectSemantic | 1.000 | 0.000 | 0.000 |
| Phase 7 | RuleBased | 0.833 | 0.317 | 1.407 |
| Phase 7 | TFIDF_LogReg | 0.889 | 0.552 | 0.932 |
| Phase 7 | TFIDF_LogReg_Calibrated | 0.861 | 0.240 | 0.397 |
| Phase 8A | NoInfo | 0.500 | 0.580 | 0.780 |
| Phase 8A | PerfectSemantic | 1.000 | 0.000 | 0.000 |
| Phase 8A | RuleBased | 1.000 | 0.178 | 0.289 |
| Phase 8A | TFIDF_LogReg | 1.000 | 0.000 | 0.000 |
| Phase 8A | gpt-4o | 1.000 | 0.032 | 0.087 |
| Phase 8B | NoInfo | 0.500 | 0.580 | 0.780 |
| Phase 8B | OracleSemantic | 1.000 | 0.000 | 0.000 |
| Phase 8B | RuleBased | 0.792 | 0.323 | 1.762 |
| Phase 8B | TFIDF_LogReg_Calibrated | 0.958 | 0.093 | 0.148 |
| Phase 8B | TFIDF_LogReg_Raw | 0.917 | 0.362 | 0.553 |
| Phase 8B | gpt-4o | 0.958 | 0.066 | 0.112 |
| Phase 9A | NoInfo | 0.500 | 0.500 | 0.693 |
| Phase 9A | OracleSemantic | 1.000 | 0.000 | 0.000 |
| Phase 9A | RuleBased | 0.917 | 0.243 | 0.372 |
| Phase 9A | TFIDF_LogReg_Calibrated | 0.833 | 0.160 | 0.226 |
| Phase 9A | TFIDF_LogReg_Raw | 0.833 | 0.377 | 0.568 |
| Phase 9A | gpt-4o | 0.944 | 0.088 | 0.174 |
