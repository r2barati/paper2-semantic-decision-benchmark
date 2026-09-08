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
| Phase 7 | NoText_Tuned | 0.333 | 1.333 | 23.026 |
| Phase 7 | NoText_Uniform | 0.333 | 0.667 | 1.099 |
| Phase 7 | PerfectSemantic | 1.000 | 0.000 | 0.000 |
| Phase 7 | RuleBased | 0.833 | 0.319 | 1.431 |
| Phase 7 | TFIDF_LogReg | 0.889 | 0.552 | 0.932 |
| Phase 7 | TFIDF_LogReg_Argmax | 0.889 | 0.222 | 3.838 |
| Phase 7 | TFIDF_LogReg_Calibrated | 0.806 | 0.319 | 0.542 |
| Phase 7 | TFIDF_LogReg_Calibrated_Argmax | 0.806 | 0.389 | 6.716 |
| Phase 7 | TFIDF_LogReg_Calibrated_ShuffledText | 0.417 | 0.849 | 4.995 |
| Phase 7 | TFIDF_LogReg_Calibrated_Sigmoid | 0.861 | 0.392 | 0.699 |
| Phase 7 | TFIDF_LogReg_FoldEnsemble | 0.889 | 0.583 | 0.977 |
| Phase 8A | NoInfo | 0.500 | 0.580 | 0.780 |
| Phase 8A | PerfectSemantic | 1.000 | 0.000 | 0.000 |
| Phase 8A | RuleBased | 1.000 | 0.178 | 0.289 |
| Phase 8A | TFIDF_LogReg | 1.000 | 0.000 | 0.000 |
| Phase 8A | gpt-4o | 1.000 | 0.032 | 0.087 |
| Phase 8B | Constant_p0.0 | 0.500 | 1.000 | 17.269 |
| Phase 8B | Constant_p0.5 | 0.500 | 0.500 | 0.693 |
| Phase 8B | Constant_p1.0 | 0.500 | 1.000 | 17.269 |
| Phase 8B | NoInfo | 0.500 | 0.580 | 0.780 |
| Phase 8B | NoText_Tuned | 0.500 | 1.000 | 17.269 |
| Phase 8B | OracleSemantic | 1.000 | 0.000 | 0.000 |
| Phase 8B | RuleBased | 0.792 | 0.323 | 1.762 |
| Phase 8B | TFIDF_LogReg_Calibrated | 0.917 | 0.138 | 0.199 |
| Phase 8B | TFIDF_LogReg_Calibrated_Argmax | 0.917 | 0.167 | 2.878 |
| Phase 8B | TFIDF_LogReg_Calibrated_Shuffled | 0.417 | 1.089 | 12.228 |
| Phase 8B | TFIDF_LogReg_Raw | 0.917 | 0.362 | 0.553 |
| Phase 8B | gpt-4o | 0.958 | 0.066 | 0.112 |
| Phase 9A | Constant_p0.0 | 0.500 | 1.000 | 17.269 |
| Phase 9A | Constant_p0.5 | 0.500 | 0.500 | 0.693 |
| Phase 9A | Constant_p1.0 | 0.500 | 1.000 | 17.269 |
| Phase 9A | NoInfo | 0.500 | 0.500 | 0.693 |
| Phase 9A | NoText_Tuned | 0.500 | 1.000 | 17.269 |
| Phase 9A | OracleSemantic | 1.000 | 0.000 | 0.000 |
| Phase 9A | RuleBased | 0.917 | 0.243 | 0.372 |
| Phase 9A | TFIDF_LogReg_Calibrated | 0.903 | 0.157 | 0.228 |
| Phase 9A | TFIDF_LogReg_Calibrated_Argmax | 0.903 | 0.194 | 3.358 |
| Phase 9A | TFIDF_LogReg_Calibrated_Shuffled | 0.500 | 0.748 | 9.409 |
| Phase 9A | TFIDF_LogReg_Raw | 0.833 | 0.377 | 0.568 |
| Phase 9A | gpt-4o | 0.944 | 0.088 | 0.174 |
