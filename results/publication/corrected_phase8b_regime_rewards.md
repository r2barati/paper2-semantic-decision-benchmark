# Phase 8B regime-specific reward effects

Rewards are means over template families, variants, and paired seeds within regime.

| Sensor | Regime | NoInfo | Sensor | Oracle-Belief Reference | Δ vs NoInfo | Signed OIV | SIVR status |
|---|---|---:|---:|---:|---:|---:|---|
| Constant_p0.0 | demand_surge | 449.8 | 322.8 | 664.3 | -127.0 | 214.6 | VALID |
| Constant_p0.5 | demand_surge | 449.8 | 522.5 | 664.3 | 72.7 | 214.6 | VALID |
| Constant_p1.0 | demand_surge | 449.8 | 664.3 | 664.3 | 214.6 | 214.6 | VALID |
| NoInfo | demand_surge | 449.8 | 449.8 | 664.3 | 0.0 | 214.6 | VALID |
| NoText_Tuned | demand_surge | 449.8 | 664.3 | 664.3 | 214.6 | 214.6 | VALID |
| OracleSemantic | demand_surge | 449.8 | 664.3 | 664.3 | 214.6 | 214.6 | VALID |
| RuleBased | demand_surge | 449.8 | 664.3 | 664.3 | 214.6 | 214.6 | VALID |
| TFIDF_LogReg_Calibrated | demand_surge | 449.8 | 609.1 | 664.3 | 159.4 | 214.6 | VALID |
| TFIDF_LogReg_Calibrated_Argmax | demand_surge | 449.8 | 607.4 | 664.3 | 157.7 | 214.6 | VALID |
| TFIDF_LogReg_Calibrated_Shuffled | demand_surge | 449.8 | 445.7 | 664.3 | -4.1 | 214.6 | VALID |
| TFIDF_LogReg_Raw | demand_surge | 449.8 | 550.2 | 664.3 | 100.5 | 214.6 | VALID |
| gpt-4o | demand_surge | 449.8 | 629.1 | 664.3 | 179.3 | 214.6 | VALID |
| Constant_p0.0 | normal | 485.7 | 439.6 | 439.6 | -46.1 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| Constant_p0.5 | normal | 485.7 | 479.7 | 439.6 | -6.0 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| Constant_p1.0 | normal | 485.7 | 456.5 | 439.6 | -29.2 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| NoInfo | normal | 485.7 | 485.7 | 439.6 | 0.0 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| NoText_Tuned | normal | 485.7 | 456.5 | 439.6 | -29.2 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| OracleSemantic | normal | 485.7 | 439.6 | 439.6 | -46.1 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| RuleBased | normal | 485.7 | 473.1 | 439.6 | -12.6 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Calibrated | normal | 485.7 | 451.3 | 439.6 | -34.4 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Calibrated_Argmax | normal | 485.7 | 439.6 | 439.6 | -46.1 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Calibrated_Shuffled | normal | 485.7 | 456.5 | 439.6 | -29.2 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Raw | normal | 485.7 | 482.2 | 439.6 | -3.5 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| gpt-4o | normal | 485.7 | 440.6 | 439.6 | -45.1 | -46.1 | NEGATIVE_ORACLE_REFERENCE_VALUE |
