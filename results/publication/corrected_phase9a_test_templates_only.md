# Phase 9A test-template-only analysis

This is the linguistic-generalization subset computed from already-saved episodes.

| Sensor | Regime | Reward | Δ vs NoInfo | Signed SIVR | Status |
|---|---|---:|---:|---:|---|
| Constant_p0.0 | ALL | 1166.7 | -166.9 | -1.481 | VALID |
| Constant_p0.5 | ALL | 1333.6 | 0.0 | 0.000 | VALID |
| Constant_p1.0 | ALL | 1725.1 | 391.5 | 3.473 | VALID |
| NoInfo | ALL | 1333.6 | 0.0 | 0.000 | VALID |
| NoText_Tuned | ALL | 1725.1 | 391.5 | 3.473 | VALID |
| OracleSemantic | ALL | 1446.3 | 112.7 | 1.000 | VALID |
| RuleBased | ALL | 1387.2 | 53.6 | 0.476 | VALID |
| TFIDF_LogReg_Calibrated | ALL | 1336.6 | 3.0 | 0.027 | VALID |
| TFIDF_LogReg_Calibrated_Argmax | ALL | 1322.0 | -11.6 | -0.103 | VALID |
| TFIDF_LogReg_Calibrated_Shuffled | ALL | 1374.1 | 40.4 | 0.359 | VALID |
| TFIDF_LogReg_Raw | ALL | 1393.7 | 60.1 | 0.533 | VALID |
| gpt-4o | ALL | 1364.6 | 31.0 | 0.275 | VALID |
| Constant_p0.0 | normal | 1166.7 | -166.9 | 1.000 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| Constant_p0.5 | normal | 1333.6 | 0.0 | -0.000 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| Constant_p1.0 | normal | 1724.3 | 390.7 | -2.340 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| NoInfo | normal | 1333.6 | 0.0 | -0.000 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| NoText_Tuned | normal | 1724.3 | 390.7 | -2.340 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| OracleSemantic | normal | 1166.7 | -166.9 | 1.000 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| RuleBased | normal | 1248.7 | -84.9 | 0.509 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Calibrated | normal | 1235.2 | -98.4 | 0.589 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Calibrated_Argmax | normal | 1166.7 | -166.9 | 1.000 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Calibrated_Shuffled | normal | 1488.4 | 154.8 | -0.927 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Raw | normal | 1378.1 | 44.5 | -0.267 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| gpt-4o | normal | 1181.6 | -152.0 | 0.911 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| Constant_p0.0 | supplier_capacity_drop | 1166.7 | -166.9 | -0.425 | VALID |
| Constant_p0.5 | supplier_capacity_drop | 1333.6 | 0.0 | 0.000 | VALID |
| Constant_p1.0 | supplier_capacity_drop | 1726.0 | 392.4 | 1.000 | VALID |
| NoInfo | supplier_capacity_drop | 1333.6 | 0.0 | 0.000 | VALID |
| NoText_Tuned | supplier_capacity_drop | 1726.0 | 392.4 | 1.000 | VALID |
| OracleSemantic | supplier_capacity_drop | 1726.0 | 392.4 | 1.000 | VALID |
| RuleBased | supplier_capacity_drop | 1525.8 | 192.2 | 0.490 | VALID |
| TFIDF_LogReg_Calibrated | supplier_capacity_drop | 1438.0 | 104.4 | 0.266 | VALID |
| TFIDF_LogReg_Calibrated_Argmax | supplier_capacity_drop | 1477.4 | 143.8 | 0.366 | VALID |
| TFIDF_LogReg_Calibrated_Shuffled | supplier_capacity_drop | 1259.7 | -73.9 | -0.188 | VALID |
| TFIDF_LogReg_Raw | supplier_capacity_drop | 1409.2 | 75.6 | 0.193 | VALID |
| gpt-4o | supplier_capacity_drop | 1547.6 | 214.0 | 0.545 | VALID |
