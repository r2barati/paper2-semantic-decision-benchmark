# Phase 9A test-template-only analysis

This is the linguistic-generalization subset computed from already-saved episodes.

| Sensor | Regime | Reward | Δ vs NoInfo | Signed SIVR | Status |
|---|---|---:|---:|---:|---|
| NoInfo | ALL | 1333.6 | 0.0 | 0.000 | VALID |
| OracleSemantic | ALL | 1446.3 | 112.7 | 1.000 | VALID |
| RuleBased | ALL | 1387.2 | 53.6 | 0.476 | VALID |
| TFIDF_LogReg_Calibrated | ALL | 1340.6 | 7.0 | 0.062 | VALID |
| TFIDF_LogReg_Raw | ALL | 1393.7 | 60.1 | 0.533 | VALID |
| gpt-4o | ALL | 1364.6 | 31.0 | 0.275 | VALID |
| NoInfo | normal | 1333.6 | 0.0 | -0.000 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| OracleSemantic | normal | 1166.7 | -166.9 | 1.000 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| RuleBased | normal | 1248.7 | -84.9 | 0.509 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Calibrated | normal | 1204.3 | -129.4 | 0.775 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| TFIDF_LogReg_Raw | normal | 1378.1 | 44.5 | -0.267 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| gpt-4o | normal | 1181.6 | -152.0 | 0.911 | NEGATIVE_ORACLE_REFERENCE_VALUE |
| NoInfo | supplier_capacity_drop | 1333.6 | 0.0 | 0.000 | VALID |
| OracleSemantic | supplier_capacity_drop | 1726.0 | 392.4 | 1.000 | VALID |
| RuleBased | supplier_capacity_drop | 1525.8 | 192.2 | 0.490 | VALID |
| TFIDF_LogReg_Calibrated | supplier_capacity_drop | 1477.0 | 143.4 | 0.365 | VALID |
| TFIDF_LogReg_Raw | supplier_capacity_drop | 1409.2 | 75.6 | 0.193 | VALID |
| gpt-4o | supplier_capacity_drop | 1547.6 | 214.0 | 0.545 | VALID |
