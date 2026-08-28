# Phase 9B boundary analysis

Descriptive robustness/boundary analysis of the fixed controller.

| Variant | Sensor | Reward | NoInfo | Δ vs NoInfo | Oracle | Oracle Δ | OIV status | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|---|
| baseline | NoInfo | 1323.6 | 1323.6 | 0.0 | 1713.6 | 390.0 | VALID | sensor_or_controller_underperforms_noinfo |
| baseline | OracleSemantic | 1713.6 | 1323.6 | 390.0 | 1713.6 | 390.0 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | RuleBased | 1436.2 | 1323.6 | 112.6 | 1713.6 | 390.0 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | TFIDF_LogReg_Calibrated | 1349.4 | 1323.6 | 25.8 | 1713.6 | 390.0 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | TFIDF_LogReg_Raw | 1352.8 | 1323.6 | 29.2 | 1713.6 | 390.0 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | gpt-4o | 1362.4 | 1323.6 | 38.8 | 1713.6 | 390.0 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | NoInfo | 1313.3 | 1313.3 | 0.0 | 1674.7 | 361.4 | VALID | sensor_or_controller_underperforms_noinfo |
| high_noise | OracleSemantic | 1674.7 | 1313.3 | 361.4 | 1674.7 | 361.4 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | RuleBased | 1416.8 | 1313.3 | 103.5 | 1674.7 | 361.4 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | TFIDF_LogReg_Calibrated | 1330.8 | 1313.3 | 17.5 | 1674.7 | 361.4 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | TFIDF_LogReg_Raw | 1342.6 | 1313.3 | 29.2 | 1674.7 | 361.4 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | gpt-4o | 1346.1 | 1313.3 | 32.8 | 1674.7 | 361.4 | VALID | semantic_value_supported_under_fixed_controller |
| long_lead | NoInfo | -875.7 | -875.7 | 0.0 | -966.6 | -90.9 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | OracleSemantic | -966.6 | -875.7 | -90.9 | -966.6 | -90.9 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | RuleBased | -902.3 | -875.7 | -26.6 | -966.6 | -90.9 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | TFIDF_LogReg_Calibrated | -862.2 | -875.7 | 13.5 | -966.6 | -90.9 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | TFIDF_LogReg_Raw | -883.5 | -875.7 | -7.8 | -966.6 | -90.9 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | gpt-4o | -862.7 | -875.7 | 13.0 | -966.6 | -90.9 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | NoInfo | 1511.8 | 1511.8 | 0.0 | 1439.5 | -72.3 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | OracleSemantic | 1439.5 | 1511.8 | -72.3 | 1439.5 | -72.3 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | RuleBased | 1489.5 | 1511.8 | -22.4 | 1439.5 | -72.3 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | TFIDF_LogReg_Calibrated | 1505.6 | 1511.8 | -6.2 | 1439.5 | -72.3 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | TFIDF_LogReg_Raw | 1506.9 | 1511.8 | -4.9 | 1439.5 | -72.3 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | gpt-4o | 1507.1 | 1511.8 | -4.7 | 1439.5 | -72.3 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| low_noise | NoInfo | 1333.5 | 1333.5 | 0.0 | 1732.7 | 399.2 | VALID | sensor_or_controller_underperforms_noinfo |
| low_noise | OracleSemantic | 1732.7 | 1333.5 | 399.2 | 1732.7 | 399.2 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | RuleBased | 1449.1 | 1333.5 | 115.6 | 1732.7 | 399.2 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | TFIDF_LogReg_Calibrated | 1362.0 | 1333.5 | 28.5 | 1732.7 | 399.2 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | TFIDF_LogReg_Raw | 1362.7 | 1333.5 | 29.2 | 1732.7 | 399.2 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | gpt-4o | 1372.5 | 1333.5 | 38.9 | 1732.7 | 399.2 | VALID | semantic_value_supported_under_fixed_controller |
| short_lead | NoInfo | 2396.9 | 2396.9 | 0.0 | 2486.8 | 89.9 | VALID | sensor_or_controller_underperforms_noinfo |
| short_lead | OracleSemantic | 2486.8 | 2396.9 | 89.9 | 2486.8 | 89.9 | VALID | semantic_value_supported_under_fixed_controller |
| short_lead | RuleBased | 2409.3 | 2396.9 | 12.4 | 2486.8 | 89.9 | VALID | semantic_value_supported_under_fixed_controller |
| short_lead | TFIDF_LogReg_Calibrated | 2212.9 | 2396.9 | -184.0 | 2486.8 | 89.9 | VALID | sensor_or_controller_underperforms_noinfo |
| short_lead | TFIDF_LogReg_Raw | 2419.7 | 2396.9 | 22.9 | 2486.8 | 89.9 | VALID | semantic_value_supported_under_fixed_controller |
| short_lead | gpt-4o | 2217.6 | 2396.9 | -179.3 | 2486.8 | 89.9 | VALID | sensor_or_controller_underperforms_noinfo |
