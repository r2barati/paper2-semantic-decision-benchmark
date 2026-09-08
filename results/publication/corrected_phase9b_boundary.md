# Phase 9B boundary analysis

Descriptive robustness/boundary analysis of the fixed controller.

| Variant | Sensor | Reward | NoInfo | Δ vs NoInfo | Oracle | Oracle Δ | OIV status | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|---|
| baseline | Constant_p1.0 | 1712.7 | 1323.6 | 389.1 | 1435.1 | 111.5 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | NoInfo | 1323.6 | 1323.6 | 0.0 | 1435.1 | 111.5 | VALID | sensor_or_controller_underperforms_noinfo |
| baseline | NoText_Tuned | 1712.7 | 1323.6 | 389.1 | 1435.1 | 111.5 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | OracleSemantic | 1435.1 | 1323.6 | 111.5 | 1435.1 | 111.5 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | RuleBased | 1436.2 | 1323.6 | 112.6 | 1435.1 | 111.5 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | TFIDF_LogReg_Calibrated | 1362.2 | 1323.6 | 38.6 | 1435.1 | 111.5 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | TFIDF_LogReg_Raw | 1352.8 | 1323.6 | 29.2 | 1435.1 | 111.5 | VALID | semantic_value_supported_under_fixed_controller |
| baseline | gpt-4o | 1362.4 | 1323.6 | 38.8 | 1435.1 | 111.5 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | Constant_p1.0 | 1673.8 | 1313.3 | 360.5 | 1410.6 | 97.2 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | NoInfo | 1313.3 | 1313.3 | 0.0 | 1410.6 | 97.2 | VALID | sensor_or_controller_underperforms_noinfo |
| high_noise | NoText_Tuned | 1673.8 | 1313.3 | 360.5 | 1410.6 | 97.2 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | OracleSemantic | 1410.6 | 1313.3 | 97.2 | 1410.6 | 97.2 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | RuleBased | 1416.8 | 1313.3 | 103.5 | 1410.6 | 97.2 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | TFIDF_LogReg_Calibrated | 1343.6 | 1313.3 | 30.3 | 1410.6 | 97.2 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | TFIDF_LogReg_Raw | 1342.6 | 1313.3 | 29.2 | 1410.6 | 97.2 | VALID | semantic_value_supported_under_fixed_controller |
| high_noise | gpt-4o | 1346.1 | 1313.3 | 32.8 | 1410.6 | 97.2 | VALID | semantic_value_supported_under_fixed_controller |
| long_lead | Constant_p1.0 | -967.1 | -875.7 | -91.5 | -878.3 | -2.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | NoInfo | -875.7 | -875.7 | 0.0 | -878.3 | -2.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | NoText_Tuned | -967.1 | -875.7 | -91.5 | -878.3 | -2.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | OracleSemantic | -878.3 | -875.7 | -2.6 | -878.3 | -2.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | RuleBased | -902.3 | -875.7 | -26.6 | -878.3 | -2.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | TFIDF_LogReg_Calibrated | -869.0 | -875.7 | 6.7 | -878.3 | -2.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | TFIDF_LogReg_Raw | -883.5 | -875.7 | -7.8 | -878.3 | -2.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| long_lead | gpt-4o | -862.7 | -875.7 | 13.0 | -878.3 | -2.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | Constant_p1.0 | 1438.9 | 1511.8 | -72.9 | 1490.2 | -21.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | NoInfo | 1511.8 | 1511.8 | 0.0 | 1490.2 | -21.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | NoText_Tuned | 1438.9 | 1511.8 | -72.9 | 1490.2 | -21.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | OracleSemantic | 1490.2 | 1511.8 | -21.6 | 1490.2 | -21.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | RuleBased | 1489.5 | 1511.8 | -22.4 | 1490.2 | -21.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | TFIDF_LogReg_Calibrated | 1502.7 | 1511.8 | -9.1 | 1490.2 | -21.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | TFIDF_LogReg_Raw | 1506.9 | 1511.8 | -4.9 | 1490.2 | -21.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| lost_sales | gpt-4o | 1507.1 | 1511.8 | -4.7 | 1490.2 | -21.6 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| low_noise | Constant_p1.0 | 1731.8 | 1333.5 | 398.3 | 1449.6 | 116.1 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | NoInfo | 1333.5 | 1333.5 | 0.0 | 1449.6 | 116.1 | VALID | sensor_or_controller_underperforms_noinfo |
| low_noise | NoText_Tuned | 1731.8 | 1333.5 | 398.3 | 1449.6 | 116.1 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | OracleSemantic | 1449.6 | 1333.5 | 116.1 | 1449.6 | 116.1 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | RuleBased | 1449.1 | 1333.5 | 115.6 | 1449.6 | 116.1 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | TFIDF_LogReg_Calibrated | 1374.8 | 1333.5 | 41.3 | 1449.6 | 116.1 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | TFIDF_LogReg_Raw | 1362.7 | 1333.5 | 29.2 | 1449.6 | 116.1 | VALID | semantic_value_supported_under_fixed_controller |
| low_noise | gpt-4o | 1372.5 | 1333.5 | 38.9 | 1449.6 | 116.1 | VALID | semantic_value_supported_under_fixed_controller |
| short_lead | Constant_p1.0 | 2486.2 | 2397.8 | 88.4 | 2214.4 | -183.4 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| short_lead | NoInfo | 2397.8 | 2397.8 | 0.0 | 2214.4 | -183.4 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| short_lead | NoText_Tuned | 2486.2 | 2397.8 | 88.4 | 2214.4 | -183.4 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| short_lead | OracleSemantic | 2214.4 | 2397.8 | -183.4 | 2214.4 | -183.4 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| short_lead | RuleBased | 2411.0 | 2397.8 | 13.2 | 2214.4 | -183.4 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| short_lead | TFIDF_LogReg_Calibrated | 2247.7 | 2397.8 | -150.1 | 2214.4 | -183.4 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| short_lead | TFIDF_LogReg_Raw | 2421.1 | 2397.8 | 23.3 | 2214.4 | -183.4 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
| short_lead | gpt-4o | 2213.4 | 2397.8 | -184.4 | 2214.4 | -183.4 | NEGATIVE_ORACLE_REFERENCE_VALUE | controller_or_environment_mismatch; oracle_reference_not_positive |
