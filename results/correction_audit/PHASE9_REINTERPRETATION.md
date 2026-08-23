# Phase 9 corrected interpretation

## Phase 9A

Phase 9A is cross-event operational transfer to a supplier-capacity-drop
environment. The original all-template evaluation contains both TF-IDF train
and test templates, so it is not clean linguistic generalization. The saved
episodes also support `phase9a_test_templates_only.csv`, which is the separate
held-out linguistic analysis.

The fixed controller uses the capacity-drop probability to amplify safety
stock. Results therefore quantify realized value under that mapping, not a
controller-independent property.

## Phase 9B boundary cases

The authoritative `phase9b_boundary_analysis.csv` distinguishes:

- `long_lead`: OracleSemantic is worse than NoInfo, so OIV is negative. This is
  a controller/environment mismatch; a negative signed SIVR is diagnostic only.
- `lost_sales`: OracleSemantic is also worse than NoInfo, so OIV is negative.
  This is an environment/controller mismatch, not evidence that the semantic
  sensor failed to identify the regime.

The remaining variants have positive OIV and are descriptive robustness
results. For each variant the table separately reports oracle reward, sensor
reward, sensor accuracy proxy, raw delta, and OIV status.
