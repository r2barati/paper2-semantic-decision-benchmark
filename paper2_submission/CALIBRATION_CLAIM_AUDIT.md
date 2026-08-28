# Calibration claim hostile check

The frozen Phase-7 source reports:

| Interpreter | Accuracy | Brier | Log-loss | SIVR |
|---|---:|---:|---:|---:|
| Raw TF-IDF | 0.889 | 0.552 | 0.932 | 0.250 |
| Calibrated TF-IDF | 0.861 | 0.240 | 0.397 | 0.648 |

## Established by the frozen design

The calibrated TF-IDF variant has lower held-out classification accuracy,
better Brier/log-loss, and higher downstream SIVR in this benchmark. The
comparison uses the same event framework/controller and the corrected frozen
analysis.

## Not established

The result does not identify calibration as the only causal explanation. The
change could reflect the full probability transformation interacting with
controller thresholds, class priors, class imbalance, or finite-sample
isotonic behavior. The evidence is consistent with, and motivates, a
calibration-to-utility interpretation; it does not establish a universal
calibration law or controller-independent causal effect.

## Manuscript wording lock

“Improved probability calibration substantially increased downstream
operational value despite not improving classification accuracy” is acceptable
when explicitly scoped to this benchmark. Avoid “calibration always improves
decisions,” “calibration causes value in general,” or conflating accuracy,
Brier score, log-loss, SIVR, and reward.

