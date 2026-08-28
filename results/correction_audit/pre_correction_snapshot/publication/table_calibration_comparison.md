# Table: Calibration Comparison — Raw vs Calibrated TF-IDF

## Phase 7 (Controlled Benchmark, 36 held-out templates)

| Metric | TFIDF_Raw | TFIDF_Calibrated |
|--------|----------:|-----------------:|
| Accuracy | 0.889 | 0.861 |
| Brier | 0.552 | 0.240 |
| AggregateSIVR | 0.250 | 0.648 |

## Phase 8B (Paper-1 Gymnasium, 24 held-out templates)

| Metric | TFIDF_Raw | TFIDF_Calibrated |
|--------|----------:|-----------------:|
| Brier | 0.181 | 0.046 |
| AggregateSIVR | 0.575 | 0.841 |
| Belief Accuracy | 0.917 | 0.958 |

**Key finding:** Calibration reduces accuracy (88.9% → 86.1%) but substantially increases operational value (SIVR 0.250 → 0.648 in Phase 7; 0.575 → 0.841 in Phase 8B). This demonstrates that classification accuracy alone does not predict downstream decision quality.
