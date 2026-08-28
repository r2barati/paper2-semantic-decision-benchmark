# Contributions

## Narrow Claim

A framework for evaluating the decision value of probabilistic semantic sensing in executable sequential operational systems.

## Specific Contributions

1. **A controlled executable benchmark** separating semantic interpretation from operational control, enabling principled measurement of how information quality affects downstream decisions (Experiment A).

2. **OIV/SIVR-style metrics** measuring how much actionable semantic information a sensor/interpreter recovers, with hierarchical bootstrap uncertainty quantification.

3. **Evidence that semantic classification accuracy alone can misrepresent downstream decision quality.** TF-IDF_Raw has higher accuracy than TFIDF_Calibrated but substantially lower SIVR.

4. **Evidence that probability calibration materially changes operational value.** Calibrated TF-IDF achieves SIVR = 0.648 (Phase 7) and 0.841 (Phase 8B) versus 0.250 and 0.575 uncalibrated.

5. **Comparison across five interpreter types** (rule-based, classical NLP, LLM, oracle-semantic, no-information) with paired evaluation.

6. **Operational-complexity transfer** into an existing multi-echelon Paper-1 Gymnasium environment, demonstrating that the phenomenon generalizes beyond the controlled benchmark.

7. **Reproducibility artifacts** with frozen templates, seeds, manifests, cached LLM predictions, and hierarchical uncertainty analysis.

## What We Do NOT Claim

- We do not claim LLMs are unsuitable for operational decision support
- We do not claim our specific benchmark numbers generalize to production systems
- We do not claim the controller design is optimal
- We do not claim the observed rankings will hold across all possible controllers
