# TODO — paper2_semantic_decision_benchmark

## Repository Separation Record

| Item | Value |
|------|-------|
| Frozen MVP path | `/Users/sanamimani/paper2_semantic_decision_mvp` |
| Source commit | N/A (no git repo in source) |
| Copy date | 2026-08-22 |
| New project path | `/Users/sanamimani/paper2_semantic_decision_benchmark` |
| Initial commit | `77c8447` |
| Baseline test count | 161 passed (original), 191 passed (with Phase 7 tests) |
| Phase-6 artifacts | `results/phase6/` (confirmation_template_manifest.json, confirmation_seeds.json, frozen_phase5_manifest.json) |
| Confirmation | All subsequent development occurs only in this project. MVP is untouched. |

## Phase 7 — Classical NLP Baseline
- [x] Implement TF-IDF + Logistic Regression classifier
- [x] Implement leakage-safe grouped split with audit
- [x] Evaluate and implement isotonic probability calibration
- [x] Semantic evaluation (accuracy, Brier, log-loss, calibration, confusion matrix)
- [x] Operational evaluation through downstream pipeline
- [x] OIV/SIVR/regret computed
- [x] Regime analysis completed
- [x] Ambiguity analysis completed
- [x] Statistical comparison completed (paired bootstrap, hierarchical bootstrap)
- [x] Phase-7 report written
- [x] Manifests saved
- [x] Full test suite passes (191 tests)
- [x] TODO.md updated

### Phase 7 Results Summary

| Item | Value |
|------|-------|
| Model | TFIDF_LogReg (uncalibrated) / TFIDF_LogReg_Calibrated (isotonic) |
| Train templates | 18 (original Phase-5) |
| Test templates | 36 (held-out confirmation) |
| CV accuracy | 72.2% |
| Held-out accuracy (raw) | 88.9% |
| Held-out accuracy (calibrated) | 86.1% |
| Brier (raw) | 0.184 |
| Brier (calibrated) | 0.080 |
| Log-loss (raw) | 0.932 |
| Log-loss (calibrated) | 0.397 |
| ECE (raw) | 0.491 |
| ECE (calibrated) | 0.160 |
| AggregateSIVR (raw) | 0.250 |
| AggregateSIVR (calibrated) | 0.648 |
| RuleBased AggregateSIVR | 0.707 |
| gpt-4o AggregateSIVR (Phase 6) | 0.767 |
| Key finding | Calibration transforms classical baseline; accuracy does not predict SIVR |

### Phase 7 Output Files
- `results/phase7_classical_baseline/PHASE7_REPORT.md`
- `results/phase7_classical_baseline/operational_results.csv`
- `results/phase7_classical_baseline/belief_metrics.csv`
- `results/phase7_classical_baseline/belief_metrics_summary.csv`
- `results/phase7_classical_baseline/sivr_by_regime.csv`
- `results/phase7_classical_baseline/aggregate_sivr.csv`
- `results/phase7_classical_baseline/ambiguity_analysis.csv`
- `results/phase7_classical_baseline/bootstrap_comparisons.csv`
- `results/phase7_classical_baseline/classification_metrics.csv`
- `results/phase7_classical_baseline/experiment_manifest.json`
- `results/phase7_classical_baseline/split_manifest.json`
- `results/phase7_classical_baseline/tfidf_logreg_model.pkl`
- `results/phase7_classical_baseline/tfidf_logreg_calibrated_model.pkl`

## Phase 8 — Paper-1 Gym Replication
- [ ] Replicate Paper-1 Gym findings in this framework
- [ ] Map Paper-1 conditions to Phase-6 conditions
- [ ] Validate benchmark consistency

## Publication Artifacts
- [ ] Generate figures
- [ ] Package benchmark for release
- [ ] Write manuscript sections
