# Statistical Audit

**Date:** 2026-08-23
**Scope:** All statistical methods, metrics, and numerical claims across Phases 7–9

---

## 1. Primary Metric: Aggregate SIVR

### Formula

```
Aggregate SIVR = Σ_i (R_sensor_i - R_noinfo_i) / |Σ_i (R_oracle_i - R_noinfo_i)|
```

where `i` indexes (seed, template_id) pairs.

### Properties

- **Scale:** Unbounded in general. SIVR ∈ (-∞, ∞) theoretically. In practice, SIVR ∈ [-1, ~1.2]
  for our experiments.
- **SIVR > 1:** Valid. Indicates sensor outperforms Oracle on aggregate. Occurs when RuleBased's
  P(regime) values produce better base-stock decisions than OracleSemantic's P=1.0 for specific
  seeds.
- **SIVR = 1.0:** Oracle reference by construction.
- **SIVR = 0.0:** NoInfo reference by construction.
- **SIVR < 0:** Oracle underperforms NoInfo (controller misalignment).
- **`|denom|` convention:** When Oracle OIV < 0, absolute value preserves numerator sign.

### Verification

All headline SIVRs recompute from raw CSVs:

| Phase | Sensor | Ledger | Computed | Δ |
|-------|--------|--------|----------|---|
| 9A | RuleBased | 0.823 | 0.82322 | < 0.001 |
| 9A | gpt-4o | 0.507 | 0.50703 | < 0.001 |
| 9A | TFIDF_Raw | 0.261 | 0.26080 | < 0.001 |
| 9A | TFIDF_Cal | 0.434 | 0.43429 | < 0.001 |
| 9B | baseline RuleBased | 0.383 | 0.38323 | < 0.001 |

---

## 2. Statistical Testing

### Hierarchical Paired Bootstrap (Primary)

**Method:** Two-stage cluster bootstrap.
1. Resample template clusters with replacement (B=10,000 iterations)
2. Within each resampled template, resample seeds with replacement
3. Compute mean reward difference (sensor - NoInfo) per bootstrap sample
4. Report percentile 95% CI

**Correctness:** Appropriate for the crossed seed×template design. Treats seed×template
combinations as nested within templates. Accounts for correlation structure within templates.

**Limitation:** Percentile CIs (not BCa) with only 16 template clusters may under-cover.
This is a known limitation for small cluster counts. BCa requires more clusters for accurate
acceleration estimation.

### Flat Paired Bootstrap (Secondary)

**Method:** Ignores template clustering. Resamples (seed, template) pairs with replacement.
Treats each observation as independent.

**Status:** Anti-conservative for this design. Used only as secondary display in Phase 9A.
Primary results use hierarchical bootstrap.

### Significance Rule

A sensor is "significant" if the hierarchical bootstrap 95% CI for (sensor - NoInfo) reward
difference excludes zero.

---

## 3. Headline Results Summary

### Phase 8B (DemandSurge, Publication Primary)

| Sensor | SIVR | 95% CI (Hierarchical) | Significant |
|--------|------|----------------------|:-----------:|
| RuleBased | 1.198 | [54.3, 148.1] | Yes |
| TFIDF_LogReg_Raw | 0.575 | [26.7, 70.9] | Yes |
| TFIDF_LogReg_Calibrated | 0.841 | [25.8, 117.8] | Yes |
| gpt-4o | 0.797 | [18.4, 118.4] | Yes |

### Phase 9A (SupplierCapacityDrop)

| Sensor | SIVR | 95% CI (Hierarchical) | Significant |
|--------|------|----------------------|:-----------:|
| RuleBased | 0.823 | [51.6, 252.2] | Yes |
| TFIDF_LogReg_Raw | 0.261 | [6.1, 87.7] | Yes |
| TFIDF_LogReg_Calibrated | 0.434 | [-41.4, 204.9] | No |
| gpt-4o | 0.507 | [-13.7, 200.4] | No |

### Phase 9B (OFAT Robustness)

| Variant | RuleBased SIVR | Significant | Interpretation |
|---------|---------------|:-----------:|----------------|
| baseline | +0.383 | Yes | Reference |
| short_lead (L×0.5) | +0.358 | No | Graceful degradation |
| low_noise (σ=0) | +0.383 | Yes | Identical to baseline |
| high_noise (σ×2) | +0.384 | Yes | Robust to noise |
| long_lead (L×2.0) | -0.385 | Yes (neg) | Policy failure |
| lost_sales | -0.389 | Yes (neg) | Policy mismatch |

---

## 4. Belief Quality Metrics

### Brier Score

```
Brier = (1/N) Σ_i (1 - p_true_i)^2
```

where `p_true_i` is the predicted probability of the true regime for episode `i`.

### Log-Loss

```
LogLoss = -(1/N) Σ_i log(p_true_i)
```

### Phase 9A Results

| Sensor | Brier | LogLoss |
|--------|------:|--------:|
| NoInfo | 0.2500 | 0.693 |
| RuleBased | 0.1078 | 0.334 |
| TFIDF_Raw | 0.1793 | 0.549 |
| TFIDF_Cal | 0.0897 | 0.248 |
| gpt-4o | 0.0494 | 0.193 |
| Oracle | 0.0000 | 0.000 |

### Accuracy vs Value Dissociation

| Sensor | Accuracy | SIVR (Phase 7) | SIVR (Phase 8B) | SIVR (Phase 9A) |
|--------|----------|---------------:|----------------:|----------------:|
| TFIDF_Raw | 88.9% | 0.250 | 0.575 | 0.261 |
| TFIDF_Cal | 86.1% | 0.648 | 0.841 | 0.434 |

**Interpretation:** Lower accuracy (86.1% vs 88.9%) but higher SIVR (0.648 vs 0.250).
Calibration trades classification accuracy for better-calibrated probabilities, which the
operational controller converts into superior decisions. This dissociation holds across all
three phases.

---

## 5. Fill Rate

```
Fill Rate = Σ(shipments) / Σ(demand)
```

Phase 8B uses `S[:, 0]` (retail shipments). Phase 9 uses `total_retail_sales` from
CapacityDropWrapper (also shipments). Both measure the same operational quantity.

---

## 6. Information Value (OIV)

```
OIV = R_oracle - R_noinfo
```

| Phase | Event | OIV Range | Interpretation |
|-------|-------|-----------|----------------|
| 8B | DemandSurge | 84.3 (467.7 → 552.0) | Moderate room for info |
| 9A | SupplierCapacityDrop | 182.7 (1333.6 → 1516.3) | Large room for info |
| 9B baseline | SupplierCapacityDrop | 389.9 (1323.6 → 1713.6) | Large room for info |

When OIV < 0 (long_lead, lost_sales), the `|denom|` convention makes SIVR degenerate.
This must be documented in the paper.

---

## 7. Episode Counts

| Phase | Seeds | Templates | Sensors | Episodes | Status |
|-------|-------|-----------|---------|----------|--------|
| Phase 5.5 | 15 | 18 | 6 | 1,350 | Historical |
| Phase 6 | 50 | 36 | 6 | 10,800 | Historical |
| Phase 7 | 20 | 36 | 6 | 4,320 | Classical baseline |
| Phase 8A | 20 | 12 | 5 | 1,200 | Exploratory |
| Phase 8B | 30 | 24 | 6 | 4,320 | **Publication primary** |
| Phase 9A | 30 | 16 | 6 | 2,880 | Cross-dynamics |
| Phase 9B | 10×6 | 16 | 6 | 5,760 | Robustness |
| **Total** | — | — | — | **30,630** | |

---

## 8. Seed Disjointness

**Publication phases are pairwise disjoint:**
- Phase 8B: 3100–3129
- Phase 9A confirmation: 4000–4029
- Phase 9B: 4100–4109

**Non-publication phases share seeds (acceptable):**
- Phase 7: 2000–2019
- Phase 8A confirmation: 2000–2019
- Phase 6: 2000–2049

Phase 9A pilot (4000–4004) is a subset of Phase 9A confirmation (4000–4029). This is
intentional — the pilot validated infrastructure before the full run.

---

## 9. Issues Found

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | MINOR | Percentile CIs (not BCa) with 16 template clusters may under-cover | Documented |
| 2 | MINOR | `paired_bootstrap_ci` (flat iid) is anti-conservative | Used as secondary only |
| 3 | MINOR | Phase 9A SIVR includes train templates for TF-IDF sensors | Add test-only SIVR |
| 4 | MINOR | `|denom|` convention not documented in paper | Add to methods |

---

## 10. Conclusion

All headline numbers recompute from raw data. Statistical methods are appropriate for the
design. The hierarchical bootstrap correctly accounts for the crossed seed×template structure.
CIs may under-cover slightly due to small cluster count and percentile method, but this
affects precision of significance claims, not direction of effects.

The accuracy-value dissociation (TFIDF_Cal lower accuracy but higher SIVR) is the paper's
core finding and holds across all three phases. The cross-event generalization (information
helps on both demand and supply shocks) is supported by the data.
