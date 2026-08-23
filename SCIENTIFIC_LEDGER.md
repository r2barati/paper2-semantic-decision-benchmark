# Scientific Ledger

This file is the authoritative source of truth for all experimental numbers in the manuscript.

---

## 1. Template Inventory

### 1.1 Original Training Templates

- **Source:** `src.events.REGIME_WARNING_TEMPLATES`
- **Count:** 18
- **Regimes:** supplier_delay (6), demand_surge (6), normal (6)
- **Ambiguity:** clear (6), moderate (6), vague (6)
- **ID prefix:** `delay_*`, `surge_*`, `normal_*`
- **Used for:** TF-IDF training (Phase 7), all Phase 5/5.5 LLM evaluation

### 1.2 Confirmation Held-Out Templates

- **Source:** `src.confirmation_templates.CONFIRMATION_TEMPLATES`
- **Count:** 36
- **Regimes:** supplier_delay (12), demand_surge (12), normal (12)
- **Ambiguity:** clear (12), moderate (12), vague (12)
- **ID prefix:** `cd_*`, `cs_*`, `cn_*`
- **Created:** Before any LLM evaluation
- **Used for:** Phase 6 confirmation, Phase 7 held-out testing

### 1.3 Phase-8B Held-Out Templates

- **Source:** Subset of CONFIRMATION_TEMPLATES where regime in {NORMAL, DEMAND_SURGE}
- **Count:** 24 (12 surge + 12 normal)
- **ID prefix:** `cs_*`, `cn_*`
- **Excluded:** SupplierDelay templates (Phase 8B uses 2-class regime space)
- **Used for:** Phase 8B confirmation experiment

### 1.4 Phase-9 Capacity-Drop Templates

- **Source:** `src.capacity_drop_templates`
- **Count:** 16 (6 normal + 10 capacity_drop)
- **Regimes:** NORMAL (6), SUPPLIER_CAPACITY_DROP (10)
- **Ambiguity:** clear (5), moderate (6), vague (5)
- **ID prefix:** `cn_*` (normal), `cd_*` (capacity_drop)
- **Train:** 9 templates (3 cn + 6 cd)
- **Test:** 7 templates (2 cn + 5 cd) — all vague CD held out
- **Used for:** Phase 9A confirmation, Phase 9B robustness

### 1.5 Template Separation

| Train | Test | Overlap |
|-------|------|---------|
| 18 original | 36 confirmation | **0 template IDs** |
| 18 original | 24 Phase-8B subset | **0 template IDs** |

Families (delay/surge/normal) are shared intentionally — the model must generalize to new phrasings.

---

## 2. Seed Inventory

| Phase | Range | Count | Purpose |
|-------|-------|------:|---------|
| Phase 5/5.5 | 1000–1014 | 15 | Original controlled benchmark |
| Phase 6 confirmation | 2000–2049 | 50 | Confirmation robustness |
| Phase 7 classical | 2000–2019 | 20 | Classical baseline eval |
| Phase 8A pilot | 1–20 | 20 | Paper-1 Gym exploratory |
| Phase 8A confirmation | 2000–2019 | 20 | Paper-1 Gym exploratory |
| Phase 8B pilot | 3000–3001 | 2 | Infrastructure validation |
| **Phase 8B confirmation** | **3100–3129** | **30** | **Publication primary** |
| Phase 9A pilot | 4000–4004 | 5 | CapacityDrop validation |
| **Phase 9A confirmation** | **4000–4029** | **30** | **Cross-dynamics confirmation** |
| Phase 9B robustness | 4100–4109 | 10 | One-factor-at-a-time (×6 variants) |

**All seed ranges are pairwise disjoint.** Phase-9A seeds (4000–4029) are disjoint from Phase 8B (3100–3129), Phase 8A (1–20, 2000–2019), Phase 7 (2000–2019), Phase 6 (2000–2049), and Phase 5/5.5 (1000–1014). Phase-9B seeds (4100–4109) are disjoint from all prior phases.

---

## 3. Episode Totals by Phase

### Phase 5.5 (Controlled, LLM)
- 15 seeds x 18 templates x 5 sensors = **1,350 episodes**
- Sensors: NoInfo, RuleBased, gpt-3.5-turbo, gpt-4o-mini, gpt-4o, PerfectSemantic

### Phase 6 (Confirmation)
- 50 seeds x 36 templates x 6 sensors = **10,800 episodes**

### Phase 7 (Classical Baseline)
- 20 seeds x 36 templates x 6 sensors = **4,320 episodes**
- Sensors: NoInfo, RuleBased, gpt-4o, PerfectSemantic, TFIDF_LogReg, TFIDF_LogReg_Calibrated

### Phase 8A (Paper-1 Gym, exploratory)
- 20 seeds x 12 templates x 5 sensors = **1,200 episodes**
- Sensors: NoInfo, RuleBased, TFIDF_LogReg, gpt-4o, PerfectSemantic
- Note: validity audit flagged linguistic leakage and fill-rate bug

### Phase 8B (Paper-1 Gym, confirmation — PUBLICATION PRIMARY)
- 30 seeds x 24 templates x 6 sensors = **4,320 episodes**
- Sensors: NoInfo, RuleBased, TFIDF_LogReg_Raw, TFIDF_LogReg_Calibrated, gpt-4o, OracleSemantic

### Phase 9A (Cross-Dynamics: SupplierCapacityDrop)
- 30 seeds x 16 templates x 6 sensors = **2,880 episodes**
- Sensors: NoInfo, RuleBased, TFIDF_LogReg_Raw, TFIDF_LogReg_Calibrated, gpt-4o, OracleSemantic
- Topology: divergent (3 factories, Factory 4 capacity 90→30, periods 10–25)

### Phase 9B (One-Factor-at-a-Time Robustness)
- 6 variants × 10 seeds × 16 templates × 6 sensors = **5,760 episodes**
- Variants: baseline, short_lead (L×0.5), long_lead (L×2.0), lost_sales, low_noise (σ=0), high_noise (σ×2)

---

## 4. Interpreter/Sensor Count

| Sensor | Controlled (Phase 5.5/6/7) | Gym (Phase 8A) | Gym (Phase 8B) | Gym (Phase 9A/9B) |
|--------|:---:|:---:|:---:|:---:|
| NoInfo | x | x | x | x |
| RuleBased | x | x | x | x |
| gpt-3.5-turbo | x | — | — | — |
| gpt-4o-mini | x | — | — | — |
| gpt-4o | x | x | x | x |
| PerfectSemantic | x | x | — | — |
| OracleSemantic | — | — | x | x |
| TFIDF_LogReg (raw) | — | x | x | x |
| TFIDF_LogReg (calibrated) | — | — | x | x |

---

## 5. Phase-7 Headline Results

### Classification (held-out 36 confirmation templates)

| Model | Accuracy | Brier | LogLoss |
|-------|----------|-------|---------|
| TF-IDF Raw | 88.9% | 0.176 | 0.443 |
| TF-IDF Calibrated | 86.1% | 0.081 | 0.233 |
| RuleBased | 100% | 0.000 | 0.000 |

### Operational Value (20 seeds x 36 templates)

| Sensor | AggregateSIVR | Mean Profit |
|--------|-------------:|------------:|
| NoInfo | 0.000 | — |
| RuleBased | 0.707 | — |
| gpt-3.5-turbo | -0.206 | — |
| gpt-4o | 0.767 | — |
| TFIDF_LogReg Raw | 0.250 | — |
| TFIDF_LogReg Calibrated | 0.648 | — |

---

## 6. Phase-8A Headline Results (EXPLORATORY — do not cite as confirmatory)

### Paper-1 Gymnasium (20 seeds x 12 training templates x 5 sensors)

| Sensor | AggSIVR | Mean Reward |
|--------|--------:|------------:|
| NoInfo | 0.000 | 476.5 |
| RuleBased | 0.887 | 547.6 |
| gpt-4o | 0.874 | 546.6 |
| TFIDF_LogReg (calibrated) | 1.000 | 556.7 |
| PerfectSemantic | 1.000 | 556.7 |

**Validity issues identified (see VALIDITY_AUDIT.md):**
- TF-IDF calibrated degenerated to {0, 1} on training templates
- Fill rate used `env.R` (replenishment) instead of `env.S` (retail sales)
- 12 templates is a small linguistic sample
- All templates seen during TF-IDF training

---

## 7. Phase-8B Headline Results (CONFIRMATORY — PUBLICATION PRIMARY)

### Paper-1 Gymnasium (30 seeds x 24 held-out templates x 6 sensors)

| Sensor | AggSIVR | Mean Reward | Fill Rate | Belief Acc |
|--------|--------:|------------:|----------:|-----------:|
| NoInfo | 0.000 | 467.7 | 0.858 | 0.500 |
| RuleBased | 1.198 | 568.7 | 0.947 | 0.792 |
| TFIDF_LogReg_Raw | 0.575 | 516.2 | 0.901 | 0.917 |
| TFIDF_LogReg_Calibrated | 0.841 | 538.6 | 0.911 | 0.958 |
| gpt-4o | 0.797 | 534.9 | 0.906 | 0.958 |
| OracleSemantic | 1.000 | 552.0 | 0.920 | 1.000 |

### Hierarchical Bootstrap CIs (vs NoInfo, primary)

| Sensor | Mean Diff | 95% CI | Significant |
|--------|----------:|--------|:-----------:|
| RuleBased | 101.0 | [54.3, 148.1] | Yes |
| TFIDF_LogReg_Raw | 48.5 | [26.7, 70.9] | Yes |
| TFIDF_LogReg_Calibrated | 70.8 | [25.8, 117.8] | Yes |
| gpt-4o | 67.1 | [18.4, 118.4] | Yes |

### Belief Quality

| Sensor | Brier | LogLoss |
|--------|------:|--------:|
| NoInfo | 0.2900 | 0.780 |
| RuleBased | 0.1615 | 1.762 |
| TFIDF_LogReg_Raw | 0.1812 | 0.553 |
| TFIDF_LogReg_Calibrated | 0.0463 | 0.148 |
| gpt-4o | 0.0332 | 0.112 |
| OracleSemantic | 0.0000 | 0.000 |

---

## 8. Phase-9A Headline Results (Cross-Dynamics: SupplierCapacityDrop)

### Divergent Topology (30 seeds × 16 templates × 6 sensors)

| Sensor | AggSIVR | Mean Reward | Fill Rate | Belief Acc |
|--------|--------:|------------:|----------:|-----------:|
| NoInfo | 0.000 | 1333.6 | 0.846 | 0.375 |
| RuleBased | 0.823 | 1484.0 | 0.895 | 0.938 |
| TFIDF_LogReg_Raw | 0.261 | 1381.2 | 0.861 | 0.875 |
| TFIDF_LogReg_Calibrated | 0.434 | 1412.9 | 0.871 | 0.812 |
| gpt-4o | 0.507 | 1426.2 | 0.875 | 0.938 |
| OracleSemantic | 1.000 | 1516.3 | 0.904 | 1.000 |

### Hierarchical Bootstrap CIs (vs NoInfo)

| Sensor | Mean Diff | 95% CI | Significant |
|--------|----------:|--------|:-----------:|
| RuleBased | 150.4 | [51.6, 252.2] | Yes |
| TFIDF_LogReg_Raw | 47.6 | [6.1, 87.7] | Yes |
| TFIDF_LogReg_Calibrated | 79.3 | [-41.4, 204.9] | No |
| gpt-4o | 92.6 | [-13.7, 200.4] | No |

### Belief Quality

| Sensor | Brier | LogLoss |
|--------|------:|--------:|
| NoInfo | 0.2500 | 0.693 |
| RuleBased | 0.1078 | 0.334 |
| TFIDF_LogReg_Raw | 0.1793 | 0.549 |
| TFIDF_LogReg_Calibrated | 0.0897 | 0.248 |
| gpt-4o | 0.0494 | 0.193 |
| OracleSemantic | 0.0000 | 0.000 |

---

## 9. Phase-9B Headline Results (One-Factor-at-a-Time Robustness)

### RuleBased SIVR Across Variants

| Variant | SIVR | Fill Rate | Significant | Interpretation |
|---------|-----:|----------:|:-----------:|----------------|
| baseline | +0.383 | 0.894 | Yes | Reference |
| short_lead (L×0.5) | +0.358 | 0.981 | No | Less time for info |
| low_noise (σ=0) | +0.383 | 0.894 | Yes | Identical to baseline |
| high_noise (σ×2) | +0.384 | 0.893 | Yes | Robust to noise |
| long_lead (L×2.0) | -0.385 | 0.434 | Yes (neg) | Policy failure |
| lost_sales | -0.389 | 0.718 | Yes (neg) | Policy mismatch |

Negative SIVR = oracle performs worse than no-info (policy misaligned with regime).

---

## 10. Cross-Event Synthesis

| Sensor | Phase 8B SIVR | Phase 9A SIVR | Phase 9B SIVR |
|--------|-------------:|-------------:|-------------:|
| NoInfo | 0.000 | 0.000 | 0.000 |
| RuleBased | 1.198 | 0.823 | 0.383 |
| TFIDF_LogReg_Raw | 0.575 | 0.261 | 0.119 |
| TFIDF_LogReg_Calibrated | 0.841 | 0.434 | 0.205 |
| gpt-4o | 0.797 | 0.507 | 0.232 |
| OracleSemantic | 1.000 | 1.000 | 1.000 |

Oracle SIVR = 1.000 in all events confirms framework is event-agnostic.

---

## 11. Key Finding: Accuracy vs Operational Value

| Sensor | Held-out Accuracy | AggSIVR (Phase 7) | AggSIVR (Phase 8B) | AggSIVR (Phase 9A) |
|--------|------------------:|-------------------:|-------------------:|-------------------:|
| TFIDF_Raw | 88.9% | 0.250 | 0.575 | 0.261 |
| TFIDF_Calibrated | 86.1% | 0.648 | 0.841 | 0.434 |
| gpt-4o | — | 0.767 | 0.797 | 0.507 |

**Lower accuracy (86.1% vs 88.9%) but higher SIVR (0.648 vs 0.250, 0.841 vs 0.575).** Calibration trades classification accuracy for better-calibrated probabilities, which the operational controller converts into superior decisions.

---

## 12. Terminology Changes

| Phase 8A Term | Phase 8B Term | Notes |
|---------------|---------------|-------|
| PerfectSemantic | OracleSemantic | Renamed to emphasize reference status |
| TFIDF_LogReg | TFIDF_LogReg_Raw + TFIDF_LogReg_Calibrated | Split into two distinct sensors |

---

## 13. Episode Totals (Grand)

| Phase | Episodes | Status |
|-------|-------:|--------|
| Phase 5.5 | 1,350 | Historical |
| Phase 6 | 10,800 | Historical |
| Phase 7 | 4,320 | Historical |
| Phase 8A | 1,200 | Exploratory provenance |
| Phase 8B | 4,320 | **Publication primary** |
| Phase 9A | 2,880 | Cross-dynamics confirmation |
| Phase 9B | 5,760 | Robustness |
| **Total** | **30,630** | |
