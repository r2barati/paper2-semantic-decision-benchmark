# Results Summary

## Primary Claims and Supporting Evidence

### Claim 1: Semantic information has measurable operational value

**Evidence:** All imperfect semantic sensors achieve SIVR > 0 with hierarchical bootstrap CIs excluding zero (Phase 8B: all four sensors significant vs NoInfo).

### Claim 2: Imperfect semantic systems recover different fractions of oracle value

**Evidence:** SIVR ranges from 0.575 (TFIDF_Raw) to 1.198 (RuleBased) in Phase 8B, and from -0.206 (gpt-3.5-turbo) to 0.767 (gpt-4o) in Phase 7.

### Claim 3: Classification accuracy alone does not determine operational value

**Evidence:** TFIDF_Raw has higher held-out accuracy than TFIDF_Calibrated (88.9% vs 86.1%) but lower SIVR in both Phase 7 (0.250 vs 0.648) and Phase 8B (0.575 vs 0.841).

### Claim 4: Probability calibration materially changes downstream value

**Evidence:** Calibration improves SIVR by +0.398 in Phase 7 and +0.266 in Phase 8B, despite reducing classification accuracy.

### Claim 5: Conventional NLP can outperform LLMs in some settings

**Evidence:** In Phase 8B, TFIDF_Calibrated (0.841) outperforms gpt-4o (0.797) in SIVR despite gpt-4o having better belief quality metrics (Brier: 0.033 vs 0.046).

### Claim 6: The phenomenon survives transfer to richer operational dynamics

**Evidence:** Phase 8B (Paper-1 Gymnasium) shows the same qualitative patterns as Phase 7 (controlled benchmark): all sensors achieve SIVR > 0, calibration helps, accuracy ≠ value.
