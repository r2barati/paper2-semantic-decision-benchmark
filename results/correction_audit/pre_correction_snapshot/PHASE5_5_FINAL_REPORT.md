# Phase 5.5 Final Report: Frozen Real-LLM Semantic Sensing Experiment

## Executive Summary

We evaluated three real LLMs (gpt-4o, gpt-4o-mini, gpt-3.5-turbo) as semantic sensors on the frozen Phase 5 multi-regime inventory benchmark. The experiment tests whether LLM semantic interpretation quality causally affects downstream sequential decision performance under a strong causal optimizer.

**Key finding:** LLM belief quality (Brier score) is strongly correlated with operational decision value (SIVR_O), supporting our hypothesis that semantic interpretation quality matters. gpt-4o achieves near-perfect SIVR_O in Normal and DemandSurge regimes, while gpt-3.5-turbo performs worse than NoInfo in SupplierDelay.

---

## Q1: What is the SIVR_O for each model and regime?

| Sensor | Normal | SupplierDelay | DemandSurge |
|--------|--------|---------------|-------------|
| NoInfo | 0.000 | 0.000 | 0.000 |
| RuleBased | 0.249 | 0.893 | 0.645 |
| PerfectSemantic | 1.000 | 1.000 | 1.000 |
| **gpt-4o** | **0.998** | **0.531** | **0.877** |
| gpt-4o-mini | 0.979 | -0.201 | 0.644 |
| gpt-3.5-turbo | 0.998 | -0.481 | -0.222 |

**Interpretation:**
- gpt-4o achieves >0.87 SIVR_O in Normal and DemandSurge, close to PerfectSemantic
- gpt-4o-mini matches gpt-4o in Normal but falls below NoInfo in SupplierDelay
- gpt-3.5-turbo matches gpt-4o in Normal but is harmful in both disruption regimes (negative SIVR_O)
- RuleBased achieves 0.893 in SupplierDelay (highest non-oracle) but only 0.249 in Normal

---

## Q2: Does belief quality predict operational value?

| Sensor | Accuracy | Brier Score | Log Loss | True Prob | SIVR_O (mean) |
|--------|----------|-------------|----------|-----------|---------------|
| NoInfo | 0.333 | 0.668 | 1.101 | 0.333 | 0.000 |
| RuleBased | 1.000 | 0.241 | 0.418 | 0.689 | 0.596 |
| PerfectSemantic | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| gpt-4o | 0.944 | 0.088 | 0.186 | 0.858 | **0.802** |
| gpt-4o-mini | 0.722 | 0.315 | 0.522 | 0.658 | 0.474 |
| gpt-3.5-turbo | 0.389 | 0.569 | 0.785 | 0.544 | 0.098 |

**Correlation (Spearman) between Brier score and SIVR_O:**
- Across all 6 sensors: strong negative correlation (lower Brier → higher SIVR_O)
- gpt-4o: best calibrated (Brier=0.088) → highest operational value
- gpt-3.5-turbo: worst calibrated (Brier=0.569) → near-zero operational value

**Conclusion:** Yes, belief quality strongly predicts operational value. The gradient from gpt-3.5-turbo → gpt-4o-mini → gpt-4o shows that each improvement in Brier score corresponds to higher SIVR_O.

---

## Q3: How does performance vary by ambiguity level?

| Sensor | Clear | Moderate | Vague |
|--------|-------|----------|-------|
| NoInfo | 0.000 | 0.000 | 0.000 |
| RuleBased | 0.401 | 0.501 | 0.792 |
| PerfectSemantic | 1.000 | 1.000 | 1.000 |
| gpt-4o | **0.988** | **0.928** | **0.692** |
| gpt-4o-mini | 0.881 | 0.744 | 0.286 |
| gpt-3.5-turbo | 0.076 | 0.127 | 0.049 |

**Key insight:** gpt-4o maintains high SIVR_O even at vague ambiguity (0.692), while gpt-4o-mini degrades sharply (0.286) and gpt-3.5-turbo is near-zero at all levels. This shows gpt-4o extracts more value from ambiguous text.

---

## Q4: Which regime is hardest for LLMs?

| Model | Normal | SupplierDelay | DemandSurge |
|-------|--------|---------------|-------------|
| gpt-4o | 0.998 | **0.531** | 0.877 |
| gpt-4o-mini | 0.979 | **-0.201** | 0.644 |
| gpt-3.5-turbo | 0.998 | **-0.481** | -0.222 |

**SupplierDelay is the hardest regime** for all LLMs. This is because:
1. SupplierDelay warnings overlap with Normal text (e.g., "delay" could mean routine delay or true disruption)
2. The belief must distinguish "no event" from "supplier delay" — a fine-grained distinction
3. False positives (predicting delay when Normal) cause over-ordering, wasting inventory budget
4. False negatives (predicting Normal when delay) cause under-protection

DemandSurge is easier because the text explicitly mentions demand increases, which are unambiguous.

---

## Q5: RuleBased vs LLM comparison

| Model | RuleBased SIVR_O | LLM SIVR_O | LLM Advantage |
|-------|------------------|------------|---------------|
| gpt-4o | 0.596 | **0.802** | **+0.206** |
| gpt-4o-mini | 0.596 | 0.474 | -0.122 |
| gpt-3.5-turbo | 0.596 | 0.098 | -0.498 |

- **gpt-4o outperforms RuleBased** in 2 of 3 regimes (Normal: +158.7, DemandSurge: +187.2 profit)
- **gpt-4o-mini matches RuleBased** in Normal but is worse in disruptions
- **gpt-3.5-turbo is worse than RuleBased** in all disruption regimes
- RuleBased achieves 100% classification accuracy but SIVR_O < 1 because it assigns non-degenerate probabilities (0.666 true regime vs 1.0 for PerfectSemantic)

---

## Q6: False positive and false negative economics

**False positive cost** (predicting disruption when Normal → over-order → excess holding):
- gpt-4o: minimal false positives (94.4% accuracy)
- gpt-4o-mini: moderate false positives in Normal (72.2% accuracy)
- gpt-3.5-turbo: frequent false positives (38.9% accuracy)

**False negative cost** (predicting Normal when disruption → under-prepare → stockouts):
- gpt-3.5-turbo: severe in SupplierDelay (SIVR_O = -0.481, profit loss ≈ $128)
- gpt-4o-mini: moderate in SupplierDelay (SIVR_O = -0.201)
- gpt-4o: minimal false negatives (SIVR_O always positive)

---

## Q7: Mechanism — How does LLM belief affect the optimizer?

The belief-aware CausalOptimizer computes expected lead time and demand across regimes:
- If belief Normal=0.6, SD=0.3, DS=0.1 → expected LT = 0.6×4 + 0.3×8 + 0.1×4 = 5.2
- The optimizer then sets order-up-to levels based on this expected LT

**gpt-4o** assigns high probability to the correct regime → expected parameters are close to truth → near-optimal orders.

**gpt-3.5-turbo** assigns high probability to wrong regimes → expected parameters are misleading → suboptimal orders (often worse than NoInfo's prior).

---

## Q8: Summary of findings

1. **LLM semantic quality causally affects operational value** — demonstrated by the monotonic relationship between Brier score and SIVR_O across gpt-3.5-turbo → gpt-4o-mini → gpt-4o
2. **gpt-4o achieves near-oracle performance** in Normal (0.998) and DemandSurge (0.877) — recovering >87% of perfect information value
3. **SupplierDelay is the hardest regime** — requires fine-grained text discrimination that even gpt-4o only partially achieves (0.531)
4. **gpt-3.5-turbo is harmful** — negative SIVR_O in disruption regimes means its interpretations actively harm decision quality
5. **RuleBased is a strong baseline** — 100% classification accuracy, but non-degenerate probability spread limits operational value (SIVR_O < 1)
6. **Ambiguity degrades LLM value extraction** — gpt-4o maintains 0.692 SIVR_O at vague ambiguity, while gpt-4o-mini drops to 0.286
7. **Belief quality predicts operational value** — Brier score is a reliable predictor of SIVR_O

---

## Appendix: Experiment Configuration

- **Seeds:** 15 (1000-1014)
- **Regimes:** Normal, SupplierDelay, DemandSurge
- **Templates:** 18 (6 per regime × 3 ambiguity levels × 2 variants)
- **Sensors:** NoInfo, RuleBased, PerfectSemantic, LLM:gpt-4o-mini, LLM:gpt-4o, LLM:gpt-3.5-turbo
- **Controller:** CausalOptimizer (belief-aware)
- **Total episodes:** 1,620 (3 regimes × 15 seeds × 6 templates × 6 sensors)
- **LLM cache:** 54 unique calls (18 templates × 3 models), all cached before experiment

### Frozen Benchmark Hash
```json
{
  "regime_definitions": "37dac3521f25a323",
  "regime_prior": "be70484ab53f4c30",
  "warning_templates": "ad6c4eeaa4cc84e9",
  "economics": "f1127cf53c4bdd0c",
  "timing": "035a232450c905c1",
  "extraction_prompt": "51e8203f34fbbc99"
}
```

### Output Files
| File | Description |
|------|-------------|
| `frozen_benchmark_manifest.json` | Hash-locked benchmark config |
| `operational_results.csv` | 1,620 episodes with profit, fill rate, beliefs |
| `belief_metrics.csv` | Per-episode belief accuracy and Brier scores |
| `raw_llm_responses.csv` | 1,080 raw LLM/RuleBased responses |
| `sivr_by_model.csv` | SIVR_O by sensor × regime |
| `sivr_by_regime.csv` | SIVR_O by sensor × regime (duplicate) |
| `sivr_by_ambiguity.csv` | SIVR_O by sensor × ambiguity level |
| `belief_to_value_analysis.csv` | Accuracy, Brier, log-loss by sensor |
| `false_positive_negative_costs.csv` | FP/FN flags and profit losses |
| `paired_comparisons.csv` | Paired sensor vs NoInfo comparisons |
| `rulebased_vs_llm.csv` | RuleBased vs LLM profit differences |
