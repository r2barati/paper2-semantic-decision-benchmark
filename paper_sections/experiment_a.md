# Experiment A: Controlled Semantic-Decision Benchmark

## Purpose

Isolate the causal relationship between semantic belief quality and sequential operational value in a controlled environment where all confounds are held constant.

## Environment

Single-SKU inventory system with Poisson demand (λ=8), 30-period horizon, and a supplier lead-time disruption (lead time 2→5 at t=18, duration 8). The disruption is a latent regime that must be inferred from textual warning signals.

## Regime Space

Three regimes: Normal (no disruption), SupplierDelay (lead-time increase). Regime prior: P(Normal)=0.70, P(SupplierDelay)=0.30.

## Warning Text

18 original templates (3 regimes × 2 ambiguity levels × 3 clarity levels) and 36 confirmation templates with genuinely different wording, created before any LLM evaluation.

## Interpreters

- **NoInfo:** Uses prior probabilities only (no text interpretation)
- **RuleBased:** Keyword-matching extractor
- **TF-IDF + Logistic Regression:** Trained on 18 original templates, evaluated on 36 held-out templates
- **gpt-4o, gpt-4o-mini, gpt-3.5-turbo:** LLM-based interpretation
- **PerfectSemantic/OracleSemantic:** Ground-truth regime (bypasses text)
- **HindsightOracle:** Perfect future demand knowledge (theoretical upper bound)
- **CausalOptimizer:** Receding-horizon LP with perfect semantic information

## Controller

Fixed heuristic policy for all conditions. CausalOptimizer and HindsightOracle use stronger policies but are only used for decomposition analysis, not for SIVR computation.

## Key Results (Phase 7)

| Sensor | AggregateSIVR |
|--------|-------------:|
| RuleBased | 0.707 |
| gpt-4o | 0.767 |
| TFIDF_LogReg_Raw | 0.250 |
| TFIDF_LogReg_Calibrated | 0.648 |

Calibrated TF-IDF (0.648) substantially outperforms raw TF-IDF (0.250) despite lower classification accuracy (86.1% vs 88.9%), demonstrating that probability calibration materially changes downstream value.
