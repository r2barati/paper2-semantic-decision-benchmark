> **Superseded in part (September 2026).** This section describes an earlier
> design. The controlled experiment now uses **three** regimes (normal,
> supplier delay, demand surge), a **40**-period horizon, and the
> `CausalOptimizer` receding-horizon controller for the primary comparison ---
> not a fixed heuristic policy. Counts are in `docs/ACCOUNTING.md`.

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

The primary controlled comparison uses `CausalOptimizer` (receding-horizon LP) for all conditions; the belief is the only thing that varies. HindsightOracle is a clairvoyant reference used for decomposition, not for SIVR. (This line previously described a fixed heuristic policy, which does not match `src/experiment_phase7.py`.)

## Key Results (Phase 7)

| Sensor | AggregateSIVR |
|--------|-------------:|
| RuleBased | 0.707 |
| gpt-4o | 0.767 |
| TFIDF_LogReg_Raw | 0.250 |
| TFIDF_LogReg_Calibrated | 0.648 |

Calibrated TF-IDF (0.648) substantially outperforms raw TF-IDF (0.250) despite lower classification accuracy (86.1% vs 88.9%), demonstrating that probability calibration materially changes downstream value.
