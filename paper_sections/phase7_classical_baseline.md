# Phase 7: Classical NLP Baseline

## Motivation

Establish a non-LLM baseline to contextualize LLM performance and test whether classical NLP methods can compete in semantic regime interpretation.

## Method

TF-IDF vectorization (max_features=500, ngram_range=(1,2), sublinear_tf=True) followed by Multinomial Logistic Regression (C=1.0, solver=lbfgs). Calibrated variant uses isotonic regression with 3-fold cross-validation on the training set.

## Dataset

- **Training:** 18 original Phase-5 templates (6 per regime × 3 ambiguity levels)
- **Testing:** 36 held-out confirmation templates (12 per regime × 3 ambiguity levels)
- **No template ID overlap** between train and test
- **Families shared** (delay/normal/surge) — model must generalize to new phrasings

## Classification Results

| Metric | Raw | Calibrated |
|--------|----:|-----------:|
| Accuracy | 88.9% | 86.1% |
| Brier | 0.552 | 0.240 |
| LogLoss | 0.932 | 0.397 |

## Operational Results (Phase 7)

| Sensor | AggregateSIVR |
|--------|-------------:|
| NoInfo | 0.000 |
| RuleBased | 0.707 |
| gpt-3.5-turbo | -0.206 |
| gpt-4o | 0.767 |
| TFIDF_Raw | 0.250 |
| TFIDF_Calibrated | 0.648 |

## Key Finding

Calibration transforms the classical baseline from a weak operational performer into a competitive one. The calibrated model achieves AggregateSIVR = 0.648 despite lower classification accuracy than the raw model (86.1% vs 88.9%). This demonstrates that classification accuracy alone does not predict downstream operational value.
