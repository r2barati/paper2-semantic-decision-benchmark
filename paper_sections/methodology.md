# Methodology

## Framework Overview

We introduce a framework for evaluating the decision value of probabilistic semantic sensing in executable sequential operational systems. The framework separates semantic interpretation from operational control, enabling principled measurement of how information quality affects downstream decisions.

## Core Design Principle

The central insight is the **perception-control separation**: semantic interpreters (LLMs, classifiers, rule-based systems) produce probabilistic beliefs about the state of the world, while operational controllers convert these beliefs into actions. By holding the controller fixed and varying only the interpreter, we estimate realized value under a specified belief-to-control mapping; this is not controller-independent information value.

## Semantic Information Value Recovery (SIVR)

The primary metric measures what fraction of an oracle-semantic reference's operational value a given sensor recovers:

SIVR(M) = [V(M) - V(NoInfo)] / [V(OracleSemantic) - V(NoInfo)]

where V(·) denotes expected operational value. The denominator is signed. If OIV is near zero, SIVR is undefined (`NaN`, status `ZERO_OR_NEAR_ZERO_REFERENCE_VALUE`). If OIV is negative, the signed diagnostic is retained but is not interpreted as information-value recovery. Values above 1 remain possible because `OracleSemantic` is a fixed-controller reference, not an upper bound.

## Decomposition

Total gap between a sensor and the hindsight optimum decomposes as:

TotalGap = SemanticRegret + ControllerGap + HindsightAdvantage

This separates the loss due to imperfect interpretation from the loss due to controller suboptimality and from the irreducible uncertainty of the operational environment.

## Statistical Inference

All comparisons use paired same-seed evaluation. The primary estimand is balanced benchmark performance; a declared deployment-prior estimate is secondary. Primary uncertainty uses regime-stratified resampling of template families, variants, and paired seeds. Simultaneous sensor comparisons use Holm-adjusted p-values; Phase 9B is reported as descriptive robustness/boundary analysis.

Belief quality is reported with the conventional multiclass Brier score,
`sum_k (p_k-y_k)^2`, together with accuracy and log-loss. `OracleSemantic`
means perfect semantic belief passed through the same fixed controller; it is
not an optimal policy, hindsight oracle, or guaranteed reward upper bound.
