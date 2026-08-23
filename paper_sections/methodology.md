# Methodology

## Framework Overview

We introduce a framework for evaluating the decision value of probabilistic semantic sensing in executable sequential operational systems. The framework separates semantic interpretation from operational control, enabling principled measurement of how information quality affects downstream decisions.

## Core Design Principle

The central insight is the **perception-control separation**: semantic interpreters (LLMs, classifiers, rule-based systems) produce probabilistic beliefs about the state of the world, while operational controllers convert these beliefs into actions. By holding the controller fixed and varying only the interpreter, we isolate the operational value of semantic information quality.

## Semantic Information Value Recovery (SIVR)

The primary metric measures what fraction of an oracle-semantic reference's operational value a given sensor recovers:

SIVR(M) = [V(M) - V(NoInfo)] / [V(OracleSemantic) - V(NoInfo)]

where V(·) denotes expected operational value. SIVR = 0 means no value recovered; SIVR = 1 means full recovery. Critically, SIVR is not bounded to [0,1] — values above 1 are possible when imperfect beliefs interact favorably with a misspecified controller.

## Decomposition

Total gap between a sensor and the hindsight optimum decomposes as:

TotalGap = SemanticRegret + ControllerGap + HindsightAdvantage

This separates the loss due to imperfect interpretation from the loss due to controller suboptimality and from the irreducible uncertainty of the operational environment.

## Statistical Inference

All comparisons use paired same-seed evaluation. Primary uncertainty quantification uses hierarchical paired bootstrap resampling (template-family × seed), which respects the hierarchical structure of the experimental design.
