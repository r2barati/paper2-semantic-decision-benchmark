# Limitations

The following are scope boundaries of this work, not fatal weaknesses.

## Synthetic Warning Text

All warning templates are synthetically generated, not extracted from real supplier emails or news feeds. The linguistic diversity is bounded by the template generation process. Real-world warning text would include noise, formatting artifacts, and ambiguous intent not captured here.

## Synthetic Inventory Environments

Both the controlled benchmark and Paper-1 Gymnasium are synthetic environments. Results do not demonstrate production deployment performance. The environments capture essential inventory dynamics (lead times, demand uncertainty, stockouts) but omit real-world complications (multiple suppliers, quality issues, pricing).

## Single SKU

The benchmark evaluates single-SKU inventory management. Multi-SKU interactions, substitution effects, and allocation decisions are not considered.

## Fixed Deterministic Lead Times

In the Paper-1 Gymnasium environment, lead times are fixed and deterministic (L=[0,4,4]). Stochastic lead times would introduce additional uncertainty that interacts differently with semantic information.

## Limited Regime Space

- Experiment A: 2 regimes (Normal, SupplierDelay)
- Experiment B: 2 regimes (Normal, DemandSurge)

The benchmark does not evaluate regime discrimination with more classes or continuous-valued disruption severity.

## Finite Template Families

Training uses 18 templates; held-out testing uses 36 (Phase 7) or 24 (Phase 8B). While the templates are genuinely distinct in wording, they share common structure and vocabulary. Real-world generalization across entirely different communication styles is not tested.

## LLM Coverage

Only three LLMs are evaluated: gpt-3.5-turbo, gpt-4o-mini, and gpt-4o. Results may differ with other models, particularly open-source or domain-specific LLMs.

## Fixed Operational Controller

The controller is intentionally fixed across conditions to isolate belief quality effects. This means:
- The controller is not globally optimal
- Its interaction with different belief distributions is non-monotonic (enabling SIVR > 1)
- Real-world controller design would co-optimize with the interpreter

## OracleSemantic Is Not a Policy-Performance Upper Bound

OracleSemantic provides the semantic reference point (perfect beliefs + fixed controller), not the best achievable policy performance. HindsightOracle and CausalOptimizer can exceed OracleSemantic. SIVR > 1 is possible and does not indicate a measurement error.

## Phase 8B Regime Space Differs from Phase 7

Phase 8B uses 2-class (Normal, DemandSurge) while Phase 7 uses 3-class (Normal, SupplierDelay, DemandSurge). Direct numerical comparison across experiments requires caution.

## No Competitors

The benchmark compares interpreters/sensors, not competing system designs. Results show the value of semantic information quality given a fixed controller architecture, not the best achievable system.
