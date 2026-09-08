> **Counts in this document are design intent and were repeatedly wrong.**
> The authoritative accounting is `docs/ACCOUNTING.md`, generated from the
> saved episode files by `python3 -m tools.generate_accounting --write`.
> Corrections applied September 2026: the controlled horizon is **40** periods
> (`P5_HORIZON`), not 30; the Phase-6 primary file holds **20** seeds, not 50;
> and Phase 7 ran **5** sensors over **3,600** episodes before the 2026 control
> arms were added, not 6 over 4,320. Where this document and `ACCOUNTING.md`
> disagree, `ACCOUNTING.md` is correct.

# Benchmark Design

## Overview

The benchmark consists of two complementary experiments that isolate and validate the relationship between semantic interpretation quality and operational decision value.

## Experiment A: Controlled Semantic-Decision Benchmark

### Purpose
Isolate the causal relationship between semantic belief quality and sequential operational value in a controlled environment where all confounds are held constant.

### Environment
- **Type:** Synthetic single-SKU inventory environment
- **Horizon:** 40 periods (`P5_HORIZON`)
- **States:** Inventory level, demand history, lead time status
- **Actions:** Order quantity
- **Disruptions:** SupplierLeadTimeIncrease (lead time 2→5 at t=18, duration 8)
- **Demand:** Poisson(λ=8), stationary except during disruption

### Regime Space
- **Normal:** No disruption expected
- **SupplierDelay:** Lead time increase at t~18
- **DemandSurge:** Not used in Experiment A

### Declared Regime Prior
- `REGIME_PRIOR`: Normal 0.35, SupplierDelay 0.35, DemandSurge 0.30.
- The corrected primary estimate is balanced over the constructed evaluation set; the prior-weighted estimate is secondary.

### Controllers
- **Heuristic policy:** Threshold-based disruption-aware controller (fixed)
- **CausalOptimizer:** Receding-horizon LP optimizer (for decomposition analysis)
- **HindsightOracle:** MILP with realized future demand (upper bound reference)

### Interpreters/Sensors
- NoInfo (prior only)
- RuleBased (keyword matching)
- gpt-3.5-turbo (LLM)
- gpt-4o-mini (LLM)
- gpt-4o (LLM)
- PerfectSemantic (ground truth)

### Key Design Properties
- Same-seed paired evaluation across all conditions
- All conditions use the identical controller (isolate belief quality)
- Warning text never affects simulator dynamics
- Interpreters receive only warning text (no simulator access)
- Controller receives only belief vector (no text or interpreter internals)

### Primary Metric
- **AggregateSIVR:** Fraction of OracleSemantic operational value recovered

### Sample Size
- 15 seeds x 18 templates x 6 sensors = 1,620 episodes (Phase 5.5)
- Phase 6 confirmation: see `docs/ACCOUNTING.md`. The saved primary file holds 20 seeds, not the 50 this line previously claimed.

---

## Experiment B: Operational-Complexity Transfer (Paper-1 Gymnasium)

### Purpose
Test whether the semantic-value phenomenon survives transfer to a richer multi-echelon supply chain environment with different dynamics, topology, and controller architecture.

### Environment
- **Package:** `gym-invmgmt` v0.1.0 (commit `a745fd5`)
- **Environment:** `GymInvMgmt/Serial-v0`
- **Topology:** RM(4) → Factory(3, C=100) → Dist(2) → Retail(1) → Market(0)
- **Lead times:** L = [0, 4, 4]
- **Capacity:** Factory C = 100
- **Horizon:** 40 periods (`P5_HORIZON`)
- **DemandSurge:** base_mu=10.0, shock_mag=2.0, shock_time=15

### Regime Space (2-class)
- **Normal:** No disruption (base_mu=10.0 throughout)
- **DemandSurge:** Demand shock at t=15 (base_mu=20.0 from t=15 onward)

### Declared NoInfo Prior
- Phase 8B NoInfo code uses P(Normal)=0.70 and P(DemandSurge)=0.30.
- The frozen Phase-8B set contains 12 Normal and 12 DemandSurge templates, so the primary benchmark estimate is balanced 50/50; the 70/30 result is reported separately.

### Controller
- **BeliefAdaptiveController:** Base-stock order-up-to policy
  - effective_mu = P(Normal) × base_mu + P(DemandSurge) × base_mu × surge_multiplier
  - target = effective_mu × avg_lead_time + safety_factor × sqrt(avg_lt × effective_mu)
  - order = max(0, target - inventory_position)
- Fixed across all conditions (isolate belief quality)

`OracleSemantic` means perfect semantic belief passed through this same fixed
controller. It is not an optimal policy, hindsight oracle, or guaranteed upper
bound on reward.

### Interpreters/Sensors
- NoInfo (prior only)
- RuleBased (keyword matching)
- TFIDF_LogReg_Raw (TF-IDF + LogReg, uncalibrated)
- TFIDF_LogReg_Calibrated (TF-IDF + LogReg, isotonic calibration)
- gpt-4o (LLM, cached predictions)
- OracleSemantic (ground truth)

### Key Design Properties
- Same-seed paired evaluation across all conditions
- All conditions use identical BeliefAdaptiveController
- Warning text never affects simulator dynamics
- Interpreters receive only warning text
- Controller reads only inventory state (env.X) and pipeline (env.Y) at current period
- No future-demand leakage: env.D is zero at reset, filled incrementally

### Validity Controls
- Held-out linguistic templates (24, never seen during TF-IDF training)
- Disjoint seed set (3100–3129, never used in any prior experiment)
- Regime-stratified hierarchical bootstrap CI (family × variant × paired seed)
- Corrected fill rate (retail sales, not replenishment orders)
- Frozen operational config (no retuning based on Phase-8B results)

### Sample Size
- 30 seeds x 24 templates x N sensors; 720 paired seed/template worlds per sensor. The episode total depends on the sensor list and is counted in `docs/ACCOUNTING.md`. 4,320 episodes across sensors is NOT 4,320 independent paired worlds.

### Relationship to Experiment A
- **Not a direct replication** — different environment, regime space, controller, and task
- **Operational-complexity transfer:** Tests whether the phenomenon (imperfect semantic beliefs have measurable operational value) generalizes to a richer operational setting
- Terminology: "operational-complexity transfer / replication" (not "exact replication")

---

## Estimands and evidence taxonomy

The primary benchmark estimand is balanced performance over the constructed
evaluation set: equal regime weight, then equal template family, variant, and
paired seed weight. The secondary deployment-prior estimand applies the prior
actually declared or passed to NoInfo. These quantities are reported
separately.

Phase 8B is evidence for held-out linguistic generalization and
operational-complexity replication. Phase 9A is evidence for cross-event
operational transfer; its original evaluation includes training templates, so
the test-template-only subset is the clean linguistic analysis. Phase 9B is a
descriptive robustness/boundary analysis of the fixed controller, not proof of
broad arbitrary-event or arbitrary-topology generalization.

## Experimental Hierarchy

```
Phase 5.5: Controlled benchmark (LLM evaluation)
    ↓ frozen
Phase 6: Confirmation with held-out templates
    ↓ validated
Phase 7: Classical NLP baseline (TF-IDF + LogReg)
    ↓ new sensor type
Phase 8A: Paper-1 Gym exploratory transfer
    ↓ validity audit
Phase 8B: Paper-1 Gym frozen confirmation
    ↓ publication primary
```

Each phase builds on frozen artifacts from prior phases. No phase modifies prior results.
