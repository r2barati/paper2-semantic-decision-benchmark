# Benchmark Design

## Overview

The benchmark consists of two complementary experiments that isolate and validate the relationship between semantic interpretation quality and operational decision value.

## Experiment A: Controlled Semantic-Decision Benchmark

### Purpose
Isolate the causal relationship between semantic belief quality and sequential operational value in a controlled environment where all confounds are held constant.

### Environment
- **Type:** Synthetic single-SKU inventory environment
- **Horizon:** 30 periods
- **States:** Inventory level, demand history, lead time status
- **Actions:** Order quantity
- **Disruptions:** SupplierLeadTimeIncrease (lead time 2→5 at t=18, duration 8)
- **Demand:** Poisson(λ=8), stationary except during disruption

### Regime Space
- **Normal:** No disruption expected
- **SupplierDelay:** Lead time increase at t~18
- **DemandSurge:** Not used in Experiment A

### Regime Prior
- P(Normal) = 0.70
- P(SupplierDelay) = 0.30

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
- 50 seeds x 36 templates x 6 sensors = 10,800 episodes (Phase 6 confirmation)

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
- **Horizon:** 30 periods
- **DemandSurge:** base_mu=10.0, shock_mag=2.0, shock_time=15

### Regime Space (2-class)
- **Normal:** No disruption (base_mu=10.0 throughout)
- **DemandSurge:** Demand shock at t=15 (base_mu=20.0 from t=15 onward)

### Regime Prior
- P(Normal) = 0.70
- P(DemandSurge) = 0.30

### Controller
- **BeliefAdaptiveController:** Base-stock order-up-to policy
  - effective_mu = P(Normal) × base_mu + P(DemandSurge) × base_mu × surge_multiplier
  - target = effective_mu × avg_lead_time + safety_factor × sqrt(avg_lt × effective_mu)
  - order = max(0, target - inventory_position)
- Fixed across all conditions (isolate belief quality)

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
- Hierarchical bootstrap CI (template-family × seed resampling)
- Corrected fill rate (retail sales, not replenishment orders)
- Frozen operational config (no retuning based on Phase-8B results)

### Sample Size
- 30 seeds x 24 templates x 6 sensors = 4,320 episodes

### Relationship to Experiment A
- **Not a direct replication** — different environment, regime space, controller, and task
- **Operational-complexity transfer:** Tests whether the phenomenon (imperfect semantic beliefs have measurable operational value) generalizes to a richer operational setting
- Terminology: "operational-complexity transfer / replication" (not "exact replication")

---

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
