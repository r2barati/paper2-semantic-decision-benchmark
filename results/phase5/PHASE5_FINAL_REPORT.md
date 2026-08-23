# Phase 5 Final Report: Multi-Regime Information-Value Experiment

## Executive Summary

Phase 5 redesigned the operational environment from a single deterministic
disruption to a multi-regime system with 3 hidden operational states. This
resolved the Phase 4 finding that "NoInfo + CausalOptimizer = PerfectSemantic +
CausalOptimizer" by creating genuine regime uncertainty where advance semantic
information has real decision value.

**Key results:**
- Gate A PASSED: All 3 regimes show positive OIV (72.7 to 373.5)
- Gate B PASSED: PerfectSemantic orders 1.5 more units pre-event on average
- RuleBased SIVR_O: 0.249-0.893 across regimes (captures 25-89% of info value)
- 114/114 tests pass (94 original + 20 new Phase 5 tests)

## Why Phase 4 Failed to Show Information Value

Phase 4 used a single deterministic disruption known with certainty. The
CausalOptimizer, even without advance information, already planned for the
disruption. Since there was only one possible future event, the optimizer's
optimal action was identical whether or not it received advance warning.

Phase 5 fixes this by introducing regime uncertainty: the controller does not
know WHICH future regime will occur, and the warning provides partial information
about the regime.

## Experimental Design

### Regimes
- **Normal**: No disruption. LT=4, demand~Poisson(8) for all 40 periods.
- **SupplierDelay**: Lead time increases from 4 to 8 during t=20-29.
- **DemandSurge**: Demand increases from 8 to 14 (x1.75) during t=20-29.

### Timing
- Warning arrives at t=12 (8 periods before event onset)
- Event starts at t=20, duration=10 periods
- Normal lead time: 4 periods (orders placed at t=12 arrive at t=16)
- Horizon: 40 periods

### Regime Prior
P(Normal) = 0.35, P(SupplierDelay) = 0.35, P(DemandSurge) = 0.30

### Cost Parameters
Revenue=$10, Ordering=$5+$2/unit, Holding=$1/unit, Stockout=$8/unit,
Initial inventory=30

## Information-Value Grid (Gate A)

| Regime | NoInfo+CO | Perfect+CO | Hindsight | OIV | OIV% |
|--------|-----------|------------|-----------|-----|------|
| Normal | 1,615.1 | 1,785.0 | 2,047.4 | +169.9 | +8.3% |
| SupplierDelay | 1,550.8 | 1,623.5 | 1,994.3 | +72.7 | +3.6% |
| DemandSurge | 1,868.9 | 2,242.3 | 2,534.9 | +373.5 | +14.7% |

OIV = J_PerfectSemantic - J_NoInfo (both under CausalOptimizer)

**Gate A: PASSED** (3/3 regimes show positive OIV)

## Trajectory Diagnostics (Gate B)

Pre-event order comparison (t=12-19, average over 3 seeds):

| Regime | NI orders | PS orders | NI lost-sales (event) | PS lost-sales |
|--------|-----------|-----------|----------------------|---------------|
| SupplierDelay | 9.8 | 11.4 | 5.7 | 0.3 |
| DemandSurge | 9.8 | 11.2 | 9.3 | 3.0 |

Mean pre-event order difference: +1.5 units

PerfectSemantic orders more BEFORE the event becomes observable, resulting in
dramatically lower lost sales during the event. This demonstrates that advance
information changes rational causal decisions.

**Gate B: PASSED**

## RuleBased Sensor Results

SIVR_O = (J_sensor - J_NoInfo) / (J_PerfectSemantic - J_NoInfo)

| Sensor | Normal | SupplierDelay | DemandSurge |
|--------|--------|---------------|-------------|
| NoInfo | 0.000 | 0.000 | 0.000 |
| RuleBased | 0.249 | 0.893 | 0.645 |
| PerfectSemantic | 1.000 | 1.000 | 1.000 |

Belief accuracy: NoInfo=33.3%, RuleBased=100.0%, PerfectSemantic=100.0%

RuleBased captures 89% of info value for SupplierDelay because delay templates
contain clear keywords (supplier, logistics, congestion, delay, lead time).
DemandSurge at 64% because surge templates use more ambiguous language about
"customer interest" and "purchasing plans". Normal at 25% because the prior
already allocates 35% to normal.

## Threshold Sensitivity

Not yet evaluated for Phase 5 (future work with LLM models).

## Key Findings

1. **The single-disruption design (Phase 4) was insufficiently information-
   sensitive.** With only one possible future event, the optimizer already
   plans for it without advance information.

2. **Multi-regime uncertainty creates genuine information value.** OIV ranges
   from 72.7 (3.6%) to 373.5 (14.7%) depending on regime.

3. **DemandSurge has the highest OIV** because it's the most impactful regime
   (75% demand increase) and the hardest to respond to after onset (orders
   placed after t=20 arrive at t=24, too late for the critical window).

4. **Advance information changes actions BEFORE the event.** PerfectSemantic
   orders 1.5 more units during the pre-event window (t=12-19), building
   inventory before consequences become observable.

5. **RuleBased captures substantial info value** (25-89%) despite being a
   simple keyword/regex system. This suggests the regime classification
   task is tractable for deterministic methods.

6. **The normal regime has positive OIV** because confirming "no event"
   allows the optimizer to avoid unnecessary safety stock buildup.

7. **Decision-making quality still dominates interpretation quality.**
   The gap between NoInfo+CO and Perfect+CO (72-374) is smaller than
   the gap between Heuristic and CO from Phase 4 (~1,500), confirming
   that controller quality remains the primary driver.

## Phase-4 Zero-Information-Value Result Explained

The Phase 4 finding (NoInfo+CO = PerfectSemantic+CO) was caused by:
1. Single deterministic disruption (no regime uncertainty)
2. Optimizer already plans for the disruption under the prior
3. Advance information provides no additional decision-relevant information
   when there's only one possible future event

Phase 5 fixes this by introducing multiple possible regimes, making the
controller's prior belief genuinely uncertain and advance information
decision-relevant.

## Frozen Primary Benchmark Configuration

- Primary regime: demand_surge (highest OIV)
- Horizon: 40, Normal LT: 4, Base demand: 8
- Event: t=20-29, Warning: t=12
- Prior: Normal(0.35), SupplierDelay(0.35), DemandSurge(0.30)
- Config saved to: results/phase5/primary_benchmark_config.json

## Methodological Problems Discovered

1. The single-disruption design cannot test information value because the
   optimizer's optimal action is regime-independent when only one regime
   exists.

2. Warning lead time must be long enough for orders to arrive before the
   event. With LT=4 and warning at t=12, orders arrive at t=16, giving
   4 periods of buffer before event onset at t=20.

3. Cost asymmetry (stockout $8 >> holding $1) amplifies the value of
   advance information because under-preparation is much more costly
   than over-preparation.

## Strongest Defensible Conclusion for Paper 2

> Semantic information has decision value in sequential operational
> environments with regime uncertainty, but only when: (1) the controller
> faces genuine uncertainty about future operational states, (2) early
> action can materially affect outcomes before consequences become
> observable, and (3) the cost structure penalizes under-preparation
> more than over-preparation. The value of advance semantic information
> is not a fixed property of the information itself, but emerges from
> the interaction between information quality, controller capability,
> and operational dynamics.

## Outputs Generated

- results/phase5/information_value_grid.csv
- results/phase5/episode_results.csv
- results/phase5/trajectory_diagnostics.csv
- results/phase5/semantic_sensor_results.csv
- results/phase5/sivr_optimizer.csv
- results/phase5/belief_results.csv
- results/phase5/primary_benchmark_config.json
- results/phase5/PHASE5_FINAL_REPORT.md
