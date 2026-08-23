# Phase 8: Paper-1 Gymnasium Replication

## Summary

Phase 8 replicates the semantic decision-value experiment inside an external
inventory management environment (Paper-1's `GymInvMgmt/Serial-v0`), demonstrating
that the Information Value phenomenon transfers beyond the controlled Phase 6/7
single-item simulator.

## Environment

- **Platform**: Paper-1 `CoreEnv`, serial scenario (RM -> Factory(C=100) -> Dist -> Retail -> Market)
- **Actions**: 3 reorder links (one per echelon boundary)
- **Demand**: Poisson with `base_mu=10.0`, shock multiplier `2.0` starting at period 15
- **Horizon**: 30 periods (warning at t=0, shock at t=15)
- **Regime space**: 2-class {Normal, DemandSurge}

## Architecture

```
Warning text (12 templates: 6 surge + 6 normal, x {clear, moderate, vague})
  -> Semantic interpreter (5 sensors)
  -> 2-class belief [P(Normal), P(DemandSurge)]
  -> BeliefAdaptiveController (base-stock policy on Paper-1 env)
  -> Operational outcome (profit)
```

The `BeliefAdaptiveController` maps beliefs to a base-stock order-up-to policy:
- `effective_mu = P(Normal) * base_mu + P(DemandSurge) * base_mu * surge_multiplier`
- Per-node target = pipeline demand + safety stock, bounded by capacity

## Results (20 seeds x 12 templates = 240 episodes per sensor)

### Aggregate Information Value (SIVR)

| Sensor | AggSIVR | Mean Reward | Belief Accuracy | Brier Score |
|--------|---------|-------------|-----------------|-------------|
| NoInfo | 0.000 | 476.5 | 50.0% | 0.290 |
| RuleBased | 0.887 | 547.6 | 100.0% | 0.089 |
| gpt-4o | 0.874 | 546.6 | 100.0% | 0.016 |
| TFIDF_LogReg | 1.000 | 556.7 | 100.0% | 0.000 |
| PerfectSemantic | 1.000 | 556.7 | 100.0% | 0.000 |

### Paired Bootstrap CIs (vs NoInfo, 95%)

| Sensor | Mean Diff | 95% CI | Significant |
|--------|-----------|--------|-------------|
| RuleBased | +71.1 | [58.7, 83.5] | Yes |
| TFIDF_LogReg | +80.1 | [63.8, 96.6] | Yes |
| gpt-4o | +70.1 | [55.4, 84.9] | Yes |

### Win Rates (vs NoInfo)

| Sensor | Win Rate | Tie Rate | Pairs |
|--------|----------|----------|-------|
| RuleBased | 51.7% | 0.0% | 240 |
| TFIDF_LogReg | 55.0% | 0.0% | 240 |
| gpt-4o | 55.8% | 0.0% | 240 |
| PerfectSemantic | 55.0% | 0.0% | 240 |

### By Regime

| Regime | Sensor | Mean Reward | SIVR |
|--------|--------|-------------|------|
| Normal | NoInfo | 493.6 | 0.00 |
| Normal | RuleBased | 475.0 | -0.42 |
| Normal | TFIDF_LogReg | 443.8 | -0.80 |
| Normal | gpt-4o | 448.6 | -0.62 |
| Normal | PerfectSemantic | 443.8 | -0.80 |
| DemandSurge | NoInfo | 459.5 | 0.00 |
| DemandSurge | RuleBased | 620.2 | 0.77 |
| DemandSurge | TFIDF_LogReg | 669.5 | 1.00 |
| DemandSurge | gpt-4o | 644.6 | 0.88 |
| DemandSurge | PerfectSemantic | 669.5 | 1.00 |

### By Ambiguity Level

| Ambiguity | Sensor | Mean Reward | SIVR |
|-----------|--------|-------------|------|
| Clear | NoInfo | 476.5 | 0.00 |
| Clear | RuleBased | 555.4 | 0.27 |
| Clear | gpt-4o | 553.7 | 0.09 |
| Moderate | NoInfo | 476.5 | 0.00 |
| Moderate | RuleBased | 507.3 | -0.13 |
| Moderate | gpt-4o | 554.9 | 0.24 |
| Vague | NoInfo | 476.5 | 0.00 |
| Vague | RuleBased | 580.0 | 0.38 |
| Vague | gpt-4o | 531.2 | 0.08 |

## Key Findings

1. **Semantic information has positive operational value in Paper-1's Gymnasium
   environment.** All three interpreters (RuleBased, TFIDF_LogReg, gpt-4o) produce
   significantly higher rewards than NoInfo (all 95% CIs exclude zero).

2. **Aggregate SIVR replicates the Phase 6/7 hierarchy.** The same ordering emerges:
   `NoInfo < gpt-4o ≈ RuleBased < TFIDF_LogReg = PerfectSemantic`. The classical NLP
   model achieves perfect information value on these training templates.

3. **Value comes from the demand surge regime.** On Normal episodes, information sensors
   actually underperform NoInfo (negative SIVR) because they correctly reduce inventory
   when no surge occurs, but the specific Poisson draws sometimes require more stock.
   On DemandSurge episodes, they substantially outperform. The aggregate SIVR is positive
   because the surge gains outweigh the normal losses.

4. **Win rates are modest (~55%) but effect sizes are large.** Individual episodes have
   high variance (Poisson demand), but the systematic advantage of correct belief
   produces large aggregate effects with tight confidence intervals.

5. **The Phase 7 finding survives in a different environment.** Calibration matters
   (gpt-4o has lower Brier than RuleBased despite similar SIVR), and raw accuracy does
   not perfectly predict operational value.

## Differences from Phase 6/7

| Aspect | Phase 6/7 | Phase 8 |
|--------|-----------|---------|
| Environment | Custom single-item P5 simulator | Paper-1 Gymnasium serial chain |
| Actions | Single order quantity | 3 reorder quantities (per echelon) |
| Regime space | 3-class {Normal, Delay, Surge} | 2-class {Normal, Surge} |
| Templates | 18 original + 36 confirmation | 12 from original (surge + normal) |
| Controller | CausalOptimizer | BeliefAdaptiveController (base-stock) |
| Inventory policy | Optimization-based | Base-stock with safety stock |

## Output Files

- `operational_results.csv` - 1200 rows (20 seeds x 12 templates x 5 sensors)
- `aggregate_sivr.csv` - Aggregate SIVR per sensor
- `belief_metrics.csv` - Brier scores per sensor
- `paired_comparisons.csv` - Paired bootstrap CIs vs NoInfo
- `regime_summary.csv` - Results broken down by regime
- `ambiguity_summary.csv` - Results broken down by ambiguity level
- `win_rates.csv` - Pairwise win rates vs NoInfo
- `experiment_manifest.json` - Full experiment configuration
