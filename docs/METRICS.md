# Metrics Reference

## Primary Metrics

### Oracle Information Value (OIV)

**Definition:**
```
OIV = V(OracleSemantic) - V(NoInfo)
```

**Interpretation:** The operational reference value of perfect semantic belief passed through the same fixed controller. It is not an optimal-policy or hindsight upper bound.

**Units:** Same as V (profit or reward, depending on environment).

**Sign convention:** Positive means the Oracle-Belief Reference outperforms NoInfo; negative means the fixed controller/environment combination performs worse with that belief.

**Edge cases:** If `|OIV| <= epsilon`, SIVR is `NaN` with status `ZERO_OR_NEAR_ZERO_REFERENCE_VALUE`. If OIV is negative, the signed diagnostic is retained with status `NEGATIVE_ORACLE_REFERENCE_VALUE`; it is not interpreted as information-value recovery.

**Canonical path:** `src.metrics.signed_sivr` and `src.metrics.information_value`.

---

### Semantic Information Value Recovery (SIVR)

**Definition:**
```
SIVR(M) = [V(M) - V(NoInfo)] / [V(OracleSemantic) - V(NoInfo)]
```

where V(·) denotes the expected operational value (profit/reward) of a given sensor/interpreter M.

**Aggregate SIVR:**
```
AggregateSIVR(M) = [E[V(M)] - E[V(NoInfo)]] / [E[V(OracleSemantic)] - E[V(NoInfo)]]
```

Computed over paired seeds. E[·] is the mean over operational seeds and templates.

**Interpretation:** When OIV is positive, what fraction of the Oracle-Belief Reference's incremental operational value does sensor M recover? It is a secondary normalization; raw reward and regime-specific deltas are primary.

**Sign convention:**
- SIVR = 0: Sensor provides no operational value
- 0 < SIVR < 1: Sensor partially recovers oracle value
- SIVR = 1: Sensor fully recovers oracle value
- SIVR < 0: Sensor actively harms decisions relative to NoInfo
- **SIVR > 1: Imperfect semantic beliefs interact with a misspecified controller to outperform the oracle-semantic reference**

**SIVR is NOT bounded to [0,1].** Values > 1 can occur because:
1. OracleSemantic is a reference point (perfect beliefs + heuristic controller), not an upper bound on policy performance
2. An imperfect belief that happens to interact more favorably with the controller's operating point can exceed the oracle reference
3. The controller is fixed and not globally optimal — its interaction with different belief distributions is non-monotonic

**Canonical path:** `src.metrics.signed_sivr(j_condition, j_noinfo, j_perfect_semantic)`; the denominator is signed, never absolute-valued.

---

### Semantic Regret

**Definition:**
```
SemanticRegret = V(OracleSemantic) - V(M)
```

**Interpretation:** Operational value lost due to imperfect semantic interpretation. Includes both misclassification effects and probability calibration errors.

**Sign convention:** Positive means M underperforms OracleSemantic. Negative means M outperforms (see SIVR > 1 discussion).

**Canonical path:** `src.metrics.semantic_regret(j_llm, j_perfect_semantic)`

---

### Controller Gap

**Definition:**
```
ControllerGap = V(CausalOptimizer) - V(OracleSemantic)
```

**Interpretation:** Additional value available from using a stronger operational controller (CausalOptimizer) with the same (perfect) semantic information. Measures the suboptimality of the heuristic policy.

**Sign convention:** Positive means the controller can improve over the heuristic with the same information. Negative means the heuristic outperforms the optimizer (possible in finite-horizon settings).

**Canonical path:** `src.metrics.controller_gap(j_causal_optimizer, j_perfect_semantic)`

**Note:** Controller Gap is defined in the controlled benchmark (Experiment A) only. In Phase 8B, the controller is the BeliefAdaptiveController and no CausalOptimizer is computed.

---

### Hindsight Advantage

**Definition:**
```
HindsightAdvantage = V(HindsightOracle) - V(CausalOptimizer)
```

**Interpretation:** Value of perfect future knowledge beyond what a strong optimizer can achieve. This is the irreducible operational uncertainty.

**Sign convention:** Non-negative. Measures the fundamental limit of causal decision-making under uncertainty.

**Canonical path:** `src.metrics.hindsight_advantage(j_hindsight, j_causal_optimizer)`

---

### Total Gap Decomposition

**Definition:**
```
TotalGap = V(HindsightOracle) - V(M)
         = SemanticRegret + ControllerGap + HindsightAdvantage
```

**Interpretation:** Complete decomposition of the gap between a given sensor's performance and the theoretical optimum (hindsight).

---

## Belief Quality Metrics

### Brier Score

**Definition:**
```
Brier(M) = E[ Σ_k (p_k - y_k)^2 ]
```

where `y` is the one-hot regime vector. A true-class-only squared error is not reported as standard Brier.

**Interpretation:** Mean squared error of probabilistic predictions. Lower is better. Ranges from 0 (perfect) to 1 (worst for binary, 2 for 3-class).

**Canonical path:** `src.metrics.standard_brier_score`, used by the phase runners and corrected audit.

---

### Log-Loss (Cross-Entropy)

**Definition:**
```
LogLoss(M) = -E[log(p_correct)]
```

**Interpretation:** Penalizes confident wrong predictions heavily. Lower is better. Unbounded above. Zero for perfect predictions.

**Canonical path:** Computed per sensor alongside Brier score.

---

### Calibration Metric (ECE)

**Definition:**
```
ECE = (1/N) Σ_k |B_k| * |acc(B_k) - conf(B_k)|
```

where B_k are equal-width confidence bins, acc(·) is the fraction of correct predictions in the bin, and conf(·) is the mean predicted confidence.

**Interpretation:** How well do predicted probabilities match empirical accuracy? ECE = 0 means perfectly calibrated.

---

## Statistical Methods

### Paired Bootstrap

**Definition:** For two conditions A and B evaluated on paired seeds:
```
Δ_i = V(A, seed_i) - V(B, seed_i)
mean_diff = E[Δ]
CI = percentile(bootstrap resamples of Δ, [α/2, 1-α/2])
```

**Use:** Seed-level uncertainty for paired comparisons.

**Canonical path:** `src.experiment_phase8b.paired_bootstrap_ci(data_a, data_b, n_boot=10000, alpha=0.05, seed=42)`

---

### Hierarchical Paired Bootstrap

**Definition:** Regime-stratified family/variant/seed resampling:
```
For each bootstrap iteration b:
  1. Apply the declared regime weights (equal weights for the primary estimand; deployment prior for the secondary estimand)
  2. Within each regime, sample template families with replacement
  3. Within each selected family, sample variants with replacement
  4. Within each selected variant, sample paired seeds with replacement
  5. Compute the weighted paired mean difference
CI = percentile over bootstrap distribution
```

**Use:** Primary uncertainty quantification in Phase 8B. Accounts for the hierarchical structure (templates nested within families, seeds nested within templates).

**Significance criterion:** CI excludes zero.

**Canonical path:** `src.metrics.family_seed_bootstrap_ci`.

---

## Fill Rate

**Definition:**
```
Fill Rate = Σ(retail_sales) / Σ(customer_demand)
```

where retail_sales are recorded at the retail link using `env.S[:, retail_link_idx]`.

**Interpretation:** Fraction of customer demand satisfied from stock. Ranges from 0 (complete stockout) to 1 (all demand met). Can exceed 1 temporarily if pipeline inventory is consumed.

**Canonical path:** `src/experiment_phase8b.run_gym_episode_corrected_fill_rate()` — uses `env.S` (retail sales), not `env.R` (replenishment orders).

---

## Notes

- All SIVR/AggregateSIVR values are computed over paired same-seed comparisons
- OracleSemantic is the normalization reference, not HindsightOracle
- HindsightOracle is the theoretical upper bound on policy performance (uses realized future demand), not the semantic reference
- The controller (BeliefAdaptiveController or heuristic policy) is intentionally fixed across conditions to isolate the effect of semantic quality
