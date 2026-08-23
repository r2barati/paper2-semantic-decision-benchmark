# Phase 8 Validity Audit

**Date**: 2026-08-22
**Auditor**: automated
**Scope**: candidate operational-complexity transfer experiment
**Preserved artifacts**: `results/phase8_gym_replication/` (CSVs, report) are NOT overwritten

---

## 1. Environment Provenance

| Field | Value |
|-------|-------|
| Package | `gym-invmgmt` |
| Version | `0.1.0` |
| Git commit | `a745fd5186a73d177dd94d283a4f8f8e8d329977` ("Initial public release") |
| Source path (runtime) | `/Users/sanamimani/Paper 1/final-github-clean/gym-invmgmt-paper/gym_invmgmt/__init__.py` |
| CoreEnv path | `/Users/sanamimani/Paper 1/final-github-clean/gym-invmgmt-paper/gym_invmgmt/core_env.py` |
| Install mode | editable (`pip install -e`) |
| Author | Reza Barati <r2barati@torontomu.ca> |
| Homepage | https://github.com/r2barati/gym-invmgmt-paper |

**Other copies on disk** (potential import confusion):

| Path | Resolves to this? |
|------|-------------------|
| `Paper 1/final-github-clean/gym-invmgmt-paper/` | **YES** (installed editable) |
| `Paper 1/final-github-clean/gym-invmgmt/` | NO (different package dir, not installed) |
| `Paper 1/paper-10/gym_invmgmt/` | NO (not installed) |
| `Documents/ChatGPT/Paper 2/paper1/gym-invmgmt-paper/` | NO (not installed) |

**Risk**: Low. Only one copy is installed. Python resolves to the correct path. No stale import risk unless someone installs a different copy.

---

## 2. Information Leakage

### 2a. Active code paths — CLEAN

| Component | Reads `env.D`? | Reads `env.demand_engine`? | Reads future t indices? | Verdict |
|-----------|---------------|---------------------------|------------------------|---------|
| `BeliefAdaptiveController.__init__` | No | No | No | Clean |
| `_precompute_params` | No | No | No | Clean |
| `get_action` | No | No | `env.X[env.period]`, `env.Y[env.period]` — current only | Clean |
| `run_gym_episode` loop | No | No | No | Clean |
| `_get_sensor_belief` (all sensors) | No | No | No | Clean |

### 2b. `env.D` verification

- `env.D` shape `(30,1)`, allocated as zeros in `_RESET()`
- Filled one row per step inside `_STEP()`: `self.D[t, i] = demand_val`
- At t=0 after reset: **zero nonzero entries**. Verified empirically.

### 2c. `env.demand_engine` — latent oracle channel (UNUSED)

`env.demand_engine.get_current_mu(t)` is a deterministic callable that returns the exact mean at any future t. At t=0, calling `mu(15)` returns 20.0 for surge envs, 10.0 for normal envs. This would constitute perfect clairvoyance if accessed.

**`BeliefAdaptiveController` does NOT call this.** But the controller holds `self.env`, which means any future code change adding a single line like `env.demand_engine.get_current_mu(15)` would silently introduce perfect leakage.

**Recommendation**: strip `self.env` from the controller or pass only `X`/`Y` rows + topology. This is a structural hardening, not a fix for current invalidity.

### 2d. `env.env_kwargs['demand_config']`

Stores a deepcopy including `effects`, `shock_time`, `shock_mag`. Accessible at t=0. Again, **not read by current controller**, but latent.

**Verdict: No active leakage. Phase 8A is valid on this audit dimension.**

---

## 3. Observation Correctness

### 3a. Observation structure (serial scenario)

`obs.shape = (15,)` with layout:

| Index | Field | Source |
|-------|-------|--------|
| 0 | Lag-1 demand | `D[period-1, 0]` (zeros at t=0) |
| 1 | Lag-1 backlog | `U[period-1, 0]` (zeros at t=0) |
| 2-4 | On-hand inventory at main nodes [1,2,3] | `X[period, [1,2,3]]` |
| 5-8 | Pipeline arrivals, link (2,1) L=4 | built from `R` |
| 9-12 | Pipeline arrivals, link (3,2) L=4 | built from `R` |
| 13 | `period` (time step) | scalar |
| 14 | `sentiment` (always 1.0) | demand engine |

### 3b. Controller observation usage

`BeliefAdaptiveController.get_action(obs, current_period)` **ignores both arguments**. It reads `env.X[env.period]` and `env.Y[env.period]` directly from the env object. The `obs` parameter is accepted but never parsed.

This means: obs-sanitization wrappers would be silently bypassed. Not a current validity issue (the controller gets correct state), but an architectural concern.

### 3c. `env.X`, `env.Y`, `env.D`, `env.R`, `env.U`, `env.S` shapes

| Array | Shape | Indexing | Meaning |
|-------|-------|----------|---------|
| `X` | `(31, 5)` | `[period, node]` | On-hand inventory (stock) |
| `Y` | `(31, 3)` | `[period, reorder_link]` | Pipeline inventory (stock) |
| `D` | `(30, 1)` | `[period, retail_link]` | Realized demand (flow) |
| `R` | `(30, 3)` | `[period, reorder_link]` | Filled replenishment orders (flow) |
| `S` | `(30, 4)` | `[period, edge]` | Retail shipments + deliveries (flow) |
| `U` | `(30, 1)` | `[period, retail_link]` | Standing backlog (stock) |

### 3d. Fill-rate metric bug (NOT a validity issue)

`gym_adapter.py:266` computes:
```python
total_sold = float(np.sum(env.R))  # sums ALL replenishment orders across 3 links
fill_rate = total_sold / total_demand
```

This is wrong — `env.R` contains replenishment orders across all echelon links, not retail shipments. Empirical verification:

- `env.D.sum() = 449.0` (actual customer demand)
- `env.R.sum() = 900.0` (replenishment across 3 links)
- `env.S.sum() = 1260.0` (all shipments)
- Reported `fill_rate = 900/449 = 2.004` — nonsensical

The correct retail fill rate would use `env.S[:, retail_link_idx] / env.D`. However, **this does not affect the primary outcome metric** (`total_reward` / `total_profit`), which is computed correctly by `CoreEnv._STEP()`. The fill rate is a secondary metric that is incorrectly reported.

**Impact on Phase 8 results**: Fill rate column is meaningless. SIVR and reward comparisons are unaffected.

---

## 4. Event Implementation

### 4a. Surge parameters

| Parameter | Value |
|-----------|-------|
| Baseline distribution | Poisson |
| Baseline mu | 10.0 |
| Surge mu | 20.0 (multiplier 2.0) |
| Warning time | t=0 (implied, since warning is static text) |
| Surge start | t=15 |
| Duration | t=15 to t=29 (15 periods, no recovery) |
| Affected node | Retail (market demand) |
| Episode horizon | 30 periods |

### 4b. Verification

- `mu(t) = 10.0` for t=0..14 (constant)
- `mu(t) = 20.0` for t=15..29 (constant)
- No recovery: surge persists to end of episode
- `env.D` is zero at t=0, filled incrementally
- Warning (static text) is always available before t=15

### 4c. Terminal-horizon artifact

The shock starts at t=15 and runs to t=29 (end of 30-period episode). There is no recovery period. This means:

- The surge is active for exactly 15 periods
- The controller has 15 periods of pre-surge time to prepare
- No post-surge normalization is needed

This is a clean event structure. No truncation artifact.

**Verdict: Event implementation is correct and clean.**

---

## 5. Pilot/Confirmation Separation

### 5a. Seed history

| Phase | Seeds | base_mu | shock_mag | event_start | num_periods |
|-------|-------|---------|-----------|-------------|-------------|
| Exploratory grid (seed=42) | 42 | 5,10,15,20 | 2.0,3.0,5.0 | 15 | 30 |
| Gate-A pilot | 1-20 | 10.0 | 2.0 | 15 | 30 |
| Final confirmation | 2000-2019 | 10.0 | 2.0 | 15 | 30 |

### 5b. Parameter evolution

1. Original adapter: `DEFAULT_BASE_MU = 20.0` → degenerate results
2. Rewritten adapter: `DEFAULT_BASE_MU = 10.0` (changed based on observed behavior)
3. Grid sweep at seed=42 confirmed base_mu=10, shock_mag=2.0 produces clean positive OIV
4. Gate-A pilot (seeds 1-20) at base_mu=10, shock_mag=2.0: passed
5. Final confirmation (seeds 2000-2019): identical parameters, disjoint seeds

### 5c. Classification

**Design = C. Same seeds used for design and confirmation** — but with an important nuance:

- base_mu and shock_mag were tuned using seed=42 + seeds 1-20
- Final confirmation used seeds 2000-2019 (completely disjoint)
- No parameters were changed after Gate-A passed
- The confirmation is a genuine holdout with respect to the Gate-A statistics

**The parameter selection was result-driven** (standard pilot practice but should be disclosed). The statistical claim (OIV > 0, SIVR ranking) is validated on unseen seeds.

---

## 6. Linguistic Generalization

### 6a. Template inventory

| Category | Count |
|----------|-------|
| Total templates | 18 |
| Surge templates | 6 |
| Normal templates | 6 |
| Delay templates | 6 (unused in Phase 8) |
| Ambiguity levels | 3 (clear, moderate, vague) |
| Templates per regime per ambiguity | 2 |

### 6b. Phase-8 test set

Phase 8 uses 12 templates: all 6 surge + all 6 normal from `REGIME_WARNING_TEMPLATES`.

### 6c. TF-IDF training overlap

The calibrated TF-IDF model was trained on the **original 18 templates** (which include these exact 12). The model has seen every Phase-8 test warning during training. This is why it produces degenerate {1.0, 0.0} probabilities — isotonic calibration on the training set saturates to point masses.

**The TF-IDF result does not demonstrate generalization to unseen text.** It demonstrates that a model trained on these exact templates can perfectly classify them and translate that into optimal operational actions.

### 6d. gpt-4o generalization

gpt-4o was NOT trained on these templates. Its responses come from general language understanding. The LLM cache shows it produces varied probabilities (0.6-1.0 for surge templates), not degenerate point masses. This is genuine linguistic generalization.

### 6e. RuleBased generalization

RuleBased is a deterministic keyword classifier. It generalizes to any text with similar keywords. Its performance on these templates reflects its keyword vocabulary, not memorization.

### 6f. Verdict

The current result represents:

- **TFIDF_LogReg**: operational-seed generalization only (no linguistic generalization — trained on exact test templates)
- **gpt-4o**: operational + linguistic generalization
- **RuleBased**: operational + keyword-based generalization

**Limitation**: Only 12 unique warning texts (6 surge + 6 normal). Twenty simulation seeds are not equivalent to twenty linguistic samples. The linguistic sample size is small.

---

## 7. TF-IDF Model Identity

### 7a. What was loaded

`experiment_phase8.py:163` loads:
```
results/phase7_classical_baseline/tfidf_logreg_calibrated_model.pkl
```

This is the **isotonic-calibrated** model. The sensor label `"TFIDF_LogReg"` does not distinguish calibrated vs raw.

### 7b. Probability behavior

The calibrated model produces degenerate probabilities for all Phase-8 templates:

| Template | P(normal) | P(surge) |
|----------|-----------|----------|
| surge_clear_1 | 0.0000 | 1.0000 |
| surge_clear_2 | 0.0000 | 1.0000 |
| surge_moderate_1 | 0.0000 | 1.0000 |
| surge_moderate_2 | 0.0000 | 1.0000 |
| surge_vague_1 | 0.0000 | 1.0000 |
| surge_vague_2 | 0.0000 | 1.0000 |
| normal_clear_1 | 1.0000 | 0.0000 |
| normal_clear_2 | 1.0000 | 0.0000 |
| normal_moderate_1 | 1.0000 | 0.0000 |
| normal_moderate_2 | 1.0000 | 0.0000 |
| normal_vague_1 | 1.0000 | 0.0000 |
| normal_vague_2 | 1.0000 | 0.0000 |

### 7c. Phase 7 comparison

| Model | Held-out Accuracy | Aggregate SIVR |
|-------|-------------------|----------------|
| Raw TF-IDF (Phase 7) | 88.9% | 0.250 |
| Calibrated TF-IDF (Phase 7) | 86.1% | 0.648 |
| Calibrated TF-IDF (Phase 8) | 100%* | 1.000 |

*On training templates (not held-out).

### 7d. Key finding

Phase 8 uses the **calibrated** model on **training templates**. The isotonic calibration collapses probabilities to {0, 1}, making it operationally identical to PerfectSemantic. The Phase 7 held-out SIVR of 0.648 is the more honest estimate of its generalization capability.

**The sensor label should be renamed `TFIDF_LogReg_Calibrated` in future reports to distinguish from raw.**

---

## 8. Controller Probability Sensitivity

### 8a. Mathematical mapping

```
effective_mu = base_mu * (1 + P(surge) * (surge_multiplier - 1))
            = 10 * (1 + P(surge))    [with base_mu=10, surge_multiplier=2]
```

This is **linear** in P(surge). No argmax, no thresholding, no discretization in the action path.

### 8b. Sensitivity diagnostic

```
P(surge) | effective_mu | tgt_retail | tgt_factory | first nonzero action
----------|-------------|------------|-------------|---------------------
   0.00   |    10.0     |   50.44    |    5.22     | t=6
   0.10   |    11.0     |   54.94    |    5.47     | t=5
   0.25   |    12.5     |   61.67    |    5.83     | t=4
   0.50   |    15.0     |   72.78    |    6.39     | t=3
   0.75   |    17.5     |   83.80    |    6.90     | t=2
   0.90   |    19.0     |   90.38    |    7.19     | t=1
   1.00   |    20.0     |   94.76    |    7.38     | t=1
```

- At t=0, all actions are zero regardless of P(surge) because initial inventory (100, 100, 200) exceeds all targets
- Divergence begins at t=1 for high P(surge), t=3-6 for low P(surge)
- Final episode profit: P=0.0 → 341.2, P=0.5 → 531.8, P=1.0 → 665.7

**The controller is genuinely magnitude-sensitive.** Calibration matters because different probability vectors produce different operational actions.

### 8c. But: saturation at t=0

The initial inventory buffer (I0=100 at retail/distributor, 200 at factory) exceeds all targets for ~3 periods. This means the first 1-3 periods produce identical actions across all beliefs. The operational divergence manifests only after the buffer is consumed.

**Impact**: The controller is sensitive, but the sensitivity is delayed. This is realistic (inventory buffers mask belief differences) but reduces the apparent effect size.

---

## 9. Why TF-IDF = PerfectSemantic

### 9a. Episode-level comparison

For every seed and template tested:
- TF-IDF probability vector = PerfectSemantic probability vector (both {0.0, 1.0} or {1.0, 0.0})
- Actions are identical period-by-period
- Rewards are identical period-by-period

### 9b. Root cause

The calibrated TF-IDF model was trained on these exact 18 templates. Isotonic calibration on the training set produces degenerate probabilities (0.0 or 1.0) because the logistic regression already separates these templates perfectly, and calibration sharpens already-confident predictions to point masses.

### 9c. Is this an error?

No. This is the expected behavior of isotonic calibration on training data. The model genuinely achieves 100% accuracy on templates it was trained on. The issue is that this does not demonstrate generalization — it demonstrates memorization.

### 9d. What would happen with raw probabilities?

The raw TF-IDF model (without calibration) would produce non-degenerate probabilities (e.g., P(surge) ≈ 0.37 for some surge templates, per Phase 7 analysis). This would produce different actions than PerfectSemantic and likely lower SIVR.

**The SIVR=1.000 for TFIDF_LogReg is an artifact of calibration on training data, not evidence of generalizable operational value.**

---

## 10. PerfectSemantic Terminology

Current documentation uses "PerfectSemantic" throughout. For publication, this should be renamed to **"OracleSemantic"** or **"oracle-semantic reference"** because:

1. It is not a guaranteed upper bound on policy performance
2. Controller misspecification can allow imperfect beliefs to occasionally produce SIVR > 1
3. "Perfect" implies no downstream errors, which is not guaranteed

**Action**: Flag for publication cleanup. Do not rename code during this audit.

---

## 11. Statistical Structure

### 11a. CI computation

CIs were computed via `paired_bootstrap_ci` from `experiment_phase6.py`:
- 10,000 bootstrap resamples
- Paired over (seed, regime, template_id) tuples
- Comparison: sensor reward vs NoInfo reward

### 11b. Pairing verification

Seeds match between sensor and NoInfo rows (confirmed: `np.array_equal(seeds_s, seeds_ni) == True` for all three comparisons).

### 11c. Hierarchical consideration

Multiple warning templates per (seed, regime) pair create nested structure. The current bootstrap treats each (seed, regime, template) as an independent observation, which **underestimates uncertainty** because:

-同一 seed + regime across different templates produces correlated rewards
- Template-level variance is non-zero for RuleBased and gpt-4o (varies by ambiguity level)
- Template-level variance is ZERO for NoInfo and TFIDF (they don't depend on template text)

A hierarchical bootstrap resampling at the seed level (within each template) would be more appropriate. Current CIs are likely anti-conservative.

### 11d. NoInfo template-invariance

NoInfo produces identical rewards across all templates for the same (seed, regime). This is correct — NoInfo uses the prior, not the text. But it means the 240 NoInfo observations contain only 40 unique rewards (20 seeds × 2 regimes), each repeated 6 times.

**Verdict: CIs are valid but likely anti-conservative. Hierarchical bootstrap recommended for publication.**

---

## 12. Scope Limitations

### 12a. Design boundaries (not bugs)

- Deterministic fixed edge lead times (L=0 for factory→dist, L=4 for dist→retail, L=4 for retail→market)
- Single SKU
- No competitors
- No stochastic lead times
- No capacity allocation across products
- No substitution effects
- Binary regime space (Normal vs DemandSurge only)
- No recovery period after surge

### 12b. What Phase 8 supports

> Transfer of semantic decision value to a richer multi-echelon operational environment with pipeline inventory, capacity constraints, and echelon-dependent lead times.

### 12c. What Phase 8 does NOT support

- Claims about performance under stochastic lead times
- Multi-SKU allocation
- Competitive markets
- Generalization to arbitrary warning text (TF-IDF is trained on test templates)
- Guarantees that the ranking (NoInfo < RuleBased ≈ gpt-4o < TFIDF = Oracle) holds on held-out text

---

## Overall Verdict

**VERDICT B — Phase 8 is valid as exploratory evidence, but requires a clean confirmation run.**

### Justification

1. **No information leakage** — the controller receives belief only through `regime_probabilities`, not through env internals
2. **Event implementation is correct** — surge timing, warning timing, and episode structure are clean
3. **Controller is magnitude-sensitive** — calibration matters, not just argmax
4. **Pilot/confirmation seeds are disjoint** — statistical claim validated on unseen seeds
5. **But**: base_mu and shock_mag were tuned on pilot seeds (standard but should be disclosed)
6. **But**: TFIDF_LogReg result is inflated by calibration on training templates (SIVR=1.000 is not generalizable)
7. **But**: only 12 warning templates (6 surge + 6 normal) — small linguistic sample
8. **But**: fill-rate metric is incorrectly computed (secondary metric, not primary)
9. **But**: CIs may be anti-conservative (non-hierarchical bootstrap)

### Minimal corrective actions

1. **Rename sensor** `TFIDF_LogReg` → `TFIDF_LogReg_Calibrated` in reports
2. **Fix fill-rate computation** to use `env.S` (retail shipments) not `env.R` (replenishment)
3. **Add disclosure** that base_mu/shock_mag were selected via pilot
4. **Add disclosure** that TFIDF was trained on the test templates
5. **Consider hierarchical bootstrap** for publication CIs
6. **Rename** `PerfectSemantic` → `OracleSemantic` in publication text

### What NOT to do yet

- Do not rerun the experiment
- Do not replace DemandSurge
- Do not add more LLMs
- Do not begin publication packaging

---

## Appendix: Latent Leakage Channels (for structural hardening)

These are present on the env object but NOT used by the current controller:

| Channel | Value at t=0 | Discriminates regimes? |
|---------|-------------|----------------------|
| `env.demand_engine.effects` | `['shock']` vs `[]` | YES |
| `env.env_kwargs['demand_config']` | Full config dict | YES |
| `env.demand_engine.get_current_mu(15)` | 20.0 vs 10.0 | YES |

**Recommendation**: pass controller only topology + current X/Y rows, not the full env object.
