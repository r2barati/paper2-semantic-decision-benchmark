# §6 EMPIRICAL MAP — FROZEN SPEC (2026-09-20, pre-execution)

Tests DEAD_ZONES.md §5 predictions: C3 mostly dead, C1 more crossings,
C0 positive-sensitive. CPU-only `decide()` sweeps; zero LLM calls; no
episodes beyond NoInfo state-harvest replays (verifiable vs frozen).

## 1. Belief grid (frozen)

2-simplex lattice step 0.025 → 861 nodes. Prior (0.35,0.35,0.30) lies
exactly on lattice. Node key = rounded triple. Reference action per state
= order at prior node.

## 2. States (frozen)

Harvest from NoInfo closed-loop replays mirroring `_run_p5_episode`
(prior belief, warning never releases, same seeds/controller): 3 true
regimes × 5 seeds (60000–60004) = 15 trajectories; sampled periods
t ∈ {0,5,10,12,15,20,25,30} (warning/event edges covered) → 120 states.
Each state = (t, on_hand, pipeline-schedule copy) with the live optimizer
instance carrying true NoInfo history. VERIFICATION GATE: harvested NoInfo
profits must match frozen noinfo episodes within 1e-9 for all 15
trajectories, else STOP.

Sweep per state: pipeline restored from snapshot before EVERY node;
`set_regime_probabilities(node)`; `order = decide(state_copy)`.
`decide` never mutates the state object (reads time/on_hand only).

## 3. Classification (frozen; tol 1e-6 on orders)

Per (state, node): DEAD if |order − prior_order| ≤ 1e-6 (no-order polytope
reported as a subset: order ≤ 1e-6 AND prior ≤ 1e-6); else SENSITIVE.
BOUNDARY node if any lattice neighbor differs in action by > 1e-6.
Slope (map viz only): max neighbor |Δorder| / 0.025.

## 4. Overlays (frozen)

Populations (rerank k=3): C0 (model none), C1-8B/14B, C3-8B/14B posteriors
+ C1/C3 test LOO full→shift pairs (frozen rows). PRIMARY = test-160
(unit = query; states shared); dev-40 reported as consistency
only. Each belief evaluated at all 120 states.

* dead_frac(consumer) = fraction of (query, state) classified DEAD.
* cross_rate(consumer) = fraction of (query, position, state) with
  action(LOO) ≠ action(full) beyond tol.
* pos_frac(C0) = fraction of (query, state) with order_C0 > prior_order
  AND SENSITIVE.

## 5. Hypotheses (frozen; query bootstrap B=2000)

* **H1:** dead_frac(C3-8B)+dead_frac(C3-14B) mean > dead_frac(C1-8B)+
  dead_frac(C1-14B) mean; paired difference CI excludes 0 (seeds 81000).
* **H2:** cross_rate(C1 both models pooled) > cross_rate(C3 both models
  pooled); paired CI excludes 0 (seed 81001).
* **H3:** pos_frac(C0) > 0.5 with 95% CI lower > 0.5 (seed 81002).
All three hold → mechanism confirmed; any fail → recorded bound on the
theory (which wall? which consumer? stated, not rescued).

## 6. Non-goals

No episodes with beliefs (no LLM states exist here); no threshold tuning
(tolerances frozen above); no solver-version claims (actions only);
no manuscript prose. Cost ≈ 500k LP solves, ~10 min on 7 workers.
