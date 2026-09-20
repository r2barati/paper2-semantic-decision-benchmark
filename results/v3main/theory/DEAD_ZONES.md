# Decision dead zones — FROZEN THEORY NOTE (2026-09-20, Workstream 1)

Status: frozen definitions, theorem, proof; empirical-map SPEC included
(§6) but no computation run yet. Corrects the plan-mode draft in exactly
the way required: the zero-value object is the ACTION-EQUIVALENCE set
D(s), never the basis chamber alone.

## 1. Setup

Belief simplex `B = {b = (b_N, b_D, b_S) : b >= 0, sum = 1}` over regimes
{normal, supplier_delay, demand_surge}. Reference belief `b_0` = benchmark
prior (0.35, 0.35, 0.30) unless stated. Operational state at period t:
`s_t = (t, on_hand, pipeline-history)`; the frozen protocol releases the
interpreted belief once at warning time with state/pipeline carried over
(`src/experiment_phase5.py:114-145`). Policy `π(b; s_t)` = first-period
order from the frozen controller. J = cumulative undiscounted profit over
the 40-period horizon at fixed seed (fixed demand draws).

Determinism lemma (audited, `src/env.py:555-633`,
`src/controller_basestock.py:57-61`): both controllers' `decide()` paths
contain no RNG; `scipy linprog(method="highs")` is deterministic given
inputs. `CausalOptimizer` is stateful (internal pipeline mutated on
`decide`), so its equivalence class is conditioned on pipeline content —
which the protocol preserves across the warning release. Same belief +
same operational state + same history => same order, exactly reproducible.

## 2. Definitions

* **Action-equivalence set:** `D(s; b_0) = {b : π(b; s) = π(b_0; s)}`.
* **Dead zone** (at state s, relative to reference b_0): any subset of
  `D(s; b_0)` with nonempty interior containing belief updates of interest.
  Evidence whose posterior stays inside `D(s)` changes NO immediate action.
* **No-order polytope** (exact special case): `{b : π(b; s) = 0}`, a polytope
  from `bounds=[(0,None)]` plus the `max(0,·)` censor (`src/env.py:605,612`).
* **Sensitive region:** set where `π(·; s)` is affine-nonconstant in belief
  (within-chamber response); small Δb gives gradual Δaction.
* **Discontinuity boundary:** belief set where the LP basis (or censoring
  branch) changes; small Δb can give large or sign-changing Δaction.

## 3. Zero-value theorem (corrected form)

**Theorem.** Fix seed, warning time, horizon, and initial state. Let
`b_t`, `b'_t` be two belief paths (e.g. prior vs evidence-induced
posterior, constant after release). If for every period t,
`π(b'_t; s'_t) = π(b_t; s_t)` with `s'_0 = s_0`, then the realized profit
paths — hence J — are IDENTICAL, i.e. sequential ΔJ = 0, despite possibly
nonzero belief change `b'_t - b_t`.

**Proof.** Induction on t. Base: same initial state. Step: same action +
same exogenous demand draw (fixed seed) => same fulfilled sales, same
inventory transition, same placed order => same pipeline evolution =>
same next state and same period profit. The controller's internal pipeline
history evolves identically because it is a deterministic function of the
same order sequence. Summation over t gives equal J. ∎

**Corollary (dead zones kill value).** If every posterior along the episode
stays in the reference action-equivalence set, operational VoI is exactly
zero. The no-order polytope is the cleanest instance: any belief update
confined to it is operationally void.

**Explicit correction recorded:** a parametric-LP basis chamber does NOT
suffice — within-chamber response is affine (gradual value change), not
zero. Zero value needs action-equivalence, i.e. dead zones; chambers give
sensitive regions; chamber walls give discontinuity boundaries. The theory
is therefore tripartite: dead zones + sensitive regions + boundaries.

## 4. Boundary taxonomy (frozen inventory instantiation)

Beliefs enter the frozen LP at exactly two places (`src/env.py:479-515`):
in-window `e_LT(b) = 4 + 4·b_D` (integer-rounded into the arrival matrix)
and `e_D(b) = 8·(b_N + b_D + 1.75·b_S) = 8 + 6·b_S` (affine demand RHS);
objective coefficients are belief-free.

1. **LT-rounding walls** (finite, enumerable): `4+4·b_D ∈ {4.5,5.5,6.5,7.5}`
   nominally `b_D ∈ {0.125,0.375,0.625,0.875}`, modulo Python bankers
   rounding (4.5→4 etc.) — computed empirically, never hand-enumerated.
2. **Demand-RHS walls** (continuous family): `sales ≤ demand` tight/slack
   transitions and inventory-balance pivots = hyperplanes in `b_S`
   conditional on `(t, on_hand, pipeline)`; swept empirically, not enumerated.
3. **Censoring kinks:** `max(0, x*_0)` and the solver-failure `0.0` fallback.
4. **Indexation:** all of the above move with `(t, on_hand, pipeline)`;
   R = H−t shrinks; event overlap vanishes for t≥30. Dead zones are
   per-`(t, state)` objects, never belief-only.
   Example separation: prior (0.35,0.35,0.30) → (5, 9.8); perfect-delay →
   (8, 8.0); perfect-surge → (4, 14.0): distinct `(A_mat, demands)`.

## 5. What the theory predicts for the observed regimes

* **C3-flat:** provenance-weighted posteriors concentrate near the prior /
  inside one chamber's dead subregion → near-zero action response → VoI≈0.
* **C1-mixed signed:** concat posteriors swing across rounding/RHS walls →
  position- and model-dependent sign flips (cf. 14B/drop1 positive,
  8B/drop2 negative).
* **C0-positive:** rule posteriors sit in the positive-order sensitive
  region with consistent direction.
  §6 tests exactly this mapping; mismatch bounds the theory instead.

## 6. Empirical-map SPEC (frozen; computation NOT yet run)

Grid over `B` (resolution frozen at execution, ≥41×41 projected to the
2-simplex) × sampled `(t, on_hand, pipeline)` states from frozen NoInfo
trajectories; each node evaluated by direct `decide()` calls ONLY (no
episodes, no sim states observed by LLMs — there are none here). Outputs:
chamber-wall map, no-order polytope slice, per-node affine slopes;
overlays of frozen C0/C1/C3 posteriors and LOO shifts with dead/sensitive/
boundary classification. Success = §5 mapping confirmed; fail = recorded
bound on the theory. Zero new LLM calls; CPU-only.

## 7. Generality note

`BeliefBaseStock` shares `decide(state)->float`; its closed form
`target = mu·lt + 1.65·√(lt·mu)` has a dead zone (`target ≤ inv_pos` → 0)
and smooth sensitivity elsewhere — the same tripartite structure with
analytic walls. The fixed heuristic in `experiment_phase5.py` takes no
belief input and is excluded from all generality claims. Solver-version
basis labels and rounding behavior are implementation caveats, not
theoretical claims: equivalence is over ACTIONS, never bases.

## 8. Non-goals

No globally-piecewise-constant policy claim; no belief-only dead zones;
no SOTA/model language; no Agentick work; no manuscript prose. This note
is mechanism, not remedy.
