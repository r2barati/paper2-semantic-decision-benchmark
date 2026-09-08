# Response to the ECIR 2027 pre-submission audit

Audit date 7 September 2026, against commit `25c04c1`. This file maps each of
the 22 findings to what was actually changed, with the command or test that
demonstrates it. Nothing here is a claim without a check.

Pre-correction results are preserved under
`results/pre_correction_archive_2026/` so every delta is auditable.

---

## Blocking findings

### 1. Wrong format and ordering — **fixed**

New LNCS entry point at `paper2_submission/manuscript_lncs/` using the official
`llncs.cls` and `splncs04.bst`. Appendix precedes references. Compiled and
checked: **12 content pages including the appendix**, references on pages 13–14.
The superseded TMLR staging directory is excluded from the reviewer artifact.

### 2. NoInfo optimizer received the true disruption — **fixed**

`CausalOptimizer` now takes an explicit `knows_true_disruption` flag, default
`False`. Only the PerfectSemantic reference sets it. Every other sensor —
including the below-threshold fallback branch — plans under the nominal lead
time. The assumed disruption's start no longer inherits the hidden true start;
it comes from the protocol's warning-release time.

*Check:* `tests/test_repairs_2026.py::TestInformationBoundary` drives the
planner over a fixed state trajectory under two different hidden disruptions and
asserts the NoInfo order sequence is identical, while PerfectSemantic's differs.

### 3. LP first-step sales constraint omitted on-hand stock — **fixed**

`b_ub[0]` now includes current on-hand inventory. The one-period LP with 100
units in stock and expected demand 8 sells 8 and orders 0, matching the
analytical solution; a three-period LP produces on-hand 92, 84, 76.

*Check:* `TestLinearProgramInventory`, including a starved case to confirm the
corrected bound does not suppress genuinely needed orders.

### 4. Arrival and warning timing did not match the declared experiment — **fixed**

Replaced the FIFO padding queue with `PipelineSchedule`, keyed by absolute
arrival period and shared by the simulator and every planner. An order placed at
*t* under lead time *L* now arrives at *t+L*, not *t+L+1*, and orders already in
transit are no longer pushed back when the lead time grows. `warning_time` is
honoured: the interpreted belief replaces the prior at period 12 while
operational state and pipeline carry over.

*Check:* `TestArrivalTiming` (impulse tests for L = 1…5, in-transit
non-delay, post-recovery contraction, planner/simulator agreement) and
`TestWarningAvailability`.

### 5. Missing no-text controls — **added, and they change the conclusions**

Constant-belief endpoints, a no-text operating point tuned only on development
seeds disjoint from every confirmation seed, shuffled-text and label-only
ablations, across Experiments C, M, S and B.

The audit's diagnostic is confirmed as a first-class result. In Experiment M the
dev-tuned no-text control returns 560.4 against 516.2 / 530.2 / 534.9 for the
TF-IDF and GPT-4o interpreters, and 552.0 for the oracle-belief reference; only
RuleBased (568.7) exceeds it. In Experiment S it returns 1725.4 against 1484.0
for the best interpreter. **In Experiment S the shuffled-text control (1421.8)
matches the genuine interpreter (1422.0)** despite belief accuracy halving,
so that transfer does not support a claim that reading the warning produced the
improvement. All of this is reported in the paper, not buried.

### 6. Obsolete estimates and omitted adverse results — **fixed**

One declared estimand throughout: `src.metrics.weighted_benchmark_return`, used
with identical weights in the numerator and denominator of SIVR. Raw paired
effects and intervals are reported first. All six boundary variants appear,
including short-lead. Under the corrected text–world pairing three of six
variants have a negative oracle information value and are marked as such rather
than given a ratio.

### 7. Release reproduction failed — **fixed**

`tools/rebuild_offline_artifacts.py` rematerialises all 213 cached semantic
responses from the tracked manifest — verified against a canonical checksum —
and retrains all 7 classifier checkpoints from tracked templates with fixed
seeds. No credentials, no network. `tools/verify_release.py` then replays stored
beliefs through the simulator (max reward error 0.0) and regenerates every table.

The anonymous supplement builds via `tools/build_anonymous_supplement.py`, and
its test suite runs from that clean export: **320 passed, 0 failed** (was 257
passed / 16 failed at the audited commit), with `PAPER2_LLM_CACHE_DIR` and
`PAPER2_PHASE5_RESULTS_DIR` pointed at empty directories and no API key set.

### 8. No retrieval contribution — **added (Experiment R)**

An entity-scoped operational report feed with wrong-entity, stale and off-topic
distractors; a fixed operational query; BM25 (implemented in-repo), TF-IDF
cosine, dense sentence-embedding retrieval with frozen embeddings, random floor
and oracle evidence selector; matched evidence budgets k ∈ {1,3,5}; the same
fixed interpreter and controller downstream.

The headline result is a dissociation: with the calibrated interpreter at k=3
every practical retrieval system is worse than retrieving nothing while the
oracle selector gains +69.0, and Kendall's τ between the nDCG and reward
orderings is +0.33. With the rule-based interpreter the orderings agree exactly.
Positioned against eRAG, CUE-R, GroGU and situational-relevance work.

---

## Important findings

### 9. Calibration confounded with ensembling — **fixed**

Three named pipelines: `single`, `fold_ensemble` (the CV fold ensemble with no
calibration map), `calibrated`. Fold vocabularies are fitted in-fold. A
sigmoid arm is included for the six-point calibration sets. Fold sizes are
recorded and reported.

Result: ensembling alone *lowers* SIVR (0.245 → 0.154) and the calibration map
then recovers past it (→ 0.338). The naive raw-vs-calibrated difference is not
attributable to calibration. Accuracy cost is 3 of 36 held-out texts once the
vocabulary leak is fixed, not 1.

### 10. Bootstrap and p-values — **fixed**

`src.metrics.crossed_bootstrap_ci` draws one shared seed sample per regime,
reused across every family and variant, and resamples the language axis
separately. p-values are centred bootstrap test inversions bounded below by
1/(B+1) and can never be reported as zero. The ledger states B and the floor.

### 11. SIVR equation omitted its weights — **fixed**

Equation 4 in the manuscript is now the weighted return, matching the
implementation. `docs/METRICS.md` declares the nesting and weights, states that
numerator and denominator must share them, distinguishes the numerical epsilon
from statistical uncertainty, and corrects the Brier maximum (2 for any K ≥ 2 in
the sum-of-squares form, not 1).

### 12. Phase 9B paired normal reports with disrupted worlds — **fixed**

`world_regime_for` makes pairing explicit. The default matches each report to
the world it describes; the false-negative arm is named and reported separately.
Rows record `template_regime`, `pairing` and `text_world_mismatch`. A latent bug
surfaced and was fixed: the oracle reference had asserted a capacity drop even in
normal worlds. Repairing the pairing flips the sign of short-lead's OIV.

### 13. Missing methods and wrong sample counts — **fixed**

`docs/ACCOUNTING.md` is generated from the saved episode files by
`tools/generate_accounting.py`. Phase 7 is recorded as 5 sensors / 3,600
episodes before the 2026 arms (not 6 / 4,320), Phase 6 as 20 seeds (not 50), the
controlled horizon as 40 periods (not 30), and Experiment A as three regimes with
`CausalOptimizer` (not two regimes with a heuristic). Episodes across sensors are
distinguished from paired worlds. Stale source documents carry correction
headers. The manuscript appendix states costs, timing, information release and
the belief-to-action mapping.

### 14. LLM provenance labels wrong — **fixed**

Provenance is reconstructed by re-deriving every cache key from all known
(model, text) pairs across three key families. Result: 65 GPT-4o, 49
GPT-4o-mini, 49 GPT-3.5-turbo, and 50 honestly labelled `unknown` — where
previously all 213 were labelled `gpt-4o-mini`. `prompt_hash` is renamed
`cache_key_hash` with its formula recorded, and the prompt surface is
fingerprinted separately.

### 15. Offline runner inconsistencies — **fixed**

Experiment-specific sensor alias tables, validated before any work runs;
`--experiment gym --sensor tfidf_raw` now works. Frozen artifacts resolve from
the repository root, so redirecting the runtime cache no longer moves them.
Cache is consulted *before* credentials, and a miss raises `MissingFrozenArtifact`
instead of silently returning a 50/50 prior under the model's own name. Table
replay is documented as replay.

### 16. Requirements and CI — **fixed**

`requirements.txt` declares the vendored simulator's own dependencies (networkx,
PyYAML) and corrects the gymnasium version claim to the actual 0.29.1 pin. CI
installs from the lock file alone on Python 3.9 and 3.11, asserts the simulator
imports from that environment, rebuilds and verifies offline artifacts, runs the
**full** suite, and runs a stored-belief replay plus publication regeneration.

### 17. HindsightOracle formulation — **fixed**

Empty-pipeline sentinels; `integrality=1` with [0,1] bounds for genuine binaries
(SciPy's `2` is semi-continuous); initial stock in the first sales bound; a
justified big-M equal to total remaining demand; explicit `RuntimeError` instead
of an undefined `_fallback_greedy`; no rounding of solver quantities, only
zeroing of residue the model's own setup indicator says is not an order.

*Check:* `verify_against_simulator()` — the MILP objective equals the simulated
return exactly (gap 0.0) across 30 seeds and all regimes.

### 18. Thin related work — **fixed**

eRAG, CUE-R, GroGU, Borlund, Okapi BM25, Sentence-BERT, nDCG,
Niculescu-Mizil & Caruana and Howard added (20 references). The Donti et al. row
in the closest-work matrix is corrected: it *does* learn probabilistic models and
*does* include a sequential task. IR rows are added to the matrix, and the
distinction is restated as the *consumer* — a control policy with asymmetric
costs and persistent state — rather than as novelty.

### 19. No anonymous licensed supplement — **built**

`tools/build_anonymous_supplement.py` produces the reviewer artifact, rewrites
author-machine paths, removes author identity from the vendored dependency's
packaging metadata, excludes git history and private planning files, and then
**re-scans the built tree**, failing if any identifying string survives. Third-party
LICENSE and NOTICE files are copied verbatim: anonymity is not achieved by
stripping copyright notices. Top-level `LICENSE` (MIT) and `NOTICE` added.

Author declarations, conflicts, concurrent-submission status and the AI-use
statement remain human facts and are not certified here.

### 20. Decisive results omitted — **fixed**

Every results table carries the reference rewards, paired raw effects,
intervals and p-values, generated from the frozen episodes by
`results/publication/generate_lncs_tables.py`. The regime-table filter bug is
fixed and now raises rather than silently emitting an empty breakdown. The
invalid zero-SIVR legacy panel is not in the submission.

### 21. Tests overstated what they established — **fixed**

`tests/test_repairs_2026.py` adds 42 behavioural regressions for the discovered
faults. Tests that encoded the old degeneracy as an expectation are rewritten to
state what is actually true, including one that records a limitation honestly:
under the gym adapter's two-class projection the calibrated interpreter *is*
still oracle-equivalent in-sample, which is why held-out templates are required.
The six-sensor roster test is replaced by one that checks no original arm was
dropped and every required control is present.

### 22. Prose and presentation — **fixed**

Rewritten around a positive research question with named experiments. Internal
project commentary, phase-number shorthand and the claim that no further science
is needed are gone. Realised returns are defined separately from their
expectation and the undiscounted finite-horizon convention is stated.

---

## What this does not fix

- Author-level declarations (conflicts, concurrent submission, AI use).
- Whether the corrected findings are worth publishing: the headline claims are
  now weaker and more qualified than before, because several of them did not
  survive their own controls. That is the point of running the controls.
