# Submission readiness

**Current status: IN REPAIR — not ready to submit.**

The previous version of this file recorded "READY FOR AUTHOR SIGN-OFF" against a
TMLR submission. That statement was contradicted by executable evidence: an
external audit of commit `25c04c1` (7 September 2026) found substantive
scientific defects, inconsistent reported results and a release whose clean
tracked export failed 16 tests. This file now tracks the repair, not a
readiness claim.

Any statement here must be reproducible by running the command it cites. Do not
restore a readiness claim that a command does not currently support.

## Target

ECIR 2027 full-paper track. Springer LNCS format, **12 content pages including
appendices**, unlimited reference pages, **appendices before references**,
double-blind, EasyChair. Abstract deadline 21 September 2026, paper deadline
5 October 2026, both 23:59 GMT. Overlength papers are desk-rejected.

## Repaired (verifiable)

Scientific and implementation defects, each with a behavioural regression test
in `tests/test_repairs_2026.py`:

- **Information boundary.** The uninformed controller no longer receives the
  environment's true disruption. `CausalOptimizer` requires an explicit
  `knows_true_disruption=True` for the oracle arm; every other sensor plans
  under the nominal lead time. NoInfo plans are now provably invariant to the
  hidden truth.
- **Optimiser LP.** The first-period sales constraint includes on-hand
  inventory. The one-period LP with 100 units in stock and mean demand 8 now
  sells 8 and orders 0, matching the analytical solution.
- **Arrival timing.** A single `PipelineSchedule` keyed by absolute arrival
  period is shared by simulator and planners. An order placed at *t* under lead
  time *L* arrives at *t+L* (previously *t+L+1*), and orders already in transit
  are no longer pushed back by a later lead-time change.
- **Warning availability.** `warning_time` is honoured: no semantic sensor acts
  before the warning is released.
- **Hindsight reference.** Empty-pipeline sentinels, genuine integer setup
  indicators (`integrality=1`, not the semi-continuous `2`), correct
  initial-stock bound, a justified order bound, explicit failure, and no
  rounding. The MILP objective now equals the simulated return exactly across
  every tested seed and regime.
- **Calibration vs ensembling.** Three named training pipelines (`single`,
  `fold_ensemble`, `calibrated`), fold vocabularies fitted in-fold, sigmoid
  alternative for the small calibration sets, fold sizes recorded.
- **Estimand.** One declared weighted benchmark return
  (`src.metrics.weighted_benchmark_return`) used in both numerator and
  denominator of SIVR.
- **Uncertainty.** Crossed bootstrap with shared seed draws across templates;
  centred bootstrap p-values bounded below by 1/(B+1), never exactly zero.
- **Phase 9B design.** Text–world pairing is explicit; the default matches each
  report to the world it describes, and the false-negative arm is named and
  reported separately.
- **Provenance.** Cache model labels reconstructed from the cache-key function;
  genuinely unknown entries labelled `unknown` rather than defaulted.
- **Offline release.** `tools/rebuild_offline_artifacts.py` reconstructs every
  semantic cache from the tracked manifest and retrains every checkpoint, with
  no credentials and no network.
- **Runner.** Experiment-specific sensor aliases, validated up front.
- **Dependencies and CI.** The simulator's own dependencies are declared; CI
  installs from the lock file alone, rebuilds artifacts, and runs the full
  suite on 3.9 and 3.11.

## Added

- **Experiment R**, the retrieval / evidence-selection study: an entity-scoped
  report feed with distractors, stale reports and routine traffic; BM25,
  TF-IDF, dense, random and oracle selectors under matched evidence budgets;
  the same fixed interpreter and controller downstream.
- **Text-free and degraded-text controls** across the controlled and gym
  experiments: constant-belief endpoints, a development-tuned no-text operating
  point, shuffled-text and label-only ablations.
- **LNCS manuscript** at `paper2_submission/manuscript_lncs/`, appendix before
  references.

## Outstanding before submission

- [ ] Final LNCS PDF compiled and page count verified at or under 12 content
      pages including appendices.
- [ ] Anonymous supplement actually built and inspected (relative paths, no git
      history, no credentials, no private planning files).
- [ ] Author, conflict, concurrent-submission and AI-use declarations completed
      from actual author activity. These are human facts and cannot be
      certified from this repository.
- [ ] Abstract registered by 21 September 2026; paper submitted by 5 October
      2026.

## Superseded

`CLEAN_REPRODUCTION_AUDIT.md` refers to commit `856a176` and to tracked cache
material that no longer exists at HEAD. `FINAL_TEST_AUDIT.md` records a
268/268 count from 23 August 2026. Neither describes the current release. Both
are retained as history and must not be cited as current evidence.
