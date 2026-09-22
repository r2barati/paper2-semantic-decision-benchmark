# Changelog

All notable benchmark changes. `BENCHMARK_VERSION` is the version authority;
this file is a human-readable pointer (no version bump in this Phase-1 pass).

## [paper2-benchmark-v2.0] — 2026-09-07

- Supersedes v1.0 (2026-08-22).
- ECIR 2027 repair: information-boundary fix (`knows_true_disruption`), LP
  first-period sales, `PipelineSchedule` arrival `t+L`, honoured `warning_time`,
  MILP integer/bounds, calibration-vs-ensembling split (`single`/`fold_ensemble`/
  `calibrated` + sigmoid), weighted `J_w`, crossed bootstrap with shared draws
  and `p >= 1/(B+1)`, Phase 9B pairing fix, cache `unknown` labels, offline
  rebuild (`tools/rebuild_offline_artifacts.py`), runner aliases, deps/CI.
- Added Experiment R (retrieval × consumer interaction) with matched budgets.
- Reframed as negative evaluation; `oracle` renamed `perfect-semantic
  reference` (not an upper bound); pinned `gym-invmgmt==0.2.1`.
- See `BENCHMARK_VERSION`, `REPRODUCIBILITY.md:3-15`,
  `paper2_submission/SUBMISSION_READINESS.md:22-87`,
  `SCIENTIFIC_LEDGER.md:106-114`.

## [Unversioned V3 work] — 2026-09-14–2026-09-21

- V3 corpus/retrieval freeze 2026-09-14 (`configs/v3/FINAL_FREEZE.md`,
  `RETRIEVAL_FREEZE_FINAL.md`): 200Q/4000D 64-node, dev40/test160,
  `BM25→Qwen3-Emb-0.6B→RRF120→Rerank-0.6B-top30`, prompts frozen.
- Belief freeze `results/v3main/belief_manifest.json` (2×12600, 0 errors).
- Sim 108k + gym 107k + delivery ledger/figures (`results/v3main/`).
- Extensions (own DESIGN + seeds + Holm families, outside frozen matrix): D0,
  G2A/G2FULL, VHAT/VHAT2, Controller-B, selective, actions/selection, map,
  ladder, TrackA (24.1k eps computed, unanalyzed), Agentick dose.
- No `BENCHMARK_VERSION` bump yet; live submission effectively v3-unversioned.

## [Phase 1 organization] — 2026-09-22

- Added `docs/experiment_registry.md`, `docs/claim_evidence_ledger.md`,
  `docs/paper2_architecture.md`, `docs/results_manifest_spec.md`,
  `docs/manifest_template.json`, `compute/README.md`,
  `compute/job_registry.yaml`, `experiments/ACTIVE_README.md`,
  `results/FROZEN_README.md`, `results/SUPERSEDED_README.md`,
  `results/INVALIDATED_README.md`, `CHANGELOG.md`, `CITATION.cff`.
- Additive only: no existing file edited or moved; no Phase 2 relocation.
