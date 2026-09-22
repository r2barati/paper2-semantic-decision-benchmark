# SUPERSEDED results — retain for provenance, never cite as findings

- `results/pre_correction_archive_2026/` — v1.0 outputs (leak in historical
  controller, LP no-sell, arrival +1, ignored `warning_time`, no text-free
  controls; see `REPRODUCIBILITY.md:10-15`). Do not mix with v2 tables.
- `results/correction_audit/pre_correction_snapshot/` — old reports/ledger/tables
  (incl. Holm p=0.0 bug, mixed all-template aggregate).
- `results/publication/table_*.md/.tex`, `SCIENTIFIC_LEDGER_CORRECTED.md` —
  pre-correction tables superseded by `corrected_*.md` + root
  `SCIENTIFIC_LEDGER.md`.
- `results/v3main_superseded/beliefs_gpt-4o-2024-11-20.*` — GPT C0-only partial
  (3424/19600, `credit_balance_exhausted`), moved after freeze gate.
- `results/v3proto/` — prototype diagnostics, no profit.
- `results/phase8_gym_replication/` — exploratory pilot (train-template overlap,
  TFIDF-on-train degenerate `{0,1}`, fill-rate `R/D` bug, shared seeds,
  non-hierarchical CIs; see `VALIDITY_AUDIT.md`). Phenomena that replicate
  (Surge gain, Normal oracle-underperformance) are cited via Phase 8B, not here.
- `results/phase5/`, `results/phase5_5/`, `results/phase6/` — frozen history;
  numbers superseded by corrected C/M tables, phenomena preserved there.
- Root legacy `results/*.csv/*.png` (`overall_summary.csv`, `episode_results.csv`,
  `phase3_5_final_report.md`, `PHASE4_FINAL_REPORT.md`, etc.) — pre-phase
  exploratory, no manifest.
- `results/threshold_sensitivity/`, `results/phase4_matrix/` vs
  `results/phase4_validation/` (identical 12-file duplication, unexplained),
  `results/phase9_synthesis/cross_event_comparison.csv` (hardcoded dicts) —
  exploratory/derived only.

Snapshots `v1/` (TMLR), `v2/` (ECIR), `legacy/` (precursors) duplicate live
paths with stale tables; see `VERSION_MAP.md`. Do not mix numbers across
snapshots. Physical relocation to `archive/` is deferred until post-ECIR.
