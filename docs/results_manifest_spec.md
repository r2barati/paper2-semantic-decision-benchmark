# Results manifest specification (docs-only for now)

No enforcement in CI yet. New results SHOULD include a manifest with these
fields; legacy/pre-correction directories are explicitly exempt (documented,
not retrofitted).

## Required fields

- `generating_commit`: full SHA of the code that produced the output
- `config_hash`: SHA of the frozen config (e.g. `configs/v3/*.yaml`,
  `prompts_freeze.json`, controller mapping)
- `input_hashes`: SHAs of input artifacts (corpus/queries/qrels/splits SHAs,
  belief parquet SHAs, `src_shas.json` entries where applicable)
- `seeds`: explicit list or range (e.g. `3100-3129`, `60000-60004`) plus
  bootstrap seed where applicable
- `command`: exact generating command (e.g. `python3 -m src.experiment_phase8b
  --seeds 30 --seed-start 3100 --no-llm`, `python3 -m tools.run_sim_v3main`)
- `data_split`: `DEV` or `TEST` (plus split reference, e.g. `splits.json`
  dev40/test160)
- `analysis_type`: `confirmatory` or `exploratory`
- `status`: `ACTIVE` / `FROZEN` / `SUPERSEDED` / `INVALIDATED`

## Notes

- Closest existing examples (partial, none meets all 8):
  `runs/v3main_tracka/manifest_tracka_b*.json` (config_hash + input_hashes +
  seeds + deps + seal, untracked, lacks repo commit/command/DEV-TEST/status);
  `results/v3main/belief_manifest.json` (file+prompt+data SHAs only);
  `results/phase8b_gym_confirmation/seed_manifest.json` +
  `linguistic_split_manifest.json` (seeds + disjointness proof).
- Biggest gap: `results/retrieval/` (Experiment R) has no manifest at all.
- Template: `docs/manifest_template.json`.
- Allowlist: `results/pre_correction_archive_2026/`,
  `results/correction_audit/pre_correction_snapshot/`,
  `results/v3main_superseded/`, `results/v3proto/`,
  `results/phase8_gym_replication/`, root legacy CSVs/PNGs — exempt.
