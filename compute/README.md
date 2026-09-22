# compute/ — local-only compute surface (untracked)

This directory contains **organizational pointers only**. Bulky compute
artifacts stay **untracked and local** per `.gitignore`:

- `runs/` (188M: `v3main` 93M, `v3` 86M trec/npz) — Kaggle fetch outputs
- `artifacts/kaggle_out/` (347M, 27 jobs) — cross-machine outputs + `output.zip`
- `kaggle/` inputs — frozen Kaggle datasets

Only small manifests are referenced from version control:

- `runs/v3main_tracka/manifest_tracka_*.json` (config_hash + input_hashes + seeds + deps + seal)
- `runs/*/kaggle_fetch_manifest.json` (job + fetched_utc + file SHAs)
- `results/frozen_llm_outputs/manifest.jsonl`, `results/v3main/belief_manifest.json`

See `compute/job_registry.yaml` for the 9 active Kaggle jobs plus the
local-only Track-B entry. See `experiments/ACTIVE_README.md` for the
do-not-move list. Nothing here is part of the frozen publication surface;
`docs/ACCOUNTING.md` and `SCIENTIFIC_LEDGER.md` define authoritative counts
and numbers.
