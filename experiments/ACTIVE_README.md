# ACTIVE experiments — do not move while jobs are running

The following paths are referenced by hard-coded `ROOT/…` resolves,
`dataset_dir` / `dest_dir` literals in `scripts/kaggle_compute.py`, basename
hash gates (`src_shas.json`, `config_hash`), and fetch/verify hooks. Moving,
renaming, or re-hashing any of them breaks Track A/B verification.

## Track A (Kaggle CPU, 7 jobs sharing one dataset)

- `scripts/kaggle_compute.py` (JOBS + `_stage_files` + `cmd_push/fetch` + verify dispatch)
- `kaggle_kernel/p2_tracka_sim.py`, `p2_tracka_b1.py` … `p2_tracka_b6.py`
- `kaggle/inputs_tracka/` (entire dir: `shard_pilot.json`, `shard_b1..b6.json`,
  `beliefs_Qwen_Qwen3-8B-AWQ.parquet`, `beliefs_Qwen_Qwen3-14B-AWQ.parquet`,
  `src_shas.json`, `src/`)
- `runs/v3main_tracka/` (`episodes_tracka_*.parquet`, `shard_manifest.json`,
  `manifest_tracka_b*.json`, `pilot_local_rerun.parquet` 100-row ground truth,
  `kaggle_fetch_manifest.json`)
- `artifacts/kaggle_out/paper2-tracka-sim-pilot/`, `paper2-tracka-sim-b1/` … `-b6/`
- `tools/gen_tracka_shards.py`, `tools/tracka_local_rerun.py`,
  `tools/run_controllerB_v3main.py`
- `src/controller_basestock.py`, `src/env.py`, `src/events.py`, `src/metrics.py`,
  `src/interpreter.py`, `src/sim_eval_v3main.py`
- Upstream frozen reads: `results/v3main/beliefs_Qwen_Qwen3-8B-AWQ.parquet`,
  `results/v3main/beliefs_Qwen_Qwen3-14B-AWQ.parquet`, `data/v3/splits/splits.json`

## Reasoning smoke + agentick probe (share `kaggle/inputs_reasoning/` under two slugs)

- `kaggle_kernel/p2_reasoning_smoke.py`, `kaggle_kernel/p2_agentick_probe.py`
- `kaggle/inputs_reasoning/` (`corpus.jsonl`, `shard.json` 6 warnings, `src/`,
  `experiments_snapshot/`)
- `experiments/reasoning_agentic/prompts.py` (prompt SHAs checked by smoke verify)
- `runs/reasoning_agentic/`, `runs/v3main_agentick/`
- `artifacts/kaggle_out/paper2-reasoning-smoke/`,
  `artifacts/kaggle_out/paper2-agentick-probe/`

## Track B / VHAT2 (local-only, no Kaggle entry)

- `tools/vhat2_train.py` (writes `results/v3main/extension_g2a/vhat2_*`)
- Inputs: `data/v3/splits/splits.json`, `results/v3main/sim/episodes.parquet`,
  `results/v3main/beliefs_*.parquet`, `kaggle/inputs_loo/`,
  `kaggle/inputs_loo_test/`, `runs/v3main/qwen3emb_main_vectors.npz`,
  `data/v3/queries.jsonl`, `data/v3/corpus.jsonl`,
  `results/v3main/beliefs_cache_awq|qwen14/`,
  `results/v3main/extension_g2a/{loo,c1_loo,vhat_oof}.parquet`
- Theory records: `results/v3main/theory/DESIGN-TRACKA.md`,
  `DESIGN-VHAT2.md`, `STEP0_AUTOPSY.md`,
  `results/v3main/extension_g2a/vhat2_gate.json`

Full per-job table: `compute/job_registry.yaml`. Full evidence statuses:
`docs/experiment_registry.md`.
