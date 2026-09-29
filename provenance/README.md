# Sem2Act V5 execution provenance

This directory contains compact, reviewed operator evidence. It does not hold
credentials, private inputs/qrels, model caches, raw provider outputs, or the
full result payload. Raw run files remain in an operator-controlled ephemeral
directory until Codex reviews the report.

Use [`templates/pilot_run_manifest.json`](templates/pilot_run_manifest.json)
to normalize each backend attempt. Attach the backend's native manifest and
preflight/canary receipts. Record their SHA-256 values, the exact execution
commit, protocol and runtime-lock hashes, qrel-free input hashes, model
revision, runtime/hardware/account identity, provider job ID, timestamps,
seed applicability, output paths and hashes, coverage, status, and failures.
Never record secret values or secret-bearing environment dumps.

The committed Kaggle input manifest is
`../versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json`; runtime locks
and execution policy are linked from the project `STATUS.md` and `RUNBOOK.md`.
Do not treat the human-readable `STATUS.md` as machine state. Run-specific
preflight, canary, and pilot records are generated outside Git unless Codex
reviews and explicitly designates a compact receipt for version control.

## Current operator evidence

- `operator_reports/kaggle_gpu_smoke_20260928.md` records the supplied
  Kaggle entitlement/runtime report. It is not a fresh V5 execution preflight.
- No V5 result-bearing job has been launched as of the initial-pilot handoff.
- Pilot outputs are operations-only and excluded from analysis. Stop after
  each pilot and return provenance to Codex before the next stage.
