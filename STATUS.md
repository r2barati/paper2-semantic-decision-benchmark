# Sem2Act / Paper 2 status

> Human-readable snapshot and index only. Follow the linked frozen protocol,
> authorization, and machine manifests for authoritative state.

- **Last reconciled:** 2026-09-28; project base HEAD before this documentation
  commit: `5bd6581`. The active V5 files referenced below are in the current
  working tree; this base SHA alone is not an execution handoff.
- **Current stage:** V5 infrastructure preflight; no result-bearing V5 job is
  recorded as started.
- **Completed:** V5 protocol/lockbox freeze and provenance gate are recorded in
  the V5 manifests. The reranker bundle is staged.
- **Running:** No V5 result-bearing job is recorded as running.
- **Blocked:** The GPU route needs a fresh account/access/quota/runtime
  preflight. The existing V5 lock fixes `V5_OWNER=siavashsimin`; the newly
  supplied `rezabarati2` account has separate operator-reported GPU smoke
  evidence, without a V5 ownership change. The launcher also contains a
  general `OWNER="rezabarati2"` binding; it is separate from `V5_OWNER`.
  The 2026-09-26 infrastructure diff records that the then-current
  `siavashsimin` credential received 403 when reading a private legacy
  `rezabarati2` kernel; that historical attempt does not characterize the
  newly supplied account. The prior V5 runtime canary under `siavashsimin`
  reported CUDA unavailable.
  The CPU route is still before runtime freeze: Qwen AWQ-to-GGUF conversion
  failed, the Llama source probe did not mount its source, and the Mistral
  source assembly was verified. Qwen, Llama, and Mistral consumer bundles are
  not staged.
- **Next authorized step:** A separately handed-off, preflight-only check that
  records the authenticated account and verifies compatibility with frozen V5
  ownership, private inputs, quota, hardware, and runtime pins. No result stage
  follows without its own exact Codex handoff and passed gates.
- **Compute status:** The operator reported that `rezabarati2` passed a smoke
  test with 2× T4, CUDA 12.8, and approximately 30 GPU-hours/week. This is
  account-specific evidence, not a fresh V5 quota/runtime preflight. The local
  Mac's recorded free disk is too small for large model artifacts; the V5 CPU
  route remains limited to its existing local/Kaggle mapping. GitHub Actions
  remains validation/analysis/build infrastructure for V5.
- **Important risks:** Account ownership and access must be reconciled without
  changing `V5_OWNER`; CUDA 12.8 evidence does not establish compatibility with
  the locked V5 runtime; a missing runtime-freeze manifest blocks CPU results.

## Authoritative records

- V5 scientific protocol: `versions/sem2act-v5/protocol/confirmation.yaml`
  and `manifests/protocol_freeze.json`.
- Current machine execution summary: `versions/sem2act-v5/manifests/EXECUTION_STATUS.json`.
- Kaggle and CPU gate detail:
  `versions/sem2act-v5/manifests/kaggle_runtime_lock.json`,
  `manifests/kaggle_preflight.json`, `manifests/cpu_runtime_lock.json`,
  `manifests/cpu_preflight_status.json`, and the corresponding runtime-freeze
  and canary manifests.
- `manifests/kaggle_ui_canary.json` is a separate machine record for
  `siavashsimin` and still reports `pending-user-ui-run`; it is not the newer
  operator-reported smoke observation for `rezabarati2`.
- The older `versions/sem2act-v5/tasks/STATUS.yaml` still describes the
  September 25 Lightning preflight state. Treat it as historical; this snapshot
  points to the later machine execution records rather than replacing them.
- Operator-reported GPU observation:
  `provenance/kaggle_gpu_smoke_operator_report_20260928.md`.
