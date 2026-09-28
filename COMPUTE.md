# Sem2Act / Paper 2 compute mapping

This file is a concise routing index. Project manifests and a fresh
Codex-issued runbook handoff govern each run. Resource observations below are
dated and do not authorize a job.

| Workload | Preferred backend | Fallback | Requirements and current state |
|---|---|---|---|
| V5 GPU inference | Kaggle, using only the V5 owner and job shape in the V5 job declarations | No automatic fallback | Frozen pipeline value: `V5_OWNER=siavashsimin`; the V5 jobs request one Tesla T4 and name a pinned runtime lock, but the current canary has not passed. The launcher also defines general `OWNER=rezabarati2`; do not conflate it with `V5_OWNER`. The operator reports a separate `rezabarati2` smoke test observed 2× T4, CUDA 12.8, and approximately 30 GPU-hours/week; see the provenance note below. Fresh authenticated-account, private-input access, quota, hardware, and exact-runtime preflight is still required. Do not change V5 owner or runtime from this evidence. |
| V5 CPU inference | Local multicore CPU as specified by `cpu_backend_v1.yaml` | Kaggle multicore CPU as specified there | Existing execution-only backend amendment; runtime freeze is absent. The local Mac disk is critically constrained. Qwen AWQ conversion currently blocks the CPU runtime freeze; Llama source access is unresolved; Mistral source assembly is verified. No consumer result run is unlocked. |
| Reranker and consumer staging | Existing staged-bundle sequence in V5 manifests | No substitute artifacts | Current machine status reports the reranker staged and Qwen/Llama/Mistral not staged. Preserve exact revisions and hashes. Fetch and verify each accepted bundle before the next authorized stage. |
| Validation, replay, analysis, paper build | GitHub Actions or local CPU within the frozen environment | TMU moon only if separately authorized and compatible | Hosted Actions are CPU-only in the existing Sem2Act guidance, with a documented six-hour/job limit; confirm current limits. They do not replace 7B–8B inference. |
| Long-running CPU development | TMU CS moon for generic CPU work | Local CPU for small jobs | Moon is not part of the current V5 backend mapping. Verify SSH/VPN, quota, `hostname`, CPU/RAM, and disk before separately authorizing any route change. |

## Kaggle identity and smoke evidence

- The frozen V5 pipeline owner remains `siavashsimin` in
  `scripts/kaggle_compute.py` (`V5_OWNER`) and the V5 runtime/job manifests.
- The same launcher has a general `OWNER="rezabarati2"` binding for other
  jobs. It is not the V5 owner binding.
- The operator-reported, GPU-verified account is `rezabarati2`. On 2026-09-28
  the operator reported a smoke-test observation of two T4 GPUs, CUDA 12.8,
  and approximately 30 GPU-hours per week. This report is recorded in
  [`provenance/kaggle_gpu_smoke_operator_report_20260928.md`](provenance/kaggle_gpu_smoke_operator_report_20260928.md).
- This account evidence does not prove access to V5-owned kernels or inputs,
  does not satisfy the fresh V5 quota/runtime check, and does not change the
  frozen owner or the V5 lock's CUDA/PyTorch pins. Report an access or runtime
  mismatch to Codex; do not rewrite ownership, locks, or model/runtime choices.

The detailed provider inventory is shared in
`${RESEARCH_OS_HOME:-$HOME/Research/research-os}/compute/providers.yaml` and
`${RESEARCH_OS_HOME:-$HOME/Research/research-os}/docs/COMPUTE_INVENTORY.md`.
