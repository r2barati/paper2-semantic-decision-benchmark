# Sem2Act / Paper 2 status

> Human-readable snapshot and index only. Frozen protocol and runtime locks are
> authoritative.

- **Last reconciled:** 2026-09-29. The execution branch is prepared for the
  first gated pilot; Codex has not executed a result-bearing job.
- **Scientific protocol:** frozen at
  `2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`.
- **Next authorized result step:** OpenCode/Moon may run only reranker shard
  000 (20 queries, 1,000 candidate scores), and only after the launcher passes
  exact-SHA, host/runtime, input/model hash, and 4-vs-16 thread parity gates.
  Stop after the shard and return its provenance to Codex.
- **Kaggle:** V5 ownership is amended to `rezabarati2`. The runtime contract
  allocates two T4s but exposes only GPU 0, fixes Python 3.12 / Torch
  `2.10.0+cu128` / CUDA 12.8, and rejects package changes to Torch/CUDA. Owner,
  quota, and non-result CUDA/model canary checks are the only Kaggle actions
  authorized in this handoff.
- **Muse/Colab:** exact runtime policy is one T4 with at least 14 GiB VRAM,
  Torch `2.11.0+cu128`, CUDA 12.8. Its full Qwen family stage is prepared but
  input-gated until a complete reranker output is accepted and a qrel-free
  Qwen bundle is staged with a Codex-issued manifest hash. The exact package
  install and non-lockbox Qwen smoke must also pass before the result kernel.
- **Not staged:** Qwen, Llama, and Mistral consumer bundles. The reranker
  qrel-free input dataset is staged as Kaggle dataset version 1.
- **CPU consumer route:** remains blocked by Qwen AWQ-to-GGUF conversion and
  unresolved Llama source access. These do not block the Transformers CPU
  reranker or GPU consumer routes.
- **No V5 result-bearing jobs** are recorded as launched.

## Authoritative records

- Protocol: `versions/sem2act-v5/protocol/confirmation.yaml` and
  `versions/sem2act-v5/manifests/protocol_freeze.json`.
- Pilot commands, required inputs, and STOP points: [`RUNBOOK.md`](RUNBOOK.md).
- Backend routing: [`COMPUTE.md`](COMPUTE.md).
- Kaggle runtime/owner: `versions/sem2act-v5/amendments/kaggle_backend_v3.yaml`,
  `versions/sem2act-v5/manifests/kaggle_runtime_lock.json`, and its freeze.
- Colab runtime: `versions/sem2act-v5/amendments/colab_backend_v1.yaml`,
  `versions/sem2act-v5/manifests/colab_runtime_lock.json`, and its freeze.
- Moon reranker-only runtime: `versions/sem2act-v5/amendments/moon_reranker_backend_v1.yaml`,
  `versions/sem2act-v5/manifests/moon_reranker_runtime_lock.json`, and its
  freeze.
- Existing broader CPU consumer gate remains in
  `versions/sem2act-v5/manifests/cpu_preflight_status.json`.
