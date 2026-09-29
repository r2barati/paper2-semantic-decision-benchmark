# Sem2Act / Paper 2 status

> Human-readable snapshot and index only. Frozen protocol and runtime locks are
> authoritative.

- **Last reconciled:** 2026-09-29. The Moon replacement pilot was executed by
  the operator and is recorded as **blocked**, not as a result: model integrity
  passed and the 4-vs-16 thread parity gate matched exactly
  (`max_absolute_score_delta: 0.0`), but the first query's capped subprocess
  exited `-24` / `SIGXCPU`. Zero queries and zero pairs were accepted, no
  ranking was produced, and nothing was concatenated. The earlier Moon
  pre-scoring stoppage at SHA `6d97cd0fe11cc87cd38661ac3a3646ebb36b7599`
  (extra `.gitattributes`/`README.md` in the model directory) remains in the
  same record. The CPU route is not authorized further.
- **Scientific protocol:** frozen at
  `2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`.
- **Next authorized result step:** OpenCode/Kaggle may run `v5-rerank-pilot`
  only — the same unchanged 20-query / 1,000-candidate shard, now on GPU. It
  requires a fresh owner/quota preflight, a `canary-passed` record, and the
  frozen pilot lock. Stop after the shard and return its provenance to Codex.
- **Kaggle:** V5 ownership is `rezabarati2`. The runtime contract allocates two
  T4s but exposes only GPU 0, fixes Python 3.12 / Torch `2.10.0+cu128` / CUDA
  12.8, and rejects package changes to Torch/CUDA. The 20-query pilot is
  authorized; the full 240-query `v5-rerank` job is still blocked pending Codex
  acceptance of the pilot.
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
- **No V5 result-bearing job is recorded as accepted.** The Moon pilot accepted
  zero queries and zero pairs, and no Kaggle result job has been run. The
  verified pilot result, if any, exists only as
  `pilot_pending_acceptance.json` until Codex accepts it.

## Authoritative records

- Protocol: `versions/sem2act-v5/protocol/confirmation.yaml` and
  `versions/sem2act-v5/manifests/protocol_freeze.json`.
- Pilot commands, required inputs, and STOP points: [`RUNBOOK.md`](RUNBOOK.md).
- Backend routing: [`COMPUTE.md`](COMPUTE.md).
- Kaggle 20-query reranker pilot:
  `versions/sem2act-v5/amendments/kaggle_reranker_pilot_v1.yaml`,
  `versions/sem2act-v5/manifests/kaggle_reranker_runtime_lock.json`, its
  freeze `versions/sem2act-v5/manifests/kaggle_reranker_runtime_freeze.json`,
  the kernel
  `versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank_pilot.py`, and the
  freeze generator `scripts/freeze_v5_kaggle_reranker_pilot.py`.
- Moon blocked-result evidence:
  `versions/sem2act-v5/manifests/moon_reranker_pilot_blocked.json`.
- Kaggle runtime/owner: `versions/sem2act-v5/amendments/kaggle_backend_v3.yaml`,
  `versions/sem2act-v5/manifests/kaggle_runtime_lock.json`, and its freeze.
- Colab runtime: `versions/sem2act-v5/amendments/colab_backend_v1.yaml`,
  `versions/sem2act-v5/manifests/colab_runtime_lock.json`, and its freeze.
- Moon reranker-only runtime: `versions/sem2act-v5/amendments/moon_reranker_backend_v1.yaml`,
  `versions/sem2act-v5/manifests/moon_reranker_runtime_lock.json`, and its
  freeze. Exact model staging is implemented in
  `scripts/stage_v5_moon_reranker_model.py`; the subprocess schedule and
  aggregation checks are implemented in `scripts/run_v5_moon_reranker_pilot.py`
  and `scripts/run_v5_cpu_rerank_shard.py`.
- Existing broader CPU consumer gate remains in
  `versions/sem2act-v5/manifests/cpu_preflight_status.json`.
