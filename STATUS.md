# Sem2Act / Paper 2 status

> Human-readable snapshot only. Last reconciled 2026-09-29 against clean
> `origin/main` base `81913d55a4c3e397726456165ed06525f8ddb074`, plus the
> operator reports linked below. The execution handoff names the canonical
> commit SHA. The linked locks, amendments, and run-specific receipts govern.

- **Current stage:** first V5 reranker operations pilots are authorized behind
  backend-specific fresh preflight and non-result-canary gates.
- **Completed:** the 240-query protocol/lockbox and qrel-free reranker input
  hashes are frozen. The input dataset is remotely verified at
  `rezabarati2/sem2act-v5-rerank-inputs`. Operator reports confirm Kaggle
  `rezabarati2` access to 2× T4, CUDA 12.8, and about 30 GPU-hours/week; Muse
  reports Colab T4/15 GB with Torch 2.11.0+cu128; OpenCode reports Moon 16 vCPU
  and 62 GB RAM. No V5 scientific job has been launched.
- **Running:** none.
- **Next authorized step:** select one backend and complete its fresh preflight
  and non-result canary, then run only its fixed 20-query × 50-candidate
  reranker pilot. Stop and send the outputs/provenance to Codex after that
  pilot; wait for review before another backend or any scale-up.
- **Blocked:** full 240-query reranking and all consumers await pilot review
  and a new stage handoff. Qwen, Llama, and Mistral consumer bundles are not
  staged. CPU consumer generation remains separately blocked by Qwen
  AWQ→GGUF conversion and unresolved Llama source access.
- **Compute status:** Kaggle V5 v3 uses owner `rezabarati2`, one requested T4,
  and explicit Torch 2.7.1+cu126/CUDA 12.6 runtime; fresh account, private
  input, quota, and CUDA canary checks are required. Colab has a distinct
  Torch 2.11.0+cu128/CUDA 12.8 policy and fresh T4/runtime/smoke checks. Moon
  has a distinct CPU-only 16-thread reranker policy with a fresh host/runtime
  check and canary that repeats at both 4/16 threads with identical fixture
  rankings; caches/builds/outputs must use node-local storage.
- **Important risks:** provider/runtime drift, private input access, Moon's
  small home-NFS quota and masked AVX2/AVX512, and consumer artifacts not yet
  staged. Any mismatch is a stop; do not switch backends silently.

## Authoritative records

- Frozen scientific protocol: `versions/sem2act-v5/protocol/confirmation.yaml`
  and `manifests/protocol_freeze.json` (protocol SHA-256
  `2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`).
- Kaggle active account/runtime/pilot gate:
  `amendments/kaggle_backend_v3.yaml`,
  `manifests/kaggle_runtime_lock_v3.json`,
  `manifests/kaggle_runtime_freeze_v3.json`, and the ignored, run-generated
  `manifests/kaggle_preflight_v3.json`.
- Colab runtime and pilot gate: `amendments/colab_backend_v1.yaml`,
  `manifests/colab_runtime_lock_v1.json`, and
  `manifests/colab_runtime_freeze_v1.json`.
- Moon CPU reranker gate: `amendments/moon_cpu_reranker_v1.yaml`,
  `manifests/moon_cpu_reranker_runtime_v1.json`, and the run-specific host and
  canary receipts produced by `scripts/moon_v5_preflight.py` and
  `scripts/run_v5_cpu_reranker_canary.py`.
- Staged qrel-free input hashes:
  `manifests/upload/v5-rerank-inputs.json` and fixed first shard
  `manifests/cpu_shards/reranker/shard-000.json`.
- Operator evidence: `provenance/operator_reports/kaggle_gpu_smoke_20260928.md`.
  The historic Kaggle v2 failure and Mac CPU preflight are retained records,
  not current V3/Moon preflight evidence.
