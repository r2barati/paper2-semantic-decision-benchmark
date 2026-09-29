# Sem2Act / Paper 2 compute mapping

This is a resource-routing index. Backend amendments and fresh run-specific
preflight receipts determine whether an execution can start. Runtime identity
and account provenance must be recorded separately from credentials.

| Resource | Intended workload | Current policy and limits |
|---|---|---|
| **Kaggle GPU — OpenCode** | Primary V5 GPU reranker pilot; later GPU consumers only after separate handoff | Active V5 owner `rezabarati2`; reranker inputs staged as `rezabarati2/sem2act-v5-rerank-inputs`; request exactly 1 Tesla T4 on `cuda:0` although 2× T4 entitlement was reported. Install/verify the exact Kaggle lock (`torch==2.7.1+cu126`, CUDA runtime 12.6, Python 3.12, Transformers 4.57.6); never rely on the provider image. Fresh account, private-input access/hash, quota, and GPU canary are required. |
| **Google Colab T4 — Muse** | Alternate 20-query reranker operations pilot | Reported T4, 15 GB, Torch 2.11.0+cu128. Separate Colab lock requires Python 3.12, that exact Torch/CUDA runtime, and Transformers 4.57.6. Do not upgrade/downgrade Torch. Record a non-secret account/profile alias. Requires authorized access to the private Kaggle input bundle. Full 240-query execution and consumers need later Codex review. |
| **TMU moon CPU — OpenCode** | CPU-only 20-query reranker operations pilot | Reported 16 vCPU/62 GB, CPU only, AVX2/AVX512 masked. Use 16 PyTorch intra-op threads and 1 inter-op thread only after fresh Moon host/runtime preflight and a canary that repeats exactly at 4/16 threads with identical fixture rankings. Keep venv, model cache, build, and outputs on verified node-local storage; do not use home NFS. Operator estimates ≈1.8 hours for 12,000 pairs at 16 threads; the 1,000-pair pilot is ≈9 minutes before startup overhead. Consumer generation is impractical and not authorized. |
| **Local Mac** | Editing, static validation, small non-result development | A historical Mac profile recorded 4 CPU threads and 8 GB RAM. It is not Moon evidence and is not an authorized V5 compute fallback. Avoid large model/cache writes to the constrained local disk. |
| **GitHub Actions** | CI, deterministic contract checks, manifests, simulation, analysis, paper/build/release checks | Validation and analysis only. Do not use hosted Actions for 7B–8B model inference. |
| **Lightning AI** | Historical V5 GPU submission attempts only | Existing attempts failed before remote job creation and are retained as history; no current fallback is authorized. |

## Workload routing

- **First GPU reranker pilot:** Kaggle is the default route because the
  qrel-free input bundle is staged there and the active account is identified.
  Muse/Colab is a separately locked alternate, not a parallel replacement once
  result-bearing scoring has started.
- **First CPU reranker pilot:** Moon is allowed only under the reranker-only
  16-thread lock and the first frozen 20-query shard. It is an operational
  pilot; do not inspect relevance outcomes or reuse it in analysis.
- **Full reranker stage:** exactly 240 queries × 50 candidates. Requires review
  of the pilot report and a later Codex handoff. Run from a clean output
  directory and rerun all 240 queries.
- **Consumer models:** stage qrel-free Qwen, Llama, and Mistral bundles only
  after the complete reranker output is fetched, hash-verified, and accepted by
  Codex. No consumer bundle is currently staged.
- **CPU consumers:** blocked separately by the unresolved Qwen AWQ→GGUF
  conversion and other missing artifact/source gates. Do not substitute a
  conversion or model.

## Identity and credentials

- Kaggle V5 operator identity: `rezabarati2`; the historical v2 owner
  `siavashsimin` is retained only in superseded records.
- Colab/Muse account alias and Moon/OpenCode execution identity belong in
  run-specific provenance. Never include tokens or credential contents.
- Expected secret names/locations are documented in the shared Research OS
  security rules and the operator runbook; this file contains no secrets.
- The Kaggle GPU observation is recorded in
  `provenance/operator_reports/kaggle_gpu_smoke_20260928.md`. It is not a fresh
  V5 preflight receipt.
