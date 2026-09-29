# Sem2Act / Paper 2 compute routing

The frozen scientific protocol is
`2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`.
Execution amendments below do not change prompts, lockbox, estimands, model
revisions, decoding, coverage, or analysis. Use only the canonical execution
SHA in [`RUNBOOK.md`](RUNBOOK.md).

| Workload | Backend and current authorization | Requirements and state |
|---|---|---|
| First reranker pilot | OpenCode on TMU Moon; one 20-query / 1,000-candidate shard | Host/runtime profile, node-local scratch, qrel-free staged reranker inputs, exact pinned reranker snapshot, and a 4-vs-16 thread fixture parity gate are enforced by `scripts/run_v5_moon_reranker_pilot.py`. Scoring uses 20 sequential one-query subprocesses, each capped at 540 soft / 600 hard CPU seconds. Stop after shard 000. |
| GPU consumer stage | Muse on Colab, Qwen first | Explicit runtime is one T4 with at least 14 GiB visible VRAM, `torch==2.11.0+cu128`, CUDA build `12.8`; no Torch/CUDA replacement. The 240-query Qwen consumer bundle is not yet available and this result stage remains conditional on a complete accepted reranker output. |
| Kaggle GPU | OpenCode, account `rezabarati2`; runtime canary only | The machine shape allocates two T4s; the lock exposes only GPU 0 and masks GPU 1. V5 image policy is Python 3.12, `torch==2.10.0+cu128`, CUDA build `12.8`; exact-package installation must preserve that tuple. A fresh owner/quota/runtime/model canary is required. No Kaggle result-stage job is authorized in this handoff. |
| CPU consumer inference | Moon, blocked | AWQ-to-GGUF conversion for Qwen has failed and Llama official source access is unresolved. This is separate from GPU consumers and the Transformers CPU reranker. Do not substitute a quantization, model, or source. |
| Consumer input staging | Kaggle datasets, after full reranker acceptance | `stage_consumer_inputs.py` creates the three full-coverage qrel-free bundles only from the complete accepted `rerank.trec`. Current Qwen/Llama/Mistral bundles are absent; no placeholder or partial bundle is valid. |

Moon observations are 16 vCPU / 62 GB RAM, CPU only, with AVX2/AVX512
masked. The 16-thread setting is authorized only when the in-run fixed
non-lockbox fixture preserves all three document scores within `1e-6` and the
ranking exactly against 4 threads. Builds, model snapshots, caches, and
temporary inputs belong on node-local scratch. Persist only the compact shard
checkpoint and provenance under the operator's quota-limited home directory.
The operator-reported Moon process limits are 600 soft / 1200 hard CPU seconds;
at about 8.99 CPU seconds per pair, one full 20-query process cannot finish.
Complete-query subprocesses preserve the frozen candidates and ordering, with
per-child CPU use and output hashes recorded in the run manifest. Model staging
uses only the 12 paths in `cpu_reranker_canary.json`; the accepted file hashes
and validator are unchanged.

Kaggle's two-T4 entitlement does not change the frozen one-GPU job shape. The
account identity, runtime image, package-preservation rule, and canary are
recorded in `versions/sem2act-v5/amendments/kaggle_backend_v3.yaml` and the
V5 Kaggle runtime lock. A positive quota alone is not a runtime pass.
