# Sem2Act / Paper 2 compute routing

The frozen scientific protocol is
`2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`.
Execution amendments below do not change prompts, lockbox, estimands, model
revisions, decoding, coverage, or analysis. Use only the canonical execution
SHA in [`RUNBOOK.md`](RUNBOOK.md).

| Workload | Backend and current authorization | Requirements and state |
|---|---|---|
| First reranker pilot | OpenCode on Kaggle GPU `rezabarati2`; one 20-query / 1,000-candidate shard | Authorized by `kaggle_reranker_pilot_v1.yaml` and locked by `kaggle_reranker_runtime_lock.json`. Identical scoring recipe, model revision, dtype, 1,024-token context, batch 32, and candidate order as the full kernel. Requires a fresh owner/quota preflight, a `canary-passed` record, and a passing freeze check. Stop after shard 000. |
| Full 240-query reranker | OpenCode on Kaggle GPU; **blocked** | Not authorized. `v5-rerank` refuses to push until Codex writes `versions/sem2act-v5/manifests/kaggle_reranker_pilot_acceptance.json` after reviewing the 20-query pilot. |
| Moon reranker route | OpenCode on TMU Moon; **blocked, evidence only** | Executed and recorded in `moon_reranker_pilot_blocked.json`: model integrity and 4-vs-16 parity passed, but the first capped subprocess exited `-24`/`SIGXCPU`. Zero queries and zero pairs accepted. Do not rerun. |
| GPU consumer stage | Muse on Colab, Qwen first | Explicit runtime is one T4 with at least 14 GiB visible VRAM, `torch==2.11.0+cu128`, CUDA build `12.8`; no Torch/CUDA replacement. The 240-query Qwen consumer bundle is not yet available and this result stage remains conditional on a complete accepted reranker output. |
| CPU consumer inference | Moon, blocked | AWQ-to-GGUF conversion for Qwen has failed and Llama official source access is unresolved. This is separate from GPU consumers and the Transformers CPU reranker. Do not substitute a quantization, model, or source. |
| Consumer input staging | Kaggle datasets, after full reranker acceptance | `stage_consumer_inputs.py` creates the three full-coverage qrel-free bundles only from the complete accepted `rerank.trec`. Current Qwen/Llama/Mistral bundles are absent; no placeholder or partial bundle is valid. |

Moon observations are 16 vCPU / 62 GB RAM, CPU only, with AVX2/AVX512 masked.
The operator-reported Moon process limits are 600 soft / 1200 hard CPU seconds;
at about 8.99 CPU seconds per pair, one full 20-query process cannot finish,
and the replacement per-query subprocess schedule still hit the CPU cap on the
first query. That route is closed and kept only as a blocked record.

Kaggle's two-T4 entitlement does not change the frozen one-GPU job shape. The
account identity, runtime image, package-preservation rule, and canary are
recorded in `versions/sem2act-v5/amendments/kaggle_backend_v3.yaml` and the
V5 Kaggle runtime lock. A positive quota alone is not a runtime pass. The
20-query pilot attaches the already remote-verified `sem2act-v5-rerank-inputs`
dataset version 1 and embeds its non-result-bearing smoke fixture by hash, so
pushing it needs no local dataset copy.
