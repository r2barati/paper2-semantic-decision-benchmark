# Frontier-anchor amendment 2026-09-14: GPT-4o → Phi-4 14B AWQ (execution impossibility, not a result-driven change)

## What happened
- The frozen matrix specified `gpt-4o-2024-11-20` (configs/v3/models.yaml) as the
  frontier anchor. After 3,424/19,600 cache fills, the OpenAI API began returning
  `credit_balance_exhausted` (insufficient_quota). No pacing or retry change can
  fix a zero balance. GPT execution is **impossible** on this machine.
- Harness (zen) models were evaluated and rejected with evidence: no endpoint or
  key is exposed to the shell (only to the agent chat loop), so 19,600 frozen
  calls cannot be served with temperature control, pinnable revisions, or usage
  records. OpenRouter flagships (Nemotron 3 Ultra 550B, GPT-OSS-120B) were
  verified to exist but their free tier (50–1,000 req/day shared) needs 20–390
  days for the matrix; paid is single-digit dollars but needs a new API account.
  User chose the free self-served option.

## Substitution (user-approved 2026-09-14)
- New frontier anchor: **stelterlab/phi-4-AWQ** @
  `075b93fe5ab0d2e86004a5d68c7575ec3bb5a88b` (AutoAWQ INT4 GEMM of
  microsoft/phi-4, MIT licence, ungated). 14B class, ~9GB weights, fits the
  proven Kaggle 16GB vLLM envelope at --max-model-len 4096.
- Why Phi-4 over Qwen3-14B-AWQ: cross-lab replication (Microsoft × Qwen) keeps
  preregistered Q6 ("replicated across two model families") meaningful; Phi-4 is
  documented for precise instruction adherence, which is what the strict-JSON
  C1/C3 schemas need. Qwen3-14B-AWQ remains the fallback (official quant, same
  thinking-toggle path as our 8B).
- UNCHANGED: corpus, qrels, splits, retrieval stack, prompts (C1 66e6890b, C3
  5966cadd), tau 0.5, consumers C0/C1/C3, controller, simulator, metric
  (trec_eval nDCG), hypotheses Q1–Q7 (Q6 now reads "Phi-4 × Qwen3-8B").

## What happens to the GPT-4o partial
- `results/v3main/beliefs_cache/` (3,424 files) is PRESERVED, not deleted, as
  superseded spend. It is excluded from every analysis and freeze manifest.
- If OpenAI credits are restored later, the GPT arm may be completed as an
  explicitly labeled extension; it cannot re-enter the frozen matrix.

## Provenance for the new arm (same gates as AWQ)
- Pinned (model, revision) in kernel + JOBS + manifest; prompt SHA asserts;
  qrel-free seal guard; temperature 0; determinism + parse probe before shards;
  per-shard manifests with src_snap SHAs; single-query assembly validation.
