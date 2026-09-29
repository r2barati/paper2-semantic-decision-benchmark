# Shared Research OS context

This project participates in the shared research control plane. Read `${RESEARCH_OS_HOME:-$HOME/Research/research-os}/agents/AGENT_START_HERE.md` and relevant shared workflows/knowledge, then read this project's instructions, current-state records, frozen protocols, and gates. Project-specific frozen protocols, preregistrations, and data-access rules override generic shared guidance. Shared methods do not transfer scientific evidence automatically.

Also read `${RESEARCH_OS_HOME:-$HOME/Research/research-os}/AGENTS.md` and this
repository's [`STATUS.md`](STATUS.md), [`COMPUTE.md`](COMPUTE.md), and
[`RUNBOOK.md`](RUNBOOK.md). `STATUS.md` is a human snapshot; frozen protocols,
authorization artifacts, and machine manifests linked from it control.

## Sem2Act / Paper 2 constraints

- The corrected `paper2-benchmark-v2.0` results are frozen and must not be
  mixed with archived v1.0 results. Sem2Act v5 is a separate confirmation
  overlay. Read its frozen protocol, protocol and lockbox manifests, execution
  status, and the backend-specific runtime lock before any operation.
- Do not change frozen prompts, schemas, estimands, lockbox, seeds, model IDs or
  revisions, decoding, dependencies, runtime pins, or result policy. Preserve
  failed infrastructure attempts as evidence. A requested code, manifest,
  dependency, or protocol change goes to Codex for an explicit amendment and
  commit.
- Historical V5 Kaggle amendments v1/v2 record owner
  `siavashsimin` and their original runtime. The user explicitly authorized
  execution amendment v3 on 2026-09-29: current V5 owner is
  `V5_OWNER="rezabarati2"`, with account ownership verified by the staged
  reranker input dataset. Preserve v1/v2 and their execution records as
  history; use v3 for new V5 Kaggle work. The separate general `OWNER` binding
  remains independent. The underlying operator observation is recorded in
  [`provenance/kaggle_gpu_smoke_operator_report_20260928.md`](provenance/kaggle_gpu_smoke_operator_report_20260928.md).
- The v3 Kaggle shape allocates two T4s and exposes only GPU 0 to PyTorch. Its
  base tuple is Python 3.12 / Torch `2.10.0+cu128` / CUDA 12.8; package
  installation must preserve Torch/CUDA. A fresh owner, quota, runtime, and
  model canary is still required before a result-stage handoff.
- Colab has a separate explicit runtime lock and also requires the installed
  package set to preserve its observed Torch/CUDA tuple. Moon has a separate
  reranker-only lock; its 16-thread profile is gated by the fixed 4-vs-16
  non-lockbox fixture parity check. Neither lock authorizes the blocked CPU
  consumer route.
  The CPU backend is a prospective execution-only route whose runtime freeze
  remains gated by complete artifact hashes and accepted non-result canaries.
  Do not replace an unsupported AWQ-to-GGUF conversion with another model,
  quantizer, or artifact.
- Fetch and verify every authorized stage before proceeding to the next. Do not
  expose lockbox inputs or qrels to a preflight, smoke test, or model bundle.
  Do not use a documentation-only commit as an execution SHA when the exact
  scientific source and manifest are absent from that commit.
