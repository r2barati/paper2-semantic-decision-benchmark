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
- In `scripts/kaggle_compute.py`, the current frozen V5 owner is
  `V5_OWNER="siavashsimin"`. Preserve that value and every owner field in the
  frozen V5 manifests. The launcher's general `OWNER="rezabarati2"` binding
  and the operator-reported GPU-verified `rezabarati2` account are distinct
  from `V5_OWNER`; never infer or apply one binding to the other. The smoke
  evidence is recorded in
  [`provenance/kaggle_gpu_smoke_operator_report_20260928.md`](provenance/kaggle_gpu_smoke_operator_report_20260928.md).
  It does not change V5 ownership. A fresh handoff preflight must record the
  authenticated identity and verify access to the frozen V5 resources; any
  owner/access/runtime mismatch stops the run and is reported to Codex.
- The V5 GPU job declarations specify their recorded runtime and one-GPU job
  shape, but the current runtime canary has not passed. A separate account's
  two-T4/CUDA-12.8 observation does not update or validate that lock.
  The CPU backend is a prospective execution-only route whose runtime freeze
  remains gated by complete artifact hashes and accepted non-result canaries.
  Do not replace an unsupported AWQ-to-GGUF conversion with another model,
  quantizer, or artifact.
- Fetch and verify every authorized stage before proceeding to the next. Do not
  expose lockbox inputs or qrels to a preflight, smoke test, or model bundle.
  Do not use a documentation-only commit as an execution SHA when the exact
  scientific source and manifest are absent from that commit.
