# Kaggle GPU smoke-test observation — operator report

- **Reconciled:** 2026-09-28
- **Evidence source:** Operator report supplied to Codex on 2026-09-28. The raw
  smoke-test output and its checksum were not supplied in this repository.
- **Account identity:** `rezabarati2`
- **Observed accelerator:** 2× NVIDIA T4
- **Observed CUDA version:** 12.8
- **Reported allowance:** approximately 30 GPU-hours per week
- **Reported result:** GPU smoke test confirmed the account's accelerator
  capability.
- **Status:** Operator-reported infrastructure evidence; not independently
  re-run during this documentation task.

## Scope and required follow-up

The frozen Sem2Act V5 pipeline owner remains `V5_OWNER="siavashsimin"` in the
project launcher and owner-bound manifests. This report records a separate
GPU-verified account and does not change V5 ownership, job shape, runtime lock,
model choices, dependencies, seeds, or experiment parameters.

The 2026-09-26 `adia_kaggle_infrastructure_diff.json` records a different,
historical event: the then-current `siavashsimin` credential received HTTP 403
when trying to read a private legacy kernel owned by `rezabarati2`. The
operator-reported smoke result here is for the newly supplied `rezabarati2`
account. Keep those identities and dates distinct; do not infer that the old
credential's access failure applies to the newly supplied account or that its
GPU evidence changes the V5 owner.

Before any authorized V5 execution, perform a fresh preflight that records the
authenticated account, verifies access to the exact frozen owner-bound kernel
and inputs, refreshes quota, and checks compatibility with the locked V5
runtime. The observed CUDA 12.8 environment does not establish compatibility
with the V5 lock's CUDA 12.6.1 / PyTorch 2.7.1+cu126 pins. If access or runtime
compatibility fails, stop and report to Codex. No scientific amendment is
made by this observation.
