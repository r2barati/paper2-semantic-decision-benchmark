# Post-Phase-1 baseline — 2026-09-22

Organizational stabilization checkpoint. Future changes are distinguished from
this reorg by diffing `git status --short` and file hashes against this record.

- Branch: `ecir2027-repairs`
- HEAD: `acae1faec3a13a3744755ddda25ec0727d0418ae` (2026-09-21, Agentick dose-response)
- Phase-1 rule: additive files only; no existing file edited, moved, or renamed;
  no Phase 2 relocation; no `.gitignore` change while Track A/B are active.

## Pre-existing dirty state (present before Phase 1, untouched by it)

Modified: `M .DS_Store`, `M scripts/kaggle_compute.py`.
Untracked (pre-existing): `DESIGN-REASONING-AGENTIC.md`, `Paper 2, Sep 19.md`,
`artifacts/`, `experiments/` (incl. `reasoning_agentic/`), `kaggle/`,
`kaggle_kernel/p2_reasoning_smoke.py`, `legacy/`, `paper2_submission/.DS_Store`,
`reports/.DS_Store`, `runs/`, `scripts/smoke_reasoning_local.py`,
`scripts/verify_reasoning_gpu_smoke.py`, `v1/`, `v2/`.

## 13 Phase-1 created paths (sha256 at checkpoint)

- `docs/experiment_registry.md` — 962a5b9254cafd1df65c9f6beb0398e2e006d54b6b88ff76fd5c9e262c7c2457
- `docs/claim_evidence_ledger.md` — c0ad82c1bc524d3637086c383cdba8d9867be21266505c9effbbdadd9d033bd2
- `docs/paper2_architecture.md` — 522c1b7d7b6806c64c7e5f9226772a44ab1f0fc997451278fcf15fc1dcce0da0
- `docs/results_manifest_spec.md` — 06e763d6d5791e112d7cb2398b6bc3d7536e50dfce8c62986ae5edb5b5811d5f
- `docs/manifest_template.json` — b294093ec3d00e1dbbcfc7563ce896c383efde52678e3e0219d13575c335e496
- `compute/README.md` — e86e4ddfd46af12d6a3489767a5cb2e3ed21deccba30864e73385e0f1cca4bb9
- `compute/job_registry.yaml` — 4f440b1592cafcdc4c990bb3f71b44352f216a116d44079837e316eb280f7f87
- `experiments/ACTIVE_README.md` — d90263ecd526637e114b8c3e4d5a561689e414f51d2436286b6211bd87ee5153
- `results/FROZEN_README.md` — 76fe8969e2e63931011ccd2160b30c9f266395dd5d74131b0078124525a0f5b8
- `results/SUPERSEDED_README.md` — 4f96f5529e2399998b1687ab44b00cebcf497faba334ef1cf421b527f53ed305
- `results/INVALIDATED_README.md` — 8efc71c6ce8ad273cc7bd79d8be024a3b0742f42e2b88a13fc0c844e85e1ee65
- `CHANGELOG.md` — 83c004e51f5d88d7b6858930c4896ea2dd27bae6fd936ce5af9483e41d7dfe2e
- `CITATION.cff` — 32021731b534b62627082f9bd9b3f78f337300201688ef3e0ecf4469865fb541

Registry schema note: `experiment_registry.md` carries separate
`lifecycle_status` (ACTIVE/FROZEN/EXPLORATORY/SUPERSEDED/INVALIDATED) and
`evidence_status` (CONFIRMATORY / … / DESIGN ONLY) columns per approved correction.
`CITATION.cff` holds only verified fields (title, version, license); authors and
identifiers intentionally omitted pending author review.

## Externally introduced file (not created by Phase 1)

- `scripts/run_reasoning_dev_pilot.py` — 21b12a90cd815037ed9b09207e5cd91b772facdc88e762521ea869389d1daff7
  Appeared between the pre-Phase-1 and post-Phase-1 status captures; untouched
  by this reorg. Included here so it is not mistaken for reorg output.

## Known defect (deferred, not fixed in Phase 1)

The three `results/*_README.md` files exist on disk (hashes above) but are
git-invisible: `.gitignore:26` (`results/**`) only whitelists exact
`README.md` / `manifest*.json`, so they would disappear from a clean clone and
cannot yet contribute to ECIR reproducibility. `.gitignore` was deliberately
left unchanged while Track A/B are active. After Track A/B finish, either
whitelist those exact filenames or move their content into tracked `docs/`
equivalents.

## Next priority (scientific, not organizational)

Finish Track A and Track B, update `docs/experiment_registry.md` and
`docs/claim_evidence_ledger.md` with their final statuses, then freeze the
ECIR claim set. No further reorganization until post-ECIR submission.
