# Sem2Act / Paper 2 decision log

Append new scientific or protocol decisions only. Keep this log append-only;
correct an entry with a dated follow-up instead of editing history. The
existing V2.0 freeze and V5 backend amendments remain canonical in their
protocol and manifest files and are linked below rather than copied here.

No scientific or protocol amendment was made for this operating-documentation
setup. In particular, V5 ownership, model choices, runtime locks, seeds, and
experiment parameters were not amended.

For every new entry, record:

- Date and issue
- Previous frozen state
- Proposed or approved amendment
- Rationale
- Affected files and manifests
- Whether scientific interpretation changes, and why
- Approval source and status
- Canonical record path and hash

Canonical history: `REPRODUCIBILITY.md`,
`versions/sem2act-v5/manifests/protocol_freeze.json`, and the individual files
under `versions/sem2act-v5/amendments/`.

## Approved prospective pilot amendments — 2026-09-29

The earlier note above referred to the prior operating-document setup only.
The following execution-only amendments were explicitly requested by the user
on 2026-09-29. They do not amend `protocol/confirmation.yaml`, its hash, the
240-query lockbox, prompts, model revisions, decoding, seeds, estimands, or
result policy. They authorize only the backend-specific first 20-query
operations pilot, behind the listed preflight/canary gates. Stop after each
pilot; Codex review is required before another backend or scale-up.

### Kaggle owner, runtime, and first pilot

- **Issue/date:** active Kaggle V5 owner and runtime policy did not match the
  operator account/dataset or enforce the intended Torch/CUDA version;
  2026-09-29.
- **Previous frozen state:** V5 v2 named `siavashsimin`; prior provider
  execution inherited the image and its canary failed with CUDA unavailable.
  The v3 source records the supplied operator report that `rezabarati2` had
  2× T4, CUDA 12.8, and about 30 GPU-hours/week. The staged qrel-free reranker
  dataset is also under `rezabarati2`.
- **Approved prospective amendment:** active V5 owner `rezabarati2`; request
  exactly one Tesla T4 on `cuda:0`; install and verify Python 3.12,
  Torch 2.7.1+cu126/CUDA runtime 12.6, and Transformers 4.57.6 before model
  loading; record provider driver separately. Require a fresh account/input/
  quota preflight and a real CUDA T4 canary before the fixed 20-query pilot.
- **Rationale:** bind execution to the verified Kaggle owner and staged input,
  and make the runtime reproducible without changing the GPU request/model
  placement or scientific factors.
- **Affected files:** `amendments/kaggle_backend_v3.yaml`,
  `manifests/kaggle_runtime_lock_v3.json`, its generated freeze, the Kaggle
  canary/reranker kernels, `scripts/kaggle_compute.py`, and the V5 operator
  runbook/status/compute records.
- **Scientific interpretation:** no change; only backend identity and runtime
  enforcement change. The pilot is excluded from analysis.
- **Approval/status:** user instruction dated 2026-09-29; approved
  prospectively, with fresh execution gates pending.
- **Canonical record/hash:** amendment SHA-256
  `4cc1299f441b1645d28398ef29c35d3188b6b39724af11091cf247dab7a3749d`;
  runtime-lock SHA-256
  `df2618f13337b37d59ae4e21cda4eb3423569190869db27614b651f07234ca6b`;
  generated freeze SHA-256
  `60e5b5eff56e1e9d69c25766e271b6c9006ba84abc88d236bf68d385ad3d4676`.
- **Historical conflict:** `amendments/kaggle_backend_v3_adia_reference.yaml`
  remains a blocked historical proposal naming the old account as token owner
  and requiring a manual-UI gate. It is superseded by this v3 amendment, not
  deleted or treated as active policy. Backend v1 had named `rezabarati2`, but
  v2 superseded it and is what the current pipeline had actually frozen as
  `V5_OWNER=siavashsimin`; both earlier amendments remain history.

### Colab/Muse runtime and first pilot

- **Issue/date:** the reported Colab Torch/CUDA runtime differs from Kaggle and
  must not be silently replaced or assumed; 2026-09-29.
- **Previous frozen state:** no separate active Colab lock for the reported
  Python 3.12/Torch 2.11.0+cu128 environment.
- **Approved prospective amendment:** accept only Python 3.12, Torch
  2.11.0+cu128 with CUDA runtime 12.8, Transformers 4.57.6, and one T4 with
  at least 14 GiB. Do not change Torch. Require a fresh runtime/account/input
  preflight and pinned-model non-lockbox fixture smoke before the same fixed
  20-query operations pilot.
- **Rationale:** retain the verified Colab image as a separately identified
  backend while isolating it from Kaggle's Torch 2.7.1+cu126 runtime.
- **Affected files:** `amendments/colab_backend_v1.yaml`,
  `manifests/colab_runtime_lock_v1.json`, its generated freeze,
  `scripts/colab_v5_preflight.py`, Colab support in the reranker kernel, and
  the V5 operator docs/runbook.
- **Scientific interpretation:** no change; alternate execution image only.
  The pilot is excluded from analysis.
- **Approval/status:** user instruction dated 2026-09-29; approved
  prospectively, with runtime and data-access gates pending.
- **Canonical record/hash:** amendment SHA-256
  `eaa6514cd6902388c1811891cd2f115714a81f0a763c644110e0c88e901e4331`;
  runtime-lock SHA-256
  `b40454497dd6a365aa1531b691eb515bab06a156e1a158d774af92d9cb4c3c4b`;
  generated freeze SHA-256
  `181a4af896ee3f8f825f2ed9e9510bcd545b2c4741e754d478381fe3f902159d`.

### Moon CPU reranker threads and first pilot

- **Issue/date:** the existing CPU configuration specified 4 threads, while
  Moon reports 16 vCPU/62 GB, CPU-only, masked AVX2/AVX512, and an estimated
  1.8 hours for 12,000 reranker pairs at 16 threads; 2026-09-29.
- **Previous frozen state:** 4-thread CPU policy and a historical local Mac
  host profile. Neither is Moon-specific evidence.
- **Approved prospective amendment:** permit 16 intra-op/1 inter-op thread
  settings for the Moon CPU reranker pilot only. Require a fresh Moon host and
  runtime preflight (16 visible vCPU, at least 48 GiB effective RAM, at least
  20 GiB local disk, local filesystem, AVX2/AVX512 masked, exact Python/Torch/
  Transformers), and a non-lockbox canary that repeats exactly at 4 and
  16 threads with identical fixture rankings. Then permit only shard 000
  (20 queries × 50 candidates); keep venv/cache/build/output on node-local
  storage. The operator estimate implies about 9 minutes of scoring for 1,000
  pairs before startup overhead.
- **Rationale:** use Moon's reported capacity without silently changing thread
  behavior; the cross-thread fixture ranking is a fail-closed compatibility
  check. Any differing ranking blocks the 16-thread pilot pending Codex review.
- **Affected files:** `amendments/moon_cpu_reranker_v1.yaml`,
  `manifests/moon_cpu_reranker_runtime_v1.json`,
  `scripts/moon_v5_preflight.py`,
  `scripts/run_v5_cpu_reranker_canary.py`,
  `scripts/run_v5_cpu_rerank_shard.py`, and the project operator docs.
- **Scientific interpretation:** no intended change to the frozen method or
  interpretation. The cross-thread fixture canary must pass before pilot
  scoring; pilot output remains excluded from analysis.
- **Approval/status:** user instruction dated 2026-09-29; approved
  prospectively, with host and cross-thread gates pending.
- **Canonical record/hash:** amendment SHA-256
  `fc795b6b516b75d0ebedf040e3b15fc364a89780d90d6027bacb82fe1c4e0b74`;
  runtime-lock SHA-256
  `9863846900ee434c16e904bb414daedf2ca45ce85a0332c1a0d7936b54f36e2a`;
  the same lock hash is checked into each run-specific Moon host/canary receipt.

### Pilot scope, consumer bundles, and AWQ-to-GGUF blocker

- **Issue/date:** define the smallest valid first execution and prevent pilot
  outputs or incomplete bundles from becoming confirmatory inputs; 2026-09-29.
- **Previous frozen state:** V5 contains 240 lockbox queries × 50 candidates;
  consumer bundles are not staged; Qwen AWQ-to-GGUF conversion and Llama
  source access remain unresolved for the CPU consumer route.
- **Approved prospective amendment:** each backend's first pilot is fixed to
  `v5-lb-q0001`–`v5-lb-q0020`, 50 candidates each, 1,000 pairs. Pilot outputs
  are operations-only and excluded from analysis. Run one backend at a time,
  stop/report after each, and require Codex review before another backend or
  larger run. Stage Qwen/Llama/Mistral bundles only after a complete 240-query
  reranker output has been fetched, hash-verified, and accepted by Codex. Keep
  the AWQ-to-GGUF issue separate; no model/converter substitution and no Moon
  consumer generation.
- **Rationale:** preserve the locked scientific scope while testing the
  minimum operational slice; avoid using partial reranker output to construct
  downstream consumer inputs.
- **Affected files:** all three backend amendments and runtime locks,
  `manifests/cpu_shards/reranker/shard-000.json`,
  `experiments/stage_consumer_inputs.py`, V5 reranker launch/fetch verifiers,
  and `RUNBOOK.md`/`STATUS.md`/`COMPUTE.md`/`AGENTS.md`.
- **Scientific interpretation:** unchanged. No pilot is analyzed; full-stage
  consumer inputs still derive only from an accepted complete reranker run.
- **Approval/status:** user instruction dated 2026-09-29; first pilots approved
  conditionally, full reranking and consumer jobs remain unauthorized.
- **Canonical record/hash:** fixed shard SHA-256
  `5a296e708773ecb52d4f83d0b265d9ec6f72e0bd0461fb63b962d092fa9dd2f5`;
  staged qrel-free input-manifest SHA-256
  `486c9763aa29d6000620bac9762cf9d5b261833226fc8bbddc5a96c138bbee24`;
  protocol hash `2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`.

### 2026-09-29 — bind Kaggle kernel provenance to the exact handoff SHA

- **Issue:** Kaggle kernels do not inherit the submitting operator's local
  environment, so the uploaded GPU pilot could otherwise have a null
  `execution_sha` despite a correct local checkout.
- **Approved implementation:** require `SEM2ACT_HANDOFF_SHA` to equal local
  `HEAD`, embed that SHA into submitted Kaggle canary/pilot source, record it in
  the local preflight receipt, and reject fetched remote manifests that differ.
- **Affected files:** `scripts/kaggle_compute.py`, the V5 Kaggle canary/pilot
  kernels, and `RUNBOOK.md`.
- **Scientific interpretation:** unchanged; this binds provenance only.
- **Approval/status:** within the user-approved exact-SHA handoff and
  provenance requirements; implemented before any V5 job was launched.

### 2026-09-29 — refresh GPU runtime source freezes after provenance guard

- **Issue:** the exact-SHA launcher and manifest checks changed the active
  Kaggle source contract after the original runtime freeze was generated.
- **Previous frozen state:** Kaggle v3 freeze SHA-256
  `60e5b5eff56e1e9d69c25766e271b6c9006ba84abc88d236bf68d385ad3d4676` and
  Colab v1 freeze SHA-256
  `181a4af896ee3f8f825f2ed9e9510bcd545b2c4741e754d478381fe3f902159d`.
- **Approved implementation:** regenerate both source freeze manifests from
  their committed tools because the shared Kaggle pilot source is also part of
  the Colab source contract.
- **New freeze SHA-256 values:** Kaggle v3
  `b414f79984e0dbb5932024cd9cf445c0ba753d7b1e5c1adbfb038940012f8f4d`; Colab
  v1 `15caf26ba3fd6fcb4be45f0f35d459a395cca9518d6841ecfb0194310813e4b6`.
- **Affected files:** both GPU runtime freeze manifests, the Kaggle launcher,
  the Kaggle canary/pilot kernels, and this decision log.
- **Scientific interpretation:** unchanged; model, protocol, runtime lock,
  and experiment parameters are unchanged.
- **Approval/status:** implementation within the user-approved provenance
  requirements; preflight still required before execution.
