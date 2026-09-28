# Sem2Act / Paper 2 operator runbook

This runbook documents the existing frozen V5 stages. It does not grant
execution authority. A fresh Codex handoff must name the exact source SHA,
protocol and manifest hashes, authorized stage, output location, and resource
limits. The current documentation SHA is not a substitute for the full
scientific source/manifest SHA. The current status is preflight-only.

## 1. Verify the handoff and frozen state

Run from the exact handoff checkout and retain the output:

```sh
git rev-parse HEAD
git status --short
python3 scripts/verify_v5_cpu_protocol.py
python3 scripts/freeze_v5_kaggle_runtime.py --check
```

Expected: `HEAD` equals the handoff SHA, frozen hashes match, and the local
verification commands pass. Use an isolated checkout; do not run from this
repository's unrelated dirty working tree. A dirty scientific tree, missing
manifest, or hash mismatch is a stop.

## 2. Refresh Kaggle preflight

The latest operator report is evidence of GPU availability for `rezabarati2`,
not a current V5 access/runtime authorization. Before any authorized V5
execution:

1. Record the authenticated Kaggle account slug from the provider account
   interface without displaying or copying credential values. Keep the frozen
   `V5_OWNER=siavashsimin` value unchanged.
2. Verify that the authenticated account can access the exact V5-owned kernel
   and private inputs in the handoff. Record both the authenticated account and
   frozen owner in the preflight receipt. If access or ownership is ambiguous,
   stop and report it to Codex.
3. Refresh quota using the existing project command:

   ```sh
   python3 scripts/kaggle_compute.py --preflight
   ```

   The command uses `kaggle quota --format json` and records a fresh
   `kaggle_preflight.json`. Expected status is `quota-passed` with a positive
   GPU balance and matching frozen protocol/runtime-lock hashes.
4. Run the authorized non-result V5 canary using the exact frozen machine
   shape/runtime. The supplied account's CUDA 12.8 smoke result does not prove
   compatibility with the frozen V5 runtime (`CUDA 12.6.1`,
   `torch 2.7.1+cu126`). Record the observed account, device count/model,
   CUDA/PyTorch versions, timestamp, and output hash. If the canary does not
   satisfy the lock, stop; do not edit the lock or fall back to another model.

The existing V5 canary command sequence is:

```sh
SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 python3 scripts/kaggle_compute.py v5-canary --push
python3 scripts/kaggle_compute.py v5-canary --watch
python3 scripts/kaggle_compute.py v5-canary --fetch
```

These commands require a separate Codex handoff authorizing the canary. The
first command is not authorized by this documentation commit. A passed
canary must be fetched and checked against its manifest before Codex can issue
a result-stage handoff.

## 3. Result-stage order and checkpoints

Only after fresh preflight, accepted canary, and an exact Codex result-stage
handoff, use the frozen order. Each `--fetch` runs the job-specific verifier
configured for that job and retains its manifest and hashes. The generic
`--test` option runs `tests/test_v3proto.py`; it is not a V5 output verifier and
must not be presented as one. Use only the V5 acceptance checks named in the
Codex handoff, then report to Codex before proceeding when the handoff requires
review.

```sh
python3 scripts/kaggle_compute.py v5-rerank --push
python3 scripts/kaggle_compute.py v5-rerank --watch
python3 scripts/kaggle_compute.py v5-rerank --fetch
```

After the verified reranker output and Codex's next stage authorization,
assemble and stage the three qrel-free consumer bundles, then execute and
verify each consumer in order:

```sh
python3 scripts/kaggle_compute.py v5-qwen --push
python3 scripts/kaggle_compute.py v5-qwen --watch
python3 scripts/kaggle_compute.py v5-qwen --fetch
python3 scripts/kaggle_compute.py v5-llama --push
python3 scripts/kaggle_compute.py v5-llama --watch
python3 scripts/kaggle_compute.py v5-llama --fetch
python3 scripts/kaggle_compute.py v5-mistral --push
python3 scripts/kaggle_compute.py v5-mistral --watch
python3 scripts/kaggle_compute.py v5-mistral --fetch
```

Do not advance if a bundle is missing, a fetch/hash/coverage check fails, a
model/runtime differs from the freeze, or the output reveals an unexpected
failure. Preserve the attempt and send the evidence to Codex. Belief assembly,
CPU replay, and one-pass analysis follow only after all accepted model outputs
and their exact manifests are present; Codex must hand off those commands and
the stage authorization explicitly.

## 4. CPU route

The CPU route remains blocked before runtime freeze. Its existing local checks
are:

```sh
python3 scripts/verify_v5_cpu_protocol.py
python3 scripts/cpu_host_preflight.py
python3 scripts/freeze_v5_cpu_runtime.py --check
```

Expected current result: protocol verification passes; runtime check remains
blocked until complete engine/conversion/GGUF hashes and all fixed non-lockbox
canaries pass. The recorded Qwen AWQ-to-GGUF conversion failure and Llama
source-mount failure remain blockers. A successful preflight is evidence for
Codex; Codex owns any runtime-freeze manifest update and commit. Do not invoke
`--freeze` or a result-bearing CPU runner from OpenCode.

## Fetch, report, and stop rules

For every remote stage, keep the provider job ID, owner/account identity,
command, timestamps, environment/runtime, seed/input identity, fetched files,
and SHA-256 receipts in the run-specific provenance. Never place secrets in
that record. Stop for owner/access mismatch, quota or runtime drift, failed
preflight/canary, hash mismatch, missing output, or a request to alter a frozen
file or parameter. Preserve outputs and failed evidence; report the exact
issue to Codex. Do not silently change `V5_OWNER`, model, runtime, dependencies,
seeds, parameters, or Git history.
