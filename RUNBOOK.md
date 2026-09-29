# Sem2Act / Paper 2 V5 operator runbook

> **Scope of this handoff:** the first fixed 20-query reranker operations pilot on
> each separately locked backend (Kaggle GPU, Colab GPU, Moon CPU), subject to
> its fresh preflight and non-result canary below. Run only one backend at a
> time. After each pilot, STOP and send the outputs to Codex; Codex must review
> before another backend or any larger run. This handoff does not authorize a
> full 240-query rerun, consumer generation, analysis, or any CPU consumer job.

The 20-query output is operational feasibility evidence only. Do not inspect
relevance metrics, compare model quality, or use it in scientific analysis.
Every later full reranker run must start in a clean output directory and rerun
all 240 queries.

## 1. Exact checkout and frozen inputs

Use a fresh isolated checkout of the execution branch at the exact SHA named
in the Codex handoff. Do not use the dirty working checkout.

```sh
git clone --branch sem2act-v5-pilot-execution-20260928 https://github.com/r2barati/paper2-semantic-decision-benchmark.git sem2act-v5
cd sem2act-v5
export SEM2ACT_HANDOFF_SHA="<exact SHA supplied in the Codex handoff>"
git checkout --detach "$SEM2ACT_HANDOFF_SHA"
test "$(git rev-parse HEAD)" = "$SEM2ACT_HANDOFF_SHA"
test -z "$(git status --porcelain)"
python3 scripts/verify_v5_cpu_protocol.py
python3 scripts/freeze_v5_kaggle_runtime.py --check
python3 scripts/freeze_v5_colab_runtime.py --check
```

Expected: the checkout is clean; the protocol hash is
`2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`;
all backend freeze checks pass. Stop on a SHA, source, lock, or protocol
mismatch.

For Kaggle, `SEM2ACT_HANDOFF_SHA` must remain set to that exact commit while
running the preflight, canary, submission, and fetch commands. The launcher
embeds the verified SHA in the uploaded canary/pilot source. The local
preflight receipt and fetched remote manifests must report that same
`execution_sha`; a mismatch stops.

The only reranker inputs are the qrel-free `queries.jsonl`,
`corpus.jsonl`, and `rerank_candidates.jsonl` in
`versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json`. Expected
SHA-256 values are recorded there. The fixed pilot is shard 000:
`v5-lb-q0001` through `v5-lb-q0020`, 50 candidates each, 1,000 pairs.
Never attach, download, or expose qrels to a preflight, canary, or pilot.

The model is `Qwen/Qwen3-Reranker-0.6B` at revision
`e61197ed45024b0ed8a2d74b80b4d909f1255473`. The execution has no applicable
random seed; record `seed: null` and the runner's deterministic-inference
note. Runtime lock and fixture hashes come from the committed manifests.

## 2. Kaggle GPU — OpenCode

Active V5 owner/account is `rezabarati2`; the old v2 owner
`siavashsimin` remains historical only. Before preflight, verify the signed-in
account slug in the Kaggle UI. Kaggle CLI/SDK credentials must already be
available through the approved external credential store; do not print them.

The active lock requests exactly one Tesla T4 on `cuda:0`, even though two
T4s were operator-reported available. The canary/kernel installs Torch
`2.7.1+cu126` and Transformers `4.57.6` before model loading, then requires
Python 3.12 and Torch CUDA runtime 12.6. Record the provider driver separately;
the operator-reported 12.8 driver/runtime observation does not replace the
Torch runtime lock.

### Preflight and non-result CUDA canary

```sh
export SEM2ACT_V5_AUTHENTICATED_ACCOUNT=rezabarati2
python3 scripts/kaggle_compute.py --preflight
SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 python3 scripts/kaggle_compute.py v5-canary --push
python3 scripts/kaggle_compute.py v5-canary --watch
python3 scripts/kaggle_compute.py v5-canary --fetch
```

Preflight freshly checks access to the staged private dataset, exact file
hashes, and positive GPU quota. The non-result canary must fetch and verify
one T4, the locked Python/Torch/CUDA/Transformers versions, and a real finite
CUDA matrix multiply. The fetch updates the local ignored preflight receipt to
`canary-passed`.

**STOP** if the account is not `rezabarati2`, dataset access/hash or quota
fails, the canary fails, or its fetched verification is incomplete. Preserve
the failure evidence and report it to Codex.

### Smallest valid Kaggle pilot

Only after the above checks pass:

```sh
SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 python3 scripts/kaggle_compute.py v5-rerank-pilot --push
python3 scripts/kaggle_compute.py v5-rerank-pilot --watch
python3 scripts/kaggle_compute.py v5-rerank-pilot --fetch
```

The kernel is attached only to
`rezabarati2/sem2act-v5-rerank-inputs`; it scores the first 20 frozen
queries. Fetch runs the committed verifier for query coverage, 50 unique
candidates/query, exact runtime/input/protocol hashes, qrels exclusion, zero
failures, and output checksums. Expected files are
`pilot_rerank.trec`, `run_manifest.json`,
`runtime_versions.json`, and the local `kaggle_fetch_manifest.json`.

**STOP after fetch, whether the pilot passes or fails.** Report the execution
SHA, account slug, Kaggle job URL/ID, preflight receipt, canary manifest,
fetched verification, and SHA-256 values. Do not launch `v5-rerank` or a
consumer kernel.

## 3. Colab T4 — Muse

The accepted Colab image is exactly Python 3.12, Torch `2.11.0+cu128` with
`torch.version.cuda == 12.8`, Transformers `4.57.6`, and one Tesla T4 with
at least 14 GiB visible memory. Do not install, upgrade, or downgrade Torch.
If the current runtime differs, stop and report it.

Keep non-secret `SEM2ACT_HANDOFF_SHA` and `SEM2ACT_ACCOUNT_ID` in the
notebook environment. Load `HF_TOKEN` and the Kaggle dataset-access
credential from Colab's approved secret store (for example its `HF_TOKEN`
and `KAGGLE_API_TOKEN` secrets); never display their values. The Kaggle
credential must be authorized to read the private qrel-free dataset.
Replace the account-alias placeholder below with the actual non-secret profile
alias shown by Colab.

From the exact checkout, use a fresh ephemeral output directory and fetch the
staged input archive. The Colab secret setup must not print values:

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
```

```sh
export SEM2ACT_HANDOFF_SHA="$(git rev-parse HEAD)"
export SEM2ACT_ACCOUNT_ID="YOUR_NON_SECRET_COLAB_PROFILE_ALIAS"
export RUN_ROOT="$(mktemp -d /content/sem2act-v5.XXXXXX)"
mkdir -p "$RUN_ROOT/input/archive" "$RUN_ROOT/input/data" "$RUN_ROOT/provenance"
kaggle datasets download -d rezabarati2/sem2act-v5-rerank-inputs -p "$RUN_ROOT/input/archive"
ZIP="$(find "$RUN_ROOT/input/archive" -maxdepth 1 -type f -name '*.zip' -print -quit)"
test -n "$ZIP"
unzip -q "$ZIP" -d "$RUN_ROOT/input/data"
python3 scripts/verify_v5_rerank_inputs.py --input-dir "$RUN_ROOT/input/data"
python3 scripts/freeze_v5_colab_runtime.py --check
python3 scripts/colab_v5_preflight.py --output "$RUN_ROOT/provenance/colab_preflight.json"
```

Expected: the verifier reports exactly the three committed hashes and
`qrels_present: false`; the preflight says `status: pass` and records
Torch/CUDA, Transformers, Python, T4 name/memory, CUDA driver, account alias,
and lock hash. Do not continue if the private dataset cannot be fetched or the
runtime/profile differs.

Run the model/fixture smoke first. It downloads only the pinned reranker and
scores the non-lockbox fixture; it does not score lockbox queries:

```sh
python3 versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank.py --backend colab --smoke-only --runtime-lock versions/sem2act-v5/manifests/colab_runtime_lock_v1.json --output-root "$RUN_ROOT/provenance/colab_canary"
```

Require `canary_manifest.json` with `status: pass`,
`result_bearing_execution_started: false`, and the exact runtime/model/lock
identity. Stop on any mismatch.

### Smallest valid Colab pilot

After both Colab preflight and fixture smoke pass:

```sh
python3 versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank.py --backend colab --pilot-limit 20 --input-root "$RUN_ROOT/input/data" --output-root "$RUN_ROOT/pilot" --runtime-lock versions/sem2act-v5/manifests/colab_runtime_lock_v1.json
```

Expected: `pilot_rerank.trec`, `run_manifest.json`, and
`runtime_versions.json`; the run manifest must show exactly 20 expected
query IDs, 1,000 pairs, the committed input hashes, the accepted Colab lock,
`qrels_read: false`, and output SHA-256 values. Hash the three outputs and
the preflight/canary receipts into a run-specific report.

**STOP immediately after the pilot.** Send the exact SHA, Colab profile alias,
runtime/preflight/canary records, output manifest and hashes, and any failure
to Codex. Do not launch the Kaggle GPU pilot in parallel or use this pilot in
analysis.

## 4. TMU moon CPU — OpenCode

Moon is CPU-only. The reported profile is 16 vCPU/62 GB RAM with AVX2/AVX512
masked. This amendment permits 16 PyTorch intra-op threads and one inter-op
thread **for the reranker pilot only**. A new host-specific preflight is
mandatory; the old Mac host manifest is not valid Moon evidence. Require at
least 48 GiB effective RAM and 20 GiB free on a verified node-local filesystem.
Home NFS is not permitted for the virtualenv, model/build cache, temporary
files, or outputs.

Before model/input use, verify the OpenCode TMU login identity. Load
`HF_TOKEN` and Kaggle dataset access only from OpenCode's approved external
secret/configuration store, with shell tracing disabled. Do not print values.

From the exact checkout, create a fresh local run directory and exact venv.
The following example places all generated files and caches under node-local
`/tmp`; the preflight must confirm that this mount is local and has adequate
free space:

```sh
export SEM2ACT_HANDOFF_SHA="$(git rev-parse HEAD)"
export TMU_USER="$(id -un)"
export RUN_ROOT="$(mktemp -d /tmp/sem2act-v5.XXXXXX)"
mkdir -p "$RUN_ROOT/local/hf" "$RUN_ROOT/local/torch" "$RUN_ROOT/local/tmp" "$RUN_ROOT/input/archive" "$RUN_ROOT/input/data" "$RUN_ROOT/provenance"
export HF_HOME="$RUN_ROOT/local/hf"
export TORCH_HOME="$RUN_ROOT/local/torch"
export TMPDIR="$RUN_ROOT/local/tmp"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export GGML_CUDA=0
export GGML_METAL=0
python3.12 -m venv "$RUN_ROOT/venv"
source "$RUN_ROOT/venv/bin/activate"
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu
python -m pip install --no-cache-dir transformers==4.57.6
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.version.cuda)'
kaggle datasets download -d rezabarati2/sem2act-v5-rerank-inputs -p "$RUN_ROOT/input/archive"
ZIP="$(find "$RUN_ROOT/input/archive" -maxdepth 1 -type f -name '*.zip' -print -quit)"
test -n "$ZIP"
unzip -q "$ZIP" -d "$RUN_ROOT/input/data"
python scripts/verify_v5_rerank_inputs.py --input-dir "$RUN_ROOT/input/data"
python scripts/moon_v5_preflight.py --node-local-root "$RUN_ROOT/local" --output "$RUN_ROOT/provenance/moon_host_preflight.json"
```

Expected runtime tuple is Python 3.12, Torch `2.7.1+cpu`, no CUDA, and
Transformers `4.57.6`. The host receipt must say `status: pass`, 16 visible
vCPU, at least 48 GiB effective memory, at least 20 GiB node-local space, local
filesystem, AVX2/AVX512 not visible, Torch 16/1 threads, and the exact Moon
lock hash. If it does not, stop; do not tune the thread count.

Run the exact fixture canary on this host:

```sh
python scripts/run_v5_cpu_reranker_canary.py --model-path Qwen/Qwen3-Reranker-0.6B --host-manifest "$RUN_ROOT/provenance/moon_host_preflight.json" --output "$RUN_ROOT/provenance/moon_reranker_canary.json"
```

Require `status: pass`, exact repeated scores at both 4 and 16 threads, the
same fixture ranking at both thread counts, the pinned model revision,
runtime-lock SHA, fixture SHA, and hash of the host receipt. The canary reports
both score vectors and their maximum absolute difference. If fixture rankings
differ, the model cannot be loaded, or any runtime/repeat check fails, stop and
preserve the evidence; do not authorize 16-thread scoring.

### Smallest valid Moon pilot

After the host preflight and canary pass, run the exact first shard in a new,
empty result directory:

```sh
python scripts/run_v5_cpu_rerank_shard.py --shard versions/sem2act-v5/manifests/cpu_shards/reranker/shard-000.json --queries "$RUN_ROOT/input/data/queries.jsonl" --corpus "$RUN_ROOT/input/data/corpus.jsonl" --candidates "$RUN_ROOT/input/data/rerank_candidates.jsonl" --model-path Qwen/Qwen3-Reranker-0.6B --output-dir "$RUN_ROOT/pilot" --moon-host-manifest "$RUN_ROOT/provenance/moon_host_preflight.json" --moon-canary-manifest "$RUN_ROOT/provenance/moon_reranker_canary.json"
```

Expected: `rankings.jsonl`, `failures.json`, and `run_manifest.json`.
The manifest must show exactly 20 queries/1,000 pairs, all three input hashes,
the host/canary/lock/shard hashes, 16 threads, CPU-only runtime, pinned model,
`qrels_read: false`, zero failures, and output hashes.

**STOP immediately after the pilot**, whether it passes or fails. Send the
execution SHA, TMU account alias, host and canary receipts, run manifest,
outputs and hashes, elapsed time, and any error to Codex. Do not start the
full 240-query CPU run or any CPU consumer generation.

## 5. Result/provenance report and global stop conditions

For every attempt, preserve the exact execution SHA, protocol hash, amendment
and runtime-lock hashes, input manifest and file hashes, model/tokenizer ID and
revision, Python/framework/CUDA or CPU identity, hardware and memory, provider
job ID, non-secret account/backend identity, start/end timestamps, status,
expected/accepted query and pair counts, seed applicability, output paths and
SHA-256 values, verification result, and failure evidence. Include preflight
and canary receipts and their hashes. Never include credential values,
environment dumps, private qrels, or lockbox labels.

Use `provenance/templates/pilot_run_manifest.json` as the normalized
reporting checklist. Keep raw outputs and receipts outside Git on the provider
or node-local run directory. Send them to Codex; Codex will decide whether any
compact receipt belongs in Git.

Stop and report on a dirty checkout, wrong identity, unavailable/private input,
hash mismatch, runtime or host drift, missing GPU/CPU capability, failed
canary, unexpected query/pair coverage, nonzero failure count, missing output,
or incomplete provenance. Preserve evidence; do not repair the code or
manifests on the execution host.

## 6. Consumer bundles and the separate CPU conversion blocker

Qwen, Llama, and Mistral consumer bundles are not staged. Do not stage bundles
from the 20-query pilot. Only after a complete 240-query reranker output is
fetched, verified, and accepted by Codex may the committed consumer builder be
used under a new explicit Codex handoff; each bundle must remain qrel-free and
be hash-verified before upload.

The Qwen AWQ-to-GGUF conversion issue is a separate CPU-consumer blocker.
Moon consumer generation is not authorized or practical. Do not substitute a
model, converter, quantizer, or artifact. The frozen GPU Qwen AWQ/vLLM path is
unchanged; the conversion blocker does not block the reranker pilot.
