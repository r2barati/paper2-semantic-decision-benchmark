# Sem2Act / Paper 2 V5 pilot handoff

**Scientific protocol:** `2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022`.
These amendments change execution configuration only. Prompts, lockbox,
estimands, model revisions, decoding, keyspace, coverage, and analysis remain
frozen. Use the canonical execution SHA supplied with this handoff; never run
from a moving branch or dirty checkout.

## 1. Checkout and shared rules

For each operator, clone the published execution branch and verify before
staging inputs or models:

```sh
git clone --branch sem2act-v5-pilot-execution-20260929 --single-branch https://github.com/r2barati/paper2-semantic-decision-benchmark.git Sem2Act
cd Sem2Act
git rev-parse HEAD
git status --porcelain -uall
git log -1 --format='%H%n%ci%n%s'
```

`HEAD` must equal the supplied canonical SHA and status must be empty. Never
put tokens or credentials in command arguments, output archives, or logs. Keep
weights, package/build caches, temporary files, and inputs on node-local disk
unless a section below explicitly names a persistent output directory.
After cloning the branch, check out the supplied SHA in detached mode and
repeat the clean-worktree check before staging or execution:

```sh
git checkout --detach <CANONICAL_EXECUTION_SHA>
git rev-parse HEAD
git status --porcelain -uall
```

## 2. Authorized first result pilot: OpenCode / Moon reranker

This is one deterministic 20-query shard: queries `v5-lb-q0001`–`q0020`, 50
candidates each, 1,000 candidate scores. It uses the unchanged Transformers
CPU reranker protocol and its exact accepted model snapshot. The run is
authorized only if the launcher passes its Moon host/runtime checks and the
non-lockbox fixture parity gate. It compares all three fixed fixture documents
at 4 and 16 threads; repeats must be exact, rankings must match, and each score
must differ by at most `1e-6`. A failed gate stops before lockbox scoring.

The prior authorized handoff SHA `6d97cd0fe11cc87cd38661ac3a3646ebb36b7599`
is superseded for Moon execution. OpenCode reported that its attempt stopped
before scoring: all accepted model files were present and hash-identical, but
the downloader also staged `.gitattributes` and `README.md`; the validator
correctly rejected those extra files. The same report measured 600 soft / 1200
hard CPU seconds per process and about 8.99 CPU seconds per scored pair. This
is resolved by the replacement staging helper and sequential one-query
subprocesses. No model hashes or scoring rules change.

Moon profile: Linux x86_64, 16 visible vCPUs, at least 60 GB RAM, AVX2/AVX512F
masked, Python `3.9.19`, `torch==2.8.0+cpu`, `transformers==4.57.6`, CPU only.
Weights, inputs, Hugging Face cache, and temporary files must be under
`/tmp/sem2act-v5`. Checkpoint and provenance go to the operator's NFS home
directory; the launcher probes an actual 10 MB write/fsync/delete on NFS to
check this account's effective quota. It checks the AVX2/AVX512 mask on every
processor record. Do not build on NFS.

Stage the exact qrel-free Kaggle input version and pinned model snapshot on
node-local scratch. The launcher checks all inputs against the committed
version-1 SHA-256 manifest. The staging helper derives its allowlist from the
frozen `cpu_reranker_canary.json` snapshot manifest, downloads only those 12
paths, and verifies the unchanged per-file and aggregate hashes. Use a fresh
model directory; it refuses to stage into a non-empty directory. Do not copy
in README or `.gitattributes` and do not weaken `validate_model`.

```sh
mkdir -p /tmp/sem2act-v5/rerank-inputs
kaggle datasets download rezabarati2/sem2act-v5-rerank-inputs --path /tmp/sem2act-v5/rerank-inputs --unzip
python3 scripts/stage_v5_moon_reranker_model.py \
  --output /tmp/sem2act-v5/models/qwen3-reranker-0.6b-12files-pilot-001 \
  --manifest-output /tmp/sem2act-v5/model-staging-pilot-001.json
```

The frozen shard is still 20 queries × 50 candidates. Moon executes one whole
query (50 pairs) per sequential subprocess, preserving the frozen query and
candidate order and reusing the same scoring implementation and 16-thread
float32 settings. The operator measurement implies about 449.5 scoring CPU
seconds per query. The launcher caps each child at 540 soft / 600 hard CPU
seconds and refuses a host limit too tight for that cap. Each child records its
CPU use and limit; aggregation verifies exactly one record per query, all 50
unique candidates per query, all 1,000 expected pairs, deterministic
concatenation, and the existing 4-vs-16 parity gate. The complete subprocess
commands, timestamps, return codes, logs and output hashes are recorded in the
run manifest.

Run exactly one pilot:

```sh
python3 scripts/run_v5_moon_reranker_pilot.py \
  --expected-sha <CANONICAL_EXECUTION_SHA> \
  --input-root /tmp/sem2act-v5/rerank-inputs \
  --model-path /tmp/sem2act-v5/models/qwen3-reranker-0.6b-12files-pilot-001 \
  --model-staging-manifest /tmp/sem2act-v5/model-staging-pilot-001.json \
  --output-root "${HOME}/sem2act-v5/pilot-reranker-001"
```

Use the fresh `pilot-reranker-001` output path so the previous stopped attempt
remains intact. Return the model staging manifest and SHA-256, plus
`pilot_manifest.json`, `pilot_authorization.json`, `moon_preflight.json`,
`moon_reranker_canary.json`, `failures.json`, `run_manifest.json`, and every
`chunks/query-*/` output and log. Return `result_bearing_started.json` and
`rankings.jsonl` when present; a pre-scoring block should instead include
`pilot_blocked.json`. The top-level run manifest must list all launched subprocesses and the
hashes of their manifests, ranking files, failure files, start markers, stdout,
and stderr. If any preflight step—including model validation—fails, return
`pilot_blocked.json` and stderr. Preserve partial checkpoints and failures. Do
not retry a failed/interrupted pilot until Codex has reviewed its evidence.

**STOP after shard 000.** Do not run shard 001 or any later shard, stage
consumer inputs, or inspect qrels. Send the output archive and provenance to
Codex for acceptance.

## 3. Muse / Colab Qwen consumer pilot

Colab has its own explicit image policy: one Tesla T4 with at least 14 GiB
visible VRAM, `torch==2.11.0+cu128`, and PyTorch CUDA build `12.8`. Capture the
base Python version in the preflight and do not replace the Colab image.
Package installation may not alter the Torch/CUDA tuple. No secondary GPU,
fallback image, or runtime substitution is allowed. Model and package caches
stay on Colab local storage; write the output archive to mounted Drive. Supply
`HF_TOKEN` through Colab Secrets only, without printing it:

```python
from google.colab import drive, userdata
import os
drive.mount("/content/drive")
token = userdata.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN is missing from Colab Secrets")
os.environ["HF_TOKEN"] = token
del token
```

This full-family Qwen stage is **input-gated and is not runnable yet**. Its
smallest valid keyspace is the complete 240-query primary family stage (720
evidence rows and 2,880 calls); the current verifier does not accept partial
consumer coverage. It requires a complete reranker result that Codex has
accepted, then a qrel-free Qwen input bundle and its exact
`consumer_inputs_qwen.json` SHA-256. No consumer bundles are staged yet. The
exact Torch/CUDA-preserving package install and fixed non-lockbox Qwen model
smoke are mandatory gates in the same job; if either fails, stop before the
result kernel and return the log for a separate runtime amendment. Do not
change the image tuple or package pins to recover.

Once Codex sends that bundle and SHA, run:

```sh
python3 scripts/colab_compute.py qwen \
  --expected-sha <CANONICAL_EXECUTION_SHA> \
  --expected-input-manifest-sha256 <CODEX_ISSUED_QWEN_INPUT_MANIFEST_SHA256> \
  --input-root /content/sem2act-v5/qwen-inputs \
  --output-root /content/drive/MyDrive/sem2act-v5/qwen-pilot-000
```

The launcher checks exact SHA/clean status, runtime tuple, T4/VRAM, bundle
hashes, and qrel-free inputs. The job runs its fixed non-lockbox model smoke
immediately before handing the bundle to the result kernel. Capture `colab_preflight.json`,
`qwen_lightning_job_manifest.json`, `qwen_smoke_manifest.json`,
`run_manifest.json`, `runtime_versions.json`, `beliefs.jsonl`,
`raw_outputs.jsonl`, the input bundle manifest, runtime/notebook snapshot, and
all SHA-256 hashes.

**STOP after the Qwen stage.** Do not run Llama or Mistral, assemble beliefs,
or analyze outcomes. Return the complete output archive and provenance to
Codex.

## 4. OpenCode / Kaggle

The V5 owner is `rezabarati2`; OpenCode must confirm its authenticated account
owns `rezabarati2/sem2act-v5-rerank-inputs` before quota preflight. Kaggle's
`NvidiaTeslaT4` accelerator maps to two T4 devices in its current kernel
accelerator list ([Kaggle CLI docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-push)).
The frozen model shape exposes only GPU 0 to PyTorch; GPU 1 is allocated but
masked. The required image tuple
is Python `3.12`, `torch==2.10.0+cu128`, CUDA build `12.8`. Exact package pins
must preserve the base Torch/CUDA tuple. The in-kernel canary verifies two
physical T4s, one visible T4, and a real CUDA matmul.

Only account/quota/runtime canary preflight is authorized now. No Kaggle
result-stage job is authorized in this handoff. Run the non-result preflight:

```sh
python3 scripts/kaggle_compute.py --preflight
```

After the account and positive GPU quota pass, the non-result CUDA/runtime
canary is the next authorized Kaggle action:

```sh
SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 python3 scripts/kaggle_compute.py v5-canary --push
python3 scripts/kaggle_compute.py v5-canary --watch
python3 scripts/kaggle_compute.py v5-canary --fetch
```

**STOP after the canary.** Return the fresh quota/owner receipt, canary output,
runtime manifest, and hashes to Codex. A passing canary does not authorize a
Kaggle result stage in this handoff.

## 4a. Kaggle 20-query reranker pilot (`v5-rerank-pilot`)

The Moon route is blocked and retained as evidence only; see
`versions/sem2act-v5/manifests/moon_reranker_pilot_blocked.json`. Kaggle is
therefore the authorized backend for the same unchanged 20-query / 1,000-candidate
shard. Authorization is
`versions/sem2act-v5/amendments/kaggle_reranker_pilot_v1.yaml`
(SHA-256 `9c90f8745721e5fe4bc06f93299ac37cff26f512af6eda7f4cb0c22b4c655fe8`),
locked by `kaggle_reranker_runtime_lock.json`
(SHA-256 `6c144ac7d1e36ba2d68914466b054aab1dd9e73e52127411d0a652680a3db34c`).
The protocol, prompt, model revision, dtype, 1,024-token context, batch 32,
candidate order, and qrel exclusion are unchanged from the full kernel.

The pilot kernel embeds the non-result-bearing smoke fixture by hash, because
the frozen input dataset version 1 carries only `corpus.jsonl`, `queries.jsonl`,
and `rerank_candidates.jsonl`. It attaches that already remote-verified dataset
directly, so no local dataset copy is needed to push.

Confirm the working tree and freeze first:

```sh
git status --porcelain          # must be empty
git rev-parse HEAD              # must be the canonical execution SHA
python3 scripts/freeze_v5_kaggle_reranker_pilot.py --check
python3 -m pytest tests/test_v5_kaggle_reranker_pilot.py -q
```

Then run the pilot, in order. The push gate refuses without a fresh positive
owner/quota preflight and a `canary-passed` record:

```sh
python3 scripts/kaggle_compute.py --preflight
SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 python3 scripts/kaggle_compute.py v5-canary --push
python3 scripts/kaggle_compute.py v5-canary --watch
python3 scripts/kaggle_compute.py v5-canary --fetch
SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 python3 scripts/kaggle_compute.py v5-rerank-pilot --push
python3 scripts/kaggle_compute.py v5-rerank-pilot --watch
python3 scripts/kaggle_compute.py v5-rerank-pilot --fetch
```

`--fetch` re-verifies provenance and writes
`versions/sem2act-v5/runtime/kaggle_rerank_pilot/pilot_pending_acceptance.json`.
It is pending, not accepted.

**STOP after the pilot.** Do not run the full 240-query `v5-rerank` job. Return
`rerank.trec`, `run_manifest.json`, `runtime_versions.json`, the pending record,
and all SHA-256 hashes to Codex. The full job stays blocked until Codex writes
`versions/sem2act-v5/manifests/kaggle_reranker_pilot_acceptance.json`; the gate
fails with `v5-rerank is not authorized` until then.

Do not submit Qwen, Llama, or Mistral in this handoff. The
Qwen/Llama/Mistral consumer bundles do not exist yet. Kaggle consumer stages
also require a later Codex handoff naming one exact remote dataset version and
upload-manifest SHA.

## 5. Consumer bundle staging and CPU consumer blocker

Do not fabricate partial consumer inputs from the 20-query Moon pilot. After a
full 240-query reranker output is accepted, Codex will provide the accepted
`rerank.trec` and a new stage handoff. Then build each family bundle once:

```sh
python3 versions/sem2act-v5/experiments/stage_consumer_inputs.py
python3 scripts/kaggle_compute.py v5-qwen --dataset create
python3 scripts/kaggle_compute.py v5-llama --dataset create
python3 scripts/kaggle_compute.py v5-mistral --dataset create
```

The builder refuses to overwrite an existing bundle or manifest. Verify each
remote dataset version and file hashes before submitting any consumer job. The
same accepted Qwen evidence bundle plus its manifest may be copied to Colab;
the Colab job separately records its Colab runtime lock/freeze hashes.

AWQ-to-GGUF remains a separate blocker for **CPU consumer inference**. The
Moon pilot uses the pinned Transformers reranker and does not convert or
quantize consumer models. Qwen AWQ conversion and Llama source access remain
open for the CPU consumer route; neither blocks GPU consumers after their own
input and runtime gates pass.

## 6. Provenance and universal stop rules

For each execution, report execution SHA/branch, protocol hash, runtime lock
and freeze hashes, backend/account, provider job or notebook identity,
timestamp, model/tokenizer revisions and snapshot hash, input manifest and
file hashes, full command, runtime versions, output file hashes, verifier
result, and `qrels_read: false`. Never record secrets.

Stop for a wrong/dirty SHA, owner or access mismatch, changed Torch/CUDA
tuple, failed host/runtime/model canary, missing or mismatched input, partial
coverage, schema/hash error, disk/quota issue, or unexpected output. After the
smallest authorized pilot on each backend, return evidence to Codex and wait
at its STOP point.
