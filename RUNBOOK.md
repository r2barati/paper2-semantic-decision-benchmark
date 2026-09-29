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

## 2. Authorized first result pilot: OpenCode / Moon reranker

This is one deterministic 20-query shard: queries `v5-lb-q0001`–`q0020`, 50
candidates each, 1,000 candidate scores. It uses the unchanged Transformers
CPU reranker protocol and its exact accepted model snapshot. The run is
authorized only if the launcher passes its Moon host/runtime checks and the
non-lockbox fixture parity gate. It compares all three fixed fixture documents
at 4 and 16 threads; repeats must be exact, rankings must match, and each score
must differ by at most `1e-6`. A failed gate stops before lockbox scoring.

Moon profile: Linux x86_64, 16 visible vCPUs, at least 60 GB RAM, AVX2/AVX512F
masked, Python `3.9.19`, `torch==2.8.0+cpu`, `transformers==4.57.6`, CPU only.
Weights, inputs, Hugging Face cache, and temporary files must be under
`/tmp/sem2act-v5`. Checkpoint and provenance go to the operator's NFS home
directory; the launcher probes an actual 10 MB write/fsync/delete on NFS to
check this account's effective quota. It checks the AVX2/AVX512 mask on every
processor record. Do not build on NFS.

Stage the exact qrel-free Kaggle input version and pinned model snapshot on
node-local scratch. The launcher checks all files against the committed
version-1 SHA-256 manifest and checks the model against the accepted snapshot
manifest:

```sh
mkdir -p /tmp/sem2act-v5/rerank-inputs
kaggle datasets download rezabarati2/sem2act-v5-rerank-inputs --path /tmp/sem2act-v5/rerank-inputs --unzip
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3-Reranker-0.6B", revision="e61197ed45024b0ed8a2d74b80b4d909f1255473", local_dir="/tmp/sem2act-v5/models/qwen3-reranker-0.6b")'
```

Run exactly one pilot:

```sh
python3 scripts/run_v5_moon_reranker_pilot.py \
  --expected-sha <CANONICAL_EXECUTION_SHA> \
  --input-root /tmp/sem2act-v5/rerank-inputs \
  --model-path /tmp/sem2act-v5/models/qwen3-reranker-0.6b \
  --output-root "${HOME}/sem2act-v5/pilot-reranker-000"
```

Return `pilot_manifest.json`, `pilot_authorization.json`, `moon_preflight.json`,
`moon_reranker_canary.json`, `result_bearing_started.json` (if present),
`rankings.jsonl`, `failures.json`, and `run_manifest.json`, with SHA-256 for
each. If the host/parity gate fails, return `pilot_blocked.json`; if an earlier
checkout, freeze, input, or model hash check stops the launcher, return its
command and stderr. Preserve partial checkpoints and failures. Do not retry a
failed/interrupted pilot until Codex has reviewed its evidence.

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

Do not submit `v5-rerank`, Qwen, Llama, or Mistral in this handoff. The
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
