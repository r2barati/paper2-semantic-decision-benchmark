# OpenCode handoff: later Llama and Mistral consumers

Llama and Mistral use the same accepted, qrel-free 720-row evidence bundle as
Qwen. Create their family-specific bundles only after the 240-query Moon
reranker verifier passes and Codex accepts that output. Use the Kaggle runtime
lock for these Kaggle paths; do not inherit Colab's Torch/CUDA image.

## Frozen routes

| Family | Model revision | Backend and precision | Result calls |
|---|---|---|---:|
| Llama | `meta-llama/Llama-3.1-8B-Instruct` / `0e9e39f249a16976918f6564b8830bc894c89659` | Transformers, float16, bitsandbytes NF4 4-bit, Kaggle one-visible-T4 policy | 1,920 |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` / `d79d1742f78eb0cc788c11e5b41a7539d7cb56ef` | Transformers, float16, bitsandbytes NF4 4-bit, Kaggle one-visible-T4 policy | 1,920 |

Each family uses the 240-query `rerank` + `oracle` analysis scope; C1 makes one
call and C3 makes three calls per evidence row. The bundle still contains all
three systems, including BM25, and exactly 720 rows. Each run also performs the
fixed two-call non-lockbox model smoke before its 1,920 result calls.

Kaggle identity is `rezabarati2`. The verified entitlement is two T4 GPUs, but
the frozen V5 runtime requests two and exposes/uses only `cuda:0` (one T4). Its
base image is `torch==2.10.0+cu128`, CUDA `12.8`, Transformers `4.57.6`, and
bitsandbytes `0.48.1`. The Qwen Colab image (`torch==2.11.0+cu128`) is not a
Kaggle substitute. OpenCode must use the clean SHA and exact Kaggle runtime
preflight in the branch handoff.

## Conditional bundle and job commands

The commands below are prepared, not authorized for use yet. A separate Codex
handoff is required after the Qwen stage is accepted. The acceptance receipt
must bind the PASS full-reranker verification JSON and `rerank.trec` SHA-256.
`RETRIEVAL_ROOT`, `FULL_RERANKER_ROOT`, and `ACCEPTANCE_RECEIPT` are the exact
accepted artifacts OpenCode returns to Codex; Codex supplies their paths and
the receipt. Build each family bundle only on the clean execution SHA that
Codex supplies; its manifest records that SHA.

```bash
python scripts/build_v5_consumer_bundle.py \
  --family llama \
  --retrieval-root "$RETRIEVAL_ROOT" \
  --reranker-root "$FULL_RERANKER_ROOT" \
  --reranker-verification "$FULL_RERANKER_ROOT/full_reranker_verification.json" \
  --acceptance-receipt "$ACCEPTANCE_RECEIPT" \
  --runtime-lock versions/sem2act-v5/manifests/kaggle_runtime_lock.json \
  --output-root versions/sem2act-v5/runtime/kaggle_llama_inputs
```

Prepare Mistral with the same generator command using `--family mistral`, output
root `versions/sem2act-v5/runtime/kaggle_mistral_inputs`, and job `v5-mistral`.
The staging and inference commands remain gated until Codex issues that
family-specific handoff. It must first refresh Kaggle identity/quota and pass
the non-lockbox V5 canary; these commands are a future operator recipe, not
current authorization:

```bash
python scripts/kaggle_compute.py --preflight
SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 python scripts/kaggle_compute.py v5-canary --push
python scripts/kaggle_compute.py v5-canary --watch
python scripts/kaggle_compute.py v5-canary --fetch
python scripts/kaggle_compute.py v5-llama --dataset create
```

After dataset creation, use the `verification_command` written to
`versions/sem2act-v5/manifests/upload/sem2act-v5-llama-inputs.json` to verify
the remote version and exact file hashes. Then the family-specific result route
is:

```bash
SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 python scripts/kaggle_compute.py v5-llama --push
python scripts/kaggle_compute.py v5-llama --watch
python scripts/kaggle_compute.py v5-llama --fetch
python scripts/verify_v5_kaggle_consumer.py \
  --family llama \
  --execution-sha "$SEM2ACT_EXECUTION_SHA" \
  --bundle-manifest-sha256 "$SEM2ACT_BUNDLE_MANIFEST_SHA256" \
  --bundle-root versions/sem2act-v5/runtime/kaggle_llama_inputs \
  --dataset-manifest versions/sem2act-v5/manifests/upload/sem2act-v5-llama-inputs.json \
  --kaggle-preflight versions/sem2act-v5/runtime/kaggle_preflight.json \
  --canary-manifest versions/sem2act-v5/runtime/kaggle_canary/canary_manifest.json \
  --output-root versions/sem2act-v5/runtime/model_outputs/llama \
  --output versions/sem2act-v5/runtime/model_outputs/llama_verification.json
```

For Mistral use job `v5-mistral`, dataset manifest
`manifests/upload/sem2act-v5-mistral-inputs.json`, bundle root
`runtime/kaggle_mistral_inputs`, output root `runtime/model_outputs/mistral`,
and sibling verifier output `runtime/model_outputs/mistral_verification.json`.
The fetch manifest now binds the execution SHA, Kaggle kernel source SHA,
runtime lock/freeze hashes, and remote-verified dataset manifest. The family
verifier checks these receipts plus model/tokenizer revision, exact 1,920 calls
(480 C1 and 1,440 C3), 960 beliefs, schemas, two-call smoke, runtime packages,
all input hashes, and every fetched output hash.

**STOP after each 1,920-call family run** and return its complete output folder
and verifier JSON to Codex for review and a separate next-stage decision.

## CPU consumer route remains separate

Moon's 16-thread authorization is reranker-only. AWQ-to-GGUF conversion and
CPU consumer execution remain blocked and require a separate source,
quantization, and runtime amendment. Do not use the AWQ artifact as GGUF, pick a
different quantization, or substitute a model to make a CPU route run.

**STOP after each 1,920-call family run** and return its complete artifact set
to Codex for verification and a separate next-stage decision.
