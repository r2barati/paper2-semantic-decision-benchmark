# Muse handoff: Sem2Act V5 Qwen on Colab

This is a **conditional** handoff. Do not create or copy the real Qwen bundle
until the full Moon reranker output has a passing full-run verifier and Codex
has written a separate `status: accepted` receipt.

After acceptance, Codex creates the Qwen bundle with this command (the three
artifact paths and acceptance receipt are supplied with that handoff):

```bash
python scripts/build_v5_consumer_bundle.py \
  --family qwen \
  --retrieval-root "$RETRIEVAL_ROOT" \
  --reranker-root "$FULL_RERANKER_ROOT" \
  --reranker-verification "$FULL_RERANKER_ROOT/full_reranker_verification.json" \
  --acceptance-receipt "$ACCEPTANCE_RECEIPT" \
  --runtime-lock versions/sem2act-v5/manifests/colab_runtime_lock.json \
  --output-root "$SEM2ACT_QWEN_BUNDLE_STAGE_ROOT"
```

The generator refuses an unaccepted verifier, mismatched acceptance SHA,
non-frozen runtime lock, or existing non-empty bundle path. It emits a
deterministic 240-query/720-row input set and reports the resulting bundle
manifest SHA-256. Copy the complete staged directory to
`/content/sem2act-v5/qwen-inputs` only after Codex returns that SHA and accepts
the bundle build record. The bundle retains `metadata.true_regime` for output
evaluation joins; prompt-key verification proves that field is not part of the
model request.

## Frozen execution and inputs

- Checkout: the clean `sem2act-v5-nextstage-prep-20260929` commit reported by Codex.
- Runtime: one Tesla T4 with at least 14 GiB visible VRAM; the validated Colab
  image tuple is `torch==2.11.0+cu128`, CUDA `12.8`. Record the base Python
  version. Do not replace the image or install a different Torch/CUDA tuple.
- Model: `Qwen/Qwen3-8B-AWQ`, revision
  `4da05a8edb55c6046cce958586c33b61da07bb79`; vLLM `0.11.0`, AWQ, float16,
  one visible GPU, eager mode, thinking disabled, seed 0, temperature 0.
- Bundle: `consumer_inputs_qwen.json`, `evidence_inputs.jsonl`,
  `model_inputs_manifest.json`, `model_spec.json`,
  `runtime_smoke_v1.json`, `consumers_v3.py`, `interpreter.py`, and `events.py`.
  Copy the complete bundle directory; compare the manifest SHA-256 with the
  value Codex provides.
- Workload: 240 query IDs × 3 retrieval systems = 720 evidence rows. The
  result-bearing stage is exactly 2,880 calls: one C1 call and three C3 calls
  per row. The separate fixed non-lockbox runtime smoke remains a preflight,
  not part of the 2,880 lockbox calls.

## Exact launch

Set `SEM2ACT_EXECUTION_SHA` to the full Git SHA supplied by Codex and
`SEM2ACT_QWEN_BUNDLE_SHA256` to the SHA-256 of `consumer_inputs_qwen.json`.
Mount Drive and place the complete input bundle at
`/content/sem2act-v5/qwen-inputs`. Supply `HF_TOKEN` through Colab Secrets.
Then run:

```bash
test "$(git rev-parse HEAD)" = "$SEM2ACT_EXECUTION_SHA"
test -z "$(git status --porcelain -uall)"
python scripts/colab_compute.py qwen \
  --expected-sha "$SEM2ACT_EXECUTION_SHA" \
  --expected-input-manifest-sha256 "$SEM2ACT_QWEN_BUNDLE_SHA256" \
  --input-root /content/sem2act-v5/qwen-inputs \
  --output-root /content/drive/MyDrive/sem2act-v5/qwen-pilot-000
```

After a successful exit, verify the returned files without launching another
stage:

```bash
python scripts/verify_v5_colab_qwen.py \
  --execution-sha "$SEM2ACT_EXECUTION_SHA" \
  --bundle-manifest-sha256 "$SEM2ACT_QWEN_BUNDLE_SHA256" \
  --bundle-root /content/sem2act-v5/qwen-inputs \
  --output-root /content/drive/MyDrive/sem2act-v5/qwen-pilot-000 \
  --output /content/drive/MyDrive/sem2act-v5/qwen-pilot-000-verification.json
```

The verifier binds the clean Git SHA, bundle manifest and file hashes, Colab runtime
lock/freeze, T4 and Torch/CUDA facts, model/tokenizer revision, smoke result,
2,880 cache keys, raw output schema, 1,440 belief rows, failure list, and SHA-256
for every output/checkpoint file. It also verifies label-independent model
request keys and the exact expected output tree. `PASS` is required for acceptance. `FAIL` or
`REVIEW` means stop and send the complete output directory to Codex.

## Checkpoint and STOP rules

Each call checkpoint is a JSON file under `cache_out/`; the run also emits
`raw_outputs.jsonl`, `beliefs.jsonl`, `failures.json`, `runtime_versions.json`,
the family smoke manifest, and the Lightning job manifest. Keep the whole output
directory intact. If Colab disconnects, any call fails, or a verifier check is
not `PASS`, do not rerun into this directory or copy partial cache files into a
new attempt. Return the checkpoint directory to Codex for review and a new
explicit recovery decision.

**STOP after this one 240-query Qwen stage.** Do not submit Llama/Mistral or
another Qwen run until Codex accepts the verifier output and issues the next
handoff.
