# OpenCode handoff: Moon reranker

## Pilot verification at the running canonical SHA

The result-bearing pilot is limited to queries `v5-lb-q0001` through
`v5-lb-q0020`, 50 candidates per query, and 1,000 candidate scores. Its
execution checkout must remain exactly
`6d97cd0fe11cc87cd38661ac3a3646ebb36b7599`.

The original runner did not save the Moon hostname. On the same Moon host, after
the runner exits, add `moon_host_provenance.json` to the returned pilot folder.
It must contain `schema_version: 1`, `status: "pass"`, the canonical
`execution_sha`, `protocol_hash`, `hostname` from `socket.gethostname()`, and
these fields copied from a fresh read-only runtime probe: `platform`, `machine`,
`python`, `torch`, `transformers`, `logical_vcpus`, `memory_bytes`,
`torch_cuda_available`, `output_filesystem`, `runtime_lock_sha256`,
`runtime_freeze_sha256`, `avx2_masked`, `avx512f_masked`,
`nfs_quota_probe_bytes_written_and_removed`, `input_files`,
`input_manifest_sha256`, `model_id`, `revision`, `model_snapshot_sha256`, and
`qrels_read: false`. Values must match `moon_preflight.json`; this receipt adds
the missing hostname binding. The probe must not load a model, modify the
runtime, or touch lockbox content. Preserve the full pilot folder and original
files.

Run the deterministic verifier on the exact three-file input bundle:

```bash
python scripts/verify_v5_moon_reranker_pilot.py \
  --execution-sha 6d97cd0fe11cc87cd38661ac3a3646ebb36b7599 \
  --pilot-dir "$MOON_PILOT_ROOT" \
  --input-root /tmp/sem2act-v5/inputs \
  --host-provenance "$MOON_PILOT_ROOT/moon_host_provenance.json" \
  --output "$MOON_PILOT_ROOT/pilot_verification.json"
```

It returns `PASS`, `FAIL`, or `REVIEW`, with reasons, 4/16-thread parity,
input/output SHA-256 values, exact key coverage, runtime facts, and the
hostname-receipt SHA-256. `REVIEW` is not authorization. Return the complete
folder, including the verifier JSON and hostname receipt, to Codex. Do not run
another shard.

## Conditional full 240-query stage

The tracked authorization package is
`manifests/moon_full_reranker_authorization_draft.json`; its status is
`draft-not-authorized`. The amendment is also a draft. The executable runner
rejects that status. Full-run execution requires a separate Codex-issued
`status: authorized` JSON outside the repository, with the clean stage SHA,
accepted pilot verifier JSON SHA-256, exact node-local input/model paths, and a
unique output path on the same NFS mount verified by the pilot.

The following is the prepared command. Codex supplies the concrete stage SHA,
NFS paths, and authorization after reviewing a pilot `PASS`:

```bash
python scripts/run_v5_moon_full_reranker.py \
  --execution-sha "$SEM2ACT_FULL_RERANKER_SHA" \
  --authorization "$MOON_NFS_ROOT/sem2act-v5/full-reranker/authorization.json" \
  --pilot-verification "$MOON_NFS_ROOT/sem2act-v5/pilot/pilot_verification.json" \
  --pilot-root "$MOON_NFS_ROOT/sem2act-v5/pilot" \
  --input-root /tmp/sem2act-v5/inputs \
  --model-path /tmp/sem2act-v5/models/qwen3-reranker-0.6b \
  --output-root "$MOON_NFS_ROOT/sem2act-v5/full-reranker/sem2act-v5-moon-reranker-240q-001"
```

Before any score is computed, the runner checks the PASS receipt SHA, exact
pilot SHA, clean checkout, same hostname, current 16-vCPU/60-GB CPU-only Moon
profile, masked AVX2/AVX512 flags, Python/Torch/Transformers versions, local
scratch, NFS, exact input hashes, model snapshot, runtime lock/freeze, and
authorization paths. It plans 12 disjoint shards of 20 queries. At 16 threads,
the expected duration is about 1.8 hours. Inputs and model cache stay under
`/tmp/sem2act-v5`; every shard checkpoint and output stays on NFS.

The runner stops on the first shard failure. Recovery is never automatic:
Codex must inspect the saved `full_run_state.json`, issue a new recovery
authorization bound to that exact state-file SHA-256, and list exactly the
incomplete shard indexes. The runner then uses the existing missing-key-only
shard logic and refuses changes to accepted keys or inputs. Each recovery
attempt gets a unique authorization and parent-checkpoint receipt. If a resumed
attempt stops again, preserve the checkpoint and wait for another explicit
Codex recovery authorization.

After a complete run, verify before sending the result back:

```bash
python scripts/verify_v5_moon_full_reranker.py \
  --execution-sha "$SEM2ACT_FULL_RERANKER_SHA" \
  --input-root /tmp/sem2act-v5/inputs \
  --run-root "$MOON_NFS_ROOT/sem2act-v5/full-reranker/sem2act-v5-moon-reranker-240q-001" \
  --output "$MOON_NFS_ROOT/sem2act-v5/full-reranker/full_reranker_verification.json"
```

The 240-query output contains the original per-query scorer digests and complete
rankings. `rerank.trec` uses a strictly monotonic rank encoding because the
consumer input builder uses TREC row order and top-3 document IDs; the encoding
is not presented as a model score. The source ranking and native-score digests
remain in `rankings.jsonl`. The verifier checks 240 queries, 12,000 candidates,
all 12 checkpoints, qrel-free manifests, and hashes for every output file.

Only after reviewing a full verifier `PASS`, Codex may create a separate
acceptance receipt for consumer-bundle construction. It must contain
`status: "accepted"`, the accepted `execution_sha`,
`full_verification_sha256`, and `source_reranker_output_sha256` from the verifier
result. That receipt authorizes bundle construction only; it does not authorize
Colab or Kaggle inference.

**STOP after the single 240-query stage.** Return all output/provenance files to
Codex. Do not create consumer bundles, submit Kaggle jobs, or start another
Moon shard until Codex verifies and separately accepts the full output.
