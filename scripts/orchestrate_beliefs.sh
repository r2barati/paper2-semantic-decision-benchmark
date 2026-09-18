#!/bin/bash
# Unattended orchestrator: AWQ b1/b2 completion -> fetch/merge -> phi4-probe +
# awq-b3 -> phi4 belief shards -> merge + single-query assembly validation.
# Execution plumbing ONLY (push/watch/fetch/merge of already-frozen artifacts).
# STOPS nonzero on any FAILED job, missing output, or failed verification gate.
# Log: /tmp/orchestrate.log
cd /Users/sanamimani/paper2_semantic_decision_benchmark || exit 1
LOG=/tmp/orchestrate.log
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
fail() { say "ABORT: $*"; exit 1; }

wait_complete() { # $1=job; returns 0 iff terminal state COMPLETE
  say "waiting on $1 ..."
  out=$(python3 scripts/kaggle_compute.py "$1" --watch 2>&1 | tail -3)
  echo "$out" >> $LOG
  echo "$out" | grep -q "COMPLETE" || { say "$1 terminal: $out"; return 1; }
  say "$1 COMPLETE"
}

fetch_job() { # $1=job
  say "fetching $1 ..."
  python3 scripts/kaggle_compute.py "$1" --fetch >> $LOG 2>&1 \
    || fail "fetch $1 failed (see log)"
  say "$1 fetched+installed"
}

push_retry() { # $1=job; retries while GPU sessions are full
  for i in $(seq 1 40); do
    out=$(python3 scripts/kaggle_compute.py "$1" --push 2>&1 | tail -1)
    echo "$out" >> $LOG
    echo "$out" | grep -q '"versionNumber": [1-9]' && { say "$1 pushed"; return 0; }
    echo "$out" | grep -q "Maximum batch" || fail "push $1 rejected: $out"
    say "push $1: slots full, retry $i/40 in 5 min"
    sleep 300
  done
  fail "push $1: retries exhausted"
}

say "=== orchestrator start ==="
wait_complete awq-b1 || fail "awq-b1 not COMPLETE"
fetch_job awq-b1
wait_complete awq-b2 || fail "awq-b2 not COMPLETE"
fetch_job awq-b2
python3 tools/merge_awq_cache.py --merge >> $LOG 2>&1 || fail "AWQ merge failed"
say "AWQ b0-b2 merged"

push_retry phi4-probe
push_retry awq-b3
wait_complete phi4-probe || fail "phi4-probe not COMPLETE"
fetch_job phi4-probe
wait_complete awq-b3 || fail "awq-b3 not COMPLETE"
fetch_job awq-b3
python3 tools/merge_awq_cache.py --merge >> $LOG 2>&1 || fail "AWQ merge failed"
say "AWQ complete (4 shards)"

# Phi-4 belief shards, two waves (2 GPU slots)
wave() { # $1=bN $2=start $3=end $4=bM $5=start $6=end
  echo "{\"start\": $2, \"end\": $3}" > kaggle/inputs_phi4/shard.json
  python3 scripts/kaggle_compute.py "$1" --dataset version >> $LOG 2>&1 \
    || fail "phi4 dataset version ($2-$3) failed"
  push_retry "$1"
  echo "{\"start\": $5, \"end\": $6}" > kaggle/inputs_phi4/shard.json
  python3 scripts/kaggle_compute.py "$4" --dataset version >> $LOG 2>&1 \
    || fail "phi4 dataset version ($5-$6) failed"
  push_retry "$4"
  wait_complete "$1" || fail "$1 not COMPLETE"
  fetch_job "$1"
  wait_complete "$4" || fail "$4 not COMPLETE"
  fetch_job "$4"
}
wave phi4-b0 0 50 phi4-b1 50 100
wave phi4-b2 100 150 phi4-b3 150 200
python3 tools/merge_awq_cache.py --merge --shard-dir runs/v3main_phi4 \
  --cache-dir results/v3main/beliefs_cache_phi4 >> $LOG 2>&1 \
  || fail "phi4 merge failed"
say "phi4 merged"

# Single-query assembly validation (cache hits only), then remove partials
PAPER2_V3_CACHE_DIR=results/v3main/beliefs_cache_phi4 \
  python3 -m src.beliefs_v3main --model stelterlab/phi-4-AWQ --queries v3-q001 \
  --consumers C1 C3 >> $LOG 2>&1 || fail "phi4 assembly validation failed"
N=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('results/v3main/beliefs_stelterlab_phi-4-AWQ.parquet')))")
[ "$N" = "42" ] || fail "phi4 validation rows=$N (expected 42)"
rm results/v3main/beliefs_stelterlab_phi-4-AWQ.parquet \
   results/v3main/beliefs_stelterlab_phi-4-AWQ.jsonl
say "phi4 single-query validation ok (42 rows, partials removed)"
say "=== orchestrator DONE: beliefs ready for freeze gate ==="
