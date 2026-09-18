#!/bin/bash
# Qwen14 belief-wave continuation. Starts patiently: waits for qwen14-probe to
# exist (polls status until non-404), watches it, fetches it (fetch
# ENFORCES probe gates — abort here means probe failed, do NOT proceed).
# Then: ensure inputs dataset -> belief waves b0/b1, b2/b3 -> merge ->
# single-query validation -> STOP (freeze-gate assembly is the next deliberate
# step, not automatic). Execution plumbing ONLY. Log: /tmp/orchestrate_q14.log
cd /Users/sanamimani/paper2_semantic_decision_benchmark || exit 1
LOG=/tmp/orchestrate_q14.log
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
fail() { say "ABORT: $*"; exit 1; }

say "=== qwen14-beliefs start: awaiting qwen14-probe ==="
for i in $(seq 1 72); do
  out=$(python3 scripts/kaggle_compute.py qwen14-probe --status 2>&1 | tail -1)
  echo "$out" >> $LOG
  echo "$out" | grep -q "COMPLETE\|RUNNING\|QUEUED" && { say "probe active: $out"; break; }
  [ "$i" = "72" ] && fail "qwen14-probe never appeared (6h)"
  sleep 300
done

wait_complete() {
  say "waiting on $1 ..."
  out=$(python3 scripts/kaggle_compute.py "$1" --watch 2>&1 | tail -3)
  echo "$out" >> $LOG
  echo "$out" | grep -q "COMPLETE" || { say "$1 terminal: $out"; return 1; }
  say "$1 COMPLETE"
}

fetch_job() {
  say "fetching $1 ..."
  python3 scripts/kaggle_compute.py "$1" --fetch >> $LOG 2>&1 \
    || fail "fetch $1 failed (see log)"
  say "$1 fetched+installed"
}

push_retry() {
  for i in $(seq 1 60); do
    out=$(python3 scripts/kaggle_compute.py "$1" --push 2>&1 | tail -1)
    echo "$out" >> $LOG
    echo "$out" | grep -q '"versionNumber": [1-9]' && { say "$1 pushed"; return 0; }
    echo "$out" | grep -q "Maximum batch" || fail "push $1 rejected: $out"
    say "push $1: slots full, retry $i/60 in 5 min"
    sleep 300
  done
  fail "push $1: retries exhausted"
}

wait_complete qwen14-probe || fail "qwen14-probe not COMPLETE"
fetch_job qwen14-probe
say "qwen14 probe PASSED gates (fetch verified)"

say "ensuring dataset paper2-v3qwen14-inputs ..."
if python3 scripts/kaggle_compute.py qwen14-b0 --dataset version >> $LOG 2>&1; then
  say "dataset versioned"
else
  say "version failed; trying create ..."
  python3 scripts/kaggle_compute.py qwen14-b0 --dataset create >> $LOG 2>&1 \
    || fail "dataset create failed"
  say "dataset created"
fi

wave() {
  echo "{\"start\": $2, \"end\": $3}" > kaggle/inputs_qwen14/shard.json
  python3 scripts/kaggle_compute.py "$1" --dataset version >> $LOG 2>&1 \
    || fail "qwen14 dataset version ($2-$3) failed"
  push_retry "$1"
  echo "{\"start\": $5, \"end\": $6}" > kaggle/inputs_qwen14/shard.json
  python3 scripts/kaggle_compute.py "$4" --dataset version >> $LOG 2>&1 \
    || fail "qwen14 dataset version ($5-$6) failed"
  push_retry "$4"
  wait_complete "$1" || fail "$1 not COMPLETE"
  fetch_job "$1"
  wait_complete "$4" || fail "$4 not COMPLETE"
  fetch_job "$4"
}
wave qwen14-b0 0 50 qwen14-b1 50 100
wave qwen14-b2 100 150 qwen14-b3 150 200
python3 tools/merge_awq_cache.py --merge --shard-dir runs/v3main_qwen14 \
  --cache-dir results/v3main/beliefs_cache_qwen14 >> $LOG 2>&1 \
  || fail "qwen14 merge failed"
say "qwen14 merged"

PAPER2_V3_CACHE_DIR=results/v3main/beliefs_cache_qwen14 \
  python3 -m src.beliefs_v3main --model Qwen/Qwen3-14B-AWQ --queries v3-q001 \
  --consumers C1 C3 >> $LOG 2>&1 || fail "qwen14 assembly validation failed"
N=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('results/v3main/beliefs_Qwen_Qwen3-14B-AWQ.parquet')))")
[ "$N" = "42" ] || fail "qwen14 validation rows=$N (expected 42)"
rm results/v3main/beliefs_Qwen_Qwen3-14B-AWQ.parquet \
   results/v3main/beliefs_Qwen_Qwen3-14B-AWQ.jsonl
say "qwen14 single-query validation ok (42 rows, partials removed)"
say "=== qwen14-beliefs DONE: ready for full assembly + freeze gate ==="
