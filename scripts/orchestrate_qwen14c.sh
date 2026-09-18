#!/bin/bash
# Qwen14 completion: b2 + b3 waves (b0/b1 already fetched+installed).
# Settle + non-fatal shard check + fetch retries + manifest assert (backstop).
# Ends: merge -> 42-row validation -> STOP for freeze gate. Log: /tmp/orchestrate_q14c.log
cd /Users/sanamimani/paper2_semantic_decision_benchmark || exit 1
LOG=/tmp/orchestrate_q14c.log
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
fail() { say "ABORT: $*"; exit 1; }

wait_complete() {
  say "waiting on $1 ..."
  out=$(python3 scripts/kaggle_compute.py "$1" --watch 2>&1 | tail -3)
  echo "$out" >> $LOG
  echo "$out" | grep -q "COMPLETE" || { say "$1 terminal: $out"; return 1; }
  say "$1 COMPLETE"
}

fetch_job() {
  say "fetching $1 ..."
  for i in 1 2 3 4; do
    if python3 scripts/kaggle_compute.py "$1" --fetch >> $LOG 2>&1; then
      say "$1 fetched+installed"
      return 0
    fi
    say "fetch $1 attempt $i failed; retry in 90s"
    sleep 90
  done
  fail "fetch $1 failed 4x (see log)"
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

version_settle_push() {
  echo "{\"start\": $2, \"end\": $3}" > kaggle/inputs_qwen14/shard.json
  python3 scripts/kaggle_compute.py "$1" --dataset version >> $LOG 2>&1 \
    || fail "dataset version ($2-$3) failed"
  say "dataset versioned for $1 [$2,$3); settling 12 min"
  sleep 720
  push_retry "$1"
}

say "=== qwen14 completion: b2 [100,150] + b3 [150,200] ==="
version_settle_push qwen14-b2 100 150
# b3 needs its own dataset version AFTER b2's push; do it sequentially:
echo "{\"start\": 150, \"end\": 200}" > kaggle/inputs_qwen14/shard.json
python3 scripts/kaggle_compute.py qwen14-b3 --dataset version >> $LOG 2>&1 \
  || fail "dataset version (150-200) failed"
say "dataset versioned for qwen14-b3 [150,200); settling 12 min"
sleep 720
push_retry qwen14-b3
wait_complete qwen14-b2 || fail "qwen14-b2 not COMPLETE"
fetch_job qwen14-b2
wait_complete qwen14-b3 || fail "qwen14-b3 not COMPLETE"
fetch_job qwen14-b3
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
say "=== qwen14 DONE: ready for full assembly + freeze gate ==="
