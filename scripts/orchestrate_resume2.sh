#!/bin/bash
# Resume orchestrator v2 (supersedes orchestrate_resume.sh).
# State: AWQ b0/b1/b2 fetched+merged (4468 files, audit logged); awq-b3 RUNNING;
# phi4-probe COMPLETE but FAILED gates (schema non-adherence, see NOTES.md) ->
# phi4 belief jobs are DEAD, never push them. Frontier fallback per amendment:
# Qwen3-14B-AWQ probe next; belief shards ONLY on explicit user go after probe.
# Execution plumbing ONLY. STOPS nonzero on FAILED job or failed verification.
# Log: /tmp/orchestrate3.log
cd /Users/sanamimani/paper2_semantic_decision_benchmark || exit 1
LOG=/tmp/orchestrate3.log
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
    say "fetch $1 attempt $i failed (transient?); retry in 5 min"
    sleep 300
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

say "=== resume-v2 start: awq-b3 RUNNING; qwen14-probe pending; phi4 jobs DEAD ==="
wait_complete awq-b3 || fail "awq-b3 not COMPLETE"
fetch_job awq-b3
python3 tools/merge_awq_cache.py --merge >> $LOG 2>&1 || fail "AWQ merge failed"
say "AWQ complete (4 shards)"
push_retry qwen14-probe
wait_complete qwen14-probe || fail "qwen14-probe not COMPLETE"
fetch_job qwen14-probe
say "=== qwen14 probe done — STOPPING for user go before any belief shards ==="
