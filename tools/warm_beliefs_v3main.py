"""Cache warmer for the V3 main GPT belief matrix (execution plumbing ONLY).

Replicates src.beliefs_v3main.run_matrix's C1/C3 call pattern exactly over a
query-index shard, populating the shared beliefs cache. Writes NO parquet/jsonl
(output assembly stays single-writer in run_matrix). Safe to run concurrently
with the main job: cache filenames are content-addressed and identical inputs
produce identical keys; cache-hit calls are free.

Frozen science untouched: same consumers, prompts, tau, model string, cache dir.
Usage:
  PAPER2_V3_CACHE_DIR=results/v3main/beliefs_cache nohup python3 -u tools/warm_beliefs_v3main.py --start 50 --end 100 > /tmp/warm_50_100.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.beliefs_v3main import KS, MODEL, _doc_view, _evidence_sets
from src.consumers_v3 import assert_frozen_prompts, consume_c1, consume_c3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--consumers", nargs="*", default=["C1", "C3"])
    ap.add_argument("--delay", type=float, default=6.0,
                    help="seconds between LLM calls (TPM pacing)")
    a = ap.parse_args()

    assert_frozen_prompts()
    docs, queries, sets = _evidence_sets()
    queries = queries[a.start:a.end]
    qmeta = {q["_id"]: q["metadata"] for q in queries}
    qtext = {q["_id"]: q["text"] for q in queries}
    systems = sorted({s for s, _, _ in sets} - {"none"})
    print(f"warmer shard [{a.start},{a.end}) model={a.model} "
          f"queries={len(queries)} systems={systems}", flush=True)

    n_fail = 0
    fails = []
    t0 = time.time()
    for i, q in enumerate(queries):
        qid = q["_id"]
        for sys_name in systems:
            for k in KS:
                key = (sys_name, qid, k)
                if key not in sets:
                    continue
                top = [_doc_view(docs, did) for did in sets[key]]
                for consumer in a.consumers:
                    t_call = time.time()
                    for retry in range(6):
                        try:
                            if consumer == "C1":
                                consume_c1(top, qtext[qid], a.model)
                            else:
                                consume_c3(top, qtext[qid], a.model,
                                           qmeta[qid]["entity_node"])
                            break
                        except Exception as e:
                            name = type(e).__name__
                            if "RateLimit" in name and retry < 5:
                                time.sleep(60 + 10 * retry)
                                continue
                            # log-and-continue: assembly retries leftovers
                            n_fail += 1
                            fails.append((qid, sys_name, k, consumer,
                                          f"{name}: {str(e)[:100]}"))
                            time.sleep(5)
                            break
                    # pace real API calls only (cache hits return in ms)
                    if time.time() - t_call > 0.5:
                        time.sleep(a.delay)
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(queries)} qid={qid} fails={n_fail} "
                  f"elapsed={el / 60:.1f}min", flush=True)
    print(f"DONE shard [{a.start},{a.end}) fails={n_fail} "
          f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    if fails:
        Path(f"/tmp/warm_fails_{a.start}_{a.end}.json").write_text(
            json.dumps(fails, indent=1))
        print(f"  {len(fails)} failures recorded (assembly will retry)", flush=True)


if __name__ == "__main__":
    main()
