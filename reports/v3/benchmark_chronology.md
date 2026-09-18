# Benchmark chronology (honest record; test qrels unseen throughout)

- **V3.0 (pools, 8 nodes):** regime-faithful pool qrels. Dev nDCG ~0.015 — dense top-10
  100% same-entity over ~328 competitors: within-entity ranking near-chance.
- **Diagnosis:** pooled-qrels × shared-corpus made 381/400 top-10 slots unjudged;
  blind 60-pair audit 0.82/0.90 (sample lacked the crux case; corrected analysis followed).
- **16-node amendment:** ~200 competitors, dev ~0.06. Still near-chance for top-k transfer.
- **V3.1 detour:** regime-agnostic re-grade implemented → saturated 0.77–0.99 (entity
  filter solved). Dropped as a grading rule; files superseded (git history).
  Lesson: neither grading rule fixes a competition-ratio defect.
- **Final (64 nodes, ~3 queries/node, guard ≤110, measured 71):** regime-faithful pool
  qrels kept; dev BM25 0.121 / dense 0.166 / RRF120 0.183 / rerank-top30 0.206.
  Measurable, differentiated, unsaturated. V3.1 complete-qrels removed (stale entities).
- **Rule changes during development:** node-count/assignment ONLY (8→16→64), each with
  documented validity reason + re-freeze. Query template, grades, instruction, prompts,
  models, metric, consumers, controllers, simulator: NEVER changed for scores.
- **What never happened:** test nDCG computed by anyone; prompt/tau/model tuning on any
  evaluation; corpus text edited for scores; LLM matrix spend (starts now, frozen).
