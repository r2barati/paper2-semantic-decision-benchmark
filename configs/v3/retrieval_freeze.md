# Retrieval freeze — dev-selected 2026-09-14, test NEVER touched

Selection metric: dev nDCG@10 (trec_eval via ir_measures), per configs/v3/tuning_scope.yaml.
Kernel: paper2-v3-encode-2b (fetched + locally re-verified exact).

## Dev table (40 queries)

| candidate | dev nDCG@10 | selected |
|---|---|---|
| bm25 | 0.0402 | frozen lexical (not tuned) |
| dense-qwen3-instruct | 0.0605 | frozen dense |
| hybrid RRF k=30 | 0.0455 | **SELECTED (best of grid; grid flat)** |
| hybrid RRF k=60 | 0.0448 | — |
| hybrid RRF k=120 | 0.0426 | — |
| rerank top-30 (of hybrid_k60) | 0.0365 | — |
| rerank top-50 | 0.0383 | — |
| rerank top-100 | 0.0485 | **SELECTED (best of grid)** |

Frozen: k_rrf=30, rerank depth=100 on hybrid_k30 candidates. (Depth sweep ran on k60
candidates per the predeclared kernel; k30-vs-k60 dev gap is 0.0007 — noise-flat, so
the final pipeline uses k30 throughout. Recorded, not re-swept: re-sweeping on dev
after seeing this would be tuning-to-noise.)

## Absolute-level reading (diagnostic, pre-registered as moderator design)

Dev retrieval is weak in absolute terms (best 0.0605; only 12% of dev queries surface
grade>=2 evidence in top-3). Relevance is retrievable (Recall@500 0.99) but buried:
expected at ~150 same-entity current competitors per node. This is ACCEPTED as the
measured difficulty, not engineered away (corpus frozen twice).

Consequence for the main experiment (pre-registered): retrieval success becomes a
MEASURED MODERATOR — per-query top-k relevance recorded for every system, and H1–H4
tested both overall and conditional on retrieval-hit vs retrieval-miss strata, with
oracle arms as positive controls. If systems differ only in noise-handling, that is
reported as noise-robustness (C3 abstention), NOT as transfer. Test qrels remain sealed
until the main frozen run.
