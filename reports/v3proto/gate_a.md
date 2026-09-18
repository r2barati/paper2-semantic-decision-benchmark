# Gate-A report — V3 Phase-1 prototype (DIAGNOSTICS, not targets)

Date: 2026-09-13. Corpus seed 20260913. No simulation/profit run (out of scope).
Nothing in this report triggered a generator revision: all findings below are
reported as-is per the frozen rule (revise only on validity defects).

## Corpus (frozen: configs/v3/dataset_proto.yaml, data/v3proto/manifests/)

- 20 queries (8 delay / 8 surge / 4 normal; 6 dev / 14 test; 4 held-out-entity R4, all test)
- 210 unique docs / 288 judged pairs; grades {0:170, 2:40, 1:40, 3:36+shared}
- 100% freshly authored (V1/V2 bank verbatim-exclusion test green)
- Dual annotation (independent text-only grader): **kappa=0.823** (g0:0.86, g1:1.00, g2:0.80, g3:1.00)
- Leakage/near-dup: no exact full-text dupes; 117 body-identical pairs = shared boilerplate across sites (by design, reported); TF-IDF-train exclusion n/a by construction

## D1–D3 retrieval diagnostics (ir_measures 0.4.3, graded qrels)

| system | nDCG@10 | R@10 | R@20 | MRR |
|---|---|---|---|---|
| random | 0.043 | 0.060 | 0.118 | 0.110 |
| bm25 (in-repo, frozen lexical) | 0.122 | 0.188 | 0.313 | 0.197 |
| dense-STANDIN (MiniLM, pipeline only) | 0.189 | 0.238 | 0.492 | 0.282 |
| hybrid-RRF (standin fusion) | 0.167 | 0.213 | 0.408 | 0.323 |
| rerank-STANDIN (MiniLM CE, mechanics only) | 0.146 | 0.197 | 0.502 | 0.296 |
| oracle-relevant | 1.000 | 1.000 | 1.000 | 1.000 |

Reading (diagnostic): task is hard for all real systems (best 0.189), large oracle
headroom (0.81), systems differ in ordering across metrics (RR favors hybrid, R@20
favors rerank-standin). Stand-in dense/rerank numbers carry NO baseline meaning;
frozen Qwen3-Embedding/Reranker-0.6B pending download (blocked: 1.7GB free disk).
Lexical Pyserini blocked (no Java); in-repo BM25 is the frozen lexical baseline.

## D4 consumer pilot (dev 6 queries x 4 systems, k=3, gpt-4o-2024-11-20, 54 cached calls)

- Belief-accuracy ranges: C0 0.67, C1 0.67, C3 1.00 → retrieval→belief variation EXISTS.
- C1+oracle: acc 1.00, Brier 0.017 → oracle evidence sufficient for the naive harness.
- C3 abstention 0.00 on dev; mechanism verified functional (abstains on
  evidence-free/offtopic-only input). tau=0.5 UNTOUCHED (predeclared, not tuned).
- Note: n=6/query-system cells are noisy (e.g. C0 dense-standin 0.00); pilot
  establishes variation existence, not effect sizes.

## D5 validity: GREEN (kappa 0.823 ≥ 0.6; leakage tests green; oracle-factual check deferred to full build)

## Environment blockers (recorded, not worked around silently)

1. Qwen3-Embedding/Reranker-0.6B: pinned (SHA in configs/v3/models.yaml), download blocked by disk.
2. Qwen3-8B: pinned (SHA b968826d), execution blocked (16GB weights vs 8GB RAM, no vLLM-on-macOS); harness speaks OpenAI-compatible base_url so wiring is code-complete.
3. Pyserini: blocked (no Java); in-repo BM25 frozen instead, cross-check retained as adoption criterion.

## Scale-up judgment

**CONDITIONAL GO:** corpus protocol is credible (D5 green), task non-degenerate
(systems spread, oracle headroom, consumer variation present). Scale to 200/4000
on a machine with disk+Java+Qwen compute; re-resolve HF SHAs at download time and
repin. Prototype IDs (v3p-*) quarantined from V3 test regardless.

## Reproduce

`python3 -m pytest tests/test_v3proto.py -q` (9 tests) ·
`python3 -m src.experiment_v3proto` (TREC runs -> runs/v3proto/) ·
`python3 -m src.pilot_v3proto` (beliefs -> results/v3proto/, ~100 cached GPT-4o-2024-11-20 calls)
