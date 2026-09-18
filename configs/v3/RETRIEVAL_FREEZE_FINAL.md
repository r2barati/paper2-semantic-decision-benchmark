# Retrieval freeze — FINAL 2026-09-14 (dev only; test sealed throughout)

Selection metric: dev nDCG@10 (trec_eval via ir_measures). All kernel numbers
locally re-verified exact before installation.

## Dev table (40 queries, 64-node corpus, regime-faithful qrels)

| candidate | dev nDCG@10 | source |
|---|---|---|
| bm25 | 0.1208 | local |
| dense-qwen3-instruct | 0.1664 | Kaggle encode-2b |
| hybrid RRF k=30 / 60 / 120 | 0.1761 / 0.1808 / 0.1831 | Kaggle encode-2b |
| rerank (k60 base) top-30 / 50 / 100 | 0.2036 / 0.1927 / 0.1945 | Kaggle encode-2b |
| rerank (k120 base) top-30 / 50 / 100 | 0.2062 / 0.1921 / 0.1884 | Kaggle rerank-refine |

## Frozen pipeline

**BM25 (in-repo, rank-bm25 cross-checked) → Qwen3-Embedding-0.6B instructed →
RRF k=120 → Qwen3-Reranker-0.6B top-30 → oracle-relevant ceiling / random floor.**

Depth decreases monotonically (30 > 50 > 100 on both bases); RRF grid monotone
(30 < 60 < 120). No further tuning authorized: `configs/v3/tuning_scope.yaml` is
spent except context-k reporting (k=3 headline, k=5 secondary).

## What this supersedes

- `retrieval_freeze.md` §dev table from the 16-node build (k=30/top100 selected there):
  that build is void (competition-ratio defect, phase2b_audit.md). This file governs.
- V3.1 regime-agnostic qrels (`src/qrels_v31*.py`, `data/v3/qrels_complete/` removed):
  saturated at 0.98 — retained in git history only.
