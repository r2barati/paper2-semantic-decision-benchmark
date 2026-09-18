# V3 metric definition — FROZEN 2026-09-13 (pre-scale-up)

nDCG (all V3 tables, all phases): **trec_eval definition via ir_measures**
(linear gains, `g / log2(rank+1)`), NOT exponential `(2^g-1)` gains.

Background: `src/retrieval.py:ndcg_at_k` (V1/V2) uses exponential gains; the two
definitions differ materially on graded qrels (v3p-q01 rerank: 0.2649 exp vs
0.2565 trec_eval). V3 standardizes on trec_eval because it is the ECIR screenable
standard and the only thing ir_measures implements.

Consequences:
- V3 nDCG columns are NOT numerically comparable to V2's nDCG column. The V2
  finding (rank-reversal pattern) is unaffected — it is rank-based — but do not
  place V2 and V3 nDCG numbers in one table without recomputation.
- Kernel-side code (no ir_measures backend available) MUST use the stdlib replica
  in `kaggle_kernel/p2_rerank_1c.py::stdlib_metrics`, whose agreement with
  ir_measures 0.4.3 is enforced to 1e-9 on every query of every committed run
  by `tests/test_v3proto.py::test_kernel_stdlib_metrics_match_ir_measures`.
- `cmd_fetch` re-verifies kernel-claimed aggregates with local ir_measures and
  REFUSES installation on any mismatch (this gate caught the 0.234/0.227 split).
