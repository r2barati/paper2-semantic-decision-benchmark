# Phase 1C report — instruction correction + remote execution (NO corpus changes)

Date: 2026-09-13/14. Instruction frozen pre-score (`configs/v3/qwen_instruction.txt`).
Rerank executed on Kaggle (P100 assigned; image torch lacks sm_60 kernels so the
kernel's in-run CPU fallback engaged — recorded in session log). Kernel v5.
**Phase 1 is now PERMANENTLY CLOSED.**

Date: 2026-09-13. Instruction frozen pre-score (`configs/v3/qwen_instruction.txt`):
"Given an operational supply-chain information need, retrieve documents relevant
to the described entity, event, and time context." Bare-query numbers retained as
implementation ablation, never as competing systems.

## Final retrieval table (trec_eval nDCG via ir_measures; see configs/v3/metrics.md)

| system | nDCG@10 | R@10 | R@20 | MRR | provenance |
|---|---|---|---|---|---|
| random | 0.043 | 0.060 | 0.118 | 0.110 | local |
| bm25 (frozen lexical) | 0.122 | 0.188 | 0.313 | 0.197 | local; rank-bm25 overlap 8–10/10 |
| dense-qwen3 bare (ablation) | 0.092 | 0.147 | 0.360 | 0.133 | local |
| dense-qwen3 INSTRUCT | 0.199 | 0.238 | 0.540 | 0.305 | local |
| hybrid-real-instruct | 0.129 | 0.207 | 0.468 | 0.212 | local |
| **rerank-qwen3-instruct** | **0.227** | 0.288 | 0.527 | 0.244 | **Kaggle v5; kernel-claimed == locally-recomputed exactly** |
| oracle-relevant | 1.000 | 1.000 | 1.000 | 1.000 | local |

Rerank-instruct is the best real system (+0.028 over instructed dense). No saturation;
oracle headroom 0.77. Difficulty accepted as observed.

## Metric Lantern (caught by the fetch gate, resolved pre-freeze)

Kernel v4 claimed nDCG 0.234 (exponential gains in its stdlib replica) vs local
ir_measures 0.227 (trec_eval linear gains) — fetch REFUSED installation. Root cause:
`src/retrieval.py:ndcg_at_k` (V1/V2) uses exponential gains while trec_eval uses linear;
they differ materially on graded qrels. Resolution frozen in `configs/v3/metrics.md`:
V3 nDCG = trec_eval definition everywhere; kernel replica proven 1e-9-exact on every
query of every committed run; V2 nDCG columns declared non-comparable (rank-based
findings unaffected). The earlier cross-check test had passed vacuously on an all-zero
query — extended to all queries with a nonzero-coverage floor.

| system | nDCG@10 | R@10 | R@20 | MRR |
|---|---|---|---|---|
| dense-qwen3 BARE (ablation) | 0.092 | 0.147 | 0.360 | 0.133 |
| **dense-qwen3 INSTRUCT** | **0.199** | 0.238 | 0.540 | 0.305 |
| hybrid-real-instruct (BM25+RRF) | 0.129 | 0.207 | 0.468 | 0.212 |

Reading: instruction fixes the deficit (0.092→0.199); RRF+BM25 drags hybrid
down (0.129) — fusion-weight tuning is dev-scope for Phase 2 (`configs/v3/tuning_scope.yaml`),
not a corpus issue.

## Retraction (caught by recipe audit, recorded honestly)

Phase-1B `rerank-qwen3 0.174` is **INVALID**: ST 5.1.2's CrossEncoder can only load a
sequence-class head, and the checkpoint's `score.weight` was randomly initialized
(warning in log). Correct path = documented causal-LM yes/no-logit recipe with the
SAME predeclared instruction. Local CPU (~3 s/pair, ~3 h) infeasible → moved to Kaggle.

## Remote execution (this machine is CPU/8GB/no-Java)

- `scripts/kaggle_compute.py rerank-1c --push/--watch/--fetch/--test` (ADIA pattern;
  machine-level `~/.kaggle/access_token`, never in repo). Kernel sources in
  `kaggle_kernel/`; frozen inputs in `kaggle/inputs/` (+ private dataset
  `rezabarati2/paper2-v3proto-inputs`). Same mechanism serves Qwen3-8B-AWQ and scale-up.
- GitHub Actions stays CI-only (no GPU, 6 h cap).
- Still blocked: Qwen3-8B execution (needs GPU host; AWQ pin frozen), Pyserini (needs Java;
  non-blocking cross-check per decision).

## Freeze status (CLOSED)

- Corpus: frozen (unchanged since Phase 1A).
- Instruction: frozen (this phase, pre-score).
- Retrieval config: FROZEN — in-repo BM25 · Qwen3-dense instructed · RRF · Qwen3-rerank top-50 ·
  oracle-relevant · random floor · trec_eval nDCG.
- No corpus change is authorized by any score in this report.
