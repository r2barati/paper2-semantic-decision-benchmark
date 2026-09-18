# Phase 1B report — production stack on the frozen prototype (NO corpus changes)

> RETRACTION (2026-09-13, see phase1c.md): the `rerank-qwen3 0.174` number below is
> INVALID — ST 5.1.2's CrossEncoder loaded a randomly-initialized score head.
> Correct recipe = documented yes/no-logit path; rerun moved to Kaggle. All other
> Phase-1B numbers stand.

Date: 2026-09-13. Same quarantined corpus (seed 20260913, 20q/210docs). No simulation.
Downloads were encode-and-delete (disk discipline); frozen artifacts are vectors/runs,
not weights.

## Model pins (re-resolved at download; SHAs confirmed, one transcription fix)

- Qwen3-Embedding-0.6B @ `97b0c61…f9eb65b3` (unchanged), dim 1024, L2-normed, NO query instruction
- Qwen3-Reranker-0.6B @ `e61197e…1255473` (**corrected**: manifest had a dropped char; the
  download rejects wrong SHAs, so the pin is self-checking)
- Reranker served via ST CrossEncoder, single-pair loop (baked config lacks pad token; batch>1 impossible)

## Retrieval comparison (ir_measures, graded qrels)

| system | nDCG@10 | R@10 | R@20 | MRR | status |
|---|---|---|---|---|---|
| random | 0.043 | 0.060 | 0.118 | 0.110 | floor |
| bm25 (in-repo) | 0.122 | 0.188 | 0.313 | 0.197 | **frozen lexical**; rank-bm25 cross-check top-10 overlap 8–10/10 |
| dense-standin (MiniLM) | 0.189 | 0.238 | 0.492 | 0.282 | pipeline only, NOT a baseline |
| **dense-qwen3 REAL** | **0.092** | 0.147 | 0.360 | 0.133 | below BM25 (see reading) |
| **hybrid-real (BM25+Qwen3, RRF)** | **0.120** | 0.163 | 0.342 | 0.210 | ≈ BM25 |
| rerank-standin (MiniLM CE) | 0.146 | 0.197 | 0.502 | 0.296 | mechanics only |
| **rerank-qwen3 REAL (hybrid top-50)** | **0.174** | 0.212 | 0.280 | 0.240 | best real system |
| oracle-relevant | 1.000 | 1.000 | 1.000 | 1.000 | headroom intact |

## Reading (diagnostic, corpus untouched)

1. **Qwen3-dense < BM25 is a method finding, not a corpus defect.** The documented Qwen3
   usage applies an *instruction prefix on the query side*; we encoded bare queries
   (manifest default). Plausible cause of the deficit; the corpus needs no change.
   Action: re-run dense/hybrid **with the documented query instruction** (query side only,
   dev-frozen wording) before freezing — vectors recompute, corpus and qrels frozen.
2. **Rerank-real best (0.174)** despite weak base: reranking hybrid top-50 recovers
   more than the dense deficit costs. R@20 drops vs standin (0.28 vs 0.50) — expected:
   CE reorders top-50 only.
3. **No saturation anywhere** (best 0.174); oracle headroom 0.83. Difficulty accepted as observed.

## Consumer pilot on production rankings (dev 6q, k=3, gpt-4o-2024-11-20)

- Ranges: C0 0.67, C1 1.00, C3 1.00 → retrieval→belief variation CONFIRMED on production stack.
- C1+oracle 1.00/0.017; C3 abstention still 0.00 (tau=0.5 untouched; mechanism verified).
- Qwen3-8B arm: BLOCKED (HF Inference API requires token; 16GB weights unrunnable here).
  Required input: HF token with inference access OR GPU host with pinned commit.

## Pyserini: BLOCKED (no Java runtime)

In-repo BM25 stands as frozen lexical baseline, cross-validated (rank-bm25 overlap 8–10/10).
Adoption criterion for Java-capable machine: Pyserini BM25 (k1=1.2,b=0.75 default) must agree
with in-repo top-10 at ≥7/10 median or the difference is documented and one is frozen.

## Freeze decision

**Stack frozen conditional on the query-instruction ablation**: in-repo BM25 · Qwen3-dense
(+instruction re-run) · RRF hybrid · Qwen3-rerank top-50 · oracle-relevant · random floor.
**Corpus frozen unconditionally** — no changes from model scores. Next: instruction ablation
→ scale to 200/4000 on capable hardware.

## Reproduce

`runs/v3proto/{dense-qwen3,hybrid-real,rerank-qwen3}.trec` (+`*_qlevel.json`, `qwen3emb_vectors.npz`);
`results/v3proto/pilot_beliefs_1b.jsonl`; `tools/rerank_qwen3_1b.py`; this report.
