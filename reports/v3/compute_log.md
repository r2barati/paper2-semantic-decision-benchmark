# Remote-compute reproducibility log (preserved failures + corrections)

Phase-1C reranker, kernel `rezabarati2/paper2-v3-rerank-1c` (private).

1. **v1 — torch churn.** Kernel pip-installed pinned transformers without `--no-deps`;
   resolver replaced the image GPU torch → `no kernel image for device`. Fix: never
   touch torch/numpy in kernels; keep image builds, `--no-deps` only for missing pkgs.
2. **v2 — same error, real cause.** Assigned card Tesla P100 (sm_6.0), unsupported by
   image torch (same failure ADIA logged). Fix: in-run GPU warmup probe + CPU fallback.
3. **v3 — ran 18/20, then died in eval.** ir_measures has no pytrec_eval backend in image.
   Fix: stdlib graded-metric replica in-kernel + local re-verification gate.
4. **v4 — fetch gate REFUSED install (0.234 vs 0.227).** Kernel replica used exponential
   gains; trec_eval uses linear. V1/V2 hand formula is exponential — V3 froze trec_eval
   (`configs/v3/metrics.md`), replica proven 1e-9-exact, V2 columns declared non-comparable.
5. **v5 — COMPLETE; kernel-claimed == locally-recomputed exactly; outputs installed.**
6. **Transcription safeguard:** reranker SHA copied with a dropped char; HF download
   rejected it; corrected in manifest (self-checking pins).

Lesson recorded: released numbers are verified outputs (fetch-gate recomputation +
exact-match install rule), never trusted framework outputs. The 0.174 CrossEncoder-head
retraction (Phase-1B) belongs to the same ledger.
