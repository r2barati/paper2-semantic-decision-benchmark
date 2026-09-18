# Venue fit: ECIR 2027 full paper

ECIR full papers: original, state-of-the-art IR research with significant
IR contribution; evaluation research, metrics, benchmarks, RAG, LLMs-for-IR
explicitly in scope; double-blind two-stage review with meta-reviewer
discussion (per ECIR 2027 call).

What ECIR should care about here (one paragraph): an executable,
regenerable evaluation instrument that scores retrieved evidence by
sequential realised return under a fixed controller — with the measured
finding that relevance ordering and control value diverge (rank-preserving
beliefs, sign-reversed utility), a tuned no-text baseline every retriever
fails, and a four-gap attribution no current RAG evaluation provides.
Negative evaluation result with full controls is an evaluation-track
contribution, not an application report: inventory is the controlled world,
named as such throughout.

Fit risks (owned): single synthetic bank without human qrels (mitigated by
model-free C0 pattern + disclosed priors); Qwen-only beliefs (mitigated by
C0 + parity nulls, not by a third model); no new ranker (explicitly not
claimed). Nearest accepted-paper analogues to cite in cover/rebuttal:
downstream-aware RAG evaluation (eRAG-style), negative-result IR studies,
benchmark-with-controls papers. If the chair frames this as "OR paper with
text", the rebuttal pointer is the IR object sentence: evidence quality
measured at induced actions, with nDCG reported on the same episodes.
