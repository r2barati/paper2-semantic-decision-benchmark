# Venue audit

**Audit date:** 2026-08-26. This is a current-policy snapshot; the author must re-check the official call immediately before submission.

## Recommendation

**Primary: TMLR.** Paper 2 is primarily an empirical-science study with a consequential-evaluation framework and a reproducible operational benchmark. TMLR's scope explicitly accommodates empirical studies, evaluation, formalization, and application work when it yields generalizable insight. Its technical-correctness emphasis and rolling review make it the best realistic immediate path.

**Conditional fallback: KDD 2027 Research Track.** The paper can fit KDD Research if rewritten as a data-mining/operational-ML study with a sharper generalization claim. The first KDD 2027 cycle is already closed as of this audit. The announced February cycle is not a general fresh-submission route: the FAQ describes it primarily for immediately preceding August-cycle resubmissions, and its exact date is still to be confirmed.

**Not primary:** KDD Datasets & Benchmarks. The artifact is useful infrastructure, but the manuscript's strongest contribution is the empirical semantic-to-utility finding, not a standalone dataset release. D&B is single-blind and has its own scope and no track transfer.

## Comparison

| Venue | Current status / format | Fit | Main risk | Decision |
|---|---|---:|---|---|
| TMLR | Rolling OpenReview submissions; double-blind; mandatory TMLR style; flexible manuscript length with appendix after references; reproducibility encouraged | 9/10 | Review may require sharper generality and claim boundaries | **Primary now** |
| KDD 2027 Research | First-cycle paper deadline July 26, 2026; 8 content pages plus references/optional appendix; double-blind; first cycle closed | 7/10 | Later February cycle is restricted/uncertain; paper may read as application/evaluation rather than data-mining research | Conditional fallback |
| KDD 2027 Datasets & Benchmarks | 8 content pages; single-blind; separate track; no transfer between tracks | 6/10 | Benchmark is not the paper's only or primary scientific object | Do not wait |
| AAAI-27 Main | Full-paper deadline July 28, 2026; 7 content pages plus 2 reference/ethics pages; double-blind; current cycle closed | 6/10 | Deadline has passed; compressed format would force loss of mechanism detail | Not available |
| NeurIPS 2026 Main | Paper deadline May 6, 2026; 8-page main paper; current cycle closed | 7/10 | Deadline passed and main-track novelty bar is high | Not available |
| ICLR 2027 Main | Abstract Sep 18 and paper Sep 25, 2026 AOE; double-blind; OpenReview | 7/10 | Very short turnaround; current paper's empirical/evaluation identity may be borderline for main track | Stretch only if author chooses a conference attempt |
| ICML 2026 Main | Paper deadline Jan 28, 2026; 8-page main body; current cycle closed | 7/10 | Deadline passed; main-track method/theory expectations | Not available |
| OR / operations journals | Usually strong fit for controlled sequential operational evidence; journal-specific formats and timelines | 8/10 | ML audience and publication delay vary by journal | Best domain fallback |

## Official sources checked

- TMLR [author guide](https://jmlr.org/tmlr/author-guide.html), [submission instructions](https://jmlr.org/tmlr/submissions.html), and [editorial policies](https://jmlr.org/tmlr/editorial-policies.html).
- KDD 2027 [Research Track call](https://kdd2027.kdd.org/research-track-call-for-papers/), [Datasets & Benchmarks call](https://kdd2027.kdd.org/datasets-and-benchmarks-track-call-for-papers/), and [FAQ](https://kdd2027.kdd.org/frequently-asked-questions/).
- [AAAI-27 submission instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/).
- [NeurIPS 2026 call](https://neurips.cc/Conferences/2026/CallForPapers) and [Evaluations & Datasets call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets).
- [ICLR 2027 call](https://www.iclr.cc/Conferences/2027/CallForPapers) and [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
- [ICML 2026 call](https://icml.cc/Conferences/2026/CallForPapers) and [author instructions](https://icml.cc/Conferences/2026/AuthorInstructions).

## Dual submission and overlap

TMLR and a conference submission must be coordinated against the exact venue policies and the manuscript's current review status. Paper 2 must remain clearly distinct from Paper 3: Paper 2 studies semantic information-to-utility transfer; Paper 3 studies contextual selection among heterogeneous resources. The author should not submit both versions simultaneously without verifying each venue's dual-submission and overlap rules.
