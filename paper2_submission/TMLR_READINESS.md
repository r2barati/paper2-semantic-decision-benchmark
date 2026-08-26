# TMLR readiness

TMLR is the recommended primary venue. Its official author guidance states
that submissions are double-blind and anonymized, use the mandatory TMLR LaTeX
style, allow flexible manuscript length with appendices, use OpenReview, and
encourage reproducibility supplements. TMLR emphasizes technical correctness
and accepts empirical studies that provide insight into learning systems.

## Fit

Paper 2 fits TMLR as an empirical-methodology study with a formal executable
evaluation framework, explicit limitations, and reproducibility artifacts. It
should not be sold as a new universal theory.

## Machine-complete

- Official TMLR style is staged under `manuscript/tmlr-style-file-main/`.
- `manuscript/main.tex` and `manuscript/appendix.tex` compile to a 6-page PDF.
- The final LaTeX log has no fatal errors, undefined citations/references, or
  overfull-box diagnostics.

## Required before submission

- Complete anonymization and OpenReview profiles.
- Add a concise ethics/broader-impact statement if required by the application.
- Verify related work, self-overlap, and concurrent-submission policy.
- Package anonymized code/data supplement up to the current official limit.
- Human-review all numerical claims and cached-model provenance.

TMLR is rolling, so there is no reason to wait for an unpublished conference
deadline. Source: [TMLR author guidelines](https://jmlr.org/tmlr/author-guide.html),
[submission instructions](https://jmlr.org/tmlr/submissions.html), and
[editorial policies](https://jmlr.org/tmlr/editorial-policies.html).
