# Multiple-comparison audit

- Phase 8B: the four non-NoInfo sensors are simultaneous comparisons against
  NoInfo. The corrected reward-effect CSV reports Holm-adjusted p-values for
  both balanced and deployment-prior estimands.
- Phase 9A: the four non-NoInfo sensors are simultaneous comparisons against
  NoInfo. Holm-adjusted p-values are reported for both estimands.
- Phase 9B: the six robustness variants and their sensor comparisons are
  treated as descriptive boundary/robustness analysis. Holm values are shown
  for transparency, but the benchmark does not claim a confirmatory
  cross-variant familywise conclusion.
- OFAT construction does not remove multiplicity: it describes how variants
  were designed, not how many inferential claims are made.

Confidence intervals are effect-estimation intervals unless a claim explicitly
uses the corresponding Holm-adjusted comparison.
