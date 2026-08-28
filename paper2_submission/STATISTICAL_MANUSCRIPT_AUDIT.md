# Statistical manuscript audit

- **Primary unit:** paired seed × template episode or declared aggregate under
  the benchmark estimand; do not treat individual agent calls as independent
  scientific units.
- **Pairing:** sensor and NoInfo/reference results share the same seed/template
  realization where the artifact supports it.
- **Primary uncertainty:** hierarchical bootstrap resampling templates/families,
  variants, and paired seeds as documented in the final audit.
- **CIs:** percentile intervals; with small template-cluster counts, report the
  limitation rather than implying exact finite-sample coverage.
- **Multiplicity:** Phase 8B and 9A simultaneous sensor comparisons use Holm
  adjustment. Phase 9B is descriptive OFAT robustness/boundary evidence.
- **SIVR:** secondary normalization; report raw reward effects first, signed
  denominator status, and balanced versus prior-weighted estimands separately.
- **Phase 9A:** do not present all-template TF-IDF results as held-out linguistic
  generalization.

No additional inferential procedure is introduced in this production pass.
