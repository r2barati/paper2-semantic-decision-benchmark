# SIVR definition audit

## Primary definition

For a sensor/interpreter `m`, NoInfo reference `n`, and OracleSemantic reference
`o`, the aggregate diagnostic is:

```text
SIVR(m) = Σ_i [R_m(i) − R_n(i)] / Σ_i [R_o(i) − R_n(i)]
```

where `i` indexes paired seed/template observations under the declared
estimand. The corrected implementation is `src.metrics.signed_sivr` and the
source-of-truth recomputations are in `results/correction_audit/sivr_recomputed.csv`.

## Edge cases

- If the oracle information value is near zero, SIVR is undefined and receives
  an explicit near-zero status.
- If oracle information value is negative, the diagnostic preserves the sensor
  numerator sign while marking the reference as negative. It is not interpreted
  as information-value recovery.
- SIVR is unbounded and can exceed one because OracleSemantic is perfect belief
  under a fixed controller, not an optimal policy or reward upper bound.
- Balanced and declared-prior estimates are distinct. The manuscript must not
  mix them.

## Manuscript policy

Use raw paired reward effects and confidence intervals as primary results. Use
SIVR as a normalized secondary diagnostic tied to this framework. Never call it
universal, causal without design qualification, or a percentage bounded by
100%.
