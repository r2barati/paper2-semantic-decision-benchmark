# Phase 9 controller audit

The executed controller uses:

```text
effective_safety = safety_factor * (1 + p_capacity_drop * 2.0)
```

The factor `2.0` is `BUFFER_AMPLIFICATION` in `src/gym_adapter9.py`. It is a
fixed design/pilot parameter, not an expected-capacity mapping and not a
controller-independent result. The full parameter provenance is in
`parameter_provenance.csv`.

The result therefore quantifies realized semantic value under this fixed
belief-to-control mapping. It does not identify the value of semantic belief
under all controllers.

`OracleSemantic` / `Oracle-Belief Reference` supplies the exact event belief to
that same controller. It is not an optimal policy, hindsight oracle, reward
upper bound, or guarantee of improvement over NoInfo.
