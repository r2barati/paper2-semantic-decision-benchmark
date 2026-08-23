# Final Test Audit

**Date:** 2026-08-23 (post-fix)
**Command:** `python3 -m pytest tests/ -v`
**Python:** 3.9.19
**Platform:** macOS Darwin 24.6.0 (x86_64)

## Summary

| Metric | Count |
|--------|------:|
| **Total collected** | 268 |
| **Passed** | 268 |
| **Failed** | 0 |
| **Skipped** | 0 |
| **Errors** | 0 |

## Test File Breakdown

| File | Tests | Category |
|------|------:|----------|
| `test_events.py` | 7 | offline deterministic |
| `test_env.py` | 7 | offline deterministic |
| `test_interpreter.py` | 7 | offline deterministic |
| `test_classical_baseline.py` | 30 | offline deterministic |
| `test_experiment.py` | 140 | offline (cached results) |
| `test_phase8_audit.py` | 29 | offline deterministic |
| `test_phase8b.py` | 24 | offline deterministic |
| `test_phase9.py` | 24 | offline deterministic |

## Classification

### Offline Deterministic Tests (128 tests)
- `test_events.py` (7): Event definitions, template structure
- `test_env.py` (7): Simulator mechanics, inventory dynamics
- `test_interpreter.py` (7): Mock interpreter schemas, probability ranges
- `test_classical_baseline.py` (30): TF-IDF model behavior, leakage prevention, metrics
- `test_phase8_audit.py` (29): Paper-1 provenance, no-leakage, controller sensitivity, TF-IDF identity
- `test_phase8b.py` (24): Linguistic separation, seed disjointness, fill-rate metric, sensor config
- `test_phase9.py` (24): CapacityDrop wrapper, templates, controller, YAML scaling, episode runner

### Cached Results Tests (140 tests)
- `test_experiment.py` (140): End-to-end experiment validation, Phase-4/5/5.5/6 reproducibility, decomposition identities, security checks

All tests in `test_experiment.py` use frozen cached results (`.llm_cache/` directory with 197 entries) and pre-computed output CSVs. No LLM API calls are made during test execution.

### API-Dependent Tests
**None.** All 268 tests run entirely offline.

## Verdict

All 268 tests pass. No failures. No skipped tests. No external API dependency required. The test suite is publication-ready.

## Post-Fix Notes

- `test_60_phase5_5_manifest_hash_consistent` previously failed due to missing `SUPPLIER_CAPACITY_DROP` in `REGIME_PARAMS`. Fixed by adding the regime to `REGIME_PARAMS`, `REGIME_PRIOR`, and `perfect_semantic_regime_belief()` in `interpreter.py`.
