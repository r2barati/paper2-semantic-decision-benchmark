# Reproduction failure root-cause audit

The prior fresh sandbox run reported 271 passed and two permission-only
failures. There were no assertion failures and no scientific-output mismatch.

| Failure | Root cause | Classification |
|---|---|---|
| Cached LLM response test | `src/interpreter.py` constructed the cache path unconditionally under repository `.llm_cache` before writing a test-only cache entry. | A + B + D: repository-write assumption, read-only sandbox, cache location not configurable. |
| Phase-5 OIV calculation test | `run_information_value_grid` defaulted to the repository's `results/phase5` directory and created/wrote its CSV there. | A + B: repository-write assumption and intentionally read-only sandbox. |

The failures are not C in the sense of overwriting a frozen scientific result:
the test invokes a function whose default output location is a frozen-tree
path, but the generated test artifact can be redirected and is not used as
publication evidence.

## Hygiene repair

Two opt-in environment variables were added without changing default paths or
calculations:

- `PAPER2_LLM_CACHE_DIR` controls the runtime cache directory.
- `PAPER2_PHASE5_RESULTS_DIR` controls the Phase-5 test/output directory.

With variables unset, the original repository-relative behavior remains. The
offline reproduction contract sets both variables to temporary writable paths.
No frozen CSV, seed, template, controller, model, or scientific computation was
changed.

