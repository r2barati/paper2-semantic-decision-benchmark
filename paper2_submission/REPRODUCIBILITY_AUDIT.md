# Reproducibility audit

## Present

- Frozen commit and tag: `paper2-benchmark-v1.0` / `0be5181926d6ab1def43ef499f8ed57ee7e31b3c`.
- Pinned requirements and benchmark runner.
- Frozen templates, seed manifests, configuration files, and cached semantic
  outputs.
- Corrected audit CSVs and publication table generators.
- Vendored Paper-1 Gym dependency at the recorded provenance.
- Offline replay path and 268-test audit.
- Opt-in `PAPER2_LLM_CACHE_DIR` and `PAPER2_PHASE5_RESULTS_DIR` controls for
  read-only-checkout reproduction; default scientific paths are unchanged.

## Author actions remaining

- Decide whether the untracked `results/phase9b_robustness/divergent_medium_lead.yaml`
  belongs outside the frozen publication package; it is not used by this paper.
- Perform a clean clone from the committed package using the temporary
  writable cache/output paths in `CLEAN_REPRODUCTION_AUDIT.md`.
- Verify all relative paths, package versions, and artifact licenses.
- Prepare an anonymized supplement with code, manifests, tables, and selected
  raw artifacts; do not upload it in this phase.
- Confirm the final model/provider identifiers for cached GPT outputs.

No frozen science is changed by these actions.
