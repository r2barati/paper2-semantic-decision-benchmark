# External artifacts

The source, tests, configuration, documentation, and reviewed aggregate
results are versioned here. The following remain local or in approved private
artifact storage:

- `.env` files, provider credentials, and virtual environments;
- LLM caches and raw LLM response files;
- serialized model checkpoints (`*.pkl`), including the classical-baseline
  checkpoints;
- large or re-creatable raw experiment outputs.

The Phase 2A local manifest is `../phase1_migration/manifests/paper2_semantic_decision_benchmark.external-artifacts.sha256`.
It records SHA-256 values without containing credential values. Raw files are
retained locally after being removed from Git tracking.
