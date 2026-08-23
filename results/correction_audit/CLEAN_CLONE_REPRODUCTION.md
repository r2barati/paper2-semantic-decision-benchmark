# Clean-clone reproduction record

This record is for the post-correction candidate commit. The historical
outputs remain under `results/`; corrected outputs are generated under
`results/correction_audit/` and `results/publication/`.

## Commands

```bash
git clone <candidate-repository> clean-checkout
cd clean-checkout
python3 -m pip install -r requirements.lock
python3 -m benchmark.run --experiment controlled --sensor tfidf_calibrated --offline
python3 -m benchmark.run --experiment gym --sensor oracle_semantic --offline
python3 results/correction_audit/recompute_corrections.py
python3 results/publication/generate_corrected_tables.py
python3 results/publication/generate_scientific_ledger.py
python3 -m pytest tests/ -q
```

The Paper-1 Gym package is resolved from
`third_party/gym-invmgmt-paper/` at commit `a745fd5`; no source edit or
absolute local path is required. The aggregate replay uses frozen episode CSVs
and `results/frozen_llm_outputs/`; no private API credential is required.

## Verification record

The commands above were run from a clean checkout of the candidate commit.
The offline controlled and Gym commands returned balanced and deployment-prior
rows, all corrected publication generators completed, the vendored Gym loaded,
and the full deterministic test suite passed with 273 tests and 0 failures.

The clean checkout contained the tracked historical raw CSVs, corrected audit
CSVs, publication tables, frozen semantic-output export, pinned requirements,
and no dependency on `/Users/...` paths. Live LLM calls remain optional and
are not part of published-number reproduction.
