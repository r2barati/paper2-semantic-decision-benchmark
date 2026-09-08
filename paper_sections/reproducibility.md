# Reproducibility Notes

## Environment

```bash
pip install -r requirements.txt
pip install gymnasium
python3 -m pip install -r requirements.lock
# The multi-echelon simulator is the published gym-invmgmt==0.2.1 package,
# installed as a pinned dependency rather than vendored.
```

## Offline Reproduction

All experimental results can be reproduced without API access. LLM predictions are cached in `.llm_cache/` (197 JSON entries).

```bash
# Run all tests (offline, ~5 minutes)
python3 -m pytest tests/ -v

# Regenerate Phase-7 classical baseline
python3 -m src.experiment_phase7

# Regenerate Phase-8B confirmation (offline)
python3 -m src.experiment_phase8b --seeds 30 --seed-start 3100 --no-llm

# Regenerate publication tables
python3 results/publication/generate_tables.py
```

## Live LLM Reproduction

```bash
# Requires API key in .env
echo "OPENAI_API_KEY=sk-..." > .env
python3 -m src.experiment_phase8b --seeds 30 --seed-start 3100 --llm-models gpt-4o
```

## Key Frozen Artifacts

- `.llm_cache/` — 197 cached LLM API responses
- `results/phase7_classical_baseline/tfidf_logreg_model.pkl` — raw TF-IDF model
- `results/phase7_classical_baseline/tfidf_logreg_calibrated_model.pkl` — calibrated model
- `results/phase8b_gym_confirmation/operational_results.csv` — episode count in `docs/ACCOUNTING.md` (720 paired seed/template worlds per sensor)
- All seed/template manifests in `results/*/manifest*.json`
