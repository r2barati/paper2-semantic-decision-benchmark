# Reproducibility Guide

## Benchmark Version

```
paper2-benchmark-v2.0
Frozen: 2026-09-07 (post-correction)
```

Version 2.0 supersedes 1.0. Version 1.0's results were produced with an
information leak in the historical controller matrix, an optimiser LP that
could not sell existing stock, arrival timing one period longer than declared,
an ignored warning-release parameter, and no text-free controls. Do not mix
v1.0 and v2.0 numbers. The v1.0 outputs are archived under
`results/pre_correction_archive_2026/`.

## Environment

| Component | Version / Identifier |
|-----------|---------------------|
| Python | 3.9.19 (`/usr/local/bin/python3`) |
| OS | macOS Darwin 24.6.0 (x86_64) |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| matplotlib | 3.9.0rc2 |
| scikit-learn | 1.6.1 |
| openai | 2.48.0 |
| gym-invmgmt | 0.1.0 (editable install, commit `a745fd5`) |
| gymnasium | 0.29.1 (pinned in `requirements.txt`; an earlier draft of this table said 1.0.0) |
| networkx | 3.2.1 (required by the vendored simulator) |
| PyYAML | 6.0.3 (required by the vendored simulator) |
| python-dotenv | 1.2.1 |
| pandas | 2.3.3 |

Supported Python: >=3.9, <3.13. CI verifies 3.9 and 3.11 in a clean
environment installed **only** from `requirements.lock`.

## Quick Start: Reproduce Main Results

### Prerequisites

```bash
python3 -m pip install -r requirements.lock
# The importable multi-echelon gym package is vendored at the exact recorded
# commit. It is added to sys.path rather than pip-installed, so ITS
# dependencies (networkx, PyYAML) are declared in requirements.txt directly.

# Reconstruct the offline artifacts that are not tracked (semantic caches and
# fitted model checkpoints). Deterministic; no API key and no network needed.
python3 -m tools.rebuild_offline_artifacts
python3 -m tools.rebuild_offline_artifacts --verify
```

Without the rebuild step a clean export of the tracked files is missing the
cached interpretations and the TF-IDF checkpoints, and part of the test suite
fails with missing-artifact errors rather than genuine failures.

### Run All Tests (offline, no API key needed)

```bash
python3 -m pytest tests/ -v
# The test count is reported by pytest; no API key is required.
```

### Run Individual Experiments

```bash
# Phase-7 classical baseline evaluation
python3 -m src.experiment_phase7

# Corrected publication replay (offline, no simulation or API key)
python3 -m benchmark.run --experiment gym --sensor tfidf_calibrated --offline
python3 -m benchmark.run --experiment controlled --sensor tfidf_calibrated --offline

# Regenerate corrected analyses and tables from frozen episode CSVs
python3 results/correction_audit/recompute_corrections.py
python3 results/publication/generate_corrected_tables.py
python3 results/publication/generate_scientific_ledger.py

# Optional live Phase-8B simulation (requires an API key only for gpt-4o)
python3 -m src.experiment_phase8b --seeds 30 --seed-start 3100 --no-llm

# Phase-8B with live LLM (requires OPENAI_API_KEY or LLM_API_KEY in .env)
python3 -m src.experiment_phase8b --seeds 30 --seed-start 3100 --llm-models gpt-4o
```

## Frozen Results

All historical and corrected raw results are tracked under `results/`. The
publication semantic-output export is under `results/frozen_llm_outputs/` and
contains hashes, model identifiers, probabilities, parsing status, and
checksums. Aggregate publication tables replay the frozen CSVs and require no
live API access.

| Directory | Description | Status |
|-----------|-------------|--------|
| `results/phase5/` | Controlled benchmark (original) | Frozen |
| `results/phase5_5/` | Real-LLM semantic sensing | Frozen |
| `results/phase6/` | Confirmation/robustness | Frozen |
| `results/phase7_classical_baseline/` | TF-IDF + LogReg baseline | Frozen |
| `results/phase8_gym_replication/` | Paper-1 Gym exploratory | Frozen |
| `results/phase8b_gym_confirmation/` | Paper-1 Gym confirmation | Frozen |

## LLM Configuration

LLM predictions are cached. To reproduce without API access:

```bash
# Published GPT results are replayed from results/frozen_llm_outputs/.
# Live calls are optional and require the API configuration below.
```

To make fresh LLM calls, create `.env`:

```
OPENAI_API_KEY=your-key-here
# or
LLM_API_KEY=your-key-here
LLM_BASE_URL=https://api.openai.com/v1
```

## Seed Manifests

| Phase | Seeds | Count | Disjoint |
|-------|-------|------:|----------|
| Phase 5/5.5 | 1000–1014 | 15 | — |
| Phase 6 confirmation | 2000–2049 | 50 | from Phase 5 |
| Phase 7 classical | 2000–2019 | 20 | from Phase 5 |
| Phase 8A pilot | 1–20 | 20 | from all |
| Phase 8A confirmation | 2000–2019 | 20 | shared with Phase 6 |
| Phase 8B pilot | 3000–3001 | 2 | from all |
| **Phase 8B confirmation** | **3100–3129** | **30** | **from all** |

## Template Manifests

| Set | Count | Source | Used In |
|-----|------:|--------|---------|
| Original training | 18 | `src.events.REGIME_WARNING_TEMPLATES` | Phase 5/5.5/6/7 training |
| Confirmation held-out | 36 | `src.confirmation_templates.CONFIRMATION_TEMPLATES` | Phase 6/7 testing |
| Phase-8B held-out (surge+normal) | 24 | Subset of confirmation templates | Phase 8B testing |

## Paper-1 Environment

- **Package:** `gym-invmgmt` v0.1.0
- **Source commit:** `a745fd5186a73d177dd94d283a4f8f8e8d329977`
- **Path:** `third_party/gym-invmgmt-paper/` (vendored at commit `a745fd5`)
- **Environment:** `GymInvMgmt/Serial-v0`
- **Topology:** RM(4) -> Factory(3, C=100) -> Dist(2) -> Retail(1) -> Market(0)
- **Lead times:** L = [0, 4, 4]
- **DemandSurge:** base_mu=10.0, shock_mag=2.0, shock_time=15, 30-period horizon
