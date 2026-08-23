# Reproducibility Guide

## Benchmark Version

```
paper2-benchmark-v1.0
Frozen: 2026-08-22
```

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
| gymnasium | 1.0.0 |
| python-dotenv | installed (version unpinned) |
| pandas | 2.3.3 |

## Quick Start: Reproduce Main Results

### Prerequisites

```bash
pip install -r requirements.txt
pip install gymnasium  # for Phase-8B results
pip install -e "Paper 1/final-github-clean/gym-invmgmt-paper"  # for Paper-1 env
```

### Run All Tests (offline, no API key needed)

```bash
python3 -m pytest tests/ -v
# 244 tests, all pass
```

### Run Individual Experiments

```bash
# Phase-7 classical baseline evaluation
python3 -m src.experiment_phase7

# Phase-8B frozen confirmation (offline, cached LLM predictions)
python3 -m src.experiment_phase8b --seeds 30 --seed-start 3100 --no-llm

# Phase-8B with live LLM (requires OPENAI_API_KEY or LLM_API_KEY in .env)
python3 -m src.experiment_phase8b --seeds 30 --seed-start 3100 --llm-models gpt-4o
```

## Frozen Results

All experimental results are pre-computed and stored under `results/`. The LLM cache at `.llm_cache/` contains 197 frozen LLM API responses, enabling full offline reproduction.

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
# All experiments use cached responses when available
# Cache location: .llm_cache/
# Cache entries: 197 (JSON files)
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
- **Path:** `Paper 1/final-github-clean/gym-invmgmt-paper/`
- **Environment:** `GymInvMgmt/Serial-v0`
- **Topology:** RM(4) -> Factory(3, C=100) -> Dist(2) -> Retail(1) -> Market(0)
- **Lead times:** L = [0, 4, 4]
- **DemandSurge:** base_mu=10.0, shock_mag=2.0, shock_time=15, 30-period horizon
