# Semantic Decision Value Benchmark

A framework for evaluating the decision value of probabilistic semantic sensing in executable sequential operational systems.

## Research Question

How much operational value can imperfect semantic interpretation of textual warning signals recover in sequential decision-making under uncertainty?

## Benchmark Architecture

```
Warning Text → Semantic Interpreter → Probabilistic Belief → Operational Controller → Action → Simulator → Value
```

The framework separates perception (interpreting text) from control (making decisions), enabling principled measurement of how information quality affects downstream performance.

## Key Metrics

- **OIV (Oracle Information Value):** Reward difference between Oracle-Belief Reference and NoInfo under the same fixed controller
- **SIVR (Semantic Information Value Recovery):** Signed secondary normalization of raw reward delta; undefined near zero OIV and diagnostic-only when OIV is negative
- **Semantic Regret:** Operational value lost due to imperfect interpretation

## Environments

### Experiment A: Controlled Benchmark
Single-SKU inventory system with supplier lead-time disruption. 3 regimes, 30-period horizon.

### Experiment B: Paper-1 Gymnasium Transfer
Multi-echelon supply chain (`gym-invmgmt` v0.1.0). 2 regimes (Normal, DemandSurge), 30-period horizon.

## Interpreters

| Sensor | Type | Requires API |
|--------|------|:---:|
| NoInfo | Prior probabilities only | No |
| RuleBased | Keyword matching | No |
| TFIDF_Raw | TF-IDF + LogReg | No |
| TFIDF_Calibrated | TF-IDF + LogReg + Isotonic | No |
| gpt-4o | OpenAI GPT-4o | Yes* |
| OracleSemantic | Ground truth | No |
| HindsightOracle | Optimal (Experiment A only) | No |

*LLM predictions are cached; reproduction is possible without API access.

## Key Findings

1. Semantic beliefs can alter downstream reward, but realized value depends on calibration, the fixed controller, and system dynamics
2. Classification accuracy does not determine operational value; the corrected Phase-7 calibration and Phase-8B reward tables report this directly
3. Probability calibration materially changes downstream value
4. The phenomenon survives transfer to a richer multi-echelon supply chain

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (offline, ~5 minutes)
python3 -m pytest tests/ -v

# Run benchmark (offline, uses cached predictions)
python3 -m benchmark.run --experiment controlled --sensor tfidf_calibrated
python3 -m benchmark.run --experiment gym --sensor oracle_semantic

# Run with live LLM (requires API key)
echo "OPENAI_API_KEY=<optional-key>" > .env
python3 -m benchmark.run --experiment controlled --sensor gpt4o --live
```

## Project Structure

```
├── src/                          # Source code
│   ├── events.py                 # Regime definitions, warning templates
│   ├── env.py                    # Inventory simulator
│   ├── interpreter.py            # Semantic interpreters
│   ├── metrics.py                # SIVR, regret, decomposition
│   ├── classical_baseline.py     # TF-IDF + LogReg
│   ├── gym_adapter.py            # Paper-1 Gymnasium adapter
│   ├── confirmation_templates.py # Held-out templates
│   └── experiment_phase*.py      # Experiment runners
├── tests/                        # deterministic benchmark and correction tests
├── results/                      # Frozen experimental results
│   ├── phase5/                   # Controlled benchmark (original)
│   ├── phase5_5/                 # Real-LLM evaluation
│   ├── phase6/                   # Confirmation/robustness
│   ├── phase7_classical_baseline/# TF-IDF + LogReg baseline
│   ├── phase8_gym_replication/   # Paper-1 Gym (exploratory)
│   ├── phase8b_gym_confirmation/ # Paper-1 Gym (confirmation)
│   └── publication/              # Manuscript-ready tables
├── docs/                         # Formal definitions, design docs
├── paper_sections/               # Manuscript source material
├── benchmark/                    # Reproducible runner
├── .llm_cache/                   # 197 cached LLM responses
├── BENCHMARK_VERSION             # Version freeze
├── SCIENTIFIC_LEDGER.md          # Authoritative numbers
├── REPRODUCIBILITY.md            # Reproduction guide
└── FINAL_TEST_AUDIT.md           # Test suite audit
```

## Limitations

- Synthetic warning text (not real supplier emails)
- Synthetic inventory environments (not production systems)
- Single SKU, fixed lead times
- 2-3 regime types (not continuous severity)
- Limited LLM coverage (gpt-3.5/4o-mini/4o only)
- Fixed controller (not co-optimized with interpreter)

## Citation

```
@article{semantic_decision_value_2026,
  title={A Framework for Evaluating the Decision Value of Probabilistic Semantic Sensing},
  year={2026}
}
```
