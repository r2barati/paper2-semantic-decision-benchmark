# Semantic Decision MVP — Phase 3.5

**How does semantic interpretation quality affect the downstream value of operational information in sequential decision-making?**

This experiment tests whether the *quality* of LLM-interpreted operational warnings causally affects inventory performance. Phase 3.5 establishes a clean causal decomposition of semantic sensing loss, controller quality, and future-information advantage.

---

## Research Hypothesis

> Better semantic interpretation of operational warnings produces greater downstream sequential decision value.

The causal chain:

```
Ground-truth disruption
        ↓
Controlled textual observation (11 templates × 3 ambiguity levels)
        ↓
Semantic interpretation (LLM or mock)
        ↓
Estimated event parameters (LT increase, duration, probability)
        ↓
Parameterized decision adaptation (policy uses INTERPRETED parameters)
        ↓
Persistent inventory consequences
        ↓
Profit / service / gap decomposition
```

---

## Information Hierarchy

Each condition has a clearly defined information set:

| Condition | Semantic Info | Policy | Optimization | Future Demand | Demand Distribution |
|-----------|--------------|--------|-------------|---------------|-------------------|
| NoInfo | None | BaseStock heuristic | None | No | No |
| LLM | Interpreted from text | DisruptionAware heuristic | None | No | No |
| PerfectSemantic | Perfect parameters | DisruptionAware heuristic | None | No | No |
| ActiveWrong | Wrong parameters | DisruptionAware heuristic | None | No | No |
| ActiveWrongOver | Overestimated parameters | DisruptionAware heuristic | None | No | No |
| CausalOptimizer | Perfect parameters | Receding-horizon LP | MPC optimization | No | Yes (Poisson λ) |
| HindsightOracle | Perfect parameters | MILP global optimizer | Full-horizon MILP | Yes | Yes (realized) |

**Key design principle**: LLM and PerfectSemantic share the *same* downstream policy (DisruptionAware). This isolates semantic quality from policy quality. CausalOptimizer uses the *same* semantic information as PerfectSemantic but a stronger optimization policy. This isolates policy quality from semantic quality.

---

## Gap Decomposition

The total performance gap from LLM to the privileged optimum decomposes cleanly:

```
TotalGap = J_HindsightOracle - J_LLM
         = SemanticRegret + ControllerGap + HindsightAdvantage

SemanticRegret     = J_PerfectSemantic - J_LLM
                     (loss from imperfect semantic interpretation)

ControllerGap      = J_CausalOptimizer - J_PerfectSemantic
                     (gain from stronger causal policy with same semantic info)

HindsightAdvantage = J_HindsightOracle - J_CausalOptimizer
                     (additional gain from knowing realized future demand)
```

**Identity**: `SemanticRegret + ControllerGap + HindsightAdvantage = TotalGap` (verified by test).

---

## Simulator Assumptions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Horizon | 40 periods | Long enough to see pre- and post-disruption effects |
| Initial inventory | 30 units | ~3.75 periods of demand |
| Demand distribution | Poisson(λ=8) | Simple, reproducible, realistic |
| Revenue per unit | $10 | |
| Ordering cost | $5 fixed + $2/unit | Fixed cost modeled exactly in HindsightOracle MILP |
| Holding cost | $1/unit/period | |
| Stockout cost | $8/unit | Lost-sales model (unsatisfied demand is lost) |
| Normal lead time | 2 periods | |
| Disrupted lead time | 5 periods | |
| Disruption start | t=18 | After the warning at t=15 |
| Disruption duration | 8 periods (t=18..25) | |
| Warning time | t=15 | 3 periods before disruption |
| Policy switch threshold | 0.70 probability | |

---

## Fixed Ordering Cost Audit

**HindsightOracle**: Uses MILP (Mixed-Integer Linear Program) with binary variables to model the $5 fixed ordering cost exactly. This is the global optimum for the deterministic full-information problem.

**CausalOptimizer**: Uses LP relaxation (ignores fixed ordering cost). This is an upper bound on what a causal MPC optimizer can achieve. The LP relaxation is documented.

---

## Warning Templates (11 templates × 3 ambiguity levels)

| Ambiguity | Templates | Expected extraction quality |
|-----------|-----------|---------------------------|
| Clear | 3 | High (exact numbers, direct notification) |
| Moderate | 4 | Medium (partial info, qualitative signals) |
| Vague | 4 | Low (uncertain, indirect, third-party reports) |

---

## Project Structure

```
paper2_semantic_decision_mvp/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── env.py           # Inventory simulator + HindsightOracle + CausalOptimizer
│   ├── events.py        # Ground-truth events + warning templates
│   ├── interpreter.py   # LLM semantic interpreter (real + mock + ActiveWrong)
│   ├── policies.py      # Decision policies (parameterized disruption-aware)
│   ├── experiment.py    # Multi-template, multi-condition experiment runner
│   └── metrics.py       # Performance metrics + gap decomposition
├── tests/
│   ├── test_env.py
│   ├── test_events.py
│   ├── test_interpreter.py
│   └── test_experiment.py
└── results/             # Generated CSVs + plots
```

---

## Setup

```bash
cd paper2_semantic_decision_mvp
pip install -r requirements.txt
```

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

---

## Running the Mock Experiment (No API Key)

```bash
python3 -m src.experiment
```

Uses deterministic keyword-based mock interpreter. No API key required.

---

## Running with a Real LLM

```bash
export LLM_API_KEY=sk-your-key
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
python3 -m src.experiment
```

Results are clearly labeled as `REAL_LLM` or `MOCK_LLM`.

---

## Results

After running the experiment, `results/` contains:

### Data
- `episode_results.csv` — Per-seed per-condition metrics
- `overall_summary.csv` — Overall condition statistics
- `template_summary.csv` — Per-template per-condition statistics
- `ambiguity_summary.csv` — Grouped by clear/moderate/vague
- `paired_comparisons.csv` — Paired statistical comparisons
- `regret_decomposition.csv` — SemanticRegret, ControllerGap, HindsightAdvantage per template
- `controller_comparison.csv` — All conditions compared
- `hindsight_gap.csv` — Gap decomposition for LLM condition
- `information_sets.csv` — Information access documentation per condition
- `semantic_interpretations.csv` — Per-template interpretation outputs
- `semantic_operational_analysis.csv` — Error ↔ outcome relationships
- `trajectory_diagnostics.csv` — Per-period behavior comparison

### Plots
- `profit_comparison.png` — Bar chart of mean profit by condition
- `sivr_by_template.png` — SIVR by warning template
- `sivr_by_ambiguity.png` — SIVR grouped by ambiguity level
- `regret_decomposition.png` — SemanticRegret + ControllerGap + HindsightAdvantage
- `semantic_error_vs_sivr.png` — Scatter: extraction error vs SIVR
- `profit_by_ambiguity.png` — Profit by condition × ambiguity
- `trajectory_diagnostics.png` — Fill rate, inventory, stockout comparison

---

## What This MVP Does NOT Include

- MCP, multi-agent systems
- LLM event generation
- Multiple event families
- News retrieval, web scraping
- Complex RL training
- Multi-echelon inventory
- GUI, database, Docker
