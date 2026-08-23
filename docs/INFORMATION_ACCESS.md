# Information Access by Condition

This table documents what each experimental condition knows and how information flows through the system.

## System Architecture

```
Warning Text
  → Semantic Interpreter (LLM / RuleBased / TF-IDF / Oracle)
    → Probabilistic Regime Belief P(regime)
      → Operational Controller (fixed policy)
        → Action (order quantities)
          → Simulator Environment
            → Utility / Reward
```

**Critical:** LLMs, classifiers, and rule-based extractors do NOT directly control inventory actions. They only produce probabilistic beliefs that the fixed controller converts to actions.

`OracleSemantic` (also called the Oracle Semantic Belief reference) supplies
perfect semantic regime belief to that same fixed controller. It is not an
optimal policy, hindsight oracle, or guaranteed upper bound on reward.

## Information Access Table

### Controlled Benchmark (Experiment A)

| Condition | Warning Text | Regime Label | Probability Belief | Future Demand | Simulator Internals | Action Authority |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **NoInfo** | No | No | Prior only | No | No | Yes (fixed policy) |
| **RuleBased** | Yes | No (heuristic) | Heuristic P | No | No | Yes (fixed policy) |
| **TFIDF_Raw** | Yes | Indirect | P(raw TF-IDF) | No | No | Yes (fixed policy) |
| **TFIDF_Calibrated** | Yes | Indirect | P(calibrated) | No | No | Yes (fixed policy) |
| **gpt-4o** | Yes | Indirect | P(LLM) | No | No | Yes (fixed policy) |
| **PerfectSemantic** | No | Yes (exact) | Degenerate {0,1} | No | No | Yes (fixed policy) |
| **OracleSemantic** | No | Yes (exact) | Degenerate {0,1} | No | No | Yes (fixed policy) |
| **HindsightOracle** | No | Yes (exact) | Degenerate {0,1} | **Yes** | **Yes** | **Optimal** |
| **CausalOptimizer** | No | Yes (exact) | Degenerate {0,1} | No | No | Optimal (receding horizon) |

### Paper-1 Gymnasium (Experiment B)

| Condition | Warning Text | Regime Label | Probability Belief | Future Demand | Simulator Internals | Action Authority |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **NoInfo** | No | No | Prior only | No | No | Yes (base-stock) |
| **RuleBased** | Yes | No (heuristic) | Heuristic P | No | No | Yes (base-stock) |
| **TFIDF_Raw** | Yes | Indirect | P(raw TF-IDF) | No | No | Yes (base-stock) |
| **TFIDF_Calibrated** | Yes | Indirect | P(calibrated) | No | No | Yes (base-stock) |
| **gpt-4o** | Yes | Indirect | P(LLM) | No | No | Yes (base-stock) |
| **OracleSemantic** | No | Yes (exact) | Degenerate {0,1} | No | No | Yes (base-stock) |

## Information Flow Details

### Text → Interpreter → Belief

- **LLM sensors** receive the warning text verbatim. They do not receive: ambiguity labels, ground-truth regime, simulator state, future information, or any operational parameters.
- **RuleBased sensor** applies keyword matching to the same text.
- **TF-IDF sensors** apply TF-IDF vectorization + logistic regression to the same text.
- **OracleSemantic/PerfectSemantic** receive the ground-truth regime label directly (bypassing text interpretation).

### Belief → Controller → Action

- The controller is **fixed** across all conditions within an experiment.
- It receives only the probability belief vector P(regime) and the current simulator state.
- It does **not** receive the warning text, the interpreter's internal representations, or any future information.
- In the controlled benchmark: Heuristic policy (threshold-based).
- In Paper-1 Gymnasium: BeliefAdaptiveController (base-stock with effective_mu = f(P(surge))).

### Latent Leakage Channels (documented, unused)

The Paper-1 Gymnasium environment has accessible attributes that could leak information if accessed:
- `env.demand_engine.get_current_mu(t)` — reveals current demand rate
- `env.env_kwargs['demand_config']` — reveals configuration including shock parameters
- `env.demand_engine.effects` — reveals active effects

These are **documented but never accessed** by any controller. The BeliefAdaptiveController reads only `env.X` (inventory) and `env.Y` (pipeline) at the current period.

## Key Design Principles

1. **Separation of perception and control:** Semantic interpreters produce beliefs; the controller converts beliefs to actions. This separation is the core of the benchmark design.
2. **Fixed controller:** The controller is identical across conditions to isolate the effect of belief quality.
3. **Paired evaluation:** Each seed produces the same stochastic demand trajectory for all conditions, enabling paired comparisons.
4. **No future leakage:** All conditions observe only current-period state. Future demand is generated incrementally and not accessible.
5. **Information monotonicity:** NoInfo ⊂ RuleBased ⊂ TF-IDF ⊂ LLM ⊂ OracleSemantic ⊂ HindsightOracle (in terms of information access).
