# Table 1: Information Access by Experimental Condition

| Condition | Warning Text | Latent Label | Probability Belief | Future Realized Demand | Simulator Internals | Action Authority |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| NoInfo | - | - | Prior only | - | - | Fixed policy |
| RuleBased | Yes | Heuristic | Heuristic P | - | - | Fixed policy |
| TFIDF_Raw | Yes | Indirect | P(raw) | - | - | Fixed policy |
| TFIDF_Calibrated | Yes | Indirect | P(calibrated) | - | - | Fixed policy |
| gpt-4o | Yes | Indirect | P(LLM) | - | - | Fixed policy |
| OracleSemantic | - | Yes (exact) | {0, 1} | - | - | Fixed policy |
| HindsightOracle | - | Yes (exact) | {0, 1} | Yes | Yes | Optimal |

**Text → Interpreter → Belief → Controller → Action → Environment → Utility**

- Interpreters receive **only** warning text (no simulator access).
- Controller receives **only** belief vector (no text or interpreter internals).
- HindsightOracle is the theoretical upper bound; OracleSemantic is the semantic reference.
