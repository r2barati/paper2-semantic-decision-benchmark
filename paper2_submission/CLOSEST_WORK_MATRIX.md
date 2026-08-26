# Closest-work matrix

| Work family | Semantic input | Probabilistic belief | Sequential environment | Controller coupling | Downstream utility | Calibration | Structural OOD | Classical/LLM baselines | Release |
|---|---|---|---|---|---|---|---|---|---|
| decision-focused learning | usually features/predictions | sometimes | often optimization, sometimes sequential | task loss/objective | yes | not central | variable | variable | variable |
| value of information | evidence/information | Bayesian or abstract | decision process | explicit | yes | not central | rarely structural | not usually LLM | theory-heavy |
| text event extraction | text | labels/probabilities | usually forecasting/extraction | weak or absent | indirect | sometimes | limited | classical/LLM | variable |
| LLM operations/forecasting | text/time series | model output | operational tasks vary | variable | often task KPI | variable | variable | LLM-focused | variable |
| ORL/OR-Gym/inventory benchmarks | structured demand/state | not semantic | yes | explicit | yes | no | scenario-dependent | OR/RL | some releases |
| Paper 2 | warning text | explicit regime belief | yes | fixed controller | paired reward, OIV/SIVR | central result | held-out templates, demand/supply transfer | rule, TF-IDF, GPT, oracle, NoInfo | frozen offline package |

This matrix supports a bounded distinction, not a uniqueness claim.
