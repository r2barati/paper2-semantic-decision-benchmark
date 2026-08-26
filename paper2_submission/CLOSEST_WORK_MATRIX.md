# Closest-work matrix

| Work | Venue/year | Semantic input | Probabilistic belief | Sequential environment | Controller coupling | Downstream utility | Calibration | Structural transfer | Baselines | Paper 2 distinction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Donti et al., Task-Based End-to-End Model Learning | NeurIPS 2017 | no | no | no | optimizer | yes | no | variable | ML | task-aware stochastic optimization; not text-to-belief sensing |
| Farahmand et al., Value-Aware Loss Function | AISTATS 2017 | no | no | MBRL | planner/policy | yes | no | no | MLE | value-aware transition loss; not semantic interpreter comparison |
| Elmachtoub & Grigas, Smart Predict-then-Optimize | Management Science 2022 | no | no | mostly optimization | optimizer | yes | no | variable | prediction | decision loss for optimization coefficients; not fixed-controller semantic utility |
| Lambert et al., Objective Mismatch | L4DC 2020 | no | no | control | policy | yes | no | limited | model learning | predictive/control mismatch in MBRL; Paper 2 studies semantic belief/control mismatch |
| Grimm et al., Value Equivalence Principle | NeurIPS 2020 | no | no | MBRL | Bellman planner | yes | no | no | MLE | formal model equivalence; Paper 2 makes no equivalent-model theorem claim |
| Zhao et al., Calibrating Predictions to Decisions | NeurIPS 2021 | class probabilities | yes | downstream loss | decision-maker class | yes | central | no | recalibration | closest calibration theory; Paper 2 is an operational sequential empirical case |
| Arumugam & Van Roy, Value of Information When Deciding What to Learn | NeurIPS 2021 | abstract information | target distribution | yes | learning/exploration | yes | no | no | theory | information acquisition theory; Paper 2 uses fixed semantic sensors |
| Voelcker et al., Calibrated VAML | ICML 2025 | no | probabilistic model | MBRL | value planner | yes | central | limited | MLE | recent value-aware/calibration threat; not text or semantic sensing |
| Boute et al., Benchmarking RL for Supply Chain Management | EJOR 2022 | no | no | supply chain | policy | KPI/reward | no | scenario transfer | RL/OR | domain benchmark lineage; no semantic/controller factorization |
| Li et al., LLMs for Supply Chain Optimization / OptiGuide | 2023 | text/LLM | model output | optimization interface | variable | KPI/what-if outcome | not central | application-dependent | LLM | closest operational-language evidence; not frozen paired semantic-sensing protocol |
| **Paper 2** | this work | warning text | explicit regime belief | finite-horizon inventory | fixed controller | paired reward/OIV/SIVR | central empirical result | held-out templates and supply-side transfer | rule, TF--IDF, GPT, oracle, NoInfo | bounded empirical characterization |

No row supports a “first” claim. The defensible distinction is the combination
of a declared text-to-belief interface, fixed controller, paired trajectory
utility, calibration analysis, and structural/boundary disclosures in one
reproducible operational protocol.
