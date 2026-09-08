# Closest-work matrix

> **Corrected September 2026.** The Donti et al. row previously recorded "no
> probabilistic model" and "no sequential environment". Both were wrong: the
> paper explicitly learns probabilistic models for downstream stochastic
> objectives and includes a multi-period energy-storage arbitrage task as well
> as inventory. Overstating a neighbouring work's limitations is a
> novelty-inflation error and is corrected here.
>
> **Information-retrieval rows added.** The earlier matrix had no IR evaluation
> work in it at all, which is why the venue-fit objection went unnoticed.

| Work | Venue/year | Semantic input | Probabilistic belief | Sequential environment | Controller coupling | Downstream utility | Calibration | Structural transfer | Baselines | Paper 2 distinction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Donti et al., Task-Based End-to-End Model Learning | NeurIPS 2017 | no | **yes** | **yes (energy-storage arbitrage; also inventory)** | optimizer | yes | no | variable | ML | learns probabilistic models for downstream stochastic objectives; the distinction is that we hold the pipeline fixed and vary the *information supplied to it*, and our input is retrieved text rather than features |
| Farahmand et al., Value-Aware Loss Function | AISTATS 2017 | no | no | MBRL | planner/policy | yes | no | no | MLE | value-aware transition loss; not semantic interpreter comparison |
| Elmachtoub & Grigas, Smart Predict-then-Optimize | Management Science 2022 | no | no | mostly optimization | optimizer | yes | no | variable | prediction | decision loss for optimization coefficients; not fixed-controller semantic utility |
| Lambert et al., Objective Mismatch | L4DC 2020 | no | no | control | policy | yes | no | limited | model learning | predictive/control mismatch in MBRL; Paper 2 studies semantic belief/control mismatch |
| Grimm et al., Value Equivalence Principle | NeurIPS 2020 | no | no | MBRL | Bellman planner | yes | no | no | MLE | formal model equivalence; Paper 2 makes no equivalent-model theorem claim |
| Zhao et al., Calibrating Predictions to Decisions | NeurIPS 2021 | class probabilities | yes | downstream loss | decision-maker class | yes | central | no | recalibration | closest calibration theory; Paper 2 is an operational sequential empirical case |
| Arumugam & Van Roy, Value of Information When Deciding What to Learn | NeurIPS 2021 | abstract information | target distribution | yes | learning/exploration | yes | no | no | theory | information acquisition theory; Paper 2 uses fixed semantic sensors |
| Voelcker et al., Calibrated VAML | ICML 2025 | no | probabilistic model | MBRL | value planner | yes | central | limited | MLE | recent value-aware/calibration threat; not text or semantic sensing |
| Boute et al., Benchmarking RL for Supply Chain Management | EJOR 2022 | no | no | supply chain | policy | KPI/reward | no | scenario transfer | RL/OR | domain benchmark lineage; no semantic/controller factorization |
| Li et al., LLMs for Supply Chain Optimization / OptiGuide | 2023 | text/LLM | model output | optimization interface | variable | KPI/what-if outcome | not central | application-dependent | LLM | closest operational-language evidence; not frozen paired semantic-sensing protocol |
| Salemi & Zamani, eRAG | SIGIR 2024 | text corpus | no (generation) | no | LLM generator | downstream generation quality | no | no | retrieval metrics, no-retrieval | evaluates retrieval through downstream LLM output; our downstream consumer is a control policy in a stochastic environment, so the quantity is a realised return and the error asymmetry comes from the environment's cost structure rather than a judge |
| CUE-R (preprint, Apr 2026) | arXiv preprint | text evidence | per-evidence utility | no | LLM generator | intervention-based utility | no | no | no-retrieval controls | intervention-based per-evidence utility in RAG; we intervene on evidence selection and measure simulated *operational consequences* instead of trace changes. Preprint, not peer reviewed |
| GroGU (preprint, Jan 2026) | arXiv preprint | text evidence | grounding utility | no | LLM generator | model-specific grounding | no | no | relevance | argues utility is consumer-specific; we measure exactly that dependence end to end and find the retrieval/utility ordering flips between two interpreters. Preprint, not peer reviewed |
| Borlund, The Concept of Relevance in IR | JASIST 2003 | -- | -- | -- | -- | situational relevance | no | -- | -- | situational/task relevance long predates this work; our contribution is an executable instrument for one consequential task, not the observation that consequences matter |
| **This work** | this work | warning text | explicit regime belief | finite-horizon inventory | fixed controller | paired reward/OIV/SIVR | central empirical result | held-out templates and supply-side transfer | rule, TF--IDF, GPT, oracle, NoInfo | bounded empirical characterization |

No row supports a "first" claim, and the IR rows above make that explicit:
downstream-aware retrieval evaluation exists, intervention-based per-evidence
utility exists, and consumer-specific grounding utility exists. The defensible
distinction is the **consumer**: a control policy acting in a stochastic
environment with an asymmetric cost structure and a state that persists across
decisions. That is what makes non-relevant documents unequally harmful, gives
the evaluation a realised-return quantity rather than an output-quality
judgement, and lets the retrieval/utility ordering be measured rather than
argued.

The second distinction is the control set. Constant-belief, development-tuned
no-text, shuffled-text, label-only, no-retrieval and oracle-selection arms are
reported for every claim, and several of them overturn readings that the
uninformed-prior comparison alone would support.
