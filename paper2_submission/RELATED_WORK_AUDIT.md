# Related-work audit

The paper should cite and distinguish five families rather than claim an empty
space:

1. **Decision-focused learning and predict-then-optimize:** these establish
   that predictive loss may not align with downstream decisions. Paper 2’s
   contribution is an executable semantic-sensing decomposition and controlled
   empirical evidence, not a new decision-focused objective.
2. **Value of information and decision theory:** these provide the conceptual
   language for information changing decisions. Paper 2 operationalizes the
   comparison through probabilistic text interpreters, a fixed controller, and
   sequential inventory trajectories.
3. **Calibration:** proper scoring rules and calibration methods are established.
   The empirical result is that calibration can change value under this
   controller even when accuracy changes little.
4. **Text/event extraction and LLM operations work:** these study extraction,
   forecasting, or operational assistance, but commonly do not isolate a fixed
   controller and downstream paired utility in the same protocol.
5. **Operational benchmarks:** ORL, OR-Gym, supply-chain RL/MARL, and inventory
   studies motivate the domain and provide lineage. Paper 2 should not resell
   Paper 1 infrastructure as its own novelty.

## Claim boundary

The exact distinction is not “semantic information has value,” which is a
decision-theoretic idea. It is the controlled empirical demonstration that
upstream classification/calibration metrics can disagree with downstream value
within a sequential operational contract, including held-out linguistic and
operational transfer and explicit controller/boundary analysis.

The closest modern comparisons should be verified by the author before
submission. No citation should be called “first” without a completed search.
