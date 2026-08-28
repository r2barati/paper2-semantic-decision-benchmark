# Paper identity

## Scientific question

**When does semantic interpretation quality translate into downstream
sequential decision value, and how are calibration, controller capability,
event type, and operational dynamics involved?**

## One-sentence finding

In controlled sequential inventory systems, semantic beliefs can materially
change operational reward, but accuracy alone does not determine value:
calibration and the fixed belief-to-control mapping mediate the result, and the
effect persists under a held-out multi-echelon transfer.

## Contributions

1. An executable framework separating text interpretation, probabilistic belief,
   a fixed controller, and sequential operational utility.
2. Evidence that classification accuracy and downstream value can disagree,
   with a particularly clear raw-versus-calibrated TF-IDF comparison.
3. Held-out operational-complexity and supply-side transfer evidence, with
   explicit controller-mismatch boundaries and reproducible artifacts.

## Non-contributions

The paper does not introduce a new universal information measure, prove that
LLMs outperform classical interpreters, optimize the controller jointly with
the interpreter, validate production supply chains, or claim universal
generalization.

## Closest prior-art category

Decision-focused learning, value of information, calibration for downstream
decisions, semantic sensing, causal policy evaluation, and operational ML
benchmarks.

## Strongest rejection argument

The framework may be viewed as a carefully executed application of known value-
of-information and decision-focused evaluation ideas to a synthetic inventory
system, with a fixed controller and limited event space. The paper must win on
clear empirical mechanism and reproducibility, not on a “first” algorithm claim.
