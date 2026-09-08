# Headline results

Regenerated 7 September 2026 from the corrected pipeline. Every number here is
in `SCIENTIFIC_LEDGER.md` or a table under
`paper2_submission/manuscript_lncs/tables/`, both generated from the frozen
episode files. The previous version of this file is superseded; it described a
pipeline with an information leak in the historical controller matrix, an
incorrect optimiser LP, off-by-one arrival timing, and no text-free controls.

## Main-text results

1. **Retrieval quality and decision utility can order systems differently
   (Experiment R).** At a matched budget k=3 with the calibrated interpreter,
   every practical retrieval system is worse than retrieving nothing --- BM25
   -74.3 [-98.9, -47.9], TF-IDF cosine -61.1, dense -1.5 [-31.1, +29.2] ---
   while the oracle evidence selector gains +69.0 [+40.2, +96.1]. Kendall's
   tau_b between the nDCG@3 and reward orderings over the four ranking systems
   is +0.33 at k=3 and 0.00 at k=1 and k=5.

2. **The dissociation has a mechanism.** Non-relevant documents differ in how
   they mislead. Random retrieval surfaces 1.33 actively misleading documents
   per episode, BM25 1.28, dense 0.87, oracle 0.67. BM25 raises nDCG by
   preferring operationally-worded reports, many of which concern a different
   node.

3. **It is consumer-dependent.** With the rule-based interpreter the two
   orderings agree exactly (tau_b = 1.0 at k=3 and k=5) and every retrieval
   system beats the no-retrieval control.

4. **A text-free control is a demanding baseline.** In Experiment M the
   development-tuned constant belief returns 560.4, above raw TF-IDF (516.2),
   calibrated TF-IDF (530.2), GPT-4o (534.9) and the oracle-belief reference
   (552.0); only RuleBased (568.7) exceeds it. In Experiment S it returns
   1725.4 against 1484.0 for the best interpreter.

5. **The shuffled-text control separates the two transfers.** In Experiment M
   shuffling text drops the calibrated interpreter from 530.2 to 451.1, below
   the uninformed prior, so reading the correct text matters there. In
   Experiment S the shuffled control returns 1421.8 against 1422.0 for the
   genuine interpreter despite belief accuracy halving from 0.875 to 0.500;
   that transfer does not support a reading-the-warning claim.

6. **Calibration is confounded with ensembling (Experiment C).** Fold
   ensembling alone lowers SIVR from 0.245 to 0.154; the calibration map then
   raises it to 0.338. The naive raw-versus-calibrated difference is not
   attributable to calibration. Accuracy cost is 3 of 36 held-out texts once
   the fold vocabulary leak is fixed.

7. **Probability quality is not always the operative variable.** The label-only
   ablation of the raw classifier reaches SIVR 0.773, above both calibrated
   arms. Brier (0.222 versus 0.552) and log-loss (2.558 versus 0.932) disagree
   about it, and neither predicts its operational advantage.

8. **Boundaries (Experiment B).** In three of six one-factor variants the
   oracle-belief reference underperforms the uninformed prior, so no normalised
   information-value statement is meaningful there; these are marked rather
   than given a ratio. Under short lead times the calibrated interpreter loses
   150 reward and GPT-4o 185 against the uninformed prior.

## Claims explicitly withdrawn

- **"A stronger controller absorbs semantic information value."** The
  historical sensor-by-controller matrix that supported this passed the
  environment's true disruption to the NoInfo controller's planner, which is
  why NoInfo and PerfectSemantic had identical rewards. The result is removed
  from the submission.
- **"Calibrated TF-IDF has higher downstream value than raw TF-IDF because of
  calibration."** True in direction, but the magnitude was confounded with fold
  ensembling; the isolated contrast is reported instead.
- **"All four non-reference interpreters produce positive paired reward
  effects, therefore semantic information has value."** The effects against the
  uninformed prior remain positive, but three of four interpreters do not beat
  a text-free control that was tuned on development seeds.
