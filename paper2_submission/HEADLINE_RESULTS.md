# Headline results

Regenerated 8 September 2026. Every number is in `SCIENTIFIC_LEDGER.md` or a
generated table under `paper2_submission/manuscript_lncs/tables/`, all produced
from the frozen episode files. The paper is framed as a **negative evaluation
result**: retrieval utility is consumer-dependent, and relevance-based
evaluation does not detect it.

## The central finding (Experiment R)

At a matched evidence budget k=3, with the interpreter and controller held
fixed:

| System | nDCG@3 | Harmful rate | RuleBased ΔJ | Calibrated TF-IDF ΔJ | Interaction |
|---|---:|---:|---:|---:|---:|
| Random | 0.090 | 0.44 | +11.6 [-11, +35] | -49.8 [-78, -19] | +61.4 |
| BM25 | 0.274 | 0.43 | +14.9 [+5, +26] | -74.3 [-99, -48] | +89.2 |
| TF-IDF cosine | 0.336 | 0.32 | +25.8 [+14, +39] | -61.1 [-88, -32] | +86.9 |
| Dense (MiniLM) | 0.617 | 0.29 | +48.4 [+32, +65] | -1.5 [-31, +29] | +49.9 |
| Oracle relevance selector | 1.000 | 0.22 | +60.6 [+48, +73] | +69.0 [+40, +96] | — |

1. **Every ranking system helps one consumer and harms the other.** The
   retriever × consumer interaction changes sign for all four, magnitude +49.9
   to +89.2.

2. **The information is present and extractable.** The oracle relevance
   selector improves *both* consumers. This is a mismatch between retrieval and
   its consumer, not an absence of useful evidence.

3. **Relevance-based evaluation detects this for one consumer and not the
   other.** Rank-reversal rate 0.00 (τ_b = +1.00) for RuleBased; 0.33
   (τ_b = +0.33) for calibrated TF-IDF. Nothing upstream distinguishes the two
   situations.

4. **The gradient is shared; the level is not.** Episode-level correlation
   between ranking quality and reward is nearly identical for both consumers
   (r = +0.372 and +0.382). Within a consumer, better evidence does produce
   better decisions. The calibrated consumer is simply below zero across the
   whole practical range. Consumer dependence here is an offset, not a reversed
   gradient — which is exactly why a relevance-only evaluation misses it.

5. **More evidence is not better.** k=3 → k=5 raises dense nDCG (0.617 → 0.691)
   and lowers its effect on the calibrated consumer (-1.5 → -18.9).

## Supporting diagnosis (Experiments C, M, S, B)

These locate the mismatch; they are not the contribution.

- **A text-free control is demanding.** A constant belief tuned only on
  development seeds returns 560.4 in Experiment M against 516.2 / 530.2 / 534.9
  for the TF-IDF and GPT-4o interpreters; only RuleBased (568.7) exceeds it. In
  Experiment S it returns 1725.1 against 1447.0 for the best interpreter.
- **Belief accuracy does not imply operational value.** In Experiment S the
  shuffled-text control returns 1415.2, *above* the genuine interpreter's
  1372.9, at roughly half the belief accuracy. In Experiment M shuffling does
  destroy the effect (530.2 → 451.1).
- **Probabilistic sophistication does not either.** The label-only ablation of
  the raw classifier reaches normalised value 0.773 against 0.245 for its own
  probabilities; Brier and log-loss disagree about it and neither predicts its
  advantage.
- **Calibration is confounded with ensembling.** Fold ensembling alone lowers
  normalised value 0.245 → 0.154; the calibration map then recovers to 0.338.
- **The perfect-semantic reference is not an upper bound.** It is beaten by a
  constant belief that reads nothing, in both transfer environments. Renamed
  from "oracle belief" accordingly.

## Claims explicitly withdrawn

- **"A stronger controller absorbs semantic information value."** The
  historical matrix supporting this passed the environment's true disruption to
  the uninformed controller's planner. Removed.
- **"Calibration raises downstream value by X."** True in direction; the
  magnitude was confounded with fold ensembling.
- **"Positive paired effects against NoInfo show semantic information has
  value."** Arithmetically true, but three of four interpreters do not beat a
  text-free control tuned on development seeds.
- **"Semantic interpretation generally improves decisions."** Not supported.
  The paper does not claim it.
