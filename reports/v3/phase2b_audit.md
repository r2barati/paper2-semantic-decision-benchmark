# 2B-answerability verdict: FAIL as pooled IR task — V3.1 spec authorized

Date: 2026-09-14. No corpus text changed. No test qrels observed (only dev.tsv +
corpus/queries.jsonl touched). No LLM spend. No scores optimized: the numbers below
are diagnostics for the verdict only.

## Evidence (mechanism, not scores)

1. Of 400 dense top-10 slots over 40 dev queries, **381 are UNJUDGED** (only 19 judged).
   100% of the unjudged are same-entity current-window ops docs, e.g. "Routine status:
   operating normally", "inventory within expected ranges", "throughput down, cover
   from safety stock" — texts a TREC assessor would judge relevant.
2. Blind audit (60 pairs, judgments recorded before scoring): 0.82 exact / 0.90 binary
   agreement — BUT the sample contained zero wrong-regime same-node docs, so it did not
   test the crux. Corrected analysis: pool-relative qrels over a SHARED corpus mean other
   queries' regime texts about the same node are unjudged→scored-0 for this query.
3. There are ZERO same-node current wrong-regime grade-0 docs by construction (all
   misleading docs are wrong-entity, i.e. header-identifiable). The failure is purely
   pooled-qrels × shared-corpus: relevance truth depends on which pool a doc was
   generated into — hidden generator metadata the query cannot express.
4. Partial signal exists (Recall@50 rises with grade; surge queries near-zero from a
   supply-biased query template), but top-k measurement is dominated by unjudged docs.

## Verdict

The IR task is **informationally underdetermined as pooled**, not merely difficult.
Spending the LLM matrix on it would measure unjudged-rate, not transfer. Phase-2
diagnostic recorded; corpus text stays frozen; qrel REGIME-AGNOSTIC re-grade authorized:

## V3.1 spec → superseded by measurement (2026-09-14)

V3.1 regime-agnostic re-grade was implemented (`src/qrels_v31*.py`, pool files rewritten,
complete judgments generated). But recomputation under complete V3.1 qrels SATURATES
(dev nDCG@10 0.77–0.99, RR 1.0): entity filtering is solved, so the task vanishes.
Regime-faithful qrels are unmeasurable past ~150 same-node competitors (dev 0.015–0.06)
and regime-agnostic qrels are trivially saturated — the defect is the COMPETITION RATIO,
not the grading rule. Resolution: 64 nodes (~3 queries/node, guard ≤110, measured 71),
preserving regime-faithful pool qrels AND the shared-reality design. V3.1 files retained
as superseded diagnostic. Final semantics decided by dev measurement on the new build
(rules v3), test sealed throughout.

- 3: own node + current + specific ops signal (ANY direction: delay/surge/specific all-clear)
- 2: own node + current + hedged/partial signal, specific normal-ops status, or correction
     (corrections are informative about conditions; their harm is stance metadata, not qrels)
- 1: own node + resolved window (unchanged)
- 0: wrong entity OR routine corporate traffic (no ops content)
- Regime-faithfulness, misleading/accidentally-correct, stance, freshness: op-labels ONLY.
- Re-grade rule is deterministic from doc/query metadata; applied to dev AND test
  (test text untouched, test grades never observed — no peeking). New SHAs, re-freeze.
- The paper's thesis is STRENGTHENED, not weakened: same relevant evidence, different
  consumers → different utility, with misleading-ness as a measured downstream moderator.
- Pooling for the main run: union of top-50 across frozen systems; any retrieved unjudged
  doc is graded by the same deterministic rule (no silent zeros).
- Human validation (300 blind test pairs) judges the V3.1 rule output — still required,
  now meaningful because the rule is query-answerable.
