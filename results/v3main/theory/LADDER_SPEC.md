# FIVE-RUNG LADDER — FROZEN SPEC (2026-09-20, pre-execution; Workstream 2)

Question (frozen): where across the utility ladder do system rankings
agree, diverge, or reverse sign? Interpreted through the dead-zone map
(§9 status: invariance suffices for zero value but does not explain
consumer regimes) and the exploratory tripartite lens (labeled as such).

## 1. Rungs (frozen)

Systems RANKED: {bm25, dense, hybrid, rerank}. Oracles (relevant/factual)
are reference points, never ranked.

* **R1 nDCG@3:** recomputed locally via ir_measures (trec_eval definition,
  frozen metric) from frozen trecs + test qrels, test-160. Depth 3 matches
  R2/R3 context width. Frozen nDCG@10 tables reported as context only.
* **R2 UDCG-inspired machine utility@3** (never "UDCG@3"): per-doc gain
  from frozen C3 per-doc judgments (8B primary; 14B sensitivity):
  u = +w on support, −0.5w on refute, 0 on na/none-event, with frozen
  weights w = confidence×entity(1.0/0.2)×fresh(1.0/0.2). Signed DCG over
  ranks 1..3 with 1/log2(rank+1) discount; IDCG = DCG of u-descending
  order; score = DCG/IDCG if IDCG>0 else 0.0 (frequency reported).
  Coverage gate: 100% of top-3 (system, test-query) docs must hit cache;
  misses documented, NEVER new calls. Consumer-free IR-side rung (one
  model's judgments as annotator; model noted).
* **R3 belief quality:** Brier primary, accuracy secondary, per
  (system, consumer, model), test means from frozen semantic_all.
* **R4 document-VoI (rerank-only deep dive, NOT a 4-system ranking):**
  per-position signed VoI + CIs reused frozen from Stage-4/A1 verdicts.
  Per-system VoI labels beyond rerank DO NOT EXIST — recorded as a stated
  limitation (no new calls to fill it).
* **R5 sequential J:** test mean dJ vs NoInfo per (system, consumer, model)
  from frozen utility_by_query (+ balanced-headline reference).

## 2. Agreement/divergence analysis (frozen)

Per stratum (C0/none, C1-8B/14B, C3-8B/14B; IR rungs single ranking):
rank the 4 systems by rung mean (ties → average rank, documented).
Adjacent-rung comparisons with query bootstrap (B=2000):
Kendall tau + pairwise sign-agreement rate over the 6 system pairs,
each with 95% CI. **Strong reversal** (claimable): a pair whose difference
CI excludes 0 on BOTH rungs with OPPOSITE signs. **Weak reversal**
(descriptive): point-estimate sign flip. Report both; claim only strong.
Seeds frozen at execution (82000/82001/...); unit = query.

## 3. Focus questions (frozen; descriptive estimands, no gate)

Q1 nDCG→UDCG-inspired: agree/diverge/reverse? Q2 UDCG-inspired→Brier?
Q3 Brier→J (with VoI positional signs as mechanism notes on rerank)?
Any strong reversals, and do they concentrate at one interface?
Interpretation may invoke the exploratory tripartite lens, labeled so.

## 4. Non-goals

No new LLM calls (coverage gate aborts to documented-missing instead);
no per-system VoI beyond rerank; no threshold tuning; no Agentick;
no manuscript prose. This ladder is measurement with preregistered
reversal definitions, not a gate — the verdict is tables + reversal list.
