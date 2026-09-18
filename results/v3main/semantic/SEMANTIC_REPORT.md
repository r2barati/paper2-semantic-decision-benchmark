# V3 main semantic-transfer results (frozen beliefs, BEFORE economic evaluation)

Source: `results/v3main/semantic/` (driver `tools/run_semantic_v3main.py`,
library `src/semantic_analysis_v3main.py`). Test split n=160 queries (dev n=40
for selection reference only). Paired query-level bootstrap, B=10,000.
No simulator, no profit/return was examined to produce this file.

## IR (test, trec_eval; matches frozen dev ordering)

| system | nDCG@10 | R@10 | R@20 | RR |
|---|---|---|---|---|
| bm25 | 0.1150 | 0.1365 | 0.2604 | 0.2102 |
| dense | 0.1754 | 0.2062 | 0.3771 | 0.2656 |
| hybrid-k120 | 0.1817 | 0.2167 | 0.4094 | 0.2857 |
| rerank-top30 | 0.1829 | 0.1938 | 0.3708 | 0.2812 |

Rerank's test edge over hybrid is +0.0012 (vs +0.024 on dev): the depth win
shrinks out of sample. Recorded as finding; no tuning (frozen).

## Semantic accuracy (test, k=3 headline)

Real retrievers sit at 0.28–0.44 for all consumers/models — barely above the
C0 rule-based reference on the same evidence (0.40–0.43) and far below oracle
evidence (C3 on oracle-factual: 0.96–0.97). C1 is frequently WORSE than C0 on
the same retrieved set (e.g. bm25/8B: 0.275 vs 0.406). C3 beats C1
systematically on oracle evidence (+0.25) but only marginally on real
retrieval (+0.00–0.07). The bottleneck pattern is consumption + evidence
quality, not model scale: 14B ≈ 8B everywhere (Q6 diffs mostly small, most CIs
cross zero).

## Q1: does better retrieval improve semantic beliefs?

Rank correlations of system orderings (4 systems; direction only — exact
permutation p = 0.083–0.92 at n=4, significance never claimed):
C1: rho 0.8–1.0; C3: rho 0.8–1.0; C0: rho -0.2–0.6. Directionally yes for
LLM consumers; C0's flatness says the rule-based reference does not exploit
better ranking.

## Q2: ranking stability across consumers

System orderings correlate strongly across C0/C1/C3 (rho 0.80–1.00,
all p < 0.003, n=14 systems incl. oracle/interventions). No rank reversal
across consumers at the semantic level.

## Q3: Retriever x Consumer interaction (head: rerank vs bm25, k=3, accuracy)

Interaction = (bm25-rerank)|cb − (bm25-rerank)|ca; negative = rerank gains
more from the consumer switch.

| model | ca→cb | interaction | 95% CI |
|---|---|---|---|
| 8B | C0→C1 | -0.1375 | [-0.2250, -0.0562] |
| 8B | C0→C3 | -0.1062 | [-0.1750, -0.0375] |
| 8B | C1→C3 | +0.0312 | [-0.0500, +0.1125] |
| 14B | C0→C1 | -0.0812 | [-0.1500, -0.0125] |
| 14B | C0→C3 | -0.1125 | [-0.1875, -0.0375] |
| 14B | C1→C3 | -0.0312 | [-0.0875, +0.0250] |

Better retrieval helps LLM consumers more than the rule-based reference
(C0→C1/C0→C3 exclude zero); C1→C3 interaction is ~zero (both LLM consumers
profit alike). Consumer dependence is C0-vs-LLM, not C1-vs-C3.

## Q4: evidence-hit vs miss (retrieval systems only, grade ≥ 2)

Hit rows beat miss rows by +0.36–0.42 accuracy / −0.15–0.21 Brier in every
consumer/model/k cell (all CIs exclude zero; k=3 hit rate 182/640 = 28%).
Evidence-hit explains a large share of semantic quality — consistent with
the oracle gap above.

## Q5: C3 − C1 gain

On real retrieval ≈ 0 (rerank k=3: +0.038/8B, +0.031/14B, CIs cross zero);
on oracle evidence +0.25–0.26 (CIs exclude zero); on random NEGATIVE
(−0.16/8B, −0.22/14B — C3's abstention fires on garbage: 94%/56% abstain).
C3's machinery works (oracle, random) but real retrieval sits in between,
where neither consumer extracts much more than the other.

## Q6: 14B vs 8B replication

Per-cell diffs (8B − 14B) cross zero in 34/42 cells (max |Δ| 0.144 on random/C3/k5; retrieval-only cells max 0.06). Direction mixed. Within-family scale does not move the
needle — reported as is.

## Q7: held-out entities (descriptive)

Held-out-entity queries score HIGHER than non-held-out (+0.05–0.15 accuracy
across cells, e.g. bm25/C1/8B: 0.38 vs 0.23). Unexpected direction —
plausibly an easier query mix in the held-out bank rule, not stronger
generalization. Flagged, not explained away; the gym transfer + sim stage
must treat held-out claims cautiously.

## Sensitivity: cross-session cache nondeterminism

36 disputed keys flipped (8B: 27 keys/91 rows; 14B: 9 keys/49 rows) with
replication exact (0.00e+00). Headline comparisons move by max |Δeffect| =
0.0021 (Brier), 0.0000 (accuracy: no argmax flips). Immaterial to every
conclusion above; audit files preserved alongside the merge.

## Preregistered answers (Q1–Q7)

1. Yes for LLM consumers (rank direction), no for C0. 2. Yes — stable
(rho ≥ 0.8). 3. Yes — C0-vs-LLM interaction, not C1-vs-C3. 4. Hit explains
~0.4 accuracy. 5. C3 ≈ C1 on real retrieval; C3 wins on oracle, loses
(abstains) on random. 6. Replicated: 14B ≈ 8B. 7. Held-out higher —
mix effect suspected, reported descriptively.
