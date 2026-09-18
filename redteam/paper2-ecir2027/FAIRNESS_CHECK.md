# Fairness check (invalidation audit): evidence sought against our own verdicts

For each remaining criticism, the overturning evidence and the search result.

1. "C1 harm is a weighting artifact." Overturned by: harm under all five
   weightings with CIs. Searched: balanced/empirical Holm-robust; normal/
   surge/delay-heavy point same direction (CIs balanced+empirical only).
   Verdict stands (scoped to BM25/dense/hybrid C1; rerank-C1 empirical n.s.
   disclosed).
2. "Interaction is noise." Overturned by: Holm over 10 preregistered cells.
   Searched: 3/10 reject, all rule-vs-LLM; C1-C3 nulls retained and reported.
   Verdict stands as scoped level-shift claim.
3. "Tuned comparator is luck/grid artifact." Overturned by: smooth dev
   gradient toward surge triples, prior ranked 19th, test replication
   +102.09 CI [+69,+138]. Searched and confirmed; prior-dependence disclosed.
4. "Text clustering voids precision." Overturned by: cluster bootstrap on
   every arm (median 1.08×); all starred claims hold under both. Searched;
   stands.
5. "Dev queries contaminate headlines." Overturned by: full test-only
   recompute (65/105 Holm; headline values shifted trivially). Searched;
   stands with dev-inclusive archived.
6. "Oracle ladder is degenerate throughout." Partially UPHELD: rel→fact
   0.00 reflects saturating oracle construction — reported descriptively,
   never as an effect. Scope the claim, done in text.
7. "A stronger retriever would erase the pattern." NOT SEARCHABLE
   pre-deadline (new bank + belief spend). Recorded as future work, not as
   a concession: rerank (best available) is neutral/negative while oracle
   helps, which bounds this objection to untested banks.
8. "Human qrels would change IR numbers." NOT SEARCHABLE (no annotators).
   Bounded: C0 pattern needs no qrels; interaction replicates across models;
   absolute nDCG quarantined. Recorded, not resolved.
