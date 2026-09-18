# Contribution: Paper 2 @ ECIR 2027

One-sentence test: **This paper is the first to show that relevance-ranked
evidence can systematically reduce sequential operating return under a
fixed controller while same-corpus oracle evidence adds large value —
measured by a frozen, regenerable retrieval→interpretation→control ladder
with signed, guarded attribution.**

- Methodological novelty: the six-node frozen ladder (no retrieval → actual
  → oracle-relevant → oracle-factual → perfect belief → hindsight) with
  matched budgets, paired world×language randomisation, and four-gap
  accounting. Excluded: SIVR (bookkeeping), crossed bootstrap (standard),
  the controller/simulator (instruments, not claims).
- Empirical novelty: sign pattern (LLM-consumer harm −35…−109 Holm-robust;
  oracle +71…+207; control gap +456.8 dominating) + rule-vs-LLM interaction
  (3/10 Holm) + tuned no-text beating every real retriever + ranking flips
  across environments.
- Dataset/benchmark novelty: 200-query operational report feed with
  regime-faithful pool judgments, frozen trecs/beliefs/episodes, SHA
  manifests, regenerable tables. Bounded: synthetic, single family, no
  human qrels.
- Application novelty: none claimed (inventory is the controlled world).

New-knowledge test (2–3 facts the community newly knows):
1. Relevance ordering can be exactly preserved into beliefs yet invert into
   operating loss under a fixed controller — with the loss located
   (selection 0.00, interpretation ∼37–140, control +456.8), not merely
   observed.
2. A tuned no-text policy beats every tested real retriever while oracle
   evidence helps — separating "retrieval fails to capture value" from
   "no value exists" — with the boundary conditions measured
   (prior-dependent, cost-asymmetric).
3. Consumer dependence of retrieval utility is a C0-vs-LLM level shift with
   retained nulls elsewhere — bounding, not extending, prior
   consumer-specificity claims.
