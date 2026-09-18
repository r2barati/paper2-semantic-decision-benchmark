# SECOND-PASS REVIEW — rewritten V3 manuscript (ECIR 2027 standard)

Method: six fresh reviewers saw ONLY `paper2_submission/manuscript_lncs/main.tex`
+ `tables/v3_*.tex` + frozen delivery parquets (no history, no prior
recommendation). All six returned reject — unanimously harsher than the
first pass, and productively so: the rejections contained verifiable
technical claims. Every verifiable claim was checked; fixes applied; B/C did
targeted re-verification. This file records meta-classification, the
claim-by-claim audit, and the final verdict.

## Meta-classification of every fresh objection

FATAL if left standing, with resolution status:
- B-F1 universal harm vs null best arm + robust positive → RESCOPED
  (C1 −41…−104 all robust; C3 3/8; +11 confronted with abstention-fallback
  mechanism). Resolved.
- B-F2 160/200 denominator → test-only recompute of ALL sim headlines;
  dev-inclusive archived. Resolved.
- B-F3 gap algebra → signed formulae + rung defs in text. Resolved.
- B-F4/F5, M-scoping items → all scoped; verified by B re-check (13 FIXED).
- C dual-prior failure → dual tables shipped; prose scoped to robust subset;
  rerank-C1 empirical n.s. explicitly stated. Resolved.
- C n=5/regime/single-family → disclosed exploratory/bank-scoped. Residual:
  MAJOR (design limit, significance judgment).
- D lineage/novelty → F1 concession + Donti/SPO engagement + SIVR hygiene in
  text. Residual: DISAGREEMENT on significance bar.
- E1 weighting+controller+VOI → VOI estimand defined; harm persists under all
  five weightings for the robust subset; delay-heavy check refutes reversal;
  tuned comparator bounds. ADDRESSED. Residual MAJOR (one controller class).
- E interventions-brittleness → REFUTED by data (perturbations stay positive;
  C1/C3 split supports design); now reported. NOT SUPPORTED as objection.
- E delay-heavy reversal → REFUTED (C1 −169/−90 persist). NOT SUPPORTED.
- E rel→fact degeneracy → disclosed as saturating descriptively. MINOR.
- E control dominance → VOI paragraph; disclosed. DISAGREEMENT on reading.
- E gym → disclosed secondary with artifact. MINOR.
- A external validity / weak retrieval / single domain → disclosed throughout;
  absolute values + oracle contrast bound the reading. MAJOR (significance).
- F fit → manuscript centers IR object; lineage conceded. DISAGREEMENT.

No FATAL issue remains standing. Highest residuals are MAJOR-by-design
(single domain/controller/family, no human validation) — disclosed
limitations bearing on significance, not correctness.

## Claim-by-claim audit (manuscript sentence → artifact → status)

| Manuscript claim | Artifact | Status |
|---|---|---|
| C1 arms −41…−104, all Holm-robust (k=3) | utility_by_arm_balanced | SUPPORTED (8/8) |
| C3 three of eight, 14B only | same | SUPPORTED |
| rerank-C3/C0 directionally negative, n.s. | same (CIs cross zero) | SUPPORTED |
| random/C3/14B +11.0, fallback mechanism | same (+10.9∗) + abstention rates | SUPPORTED (+mechanism stated) |
| oracle +178.9 [+135,+221] | same | SUPPORTED |
| gaps −21.4/0.00/36.4/456.8 | four_gaps (coherent, exact) | SUPPORTED |
| tuned 1881.1, +102.1, SIVR 0.474 | extension_notext_tuned_J | SUPPORTED (extension-labeled) |
| tuned beats every real-retriever arm | both priors max real < 1881 | SUPPORTED (scoped) |
| interaction 3/10, rule-vs-LLM | headline_holm + magnitudes | SUPPORTED |
| Q1 direction-only, exact p | q1_corr (0.083–0.92) | SUPPORTED (no significance) |
| Q6 34/42 null, max 0.144, scattered | q6_xmodel | SUPPORTED |
| Q4 +0.36–0.42 retrieval-only | recomputed (artifact regen) | SUPPORTED |
| held-out higher + mix suspected | q7_heldout | SUPPORTED (numbers) + explanation NOT TESTED |
| nondeterminism ≤0.0021 | sensitivity_summary | SUPPORTED |
| 160 test queries, eff N≈137, 63 texts | splits/queries recomputed | SUPPORTED |
| regime spans +150–206/−225–−157/−129–−79 | regime_exploratory recomputed | SUPPORTED (exploratory-labeled) |
| interventions positive + C1/C3 split | utility tables | SUPPORTED |
| gym NoInfo sweep + flip | gym_utility | SUPPORTED (artifact-disclosed) |
| IR 0.115–0.183, shrink +0.024→+0.001 | ir_test_means + dev table | SUPPORTED |
| SIVR bookkeeping, no novelty | defined once + guards | SUPPORTED (as hygiene) |
| "No prior study measures rule-vs-LLM interaction…" | narrowed + "not aware" | SUPPORTED BUT NARROW |
| Empirical scoping sentence | empirical tables | SUPPORTED |
| k=5 sentence (C1 10/10, C0 n.s., C3 5/10) | balanced table | SUPPORTED |

## Final verdict: SUBMISSION-READY (conditional)

Conditions, all outside the manuscript: (1) register the ECIR abstract before
Sep 21 under the V3 story; (2) no further edits except copy-editing
(any content change re-opens this verdict). F1/F2 demonstrably resolved in
text; no unresolved fatal issue; no claim exceeds its evidence as audited
above; IR contribution stands without the inventory application (which is
framed as the controlled world throughout); statistics match the final audit;
the strongest E argument (weighting+controller+VOI) was addressed
experimentally (dual priors, delay-heavy check, tuned comparator, VOI
estimand) and its specific falsifications failed on data. The remaining
MAJOR items are significance judgments a fair chair can weigh — that is what
peer review is for, and the manuscript gives it both sides in full.
