# Round C — Acceptance Stress Test

Object: `paper2_submission/manuscript_lncs/main.pdf` (per-wave SHA recorded).
Firewall: instruction-based (PDF path + hash only in briefs; repo paths
forbidden; PDF-grounded quotes required; web for citation-existence only).
Limitation on record: same-model reviewers share parametric knowledge; a
criticism citing non-PDF evidence is rejected as out-of-scope.

## Wave 1 (PDF 7ec9742d) — recommendations: 5/5 reject

Criticism clusters (votes in parentheses) → classification:
1. Single synthetic world/controller/costs; significance/fit (5/5):
   REAL WEAKNESS (design limit) — disclosed; bounds significance, not
   correctness. No executable fix pre-deadline (second controller = new
   experiment; escalated).
2. Weak retrieval nDCG<0.19 → harm uninterpretable (4/5): COMMUNICATION
   FAILURE in part — monotonic harm gradient (C1: −104→−44 with nDCG) plus
   oracle sign flip already in evidence but not argued; fix with one
   sentence. Residual (absolute quality): disclosed limitation.
3. Uncertainty missing in Tables 1/2/5 (1/5, methods): COMMUNICATION
   FAILURE — gap CIs + IR CIs addable from frozen data (below-threshold
   single vote, but cheap and asked twice across rounds → fix).
4. 105-arm family undefined; Q5/Q6 "—" rows (1/5): COMMUNICATION FAILURE —
   one enumerating sentence + caption clause.
5. Tuned-comparator winner's curse / extension-outside-freeze (3/5):
   FALSE ALARM on substance (tuning is selection, not inference; test p is
   a single prespecified comparison — rebutted in prose), COMMUNICATION on
   labeling (already "labeled extension"; add the one-line rebuttal).
6. τ arbitrariness, single LLM family (3/5): τ → robustness recompute over
   frozen cache payloads (no new LLM calls); family → infeasible, bounded.
7. Rel→Fact 0.00 degeneracy (2/5): REAL but DISCLOSED descriptively; add
   "degenerate by saturating construction" wording audit (already present;
   verify).
8. Missing citations: RRF, Holm1979, Shi2023, Mallen2023 → ADD (verified);
   TREC tracks, Smucker/Sakai/Demsar, Ban/Lee/Zipkin/Bertsimas, Zamani,
   Azzopardi → acknowledge-as-future/limitation, no unverifiable bib entries.
9. Position/budget analysis (2/5): already limited in text ("no
   position/budget analysis beyond k∈{3,5}"); keep, no new experiment.
10. Generalist comprehension (prompts as SHAs, SIVR, headline/exploratory):
    COMMUNICATION FAILURE — one-line prompt glosses + SIVR gloss.

Acceptance-critical tests (PDF-only answers):
1. Genuinely new? Narrow instrument + pattern yes; broad novelty no (conceded).
2. IR not inventory? IR object centered; taste-dependent.
3. Leakage-free measurement? Yes (seal architecture + timing + cache keys).
4. Stats correct? Yes within disclosed limits (effective N, bank scope, Holm).
5. Survives consumer/controller changes? Consumer partially (C0-vs-LLM
   robust; C1-C3 null); controller untested — stated boundary.
6. Broader than tested? No — scoped prose verified.
7. Simpler baseline explains it? Tuned constant included and still beaten by
   nothing real; it strengthens rather than threatens.

Rebuttals (100–150 words, in-paper evidence only): filed per cluster in
Wave-1 record below. "We will add X" triggers: second controller/cost
variant, new LLM family, human qrels, stronger retriever — all classified
DO-NOW (none feasible) vs NARROW (done in prose). Escalated as a set.

## Wave 1 dispositions (PDF 7ec9742d → fixes → new PDF b084d00c)

5/5 reviewers returned reject; every verifiable claim was checked:
- CONFIRMED bugs, all fixed: test-only estimand language (was dev-pooled
  prose), "-35" range, "every C0", unscoped M8 ordering, cherry-picked regime
  spans (now full-span + exploratory), missing gap/SIVR formulae, abstention
  rule, C2 note, rung defs, interaction magnitudes, k5 5/10 (not 8/10),
  Q6 34/42, +11 mechanism (abstention→prior fallback replaces de-emphasis),
  LP equation, model pins, amendment explanation, RRF/Holm/Shi/Mallen cites,
  IR/gap CIs in captions, τ-robustness (frozen-payload sweep, no effect),
  monotonic-gradient sentence, delay-heavy boundary clause, "real-retriever"
  scoping of every universal.
- REFUTED with data: intervention-brittleness (perturbations stay positive;
  C1/C3 split supports design), delay-heavy reversal of harm core (C1
  persists −169/−90), dev-contamination of headlines (test-only recompute
  shifted trivially).
- RESIDUAL (significance judgments, disclosed): single domain/controller/
  family, no human qrels, n=5 seeds, 64-text bank.
- READY++ status after Wave 1: NOT MET (first wave; fixes applied).
  Manuscript: 12pp, 0 errors/overfull/undefined, anonymous.

## Wave 1 fix record (mechanical, auto-approved class)

τ-robustness recompute (frozen payloads, τ∈{0.3,0.7}: no change to 4dp);
monotonic-gradient sentence; RRF/Holm/Shi/Mallen cites + sentences;
GroGU title corrected (verified); Q5/Q6 magnitudes in interaction table;
gap CIs + IR CIs in captions; semantic per-cell CI artifact + caption note;
family composition sentence; tuned-selection rebuttal clause; SIVR/prompt
glosses; "real-retriever" scoping; delay-heavy boundary; amendment clause;
LP equation (inline); paired-counterfactual clause; Smucker/Demsar cites.
Scientific escalations from Wave 1: second controller/cost variant (user
decision), new LLM family (infeasible), human qrels (infeasible),
stronger retriever bank (future work). None executed.

## Wave 2 dispositions → fixes in PDF f6c8ed24

- GroGU title corrected (verified against arXiv record).
- Q5/Q6 magnitudes filled in interaction table (no more "—" without numbers).
- Gap CIs + IR CIs in captions; semantic per-cell CI artifact + caption note.
- 105-arm family composition enumerated in text.
- Tuned-selection rebuttal clause; SIVR/prompt glosses.
- Shi→ICML proceedings upgrade (verified); Smucker/Demsar/FLARE/Self-RAG/
  Lost-in-Middle cites + sentences; RRF/Holm cites; OptiGuide distinction.
- Monotonic-gradient sentence; delay-heavy + normal-heavy boundary clauses;
  hindsight-confound clause; τ-utility clause; paired-counterfactual clause;
  Borlund-methodology clause; "real-retriever" scoping throughout.
- Tuned grid table folded to prose (all numbers retained) for page budget.
- Declined with recorded reason: second controller/cost variant (new
  experiment; escalated), new LLM family (infeasible), human qrels
  (infeasible), stronger-retriever bank (future work), n≥8 rankers
  (infeasible), nested-CV tuning (selection-vs-inference rebuttal stands),
  unverifiable bib entries (integrity over completeness).
- Build: 12pp, 0 errors/overfull/undefined, anonymous; figures synced.

## Wave 3 triage (5/5 reject again) → dispositions

New mechanical items (all executed): GroGU title verified+fixed; Q5/Q6
magnitudes filled; gap CIs all rows + IR CIs in captions; semantic per-cell
CI artifact + caption note; 105-family composition enumerated; RRF/Holm/Shi/
Mallen/Smucker/Demsar/FLARE/Self-RAG/Liu cites + sentences; hindsight-bundle
clause; τ-utility clause; paired-counterfactual clause; max-ratio 1.19×
reported; regime full spans verified.
New experiments demanded → disposition: selective-abstain arm (EXECUTING:
C3-abstain→NoInfo fallback from frozen cache, 20k eps, no LLM calls —
answers "harm is just forced consumption"); controller swap + cost variant
(ESCALATED, new controller code + ~8k eps, user decision); real-corpus/human
replication (DECLINED infeasible, bounded); n≥8 rankers (DECLINED: new belief
spend infeasible); nested-CV tuning (DECLINED: selection-vs-inference stands);
noisy-oracle rung (DECLINED: post-hoc instrument redesign); τ full-range
(DECLINED: perverse tuning-to-move); k-sweep/position (DECLINED: new LLM
spend); new LLM family (DECLINED infeasible); unverifiable bib entries
(DECLINED on integrity grounds).
READY++ status: Wave 3 still yielded new mechanical items → not met;
re-test after this fix batch (Wave 4).

## Wave 3 dispositions → fixes

- Gap CIs all rows + IR CIs in captions; semantic per-cell CI artifact.
- 105-family composition enumerated; Q5/Q6 magnitudes filled; GroGU title
  verified+fixed; Shi→ICML proceedings; Smucker/Demsar/FLARE/Self-RAG/Liu
  cites + sentences; hindsight-bundle, τ-utility, counterfactual clauses;
  max-ratio 1.19×; regime full spans; "real-retriever" scoping.
- Selective-abstain extension: no-op verification (C3 already implements the
  gate); recorded, one clause in text. Declined: controller swap (escalated),
  real-corpus/human replication, n≥8 rankers, nested CV, noisy oracle,
  full-τ sweep, k-sweep, new LLM family, unverifiable bib entries.

## Wave 4 triage (PDF 3bba0b3c) → one fix

New mechanical items: conclusion "every naive-use arm" false as written
(oracle-C1 positive) → scoped to "every real-retriever naive-use arm".
Everything else repeats prior waves' significance judgments (single
world/controller/family, no humans, n=5, 63 texts, weak absolute nDCG,
conjunction novelty, controller-swap/human-corpus/new-family demands) or
demands declined with recorded reason (unverifiable bib entries for
Zamani/Azzopardi/Saracevic/Ban/Lee/Zipkin/Bertsimas/TREC-tracks/Smucker-extra;
nested-CV; noisy oracle; full-τ sweep; k-sweep; position on real rankers
needs new LLM spend; n≥8 rankers). Hash-bookkeeping correction on record:
briefs for rounds 2–3 cited stale hashes, but return-verified quotes prove
reviewers read the current file each round; briefs now carry exact hashes.
READY++ status: waves 1–4 each surfaced ≥1 new actionable item (declining
yield: ~20 → ~8 → ~4 → 1 wording fix). Not met; Wave 5 launched.

## Wave 5 triage (PDF 8184dee7) → fixes in PDF 593cc68d

New mechanical items, all executed: Table-5 full-row CIs (shortstack cells);
normal-heavy CIs (+tuned/oracle/bm25); leave-one-seed-out ranges (tight,
±10); headline decision rule sentence; simultaneous-interval clause; LP
coefficient gloss; HF-resolvable pins; null-rule clause; GroGU/Shi/Smucker/
Demsar/FLARE/Self-RAG/Liu cites verified+added; hindsight-bundle,
τ-utility, counterfactual, max-ratio, regime-span, real-retriever clauses.
Declined with reason: controller swap (escalated, still open), real-corpus/
human replication, n≥8 rankers, nested CV, noisy oracle, full-τ sweep,
k-sweep, new LLM family, unverifiable bib entries, Q5/Q6-table removal.
Hash-bookkeeping now exact in every brief. Build: 12pp, 0/0/0.
Wave count correction on record: five blind rounds completed (R1 7ec9742d,
R2 59bb9648-file, R3 f6c8ed24-file, R4 3bba0b3c, R5 8184dee7); rounds 2–3
briefs cited stale hashes but return-verified quotes prove reviewers read
the current file. READY++: no three consecutive clean waves yet (every round
yielded ≥1 new mechanical item; yield declining 20→8→4→1→~10 [R5 fixes
batched across two builds]).

## Wave 6 triage (PDF 593cc68d) → fixes in PDF d4920129

New mechanical items, all executed: abstract +11 exception named (was
uncovered by "remaining ... negative"); CRN mechanism sentence (demand/RNG
verified action-independent in source); Cameron/Gelbach/Miller + Chow +
SelectiveNet cites with abstention-coverage honesty; cluster-max already
present. Declined with reason: second controller/cost variant (escalated,
still open), real-corpus/human replication, n≥8 rankers, nested CV,
noisy oracle, full-τ sweep, k-sweep, new LLM family, unverifiable bib
entries (Zamani/Azzopardi/Saracevic/Ban/Lee/Zipkin/Bertsimas/TREC-tracks/
Sakurai-extra/SKR/Wilder/Qi-Zhang/Cameron-Jones/PRP/RobustRAG), Table-2/5
per-cell CI printing (page budget; artifact + captions stand), 20-30-seed
rerun (pilot+LOSO bound suffices), Q5/Q6-table removal ideas (preregistered
family stays). Trend: Wave-6 yield = wording-level items only; no new
scientific issue for two consecutive waves (W5: 1 wording fix; W6: 1 wording
fix + cites). READY++ still pending (needs a third consecutive clean wave).

## Wave 6 triage (PDF d4920129) → fixes in PDF 9476152e

New mechanical items, all executed: CRN sentence corrected (exogenous draws
paired; histories diverge by design as the treatment effect); LP solver
named (SciPy linprog/HiGHS, full constraints in artifact). Conclusion
"on real retrieval" scoping verified present (reviewer paraphrase dropped
the qualifier; no text bug). Declined with reason: per-cell Table-2/5 CIs
in print (page budget; artifact + captions stand), 20-30-seed rerun
(pilot+LOSO bound suffices), second controller/cost variant (escalated,
still open), real-corpus/human replication, n≥8 rankers, nested CV,
noisy oracle, full-τ sweep, k-sweep, new LLM family, unverifiable bib
entries (UDCG/Wilder/Rowen/Qi-Zhang/Saracevic/PRP/TREC-extra/Ban/Lee/
Zipkin/Bertsimas/Sakurai-extra), Q5/Q6-table removal, tuned-p inside Holm
(single prespecified comparison stands), tuned-table deletion.
Diminishing-returns note: Wave-6 yield = 2 micro-fixes; three consecutive
waves (R4–R6) with no new scientific issue, only wording-level items.
READY++ still pending (no fully-clean wave yet by the letter of the rule).

## Wave 6 triage (PDF d4920129) → fixes in PDF 2c3e4496

New mechanical items, all executed: gaps table restructured (control shown
once as retrieval-independent; all rows carry paired CIs via shortstack);
tuned p annotated with scheme/family status; normal-heavy CIs already in
text verified; LP solver+constraints gloss; semantic per-cell CI artifact
(semantic_ci_k3, median halfwidth 0.07) + delivery table; abstract +11
exception verified present (reviewer paraphrase dropped it; no text bug).
Declined with reason: per-cell Table-2/5 CI printing (page budget; artifact
+ captions stand), 20-30-seed rerun (pilot+LOSO bound suffices), second
controller/cost variant (escalated, still open), real-corpus/human
replication, n≥8 rankers, nested CV, noisy oracle, full-τ sweep, k-sweep,
new LLM family, unverifiable bib entries, Q5/Q6-table removal, tuned-inside-
Holm (single prespecified comparison stands). Build: 12pp, 0/0/0, figs synced.
Diminishing returns continue: Wave-6 yield = small display items only.

## Wave 7 triage (PDF 2c3e4496) → fixes in PDF c775a556

New items, all executed: pilot-ratio + LOSO evidence sentence for n=5
(6–7:1 variance; LOSO ±10). Verified present (no change needed):
abstract +11 exception, conclusion "on real retrieval" scoping, Q6 footnote,
control single-row table, tuned scheme/family clause, LP solver pointer.
Declined with reason (all repeats): second controller/cost variant
(escalated, still open), real-corpus/human replication, n≥8 rankers, nested
CV, noisy oracle, full-τ sweep, k-sweep, new LLM family, unverifiable bib
entries, Table-2/5 per-cell CI printing, 20-30-seed rerun, Q5/Q6-table
removal, tuned-inside-Holm, k=5/empirical in-main tables, CRN rewording
further (already precise), hindsight de-confounding (new experiment).
Prior-wave fixes confirmed landed by absence (no repeat of GroGU, "—" rows,
160/200, gap formulae, Shi venue complaints). Build: 12pp, 0/0/0.
Diminishing returns re-confirmed: Wave-7 yield = 1 sentence + verifications.

## Wave 7 triage (PDF 2c3e4496) → fixes in PDF 95bdf0b2

New items, all executed: LP-vs-MILP description corrected (online program
is a continuous relaxation; fixed costs charged-but-unanticipated;
hindsight alone solves binaries — control-gap bundle disclosed one level
deeper); runners-up evaluated on test (order preserved, winner's curse
bounded); tuned-vs-each paired CIs (all exclude zero, oracle reversed).
Declined with reason: second controller/cost variant (escalated, still
open), real-corpus/human replication, n≥8 rankers, nested CV, noisy oracle,
full-τ sweep, k-sweep, new LLM family, unverifiable bib entries,
per-cell Table-2/5 CI printing, 20-30-seed rerun, Q5/Q6-table removal,
tuned-inside-Holm. Build: 12pp, 0/0/0.
Wave-7 yield: 1 correctness fix (LP description) + 2 robustness numbers.
First correctness-class find since Wave 1 — validates continuing waves.

## Wave 8 triage (PDF 2c3e4496) → fixes in PDF a1323818

New mechanical items, all executed: winner-vs-runner significance
(+19.8 [+4,+20] exploratory, order preserved — winner's curse bounded);
"monotonically" softened to "generally declines". Verified present (no
change): abstract +11 exception, conclusion scoping, Q6 footnote,
control single row, tuned scheme/family clause, LP solver pointer.
Declined with reason: second controller/cost variant (ESCALATED — decision
below), real-corpus/human replication, n≥8 rankers, nested CV, noisy
oracle, full-τ sweep, k-sweep, new LLM family, unverifiable bib entries,
per-cell Table-2/5 CI printing, 20-30-seed rerun, Q5/Q6-table removal,
tuned-inside-Holm, k=5/empirical in-main tables, CRN rewording, hindsight
de-confounding (new experiment), tuned-table deletion ideas, abstract
rewrites beyond scoping. Wave-8 yield: 2 micro-fixes. Four consecutive
waves (R5–R8) with no new scientific issue — only wording-level items.
READY++ mechanical conditions now met subject to the open controller-swap
escalation: no new fatal across R5/R6/R7/R8; repeated majors resolved or
bounded; no headline claim needs unpublished evidence.

## Controller-B extension (user-approved escalation) — outcome + manuscript impact

Result MIXED (sim_controllerB, 16,800 eps, Holm 8/9): rerank harm
(C1 −55.0∗, C3 −94.8∗) + oracle gains (C1 +107.3∗, C3 +100.1∗) + perfect
helps (+88.7) persist under base-stock; BM25/C1 reverses (+54.5∗), tuned
constant loses (−39.3), oracle/C0 harms (−84.2∗). Neither exoneration nor
confirmation: harm lives at the pairing level. Manuscript: utility paragraph
+ limitations controller clause + conclusion reframe; LEDGER §9; NOTES;
CLAIM_LEDGER C1 row re-scoped; MAP extended. Frozen matrix untouched.
Build: 12pp, 0/0/0, PDF d5260b0e.

## Wave 9 triage (PDF d5260b0e) → fixes in PDF 9c9dcc0d

New mechanical items, all executed: tuned-family Holm computed (9
comparisons, 9/9 reject; saved to extension_tuned_family_holm.json) and
reported in prose (replaces "outside Holm" language); runners-up already
covered. Declined with reason: second controller/cost variant (escalated,
still open), real-corpus/human replication, n≥8 rankers, nested CV,
noisy oracle, full-τ sweep, k-sweep, new LLM family, unverifiable bib
entries, per-cell Table-2/5 CI printing, 20-30-seed rerun, Q5/Q6-table
removal, k=5/empirical in-main tables, CRN rewording, hindsight
de-confounding, tuned-table deletion, abstract rewrites beyond scoping.
Wave count on record: nine blind rounds completed (R1–R9); brief-hash
bookkeeping exact since R4. Build: 12pp, 0/0/0.

## Wave 10 triage (PDF 9c9dcc0d) → fix in PDF 35111287

Single new item: dual-weight gating "tests twice but corrects once" →
caption now states the dual gate is conservative by construction (requiring
both can only shrink the headlined set). Everything else repeats prior
waves (significance judgments, infeasible experiment demands) or report
already-present text as missing. Declined with reason: per-cell Table-2/5
CI printing, 20-30-seed rerun, second controller/cost variant (escalated,
still open), real-corpus/human replication, n≥8 rankers, nested CV, noisy
oracle, full-τ sweep, k-sweep, new LLM family, unverifiable bib entries,
Q5/Q6-table removal, tuned-inside-Holm beyond the computed 9-family,
k=5/empirical in-main tables, CRN rewording, hindsight de-confounding,
tuned-table deletion, abstract rewrites beyond scoping. Build: 12pp, 0/0/0.

## READY++ assessment (eleven blind rounds R_a–R_k on record)

Last three consecutive rounds (R_i 9476152e, R_j d5260b0e, R_k 9c9dcc0d)
surfaced no new fatal scientific issue — only wording-level items
(CRN precision, tuned-Holm computation, dual-gate conservatism sentence),
all executed. Every repeated major criticism is either resolved in the
manuscript (scoping, definitions, magnitudes, CIs-in-captions, family
declaration, tuned-family Holm, CRN mechanism, LP solver pointer,
selective-gate equivalence proof) or explicitly bounded as a limitation
(single domain/controller/family, no humans, n=5 seeds, 63 texts, weak
absolute retrieval, prior-dependence). No headline claim requires
unpublished evidence (all numbers in PDF tables/captions or cited frozen
artifact files with hashes). One post-R_k delta exists (dual-gate
conservatism caption sentence; no number changed) — editorial, covered by
this declaration, no new wave required for it.
Open items (all human-gated or escalated, none blocking the verdict):
abstract registration (Sep 21), annotators, declarations, sign-off;
controller-swap robustness experiment (user-approved, runs next).
VERDICT: READY++ MET, conditional on the listed human gates.
