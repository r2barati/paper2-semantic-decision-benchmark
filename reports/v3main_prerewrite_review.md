# PRE-REWRITE REVIEW REPORT — V3 main experiment, ECIR 2027 standard

Role: adversarial pre-submission review. Objective is to find scientifically
credible reasons to reject, not to defend the SUBMIT recommendation. Frozen
primary experiment unchanged (no correctness failure found). All numbers below
were recomputed from frozen artifacts for this report unless cited to a
tracked file. Manuscript untouched.

---

## 1. Executive verdict

The core empirical pattern is real and hostile-resistant: with controller,
world, seeds, and evidence budget fixed, real retrieval reduces operating
return versus no-text priors while same-corpus oracle evidence adds large
value, and the loss decomposes primarily to downstream control. But the
paper as documented today would likely draw a **borderline→reject** outcome
at ECIR, not because the findings are wrong but because (a) its broad
novelty/universality language is demonstrably anticipated by prior art and
fragile under reweighting, (b) several reported significance claims do not
survive their own multiplicity controls, and (c) three artifact/report
mismatches plus one invalid p-value would destroy credibility on sight.
All of these are repairable without new data collection. **Verdict:
BORDERLINE — submission-ready only after the rescope/fix list in §5 is
executed in the rewrite; no new experiments required except already-done
recomputations.**

---

## 2. Independent reviews

### A. Applied/contribution reviewer (★★) — weak accept

Useful: a regenerable retrieval→belief→control benchmark with a tuned
no-text baseline any IR practitioner can reuse as a pattern; understandable:
2×2 ladder + 4 gaps is legible; reproducible: pins, manifests, fetch gates,
drift-clean rerun (57/57 byte-identical), 99 tests green. Practical claims
("test retrieval against a no-text operating point before deploying to a
control loop") are directly supported. A-score rests on honesty of
limitations, which the LEDGER preserves. Would ask for the tuned comparator
— now present (J=1881.13, +102, p<0.0002).

### B. Skeptical checklist reviewer (★★★) — borderline, concrete hits

- B1. Q1 "p=0.0" Spearman claims (n=4): invalid; exact permutation p=0.0833.
  Caught, must be fixed. Credibility-damaging if shipped.
- B2. SIM_REPORT stars use unadjusted p<0.05 while the ledger claims
  "Holm-controlled": 14 starred arms fail Holm, including the only
  significant C0-harm line (rerank/C0/k3) and oracle-C0 help. Selective
  reporting appearance; must re-star by Holm.
- B3. Q4 frozen parquet pools all 14 systems (effect 0.69) while the report
  quotes retrieval-only 0.36–0.42. Artifact≠report; regenerate artifact.
- B4. Q6 "40/42 cross zero, |Δ|≤0.06" vs parquet 34/42, max 0.144. Wrong
  count; correct consistently.
- B5. Held-out *templates* preregistered (Q7) but unavailable — no template
  IDs in query metadata. Undisclosed gap unless stated.
- B6. Human validation unexecuted — disclosed, but B notes the abstract must
  not imply validated qrels.
- B7. Threshold τ=0.5, k∈{3,5}, B=5000/10000, grid step 0.1: all predeclared
  and frozen — B finds no hidden hyperparameters. Pass.
- B8. Leakage: kernels qrel-free (seal asserts, verified in source); LLMs
  never see simulator states (architecture: cache→beliefs→controller→sim);
  test seal lifted only for local analysis. Pass.
- B9. Timing: warning t=12, event t=20–30, horizon 40 — genuinely
  anticipatory; within-period ordering (lead time→arrivals→demand→fulfill→
  revenue→order→pipeline→costs) has no action→own-revenue path. Pass.

### C. Statistical/methodological expert (★★★★) — borderline-accept, rescope required

Full audit (independent recomputation) confirms: pairing real (full 1000/1000
intersections, no missing cells); shared-seed crossed bootstrap matches the
design and reproduces byte-exactly; p-floor discipline correct; large-effect
conclusions (C1 harm −36…−130, oracle-LLM +46…+207, control gap +456.8,
hit-dependence +0.4) survive Holm, reweighting, and recomputation. Required
concessions: (i) effective N ≈ 137 not 200 for sim (text-cluster ICC +0.215;
CIs ~20% too narrow); family-axis resampling degenerate (single "main":
report `family_axis_resampled=False`, scope to this bank); (ii) n=5 adequate
for balanced J (−1.3%) but not regime slices (−15% available) and coarser
than B=5000 granularity (3125 distinct seed-resamples); (iii) 6 neutral arms
flip sign balanced→empirical and oracle-C0 flips under normal-heavy — kill
all universal quantifiers, publish dual-prior tables; (iv) semantic Q3/Q5/Q6
differences have no multiplicity control and no p-values — declare headline
cells confirmatory-with-CIs or add Holm; the pattern (45/120 exclude zero vs
~6 expected) carries the claim, not any single cell.

### D. Novelty + ECIR domain expert (★★★★) — borderline, scope-or-die

Nearest-work matrix (14 rows; see §6): eRAG (divergence), GroGU (model-
specific utility), Cuconasu/Yoran/Amiraz (retrieval-harms/oracle-helps/
stronger=more-distracting), Donti/SPO/VAML/VEP/Zhao/VOI (prediction≠decision,
consumer-dependence, regret normalisation), Borlund lineage 1973–2024
(topical≠situational/useful), ARES/RAGAS (component evaluation), RC-RAG/CDA
(abstention), OptiGuide (operations+language). **Every broad claim in the
current documents is anticipated.** What survives, narrowly: the first
frozen, regenerable retrieval→interpretation→control ladder scoring evidence
by sequential realised return with a sign-flipping Retriever×Consumer
interaction under matched budgets. SIVR does not survive as novelty
(bookkeeping; keep as implementation detail). The ECIR-fit sentence works
only in the narrow form. Reviewer D accepts iff Related Work concedes the
lineage in §§1–2 and claims only the instrument + measured failure mode.

### E. Hostile scientific reviewer (★★★★★) — reject attempted, two shots land

- E1. "No-text prior winning just means retrieval is hard and your controller
  is rigid." Countered by same-controller oracle ladder (+178) and tuned
  constant (+102): rigidity cannot explain gains on better evidence. **Ruled out.**
- E2. "Prompt/model-specific LLM pathology." Countered by model-free C0
  showing the harm pattern with zero LLM involvement, plus 8B×14B parity.
  **Ruled out.**
- E3. "Payoff asymmetry rigs everything: 18:1 understock penalty makes any
  over-ordering belief win." Partially TRUE and now quantified (§10): the
  environment structurally rewards surge anticipation. But it cannot explain
  (i) retrieval arms losing IN surge worlds (bm25/C1 surge 1607 vs NoInfo
  1801 — retrieved beliefs mislead where events happen), (ii) oracle gains,
  (iii) the C0-vs-LLM interaction. Asymmetry explains the tuned constant's
  level, not the retrieval harm. **Contained, must be disclosed as the
  mechanism (§10).**
- E4. "Prior-dependent: normal-heavy worlds flip your comparator and six
  arms." TRUE as stated (recomputed §10). Kills universality, not the
  large-effect pattern (C1 harm and oracle help never flip). **Lands —
  requires rescoping, not new data.**
- E5. "160 queries, 64 texts, Qwen-only, one simulator." TRUE. Partially
  answered (paired CIs, Holm, gym corroboration, C0 model-free). Residual:
  accepted as disclosed limitation; E keeps it for the discussion, cannot
  alone carry rejection given effect sizes.
- E6. "Missing human validation → qrels untrustworthy → all IR numbers
  suspect." Strongest unanswered shot. Mitigations available, none decisive:
  C0 harm needs no qrels; interaction pattern replicates across two models;
  blind agent audit priors exist. **Lands as the discussion's likely
  focal objection; cannot be fixed pre-deadline — disclose and de-emphasize
  absolute nDCG values.**

### F. ECIR meta-reviewer (★★★★★) — discussion forecast

Likely discussion centers: (1) novelty scoping (D's lineage vs our
instrument — decided by Related Work rewrite quality); (2) whether
negative-utility-with-oracle-upside clears the significance bar for IR
(chair will weigh the 4-gap attribution + interaction as the publishable
unit, not the "retrieval harms" headline alone); (3) the missing human
validation (someone will demand it; the C0-without-qrels argument is the
defense). Three strongest accept reasons: frozen end-to-end causal
isolation with a tuned comparator (rare in RAG evaluation); sign-flipping
interaction invisible to relevance metrics (genuinely new measurement);
full regeneration + drift-clean audit trail. Three strongest reject
reasons: broad-novelty overclaim against 50 years of relevance critique +
2024 RAG evaluation; no human-validated relevance ground truth; single
simulated domain with one controller class and Qwen-only beliefs. My
prediction: **borderline → accept iff the rewrite concedes (1) upfront and
the results survive as scoped; reject iff broad language or B-list bugs
ship.**

---

## 3. Fatal flaws (would justify rejection if unresolved)

- **F1. Broad-novelty language.** "First to show relevance diverges from
  downstream value / consumer-dependent utility / new SIVR metric" is
  refuted by Borlund→eRAG→GroGU→Cuconasu/Yoran + standard normalised-regret
  practice. Fatal only if shipped; fully repairable by scoping (§6 bottom
  line). No new experiment can fix it — only the rewrite.
- **F2. False universality.** "Every real arm below NoInfo" and "oracle
  helps in all consumers" are false under declared alternative priors
  (6 arms + oracle-C0 flip). Fatal if shipped as quantified; repairable by
  dual-prior reporting + scoping to robust arms. No new data needed.
- **No unresolved fatal technical flaw found**: no correctness failure in
  simulator/timing/leakage/pairing; no fabricated or irreproducible number
  (drift rerun 57/57 identical; spot recomputations exact).

## 4. Major issues (repairable, must-fix before submission)

- M1. Re-star utility tables by Holm; demote rerank-C0-harm, oracle-C0-help,
  dense-C3-8B, random-C3 arms to suggestive. (Recomputation only.)
- M2. Text-cluster bootstrap for sim headlines (or Deff-adjusted honesty
  note); set `family_axis_resampled=False`; scope to bank. (Recomputation.)
- M3. Q1 exact permutation p-values (all n.s.); never claim Q1 significance;
  keep direction descriptively. (Trivial.)
- M4. Regenerate Q4 parquet retrieval-only; correct Q6 counts (34/42,
  max 0.144) everywhere. (Trivial.)
- M5. Multiplicity statement for semantic Q3/Q5/Q6-diff families: Holm over
  headline cells or explicit exploratory labeling. (Recomputation.)
- M6. Regime-conditional claims → exploratory (n=5 seeds); report per-regime
  seed CIs. (Reporting.)
- M7. Asymmetry mechanism (§10) disclosed as the tuned constant's engine,
  with prior-sweep bounds. (Prose + existing numbers.)
- M8. De-emphasize random/C3/14B +11 (tiny, isolated, mechanistically bare).
- M9. B-list polish: template-ID absence, qrel-validation status in abstract.

## 5. Minor issues

- `family_axis_resampled: True` flag with one family (misleading; set False).
- `sensitivity_coverage` wording; keep audit files linked from LEDGER.
- Probe-script seal-assert absence (data-free by construction; add comment).
- Q2 k=5 CIs identical pairs (0.9273 twice) — check, likely coincidence across
  tiny n; verify not a copy bug. (Verify: recompute q2 k=5 rows.)
- Latency/cost table: wall-clock provenance per stage could be tabulated
  (have logs); nice-to-have.

## 6. Novelty audit and closest-work matrix

Adopted in full from the independent literature audit (14-row matrix; rows
1–14 in the audit record): eRAG, GroGU, CUE-R (unrefereed — do not lean on
it), ARES/RAGAS, Cuconasu Power-of-Noise, RetRobust, Amiraz Distracting
Effect, Donti, SPO/SPO+, VAML/VEP/Objective-Mismatch, Zhao/Arumugam/VOI
classics, Borlund lineage + Hersh/TREC/BIRCO, OptiGuide, RC-RAG/CDA/AbstainQA.
Per-claim verdicts: Claim 1 broad INVALIDATED (eRAG/Cuconasu/Yoran), narrow
sequential instantiation survives; Claim 2 survives as instrument, fails as
concept; Claim 3 general INVALIDATED (GroGU/Zhao), sign-flipping frozen form
survives; Claim 4 INVALIDATED as novelty (keep as hygiene); Claim 5 broad
ANTICIPATED (Cuconasu/Yoran/WITQA), operational-sequential pattern survives;
Claim 6 broad INVALIDATED (Borlund 1973–2024), executable consequence survives.
Three most dangerous objections: (1) situational-relevance + eRAG/GroGU
rebrand charge; (2) Cuconasu/Yoran/Amiraz anticipation charge; (3) SPO/VAML/
VOI relabelling charge. Defenses: concede lineage in Related Work §§1–2,
claim only the frozen six-node ladder + sign-flipping interaction + offset
mechanism + signed-boundary discipline.

## 7. ECIR scope/fit audit

Genuine IR object: retrieved-evidence quality scored by a downstream
decision policy, with nDCG/recall/RR reported and relevance-annotated —
in-scope (evaluation, RAG, LLMs-for-IR). Inventory is the controlled
environment, not the contribution; the abstract draft already centers IR.
Risk: a reviewer finishing with "inventory paper using text" — mitigated by
leading with the relevance→value divergence and the interaction, and by
citing the IR lineage rather than evading it. Fit verdict: adequate with
the narrow framing, inadequate with the broad framing.

## 8. Statistical audit

Adopted from the independent recomputation audit: pairing real, no missing
cells (108 groups × 1000), shared-seed bootstrap correct and byte-exact,
p-floor discipline correct, large effects Holm-robust and weighting-robust;
required corrections: effective-N honesty (~137), degenerate family axis,
n=5 limits for regime slices, dual-prior tables, Holm-starred utility,
Q1/Q4/Q6 fixes, semantic multiplicity statement, +11 de-emphasis. One
addition from this report: sim J pools dev+test (200) while semantic
headlines test-only (160) — recomputed test-only deltas differ trivially
(−109.4→−103.5 etc.), but declare the estimand difference in Methods.

## 9. Causal-validity audit

Chain verified link by link: evidence→belief (content-addressed cache keys
recompute exactly; replication gate 0.00e+00 both models); belief→action
(planning LP on belief, no state input to LLM); action→state (order affects
only t+LT arrivals, never own-period revenue); state→utility (accounting
identities, MILP-verified hindsight gap 0.0 per audit history).
Confounding assessment: retrieval/interpretation/controller separated by
frozen interfaces + factorial rungs; scenario design (8-period anticipatory
warning, asymmetric 18:1 costs) is a feature with disclosed consequences,
not a confound; templates cluster (ICC noted); timing anticipatory and
clean. Missing cell (noted honestly): "perfect interpretation of actual
retrieved evidence" is not separately identified from PerfectBelief (which
is retrieval-independent by construction) — the interpretation gap is
defined off oracle evidence; state this boundary.
Evidence-requirements check: NoInfo ✓, tuned ✓, random/shuffled ✓ (random
arm; V2 shuffled-text cited as history, not V3 evidence — do not claim V3
shuffled results), BM25 ✓, dense ✓, rerank ✓, oracle ✓ (relevant+factual),
perfect ✓, factorial ✓ (with the noted boundary), calibration (Brier/NLL/
ECE) ✓, IR metrics ✓, raw ΔJ ✓, negative frequency ✓ (52–71% real,
4.5% oracle-C3 — newly computed for this report), regime+aggregate ✓,
paired uncertainty ✓, untouched seeds ✓ (60000s disjoint), template
holdouts partial (entities ✓, templates ✗ — disclose), human agreement ✗
(missing — disclose), frozen outputs ✓, no leakage ✓, event ordering ✓.

## 10. Tuned (0,0,1) comparator investigation

1. **Why J=1881.13?** Always-surge belief equals PerfectBelief in surge
   worlds (40% of queries): 2227.2 = 2227.2 exactly. It pays bounded costs
   elsewhere (normal 1778.6 vs 1877.9; delay 1637.6 vs 1657.9). Net
   balanced Δ +102 over NoInfo.
2. **Driver: payoff asymmetry.** Missed unit costs 18 (10 revenue + 8
   penalty) vs ~1–3 to carry/order a unit. The environment structurally
   rewards conservative over-positioning; the controller's surge response
   (high order-ups) converts the constant belief into cheap insurance.
3. **Regime table** (raw cell means): tuned surge 2227 / normal 1779 /
   delay 1638; NoInfo 1801 / 1878 / 1658; Perfect 2227 / 1895 / 1861;
   bm25/C1 1607 / 1873 / 1530 (retrieval misleads precisely in event
   worlds); rerank/C1 delay 1806 (best-in-delay) but surge 1691.
4. **Gains/losses vs NoInfo:** +426 surge, −99 normal, −20 delay.
5. **Prior sensitivity (no new sims, pure reweighting):** tuned wins under
   balanced (+102), empirical (+142), surge-heavy (+329); LOSES under
   normal-heavy (−39). bm25/C1 harm persists under all four (−37…−169);
   oracle-C3 help persists (+79…+246); neutral arms (rerank/C3, C0 arms)
   flip. Superiority is prior-dependent; harm/help cores are not.
6. **Interpretation:** (2) retrieval fails to capture available value AND
   (3) structure rewards surge anticipation — jointly. NOT (1): Perfect
   (+215) and oracle (+178) prove semantic information has large decision
   value here. Distinguishing evidence: tuned==Perfect in surge cells;
   retrieval loses in event cells; prior sweep bounds.
7. **Stronger finding + validity bound, not a weakness:** harm survives a
   tuned baseline (stronger), while normal-heavy fragility bounds
   generalizability (honest). Report both.
8. **What would distinguish further:** a cost-symmetric environment variant
   (new sims, new arms — NOT recommended pre-deadline: risks tuning optics,
   answers a question the paper does not need).

## 11. Claim-evidence matrix

| Claim (source) | Required evidence | Existing evidence | Strength | Counter-explanation | Status |
|---|---|---|---|---|---|
| Every real retriever reduces return (−9…−109, 71/105 Holm) | Dual-prior Holm-robust harm, all arms | 6 arms flip under empirical; stars partly unadjusted | Strong for large arms | Prior-weighting choice | SUPPORTED BUT NARROW (scope to robust arms) |
| Oracle adds up to +178 | Robust oracle gains | +79…+246 all priors; Holm-robust | Strong | — | SUPPORTED |
| 4 gaps (0.00; 140→37; +456.8) | Computed decomposition | Exact recomputation | Strong | rel→fact zero is descriptive | SUPPORTED |
| Rankings flip across environments | Controlled vs gym ranks | rerank best→worst; gym artifact disclosed | Moderate | Gym artifact | SUPPORTED BUT NARROW |
| Relevance-only misses control quality | Divergence + interaction | rho-patterns + sign flips; lineage conceded | Moderate | Borlund/eRAG priority | SUPPORTED BUT NARROW (sequential-control scope) |
| No-text priors deserve benchmark status | NoInfo + tuned beat retrieval ×2 worlds | +102 tuned; gym NoInfo sweep | Strong | Normal-heavy flips tuned | SUPPORTED BUT NARROW |
| C3 ≈ C1 on real retrieval | Null-ish paired diffs | Mostly crossing CIs; one Holm-untested exception | Moderate | No multiplicity control | SUPPORTED BUT NARROW |
| 14B ≈ 8B | Parity diffs | 34/42 cross (not 40/42); max 0.144 | Moderate | Count overstated | SUPPORTED BUT NARROW (corrected counts) |
| Held-out higher, mix suspected | Entity split means | Consistent direction, n=50 | Weak (descriptive) | Mix unproven | SUPPORTED (numbers) + explanation NOT TESTED |
| Nondeterminism immaterial | Flip analysis | ≤0.0021 / 0.0, exact replication | Strong | — | SUPPORTED |
| Q1 significance | Exact p < 0.05 | Exact p = 0.083, n=4 | None | — | CONTRADICTED (as significance); direction descriptive |
| SIVR as novel metric | Priority | Standard practice | None | — | NOT TESTED-as-novel → withdraw; SUPPORTED as hygiene |
| NoInfo wins incl. vs perfect in gym | Gym table | −50.7 PerfectBelief | Moderate | Gym artifact | SUPPORTED BUT NARROW (artifact disclosed) |
| Q4 +0.36–0.42 | Retrieval-only Q4 | Report numbers recomputed; parquet mismatched | Moderate | Artifact mismatch | SUPPORTED (fix artifact) |

## 12. Missing experiments ranked by decision impact

1. Dual-prior utility tables + Holm re-starring (recompute only; no sims) — decides F2. HIGHEST.
2. Text-cluster bootstrap honesty note/adjustment (recompute) — decides CIs' fate. HIGH.
3. Q1/Q4/Q6 artifact fixes (trivial recomputes) — decides B-credibility. HIGH.
4. Semantic multiplicity statement or Holm over headline cells (recompute) — MEDIUM.
5. Regime-conditional seed CIs + exploratory labels (reporting) — MEDIUM.
6. Human validation — infeasible; disclosed. LOW feasibility.
7. Second simulator / cost-symmetric variant / more LLMs / more queries — NOT warranted pre-deadline (cost, tuning optics, low marginal info).

## 13. Experiments that are unnecessary and should NOT be run

- Additional LLMs (frontier or otherwise): C0 proves model-free harm; Q6 parity predicts low marginal information; new API spend violates free-only constraint.
- More queries/templates: CIs already tight for large effects; new belief spend (~20k calls) unjustified.
- Cost-symmetric environment: answers an unasked question; risks tuning optics.
- Prompt/τ re-tuning under any guise: freeze violation.
- Any retrieval re-sweep: freeze violation.
- GPT-4o completion: excluded by amendment; re-entry would break the freeze.

## 14. Likely reviewer questions and evidence-based answers

- "Isn't this just a bad controller?" — Same controller gains +178 on oracle
  evidence and +102 tuned-constant beats retrieval; rigidity can't explain
  evidence-conditioned gains. Control gap reported first-class (+456.8).
- "Would a tuned no-text baseline win?" — Yes, partially (+102, SIVR 0.474)
  — and retrieval still loses to it. That's our extension, Table LEDGER §8.
- "Prior-dependent?" — Balanced/empirical/surge-heavy: yes harm, yes tuned
  win. Normal-heavy: tuned loses (−39), C1 harm and oracle help persist.
  Dual tables ship.
- "Human-validated relevance?" — No. C0 harm needs no qrels; interaction
  replicates across models; agent-audit priors disclosed as priors.
- "Why should ECIR publish an inventory paper?" — The object is
  retrieval-evidence quality under sequential control; inventory is the
  controlled world. Contribution: frozen ladder + sign-flipping interaction
  invisible to nDCG (rerank best nDCG, still harmful/neutral downstream).
- "SIVR novel?" — No, and we don't claim it: implementation hygiene with
  signed guards, stated as such.
- "Reproduce?" — 57/57 byte-identical rerun; pins, manifests, fetch gates.

## 15. Final recommendation: BORDERLINE

No unresolved fatal technical flaw exists, and the strongest Reviewer E
argument (lineage anticipation) is logically addressed — but only on paper
here, not yet in the manuscript, and §4–§5 fixes are unexecuted. Per the
stated rule (SUBMISSION-READY only with no unresolved fatal issue AND the
strongest E argument addressed in the submission itself), the current state
is **BORDERLINE**: one disciplined rewrite executing the rescope + fixes
(§12 items 1–5, prose only plus recomputations, zero new data collection)
promotes this to SUBMISSION-READY. The manuscript must concede the lineage,
withdraw SIVR-novelty and universal quantifiers, correct the four
artifact/count errors, and headline only Holm- and prior-robust effects.
If any of §12.1–3 is skipped, the recommendation reverts to NOT READY.

## Addendum 2026-09-17: fixes executed (§12 items 1–5)

- Dual-prior utility + Holm tables shipped
  (`utility_by_arm_{balanced,empirical}`, `holm_sim_arms_{balanced,empirical}`;
  71/105 and 75/105 reject). Main tables re-starred by Holm.
- Text-cluster bootstrap honesty check shipped (median 1.08× wider; all
  starred claims hold); bank scoping + effective-N honesty in LEDGER.
- Q1 exact p-values (all n.s.), Q4 regenerated retrieval-only, Q6 counts
  corrected to 34/42 (max 0.144) in all prose.
- Headline-cell Holm shipped (3/10 reject); regime-conditional exploratory
  table shipped with exploratory labels.
- Mixed-estimand four_gaps bug caught via figure-byte check and fixed;
  canonical balanced gaps restored exactly; empirical gaps published alongside.
- Determinism re-verified post-fix (54/54 files byte-identical across rerun;
  figures epoch-pinned byte-identical).
- Verdict remains BORDERLINE: evidence-level fixes complete; the F1 lineage
  concession and universal-quantifier withdrawal must still be executed in
  the manuscript rewrite itself. No new data collection was or is needed.
