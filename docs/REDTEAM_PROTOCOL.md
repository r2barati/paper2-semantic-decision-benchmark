# Mandatory pre-submission red-team protocol

Governing principle: *review your own paper as though you wanted to reject
it, then actively search for evidence that defeats each criticism.* The goal
is not to make a paper criticism-proof; it is to ensure every major claim is
novel, traceable, reproducible, appropriately qualified, and supported by the
experiment actually performed.

Run TWICE per paper: (a) at paper inception, when prior-art or design fatals
are still cheap to fix; (b) on the frozen manuscript immediately before
submission. Both rounds are recorded under `redteam/<paper>/`.

Conventions in this repo: frozen artifacts live under `results/` with SHA
manifests; analysis drivers under `tools/` regenerate every table/figure from
stored episodes (`tools/gen_*`, `tools/run_*`, `tools/build_*`); the
manuscript lives under `paper2_submission/`; model/data pins live in
`configs/`. Adapt paths per paper; preserve the phase/gate structure.

## Phases

| # | Phase | Procedure (repo-specific) | Output | Gate |
|---|---|---|---|---|
| 0 | Freeze candidate | Record manuscript version, `git rev-parse HEAD`, data/dataset manifest SHAs, table/figure hashes, venue + deadline. No silent improvements during audit. | `AUDIT_SNAPSHOT.md` | Exact object identifiable |
| 1 | One-sentence contribution | Complete "first to show/develop/establish ___ under ___"; separate methodological / empirical / dataset / application novelty. | `CONTRIBUTION.md` | Concrete claim, no marketing language |
| 2 | New-knowledge test | List the 2–3 facts the community would newly know (mechanism, boundary, interaction, failure mode, benchmark property — not "X performs better"). | in `CONTRIBUTION.md` | ≥1 consequential new fact survives |
| 3 | Closest-prior-art search | Search keywords + synonyms + adjacent fields + older terminology; backward/forward citations; recent papers from closest authors. Web search mandatory. | `PRIOR_ART_MATRIX.md` | No obvious paper establishes the contribution |
| 4 | Closest-paper confrontation | For 3–5 closest: RQ, assumptions, data/env, information access, method, baselines, metrics, unit, validation, findings, limits + "what can mine establish that this cannot?" | in `PRIOR_ART_MATRIX.md` | Difference is scientific, not newer-data |
| 5 | Self-overlap audit | Diff against own papers/preprints/benchmarks (here: v1 vs v2 vs V3 lineage). New platform run ≠ new paper. | `SELF_OVERLAP.md` | Contribution separable from prior publications |
| 6 | Claim inventory | Extract every meaningful claim (title, abstract, highlights, intro, Results, Discussion, Conclusion). Number C1…Cn. | `CLAIM_LEDGER.csv` | Every headline claim represented |
| 7 | Claim→evidence audit | Per claim: exact table/figure/test, unit, n, uncertainty, alternative interpretation + "what evidence would falsify this claim?" | `CLAIM_LEDGER.csv` | No major claim rests on narrative alone |
| 8 | Reverse traceability | For every table/figure ask whether prose says more than shown. Flag proves/causes/generalizes/robust/superior/SOTA/frontier. Rule: **no Abstract/Highlights/Conclusion sentence may exceed the strongest surviving ledger row.** | Overclaim report (in ledger `status` col) | Conclusions ≤ evidence |
| 9 | Data/protocol integrity | Reconstruct inclusion/exclusion, splits, duplicates, labels, leakage, test reuse, selection-after-outcomes. Check seal discipline for held-out data. | `DATA_AUDIT.md` | Held-out genuinely held out |
| 10 | Reproducibility audit | Fresh environment from README/code only: versions, hyperparameters, seeds, deps, hardware, eval commands. Rerun table/figure drivers; byte-diff. Forbid head-only parameters. | `REPRO_AUDIT.md` | Third party could reconstruct |
| 11 | Result provenance | Trace every headline number to machine output (no hand transcription). Prefer generated manuscript tables (`tools/gen_*`). | `RESULT_PROVENANCE.md` | No unexplained hand-entered numbers |
| 12 | Statistical audit | Unit, independence, pairing, CIs, seeds, effect sizes, multiplicity, clustering, bootstrap unit vs data-generating process. | `STATISTICAL_AUDIT.md` | Inference matches design |
| 13 | Baseline audit | Would a skeptic demand a stronger baseline? Simple + strong + ablations isolating the claim; fair tuning. | `BASELINE_MATRIX.md` | Advantage not attributable to weak comparator (tuned no-text rule) |
| 14 | Robustness audit | Predefined sensible perturbations (seeds, hypers, metrics, priors, preprocessing). Try to make the headline disappear. | `ROBUSTNESS.md` | Conclusion survives or is narrowed honestly |
| 15 | Negative/control audit | Controls that should fail: shuffled, no-info, constant, random, oracle, leakage, placebo. | `CONTROLS.md` | Sensible behavior both directions |
| 16 | Alternative explanations | Strongest competing story per result (selection/metric/dataset/confounder/simulator/capacity/leakage). | `ALTERNATIVES.md` | Excluded experimentally or acknowledged |
| 17 | Generalization boundary | In-distribution vs OOD vs sim vs real. Abstract/conclusion never exceed tested population. | `BOUNDARIES.md` | No overgeneralization |
| 18 | Venue fit | Read aims/scope + recent accepted papers. Name the exact contribution the venue should care about. | `VENUE_FIT.md` | Fit explicit |
| 19 | Presentation/references | Every citation supports its sentence; closest art + foundations present; tables self-contained; numbers agree everywhere. | `PRESENTATION_QC.md` | No inconsistencies |
| 20 | Hostile independent review | Six personalities (applied ★★, checklist ★★★, methods ★★★★, novelty ★★★★, hostile ★★★★★, chair ★★★★★) on the frozen manuscript only, no internal rationale. Ask each for the strongest defensible rejection reason. | `INDEPENDENT_REVIEWS/` | Criticisms resolved or incorporated |
| 21 | Invalidation audit | Per remaining criticism: "what evidence would overturn it?" Then look for that evidence. | `FAIRNESS_CHECK.md` | Own negative assessment stress-tested |
| 22 | Strongest rejection | 300–500 word expert rejection with prior art + experiments used against you. | `STRONGEST_REJECTION.md` | Convincing evidence-based response exists |
| 23 | Strongest acceptance | Only after 22: why it matters, what is new, establishing evidence. | `ACCEPTANCE_CASE.md` | Survives the rejection case |
| 24 | Final gate | Classify unresolved: fatal / major-fixable / minor. READY / REVISE / STOP. | `FINAL_ASSESSMENT.md` | No unresolved fatal |

## Novelty adversarial process (per closest work)

> Prior work establishes: X under assumptions A.
> Our paper establishes: Y under assumptions B.
> The scientifically meaningful difference is: Z.
> Why Z matters: ___.
> Experiment/result uniquely establishing Z: ___.

Weak last two blanks → do not submit yet. New paper needs a new scientific
claim, not another run on the same platform. Run technical-validity and
novelty/significance as INDEPENDENT gates (an idea can fail either).

## Simulator/RL module (mechanism isolation)

Audit simulator mechanics, reward design, timing, semantics, costs, horizon,
information access, oracles. Target pattern: full method beats baseline →
ablation removes advantage → targeted intervention restores it →
alternative/control does not. Distinguish simulation → algorithmic →
decision → real-world validity; never let one layer imply the next.

## Retrieval/semantic/LLM module (four layers)

Measure independently: semantic quality → retrieval quality →
interpretation/belief quality → downstream decision value. Include
no-information, random/shuffled, heuristic, strong conventional,
model-based, oracle controls. Never assume transfer downstream: nDCG gains
are evidence about retrieval, not about decisions.

## Reproducibility pre-submission test

Fresh environment, reviewer role, README only:
`README → environment → data → run → evaluate → generate_tables`.
One command regenerates derived tables/figures from saved outputs. Record:
commit + dataset/version + config hash + seeds + environment + raw-output
locations + generation script. Head-only parameters = not reproducible.

## Major-revision-vs-reject self-test

Pretend the frozen draft is someone else's: could it become publishable
preserving RQ, design, results? Yes → major revision before submission.
No (needs new RQ/experiment/dataset/claim) → internal reject, redesign.

## Absolute stop conditions — do NOT submit while any holds

1. Novelty unresolved (delta from closest paper unexplained).
2. Central claim untraceable (no specific experiment/result).
3. Result unstable (reasonable choices reverse it, unexplained).
4. Reproducibility broken (cannot regenerate from frozen artifacts).
5. Contribution/venue mismatch.

## Submission package layout (`redteam/<paper>/`)

`CONTRIBUTION` → `PRIOR_ART_MATRIX` → `CLAIM_LEDGER.csv` → `DATA_AUDIT` →
`REPRO_AUDIT` → `STATISTICAL_AUDIT` → `ROBUSTNESS` → `CONTROLS` →
`BOUNDARIES` → `STRONGEST_REJECTION` → `FAIRNESS_CHECK` → `FINAL_ASSESSMENT`
(+ `SELF_OVERLAP`, `VENUE_FIT`, `BASELINE_MATRIX`, `ALTERNATIVES`,
`RESULT_PROVENANCE`, `PRESENTATION_QC`, `INDEPENDENT_REVIEWS/`,
`AUDIT_SNAPSHOT.md`, `gates.md` for human-owned steps).

The manuscript is the public compression of this package.
