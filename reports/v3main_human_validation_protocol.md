# V3 main blind human qrel validation protocol — mandatory submission gate

Status: MANDATORY SUBMISSION GATE. The main experiment must not be submitted
until this protocol has been executed, frozen, and passed. No compute
performed by writing this document. No frozen config, prompt, model pin,
corpus, qrel, or source file edited.
Normative predecessor: `reports/v3/qrel_validity.md` (frozen record). Where
this protocol and the predecessor differ in operational detail, the
predecessor's guarantees (test-only sampling, per-grade kappa floor, no
silent relabeling) govern and this document implements them.

On-record priors that motivate but do not replace this gate: blind 60-pair
agent audit 0.82 exact / 0.90 binary agreement in
`reports/v3/benchmark_chronology.md` and `reports/v3/phase2b_audit.md`, and
independent-grader kappa 0.874 (n=3881) in `reports/v3/qrel_validity.md`,
both cited on record in `configs/v3/FINAL_FREEZE.md`. The 60-pair sample
contained zero wrong-regime same-node documents and therefore did not test
the crux case; the 0.874 comes from a text-only grader sharing the
generator's authorship and guards against construction bugs, not shared
blind spots. (Source: `reports/v3/phase2b_audit.md`,
`reports/v3/qrel_validity.md`, `configs/v3/FINAL_FREEZE.md`.)
Neither number substitutes for blind human judgment.

## Sampling plan

### Population and eligibility

Candidate pairs are drawn from TEST ONLY: test queries in
`data/v3/splits/splits.json` crossed with documents in `data/v3/corpus.jsonl`,
labeled by `data/v3/qrels/test.tsv`. Dev queries, dev qrels, training
material, and any annotator-visible generator metadata are ineligible.
(Source: `reports/v3/qrel_validity.md`, `data/v3/splits/splits.json`,
`configs/v3/FINAL_FREEZE.md` test-seal entry.)

### Exact n

Target n = 300 query-document pairs, matching the frozen plan of 75 pairs at
each grade 0/1/2/3 in `reports/v3/qrel_validity.md`. No smaller n passes this
gate; a larger n is permitted only by pre-registered amendment before
annotation starts, never in response to interim agreement.

### Stratification

Stratify jointly, in priority order, by:

1. Frozen grade (75 × 0/1/2/3), the binding quota.
   (Source: `reports/v3/qrel_validity.md`.)
2. Regime (supplier_delay / demand_surge / normal, proportional to the
   `configs/v3/dataset_full.yaml` regime stratification of 80/80/40).
3. Entity class (held-out entities `H1`–`H16` vs standard entities, per
   `data/v3/splits/splits.json`), guaranteeing held-out coverage at every
   grade.
4. Retriever agreement (unanimous top-10 vs disputed documents across the
   four frozen systems BM25 / dense / hybrid-k120 / rerank-top30 in
   `configs/v3/RETRIEVAL_FREEZE_FINAL.md`), oversampling disputed and
   same-entity current-window documents so the crux case the 60-pair audit
   missed is tested. (Source: `reports/v3/phase2b_audit.md`.)

Sampling is without replacement, uniformly at random within each stratum
cell, with the random seed and the sampled pair list frozen before
annotators are contacted.

### Power rationale

The n = 300 target with 75 per grade is powered to test the §5 pass bar,
not to re-estimate the 0.874 prior precisely. For pairwise accuracy near
0.82 at n = 300, the 95% Wilson interval half-width is about ±0.045,
sufficient to separate the 0.6 floor from chance-adjacent agreement. For
Cohen's kappa near 0.8 with four balanced grades, the large-sample standard
error is on the order of 0.03–0.04, so a per-grade observed kappa below 0.6
cannot be attributed to sampling noise at n = 75 per grade. If the true
human–qrel kappa were 0.6, n = 300 gives a lower confidence bound cleanly
above 0.5, which is the minimum separation this gate requires. Cells that
end thin after stratification (e.g. grade-1 held-out disputed) are reported
with exact intervals and wider uncertainty rather than filled by relaxing
the grade quota.

## Annotator instructions

### Who annotates

Two independent judges, recruited outside the build team, each judging all
300 pairs alone. A third adjudicator resolves disagreements by discussion
only after independent judgments are frozen. (Source role design:
`reports/v3/qrel_validity.md`.)

### Grade rubric (frozen definitions only)

Judges receive the written rubric below, which restates the frozen
entity-recency-aboutness rule in `configs/v3/dataset_full.yaml`
(`3 own+current+faithful; 2 own+current+hedged; 1 own+resolved; 0 else`).
No other instruction, example, or system output may supplement it.

Grade 3 (fully relevant): the document concerns the operator's OWN node,
describes the CURRENT window (not resolved or closed), and carries a
specific faithful operations signal about the query's event, in any
direction (delay, surge, or specific all-clear).

Grade 2 (partial): own node plus current window, but the signal is hedged,
partial, or a specific normal-operations status or correction that is still
informative about conditions. Stance metadata (support vs refute) and harm
assessments are out of scope for grading.

Grade 1 (stale): own node, but the window is resolved or closed. A document
that would otherwise grade 2 or 3 but describes a past window grades 1.

Grade 0 (not relevant): wrong entity, or routine corporate traffic with no
operations content, or contradictory off-topic material. When in doubt
between entity mismatch and staleness, entity mismatch governs.

Judges decide on entity, recency, and aboutness from the query and document
text alone. Construction-kind labels, pool membership, retriever ranks,
frozen grades, and the other judge's labels are never shown.

### Worked procedure per pair

Read the query text. Read the document text. Answer three questions in
order (own node? current window? specific faithful signal?), then assign
exactly one grade 0–3 plus a one-line textual justification quoting the
decisive span. Abstention is not permitted; uncertainty goes in the
justification.

## Blinding

Pairs are presented in a single randomized order with blind identifiers;
no query IDs, document IDs, grades, system names, scores, ranks, pool
membership, entity-class flags, regime labels beyond what the text itself
states, generator metadata, or kind labels are visible.
(Source blinding requirement: `reports/v3/qrel_validity.md` — blind to
generator metadata, kind labels, and qrels.)

The annotation spreadsheet and adjudication log record only blind IDs. The
unblinding key mapping blind IDs to (`query_id`, `doc_id`, frozen grade) is
generated from `data/v3/qrels/test.tsv` and `data/v3/splits/splits.json`,
sealed before annotation, and opened only after both judges' independent
labels are frozen.

## Agreement metrics

### Primary metrics (all reported with 95% confidence intervals)

Human–qrel agreement per judge: Cohen's kappa overall, kappa per grade
(one-vs-rest), and pairwise exact-match accuracy versus the frozen qrels,
each with a 95% interval (analytic for kappa, Wilson for accuracy).

Inter-annotator agreement: Cohen's kappa (raw independent judgments) and
adjudicated kappa after discussion, plus raw disagreement rate. Both raw
and adjudicated values are reported; the adjudicated value never replaces
the raw value. (Source reporting convention: `reports/v3/qrel_validity.md`.)

### Slices (all reported, no selection)

All §4 metrics are repeated sliced by regime, by entity class (held-out vs
standard), and by retriever-agreement cell (unanimous vs disputed). The
grade-3 miss pattern (frozen-3 judged 0/1, and frozen-0/1 judged 3) is
tabulated as a confusion matrix over the 300 pairs.

### Priors are context, not comparators

The 0.82/0.90 agent audit and the 0.874 independent-grader kappa are
reported in the validation report's background section with their caveats
(sample lacked the crux case; shared authorship), and are never used as
pass thresholds, covariates, or adjudication tie-breakers.
(Source: `reports/v3/phase2b_audit.md`, `reports/v3/qrel_validity.md`.)

## Pass and fail gate criteria

### Pass (all required)

Human–qrel kappa ≥ 0.6 for EACH judge at EVERY grade (one-vs-rest), and
overall kappa ≥ 0.6 for each judge. (Source floor: `reports/v3/qrel_validity.md`
and `configs/v3/gate_a.yaml` validity floor 0.6 per category.)

No systematic grade-3 miss pattern: frozen grade-3 documents are not
concentrated in human grades 0/1 within any regime or entity-class slice,
and the confusion matrix shows no slice where grade-3 recall falls below
0.5. (Source pattern check: `reports/v3/qrel_validity.md`.)

No slice veto: no regime, entity-class, or retriever-agreement slice shows
kappa below 0.4 or accuracy below 0.6 on n ≥ 20; thinner cells are flagged
as inconclusive rather than passed.

### Fail (any one)

Any per-grade kappa below 0.6 for either judge; any systematic grade-3 miss
pattern; any slice veto; any unblinding or protocol deviation before both
judgments froze; or n below 300 without a pre-registered amendment.

## Failure procedure

A failure blocks submission. The team publishes the validation report with
the failure stated plainly, discusses the implication for every frozen
result that depends on qrels, and revises the corpus construction rules
plus re-freezes (new SHAs, new splits if needed) under the
`configs/v3/FINAL_FREEZE.md` violation rule — never by editing grades,
documents, or labels in place. `configs/v3/tuning_scope.yaml` lists qrels,
grade definitions, corpus contents, and split assignment as
`explicitly_not_tuned`; silent relabeling or re-tuning to raise kappa is a
protocol violation, not a fix. The failed annotation package stays in the
record alongside the revised build. (Source: `reports/v3/qrel_validity.md`
failure rule, `configs/v3/tuning_scope.yaml`, `configs/v3/FINAL_FREEZE.md`.)

## Records and freezing

Before main-experiment analysis is submitted, freeze in `data/v3/annotation/`
(the directory planned in `reports/v3/qrel_validity.md` and currently absent):
annotator instructions as given, the sealed sampling seed and pair list, both
judges' raw frozen judgments, the adjudication log, the unblinding key, the
agreement computations with intervals, and the signed pass/fail verdict.
`ls` at drafting time confirms `data/v3/annotation/` does not yet exist;
creating it with the above contents is the observable proof this gate has
been executed.
