# V3 main threats to validity — structured audit (docs only)

Status: submission-gate audit record. No compute performed. No frozen
config, prompt, model pin, corpus, qrel, or source file edited.
Honest-limitation style follows `AUDIT_RESPONSE_2026.md` §§5, 20–22:
adverse controls are reported as first-class results, tests state what
they actually establish, and weaker qualified claims are preferred over
buried limitations.

Sources below were read fully before drafting: `configs/v3/FINAL_FREEZE.md`,
`configs/v3/MODEL_AMENDMENT_PHI4.md`, `reports/v3/benchmark_chronology.md`,
`reports/v3/qrel_validity.md`, `results/v3main/NOTES.md`,
`results/v3main/beliefs_cache_awq_nondeterminism.json`.
Each factual repo claim cites a real path verified with `ls`/`rg` during drafting.

## Scope and method

### What this document is

A structured threats audit for the V3 main experiment, organized as
construct, internal, external, statistical, and conclusion validity.
Each threat entry has four fields: what it is, what was done, residual
risk, where disclosed.

### What this document is not

It does not recompute any metric, touch test qrels beyond citing their
sealed status, or amend any freeze. Corpus, qrels, splits, retrieval
stack, prompts, tau, consumers, controller, simulator, and metric pins
listed in `configs/v3/FINAL_FREEZE.md` and `configs/v3/MODEL_AMENDMENT_PHI4.md`
are taken as given.

## Construct validity

### Regime-faithful pool qrels

What it is: qrels are pooled from frozen retrieval systems over a shared
corpus, and relevance is regime-faithful (the document must concern the
query's own entity and current window to score above zero). A retrieved
same-entity current-window document from another query's regime pool is
unjudged and scores zero for this query. (Source: `reports/v3/phase2b_audit.md`.)

What was done: the pooled-qrels × shared-corpus mechanism was diagnosed
explicitly — 381/400 dense top-10 slots unjudged over 40 dev queries, all
unjudged being same-entity current-window operations text — and recorded
as an informational-underdetermination verdict rather than a difficulty
level. A regime-agnostic re-grade detour was implemented and then dropped
because it saturated (dev nDCG@10 0.77–0.99) and answered a different
question. The final 64-node build keeps regime-faithful pool qrels.
(Source: `reports/v3/phase2b_audit.md`, `reports/v3/benchmark_chronology.md`,
`configs/v3/RETRIEVAL_FREEZE_FINAL.md`.)

Residual risk: nDCG differences between systems partly reflect unjudged-rate
under pooling, not only relevance transfer. Pooling for the main run is the
union of top-50 across frozen systems with deterministic re-grading of any
retrieved unjudged document by the same rule (no silent zeros), which bounds
but does not eliminate the pool-coverage threat.
(Source: `reports/v3/phase2b_audit.md`.)

Where disclosed: `reports/v3/phase2b_audit.md`,
`reports/v3/benchmark_chronology.md`, and the pooling note in
`reports/v3/phase2b_audit.md` must be cited in any paper methods section.

### Grade definitions

What it is: grades are entity-recency-aboutness-only. The frozen rule is
`3 own+current+faithful; 2 own+current+hedged; 1 own+resolved; 0 else`.
(Source: `configs/v3/dataset_full.yaml`.)

What was done: the per-query mix (2/2/2 grade-3/2/1 plus wrong-entity,
contradictory, and off-topic grade-0 quotas), the shared cross-query
distractors, and the forbidden giveaway
(`no-joint-regime-plus-full-parameters`, enforced by generator assertion
in `src/corpus_v3.py`) were frozen before generation in
`configs/v3/dataset_full.yaml`. `configs/v3/tuning_scope.yaml` lists
`grade_definitions` under `explicitly_not_tuned`. No qrel, grade, or
document was edited in response to any kappa number.
(Source: `configs/v3/dataset_full.yaml`, `configs/v3/tuning_scope.yaml`,
`reports/v3/qrel_validity.md`.)

Residual risk: grade boundaries (faithful vs hedged vs resolved) require
human judgment on hedged and correction texts; the frozen text-only
independent grader guards against construction bugs, not shared
authorship blind spots. Blind human validation on test-only pairs is the
mandated gate before submission claims.
(Source: `reports/v3/qrel_validity.md`.)

Where disclosed: `configs/v3/dataset_full.yaml`,
`reports/v3/qrel_validity.md`, and
`reports/v3main_human_validation_protocol.md` (companion gate document).

## Internal validity

### Competition-ratio history (8 → 16 → 64 nodes)

What it is: the number of same-node competitors per query changed twice
during benchmark development: 8-node build (~328 competitors, dev nDCG
~0.015), 16-node build (~200 competitors, dev ~0.06), final 64-node build
(~3 queries/node, guard ≤110, measured 71; dev BM25 0.121 / dense 0.166 /
RRF120 0.183 / rerank-top30 0.206). (Source: `configs/v3/dataset_full.yaml`,
`reports/v3/benchmark_chronology.md`, `configs/v3/RETRIEVAL_FREEZE_FINAL.md`.)

What was done: each change was restricted to node-count/assignment, each
with a documented validity reason plus re-freeze. Query template, grades,
instruction, prompts, models, metric, consumers, controllers, and simulator
were never changed for scores. Corpus was regenerated with new node names
at 64 nodes; test text is new; test metrics remained sealed until the main
analysis lift. (Source: `reports/v3/benchmark_chronology.md`,
`configs/v3/dataset_full.yaml`, `configs/v3/FINAL_FREEZE.md`.)

Residual risk: development history spans three corpus scales, so dev numbers
across scales are not comparable and must never be trended as progress.
Only the 64-node build supports frozen-matrix claims; earlier builds are
superseded diagnostics retained in git history.
(Source: `reports/v3/benchmark_chronology.md`,
`configs/v3/RETRIEVAL_FREEZE_FINAL.md`.)

Where disclosed: `reports/v3/benchmark_chronology.md`,
`configs/v3/dataset_full.yaml` entities comment,
`configs/v3/RETRIEVAL_FREEZE_FINAL.md` supersession note.

### Embedding re-encode numerical noise

What it is: document embeddings re-encoded between `runs/v3` and
`runs/v3main` show GPU numerical noise only: identical ids/shape, mean
cosine ~1.0 (min 0.9977), mean absolute difference 4e-4.
(Source: `results/v3main/NOTES.md`.)

What was done: the difference was measured and recorded; main trecs under
`runs/v3main` are declared authoritative for the main experiment.
(Source: `results/v3main/NOTES.md`.)

Residual risk: negligible for ranking claims at the reported three-decimal
nDCG precision, but exact bitwise reproduction across GPU sessions is not
claimed.

Where disclosed: `results/v3main/NOTES.md`.

### Phi-4 community-quant provenance

What it is: the frontier anchor was amended from `gpt-4o-2024-11-20`
(pinned in `configs/v3/models.yaml`) to `stelterlab/phi-4-AWQ` at revision
`075b93fe5ab0d2e86004a5d68c7575ec3bb5a88b`, an AutoAWQ INT4 GEMM of
`microsoft/phi-4`, MIT licence, ungated, 14B class. Rationale: cross-lab
replication (Microsoft × Qwen) preserves preregistered Q6; Phi-4 is
documented for precise instruction adherence needed by the strict-JSON
C1/C3 schemas; Qwen3-14B-AWQ remains the stated fallback.
(Source: `configs/v3/MODEL_AMENDMENT_PHI4.md`, `configs/v3/models.yaml`.)

What was done: provenance gates identical to the AWQ arm were required:
pinned (model, revision) in kernel plus JOBS plus manifest, prompt SHA
asserts, qrel-free seal guard, temperature 0, determinism plus parse probe
before shards (staged as `kaggle_kernel/p2_phi4_probe.py` plus
`kaggle_kernel/p2_phi4_beliefs.py`), and
re-shard manifests with source-snapshot SHAs.
(Source: `configs/v3/MODEL_AMENDMENT_PHI4.md`, `results/v3main/NOTES.md`.)

Residual risk: `stelterlab/phi-4-AWQ` is a community quantization, not a
vendor-official quant like the frozen `Qwen/Qwen3-8B-AWQ` official 4-bit
AWQ variant pinned at `4da05a8edb55c6046cce958586c33b61da07bb79` in
`configs/v3/models.yaml`. Quantization-method effects cannot be separated
from family effects within this design; claims must say "Phi-4 AWQ
(community quant, revision pinned)" and never "Phi-4" unqualified.

Where disclosed: `configs/v3/MODEL_AMENDMENT_PHI4.md`,
`results/v3main/NOTES.md`, `configs/v3/models.yaml`.

### OpenRouter and harness alternatives considered and rejected

What it is: after the OpenAI API began returning `credit_balance_exhausted`
(`insufficient_quota`) at 3,424/19,600 cache fills, two alternative
frontier routes were evaluated. (Source: `configs/v3/MODEL_AMENDMENT_PHI4.md`,
`results/v3main/NOTES.md`.)

What was done with reasons: harness (`zen`) models were rejected because no
endpoint or key is exposed to the shell (only to the agent chat loop), so
19,600 frozen calls cannot be served with temperature control, pinnable
revisions, or usage records. OpenRouter flagships (Nemotron 3 Ultra 550B,
GPT-OSS-120B) were verified to exist but rejected because the free tier
(50–1,000 req/day shared) needs 20–390 days for the matrix; paid is
single-digit dollars but needs a new API account. The user chose the free
self-served Phi-4 option. GPT execution on this machine is recorded as
impossible, not deferred. (Source: `configs/v3/MODEL_AMENDMENT_PHI4.md`.)

Residual risk: the frontier anchor is weaker and architecturally narrower
than the preregistered GPT-4o anchor; frontier-generality claims are
downgraded accordingly and Q6 now reads "Phi-4 × Qwen3-8B".
(Source: `configs/v3/MODEL_AMENDMENT_PHI4.md`.)

Where disclosed: `configs/v3/MODEL_AMENDMENT_PHI4.md`.

### GPT-4o partial superseded and excluded

What it is: 3,424 GPT-4o cache files were filled of 19,600 expected LLM
calls (4,200 C1 plus 15,400 C3-doc) before warming was stopped on billing
exhaustion; warmers were killed before false-failure logging so no fail
files were written. (Source: `results/v3main/NOTES.md`,
`configs/v3/MODEL_AMENDMENT_PHI4.md`; file count of
`results/v3main/beliefs_cache/` verified as 3,424 entries with `ls`.)

What was done: the partial cache under `results/v3main/beliefs_cache/` is
preserved, not deleted, as superseded spend. It is excluded from every
analysis and freeze manifest. If OpenAI credits are restored later, the GPT
arm may be completed only as an explicitly labeled extension; it cannot
re-enter the frozen matrix. (Source: `configs/v3/MODEL_AMENDMENT_PHI4.md`.)

Residual risk: none to the frozen matrix if the exclusion is honored; the
risk is procedural (accidental inclusion of superseded files in a future
assembly). Assembly validation must assert keys against the frozen kernels
only.

Where disclosed: `configs/v3/MODEL_AMENDMENT_PHI4.md`,
`results/v3main/NOTES.md`.

### Cross-session vLLM non-determinism with first-wins policy

What it is: identical query text repeats across query IDs (64 unique texts
over 200 queries) and retrieval is deterministic on text, so identical
C1/C3 inputs share content-addressed cache keys by design (1,285 shared
keys). Across vLLM sessions at temperature 0, 24/1,285 shared keys (1.9%)
differ in raw output and 15/1,285 (1.2%) differ in parsed belief. The merge
policy is deterministic first-wins in shard-start order. All 24 divergences
are logged with kept and later raw strings and a `same_parsed_belief` flag:
9 same-belief or unused-field-only, 15 diff-belief.
(Source: `results/v3main/NOTES.md` 17:50 merge-collision entry,
`results/v3main/beliefs_cache_awq_nondeterminism.json` with `policy`
`first-wins in numeric shard-start order`, `n_diverged_files` 24; counts
9 true / 15 false verified by direct read of the JSON.)

What was done: the collision was root-caused, the audit file above was
written, and the partial cache was wiped and re-merged clean (4,468 files
from b0–b2; `results/v3main/beliefs_cache_awq/` verified as 4,468 entries
with `ls`). A sensitivity hook (`with_alternative_cache` in
`src/semantic_analysis_v3main.py`) supports re-running every analysis under
an alternative cache resolution. (Source: `results/v3main/NOTES.md`,
`src/semantic_analysis_v3main.py`.)

Residual risk: 15 parsed-belief divergences (~1.2% of shared keys) flow into
downstream semantic metrics under the first-wins resolution. Sensitivity
analysis at the semantic stage is required, not optional: report the
headline result under first-wins plus the range under the alternative
resolution before any transfer claim is submitted.
(Source: `results/v3main/NOTES.md`, `src/semantic_analysis_v3main.py`.)

Where disclosed: `results/v3main/NOTES.md`,
`results/v3main/beliefs_cache_awq_nondeterminism.json`,
`src/semantic_analysis_v3main.py` module docstring and
`with_alternative_cache`.

## External validity

### Held-out entities and templates

What it is: the frozen splits in `data/v3/splits/splits.json` hold out 16
entities (`H1`–`H16`), allocate 40 dev and 160 test queries with zero
held-out-entity queries in dev, and reserve ~20% of bank texts as test-only
(`bank index mod 5 == 0`). (Source: `data/v3/splits/splits.json`,
`configs/v3/dataset_full.yaml` `variant_holdout` and `splits` sections.)

What was done: split rules (40/160 counts, held-out count 16, stratification
16/16/8 regimes on dev, test-only bank mechanism, quarantine of `v3p-*`
prototype IDs) were frozen before generation in
`configs/v3/dataset_full.yaml`; short SHA prefixes of corpus, queries,
qrels, splits, and instruction are recorded in
`configs/v3/FINAL_FREEZE.md`. Dev constraints (no held-out-entity queries
in dev, no test-only bank variants in dev pools) are enforced by test.
(Source: `configs/v3/dataset_full.yaml`, `configs/v3/FINAL_FREEZE.md`.)

Residual risk: held-out entities are synthetic nodes from the same generator,
so the test measures within-distribution template and entity generalization,
not deployment-domain transfer. Claims must say "held-out synthetic entities
and bank variants" and never "out-of-domain generalization."

Where disclosed: `data/v3/splits/splits.json`,
`configs/v3/dataset_full.yaml`, `configs/v3/FINAL_FREEZE.md`.

### Gym transfer as secondary

What it is: downstream simulator (gym) transfer is out of scope for Phase 1
and secondary for V3. (Source: `configs/v3/gate_a.yaml`
`out_of_scope_phase1` lists `gym transfer`; `docs/BENCHMARK_DESIGN.md`
describes the simulator package and paired evaluation conventions.)

What was done: scoped explicitly as a secondary deployment-prior estimand
distinct from the primary balanced-benchmark estimand, following the
`AUDIT_RESPONSE_2026.md` §5 precedent that transfer claims require no-text
and shuffled-text controls before they are believed.
(Source: `configs/v3/gate_a.yaml`, `AUDIT_RESPONSE_2026.md` §5.)

Residual risk: any gym-transfer number without the §5 control family
(no-text operating point, shuffled-text and label-only ablations) repeats
the exact failure mode documented in `AUDIT_RESPONSE_2026.md` §5, where a
shuffled-text control matched the genuine interpreter. Gym transfer must be
labeled secondary and controlled, or omitted.

Where disclosed: `configs/v3/gate_a.yaml`, `AUDIT_RESPONSE_2026.md` §5,
`docs/BENCHMARK_DESIGN.md`.

## Statistical conclusion validity

### Unit of analysis is the query, pairing is within-query

What it is: all paired statistics in the V3 main scaffolding operate at
query level with pairing on `query_id` (inner join; queries present in both
arms). Per-query IR metrics (nDCG@10, Recall@10/20, RR) use the trec_eval
definition via `ir_measures` 0.4.3. (Source: `src/semantic_analysis_v3main.py`
module docstring, `ir_per_query`, `paired_contrast`; metric definition in
`configs/v3/metrics.md`.)

What was done: helpers (`paired_contrast`, `q3_retriever_consumer_interaction`,
Q1–Q7 scaffolding) pin slice keys and contrast within overlapping queries
only. (Source: `src/semantic_analysis_v3main.py`.)

Residual risk: 64 unique texts over 200 query IDs (see non-determinism entry
above) mean text-level clustering exists; query IDs sharing text are not
independent draws. Confidence intervals that treat query IDs as independent
are slightly narrow. The direction is known (overprecision) and must be
stated; cluster-robust or text-aggregated intervals are a listed extension,
not a silent fix.

Where disclosed: this document, `src/semantic_analysis_v3main.py`,
`results/v3main/NOTES.md` (64-texts note).

### Paired bootstrap

What it is: uncertainty for paired contrasts is the bootstrap distribution
of the mean of paired differences, resampling query indices with
replacement, deterministic given seed; defaults `n_boot=10000`, 95% level
in `paired_bootstrap_ci`. Unpaired group differences use the independent
`unpaired_diff_ci`. (Source: `src/semantic_analysis_v3main.py`
`paired_bootstrap_ci`, `unpaired_diff_ci`.)

What was done: the crossed and family-seed bootstrap conventions of the
project (`src.metrics.crossed_bootstrap_ci` with one shared seed sample per
regime, test-inversion p-values bounded below by `1/(B+1)`) are the declared
lineage for any p-value reporting. (Source: `AUDIT_RESPONSE_2026.md` §10,
`docs/METRICS.md`.)

Residual risk: bootstrap CIs inherit the query-independence caveat above and
are conditional on the first-wins cache resolution until the sensitivity
analysis is run. Report `n`, `n_boot`, seed, and CI level with every
interval; never report a p-value as exactly zero.
(Source: `src/semantic_analysis_v3main.py`, `AUDIT_RESPONSE_2026.md` §10.)

Where disclosed: `src/semantic_analysis_v3main.py`, `docs/METRICS.md`,
`AUDIT_RESPONSE_2026.md` §10.

### Holm adjustment for simultaneous comparisons

What it is: simultaneous sensor and retriever comparisons use Holm step-down
adjustment (`holm_correction` in `src/semantic_analysis_v3main.py`;
project convention in `SCIENTIFIC_LEDGER.md` and `docs/BENCHMARK_DESIGN.md`
lineage: Holm within each experiment's interpreter family).
(Source: `src/semantic_analysis_v3main.py`, `SCIENTIFIC_LEDGER.md`.)

What was done: the adjustment helper returns raw p, Holm-adjusted p, and
the reject-at-0.05 flag together so unadjusted and adjusted values cannot be
confused. (Source: `src/semantic_analysis_v3main.py`.)

Residual risk: Holm controls family-wise error at the cost of power on the
small system counts (four retrieval systems); rank-correlation helpers
(Q1/Q2) are small-n by design and must be reported with `n` and without
multiplicity-adjusted significance stars.

Where disclosed: `src/semantic_analysis_v3main.py`, `SCIENTIFIC_LEDGER.md`.

## Conclusion validity

What it is: the risk of drawing transfer or frontier-generality conclusions
the design cannot support — in the `AUDIT_RESPONSE_2026.md` §§20–22 sense:
omitting decisive adverse panels, overstating what tests establish, and
writing prose stronger than the estimates. (Source: `AUDIT_RESPONSE_2026.md`
§§20–22.)

What was done: this audit pre-registers the downgrades. Rerank test gain
over hybrid shrinks versus dev (dev ordering preserved) and is recorded as
a finding, not a tuning trigger. The V3.1 saturation and the 8/16-node
near-chance floors are kept in the record rather than rewritten.
Calibration-style overclaims are out of scope because no calibration map is
fit on test. (Source: `results/v3main/NOTES.md`, `reports/v3/benchmark_chronology.md`,
`AUDIT_RESPONSE_2026.md` §§21–22.)

Residual risk: the submission still lacks the two mandated empirical gates
(human qrel validation; non-determinism sensitivity analysis). Until both
are complete, conclusions are conditional and the paper must carry explicit
limitations sections rather than relegating them to an appendix.

Where disclosed: this document and the companion
`reports/v3main_human_validation_protocol.md`.

## Disclosure map

Each row names the threat, the primary on-record location, and the
paper section that must cite it.

Threat: regime-faithful pooling. Record: `reports/v3/phase2b_audit.md`.
Paper section: methods (qrels) plus limitations.

Threat: grade-boundary judgment. Record: `reports/v3/qrel_validity.md`.
Paper section: methods (grades) plus limitations.

Threat: competition-ratio history. Record: `reports/v3/benchmark_chronology.md`.
Paper section: methods (corpus) plus appendix chronology.

Threat: embedding noise. Record: `results/v3main/NOTES.md`.
Paper section: reproducibility note.

Threat: Phi-4 community quant. Record: `configs/v3/MODEL_AMENDMENT_PHI4.md`.
Paper section: methods (models) plus limitations.

Threat: rejected alternatives. Record: `configs/v3/MODEL_AMENDMENT_PHI4.md`.
Paper section: methods (frontier anchor) footnote.

Threat: GPT-4o partial exclusion. Record: `configs/v3/MODEL_AMENDMENT_PHI4.md`.
Paper section: methods (frontier anchor) footnote.

Threat: vLLM non-determinism plus sensitivity. Record:
`results/v3main/beliefs_cache_awq_nondeterminism.json`,
`results/v3main/NOTES.md`, `src/semantic_analysis_v3main.py`.
Paper section: methods (beliefs) plus limitations with the sensitivity range.

Threat: held-out scope. Record: `data/v3/splits/splits.json`,
`configs/v3/dataset_full.yaml`. Paper section: methods (splits) plus limitations.

Threat: gym secondary. Record: `configs/v3/gate_a.yaml`.
Paper section: secondary analysis or omitted.

Threat: query-level units, bootstrap, Holm. Record:
`src/semantic_analysis_v3main.py`, `configs/v3/metrics.md`,
`SCIENTIFIC_LEDGER.md`. Paper section: statistical methods.

Threat: human-validation gate open. Record: `reports/v3/qrel_validity.md`,
`reports/v3main_human_validation_protocol.md`. Paper section: limitations
until the gate passes.

## Doc-vs-repo discrepancies found

Discrepancy 1: `data/v3/annotation/` planned in `reports/v3/qrel_validity.md`
as the frozen home for annotator instructions, raw judgments, adjudication
log, and kappa does not exist (`ls` returns no such directory). This is the
open human-validation gate, not a contradiction: the gate has not been
executed. Tracked in the companion protocol document.

Discrepancy 2: `configs/v3/FINAL_FREEZE.md` pins `gpt-4o-2024-11-20` in the
models line, while `configs/v3/MODEL_AMENDMENT_PHI4.md` (same date,
user-approved) supersedes the frontier anchor with `stelterlab/phi-4-AWQ`.
Read the amendment as superseding the freeze on this one line; the GPT
partial is preserved and excluded per the amendment, not silently dropped.

Discrepancy 3: `reports/v3/benchmark_chronology.md` states test nDCG had been
computed by nobody; `results/v3main/NOTES.md` later reports IR test numbers
(bm25 0.1150 / dense 0.1754 / hybrid-k120 0.1817 / rerank-top30 0.1829) after
the seal lift for main analysis only. Consistent under the
`configs/v3/FINAL_FREEZE.md` seal-lift condition; cite both with dates
rather than quoting the chronology line alone.

Counts verified during drafting: `data/v3/splits/splits.json` holds 40 dev
plus 160 test queries, 16 held-out entities, and test-only rule `bank index
mod 5 == 0`; `results/v3main/beliefs_cache/` holds 3,424 files;
`results/v3main/beliefs_cache_awq/` holds 4,468 files;
`results/v3main/beliefs_cache_awq_nondeterminism.json` holds 24 diverged
files (15 parsed-belief differences, 9 same-belief or unused-field-only).

## Addendum 2026-09-17: results-stage residual risks (no new mitigations possible under freeze)

- Nondeterminism audit closed: 36 disputed keys flipped (91 + 49 rows),
  replication exact, headline effects move ≤0.0021 Brier / 0.0 accuracy.
  Residual: none material; files preserved.
- C0 duplication across model parquets was asserted identical (scalar + doc_ids)
  before dedupe; latency_s excluded as wall-clock by design.
- IR means are macro over qrel queries only (the driver initially diluted over
  run queries; caught by comparison with the frozen dev table, fixed, rerun).
- Held-out wrong sign, gym PerfectBelief<NoInfo artifact, Qwen-only frontier,
  160-query test, simulated world: all reported in LEDGER §7 / FINAL_ASSESSMENT,
  not mitigated. Human qrel validation still unexecuted (mandatory gate).
