# V3 main run notes (execution record, not science)

- 2026-09-14: test seal lifted for MAIN analysis only (FINAL_FREEZE condition).
  Kernels still run qrel-free (hard abort); local analysis computes test metrics.
- IR test (trec_eval, 160 test q): bm25 0.1150 / dense 0.1754 / hybrid-k120 0.1817 /
  rerank-top30 0.1829. Dev ordering preserved (dev 0.1208/0.1664-0.1686/0.1826-0.1831/0.2062-0.2065).
  Rerank test gain over hybrid shrinks vs dev — finding, not a tuning trigger.
- Re-encode noise: runs/v3 vs runs/v3main doc embeddings identical ids/shape,
  mean cos ~1.0 (min 0.9977), mean |diff| 4e-4. GPU numerical noise only.
  Main trecs (runs/v3main) are authoritative for the main experiment.
- GPT matrix: 19600 expected LLM calls (4200 C1 + 15400 C3-doc). 4 paced warmers
  (6s delay, 429-backoff, resume via content-addressed cache) running sharded.
- AWQ: b0 COMPLETE server-side (2100 calls, 0 fail, 2065 files); fetch in progress.
  b1-b3 pending push after b0 fetch (shard.json per-shard dataset versions).
- No sim before belief freeze (freeze_beliefs_v3main gate). C0-only plumbing
  validated (oracle >> real for RuleBased; order-reverse == oracle as expected).
- 2026-09-14 ~16:36 EDT: GPT warming STOPPED — OpenAI API returns
  credit_balance_exhausted (insufficient_quota). 3424/19600 GPT cache files
  preserved; warmers killed before false-failure logging (no fail files).
  GPT matrix BLOCKED on billing (human: add credits). AWQ path unaffected.
- Runner fix: _verify_awqbeliefs floor >3000 rejected valid 50-query shards
  (b0: 2065 files, 2100 calls, 0 fail). Floor lowered to >500 + added
  n_cache_files <= n_llm_calls consistency check. Science pins untouched.
- b0 fetched+installed+merged after fix (src_snap byte-identical, seal ok).
- AWQ plumbing validated: single-query (v3-q001) assembly from merged b0 cache,
  42 rows all cache hits, keys match kernel. Partial outputs removed (final
  assembly only after all 4 shards + GPT billing resolved + freeze gate).
- Frontier amendment (user-approved): GPT-4o superseded by stelterlab/phi-4-AWQ
  @075b93fe (see configs/v3/MODEL_AMENDMENT_PHI4.md). Prompts/tau/consumers/
  corpus/retrieval unchanged; Q6 reframed Phi-4 x Qwen3-8B. GPT partial cache
  preserved, excluded from analysis.
- Phi-4 execution staged: p2_phi4_probe.py + p2_phi4_beliefs.py, JOBS
  phi4-probe/phi4-b0..b3, dataset kaggle/inputs_phi4 (qrel-free, seal-guard
  kernels). Probe push waits for a free GPU slot (awq-b1/b2 occupying 2/2).
- 2026-09-14 ~17:50 EDT: AWQ merge collision root-caused. Same query text repeats
  across 4 query IDs (64 unique texts / 200 queries) + retrieval deterministic on
  text => identical C1/C3 inputs share content-addressed cache keys by design
  (1285 shared keys). Cross-session vLLM non-determinism: 24/1285 shared keys
  (1.9%) differ in raw, 15/1285 (1.2%) differ in parsed belief. Merge policy:
  deterministic first-wins in shard-start order; all 24 logged in
  beliefs_cache_awq_nondeterminism.json (9 same-belief/unused-field-only, 15
  diff-belief). Sensitivity analysis required at semantic stage. Partial cache
   wiped + re-merged clean: 4468 files from b0-b2.
- 2026-09-14: sim-eval scaffold (src/sim_eval_v3main.py, code only, zero sim
  execution): controller-only smoke belief-row -> RegimeInterpretation ->
  CausalOptimizer ACTION on C0 bm25@k3 v3-q001/v3-q002/v3-q003 ok (3/3,
  repeat-deterministic, ~4.5s cold/~0.1s warm, in-memory, 0 files, 0 returns).
  Episode/ladder/4-gap/SIVR paths all raise FreezeGateError without a
  freeze-manifest-verified beliefs dir (tools/freeze_beliefs_v3main.py).
- 2026-09-14 ~18:45 EDT: Phi-4 probe COMPLETE but FAILED its gates
  (log /tmp/phi4-probe_kernel.log): C1 output ignores the frozen regime schema
  (keys ["operational_regime"], value "Congestion Management" — no
  normal/supplier_delay/demand_surge probs, so consume_c1 would KeyError);
  C3 output ignores the frozen per-doc schema (keys
  ["congestion","impact","operator","planning_window"] — no
  entity_match/event/fresh/stance/confidence); C3 temp-0 back-to-back
  identical=false. Prompts frozen => no prompt adaptation allowed. Phi-4 @075b93fe
  REJECTED as frontier anchor (gate failure, not an evaluation result).
  Fallback per MODEL_AMENDMENT_PHI4.md: Qwen3-14B-AWQ probe next.
- 2026-09-14 ~19:00 EDT: resume-orchestrator v2 live (log /tmp/orchestrate3.log):
  waits awq-b3 -> fetch/merge (AWQ complete) -> qwen14-probe (Qwen/Qwen3-14B-AWQ
  @31c69efc, official quant, ungated) -> STOPS for user go. phi4-b0..b3 jobs
  dead (probe gate failure). Runner repaired (PHI4_REVISION constant) + added
  qwen14-probe job/verifier; runner JOBS=15, lint clean.
- 2026-09-14 ~19:05 EDT (user: proceed per recommendation): frontier =
  Qwen3-14B-AWQ @31c69efc (official quant, ungated, same template/toggle as
  validated 8B; gpu-mem 0.85 for 14B). Q6 reframed as within-family scale check
  (8B x 14B); core paper claims (transfer + consumer dependence) unaffected.
  Staged: p2_qwen14_beliefs.py, kaggle/inputs_qwen14, JOBS qwen14-b0..b3 +
  verifier, scripts/orchestrate_qwen14.sh (waits probe -> gates -> waves ->
  merge -> 42-row validation -> STOP for freeze gate). GPT-OSS/Nemotron paid
  + OpenAI-credit options deferred (new accounts/keys, weaker pinning).
- Free-only constraint (user, 2026-09-14): nothing paid anywhere in the path.
  Current plan complies: Kaggle free GPU + local compute + Apache-2.0/MIT open
  weights only. Rejected-paid options stay rejected (OpenAI credits, OpenRouter
  paid). Retry note: awq-b3 fetch hit a transient ChunkedEncodingError at
  19:31; fetch_job now retries 4x (install only post-verify, so idempotent).
- 2026-09-14 ~20:17 EDT: qwen14-probe PASSED gates (identical=true, parses=true,
  exact C3 schema keys; rev 31c69efc) -> fetch-verified -> dataset
  paper2-v3qwen14-inputs created -> qwen14-b0/b1 pushed (v1) and running.
  AWQ arm complete (b0-b3 fetched+merged). Frontier belief matrix in flight.
- 2026-09-15 ~01:35 EDT: qwen14-b1 root-caused: dataset version/push race — b1
  attached the pre-version dataset and recomputed shard [0,50] (same 2065 keys;
  fetch gate caught it: expected cache_shard_50_100.zip missing). b0 [0,50]
  install valid, kept (first-wins). Fix: 12-min version-settle + early
  log shard check in scripts/orchestrate_qwen14b.sh; fetch manifest assert
  remains the backstop. No frozen artifact touched.
- 2026-09-15: qwen14-b1 recovery SUCCEEDED with settle fix (log shard [50,100],
  2100 calls, 0 fail, 2067 files, 77 min). The 02:16 abort was a false alarm:
  kernel-log API lagged 30+ min while the run was healthy; expect_shard must
  never abort on empty logs (fetch verifier is the real gate). Patched accordingly.
- 2026-09-15 ~16:15 EDT: BELIEF FREEZE OK. Assembled 2x12600 rows (AWQ-8B +
  Qwen14, C0/C1/C3) with ZERO cache writes (mtime snapshot proof: 0 new, 0
  modified across 2x5509 files) — pure frozen-cache assembly. GPT C0-only
  partial moved to results/v3main_superseded/ (gate globs beliefs_*.parquet).
  Gate: exact key coverage, prob ranges/sums, abstain-prior, evidence integrity
  vs frozen trecs/arms, C0 LLM-free — 0 errors both files. belief_manifest.json
  written (SHA-256). Simulation gate now OPEN.
- 2026-09-15: SEMANTIC STAGE DONE (saved BEFORE any econ): results/v3main/
  semantic/ (ir, semantic_all 21000 rows post-C0-dedupe, agg, Q1-Q7,
  sensitivity deltas + summary, SEMANTIC_REPORT.md). Driver bugs fixed en
  route (C0 model-free slicing, IR macro over qrel queries, C0 twin-dedupe
  with identity proof, doc_ids join for Q4, numpy-guard in add_hit_indicator).
  Headline: real retrieval 0.28-0.44 acc (C0 0.40-0.43, oracle-C3 0.96+);
  C1<C0 on same evidence; C3≈C1 on real retrieval; 14B≈8B; hit-miss +0.4;
  held-out higher (mix suspected); nondeterminism immaterial (max 0.0021).
- 2026-09-15: SIM pilot (tools/pilot_sim_cost.py, excluded from J): 960 eps,
  mean 0.69s/ep, Hindsight MILP 0.05s, paired-diff var query:seed = 6.3:1.
  Full run set to n_seeds=5 (60000-60004, disjoint from 2000s/3100s/5000s
  ranges): 108k episodes ≈ 3h on 7 workers; n=10 would buy only ~1.5%.
- 2026-09-16: SIM ANALYSIS DONE: results/v3main/sim/ (utility_by_arm 105,
  utility_by_query, four_gaps, SIM_REPORT.md). B=5000 crossed CIs; SIVR vs
  PerfectBelief AND matched OracleFactual (user: both). Headline: real
  retrieval harms vs NoInfo (C1 -35..-109, C0 -5..-37); oracle helps
  (+38..+178); relevance->factual gap 0.00; control gap +456.8 dominates.
- 2026-09-16/17: GYM DONE (107k eps, gym_utility.parquet, GYM_REPORT.md):
  NoInfo wins everywhere incl. vs PerfectBelief (-50.7); corroborates
  NoInfo-robustness only. Rankings flip vs controlled (rerank best->worst):
  environment dependence, kept secondary.
- 2026-09-17: DELIVERY BUILT (results/v3main/delivery/): 7 MD tables, 4 Holm
  tables (sim 71/105 reject), cost/latency ($0 API), 4 figures PNG+PDF,
  LEDGER.md, FINAL_ASSESSMENT.md (10 hypotheses, conclusion, critique,
  SUBMIT recommendation). Threats doc addendum appended. Q6 p-values added
  to module (tests still 33 green).
- Extension NoText_Tuned DONE: dev grid (66 triples, seeds 61000-61004) ->
  (0,0,1), dev_J 1885.6 -> test J 1881.13, +102.09 over NoInfo (p<0.0002),
  SIVR 0.474. Labeled extension; frozen matrix untouched. (Also fixed a
  macOS-spawn Pool hang in the throwaway evaluator: missing __main__ guard.)
- 2026-09-17: DRIFT RERUN CLEAN: all 4 analysis drivers rerun end-to-end
  (semantic, delivery, sim B=5000, gym); 57/57 output files byte-identical
  (sha256 pre/post). Tracked code unchanged since freeze (only .DS_Store /
  .gitignore diffs). Tests: 49 v3/semantic + 50 repairs green.
- 2026-09-17: figures byte-reproducible (2 consecutive builds, fixed
  SOURCE_DATE_EPOCH, sha256-identical PDFs). Rerun-the-world drift: CLEAN.
- Adversarial pre-submission review DONE (reports/v3main_prerewrite_review.md,
  15 sections): verdict BORDERLINE. No fatal technical flaw; rescope+fixes
  required (dual-prior tables, Holm re-star, cluster-bootstrap honesty, Q1/Q4/Q6
  artifact fixes, lineage concession, SIVR-novelty withdrawal). Tuned-(0,0,1)
  special audit: wins via surge-cell equality with Perfect + asymmetric costs;
  prior-dependent (loses normal-heavy); harm/help cores prior-robust.
  Independent novelty (14-row matrix) + stats audits adopted. No manuscript
  changes made. No new experiments warranted except listed recomputations.
- 2026-09-17: CAUGHT+FIXED a real bug (fig3 byte-mismatch hunt): the dual-prior
  rewrite left the 4-gap block reading loop-scoped J_noinfo (empirical) while
  J_of defaulted balanced -> mixed-estimand gaps. Fixed per-vector gaps
  (four_gaps_{balanced,empirical}; canonical=balanced restored exactly:
  -21.4/0.00/36.9/456.8). All 4 fig PDFs match the pre-contamination
  epoch-pinned baseline byte-for-byte. Delivery now carries empirical utility
  + gaps tables, Holm-starred main table, cluster CIs, headline Holm,
  regime-exploratory table.
- 2026-09-17: FRESH-REVIEW FIXES: sim headlines recomputed TEST-ONLY (160 held-out
  queries); dev-inclusive tables archived as *_devinclusive.*. B-F2 resolved.
  Delay-heavy prior check (no new sims): C1 harm persists (-169/-90),
  rerank/C3 -48, oracle +246 — E's reversal claim refuted for the harm core.
  Interventions audited: perturbations of oracle evidence stay positive
  (duplicate-top C3 +185/+215); contra-injection splits C1 (+28ns) vs C3
  (+165*): provenance-awareness working as designed, not brittleness.
- Second pass CLOSED: 6 fresh reviewers (manuscript-only) all rejected; every
  verifiable objection checked -> test-only recompute, scoping, definitions,
  magnitudes, mechanism sentences, LP/pins/amendment text. B re-check: 13
  FIXED + 3 PARTIAL (all closed above); C re-check: core PASS + 2 flags
  (fixed). Meta: no FATAL standing; verdict SUBMISSION-READY conditional on
  abstract registration + content freeze. Record: reports/v3main_secondpass_review.md.
- Red-team protocol adopted (docs/REDTEAM_PROTOCOL.md) + Paper-2 instance
  backfilled (redteam/paper2-ecir2027/: 12 files). Fresh-env rebuild PASS
  (54/54 files + 99 tests; fig false alarm closed as stale baseline;
  fonttools pinned). Manuscript: 12pp, 0 errors/overfull, 22/22 cites resolve.
- Selective-abstain extension (Round C): C3-abstain->NoInfo arms (20k eps,
  frozen cache, paired seeds) reproduce C3 EXACTLY (all cells identical) —
  vacuous as designed, because C3 abstention already maps to the prior via
  to_regime_interpretation. Recorded as a no-op verification
  (results/v3main/sim_selective/), not a new finding: the demanded
  "selective gate" experiment is already inside the evaluated C3 consumer.
- Wave-7 fixes: LP-vs-MILP description corrected (online program is a
  continuous relaxation; fixed costs charged but unanticipated; hindsight
  alone solves binaries — control gap bundles relaxation+foresight).
  Runners-up evaluated on test (1869.1/1869.1 vs 1881.1, order preserved).
  Tuned-vs-each paired CIs (all exclude zero; oracle-C3 reversed).
- Controller-B extension DONE (user-approved Round-C escalation):
  base-stock policy, 16,800 eps, same beliefs/seeds/protocol. Result MIXED
  (not exonerating, not confirming): rerank harm + oracle help persist;
  BM25/C1 flips to +54.5*, tuned loses -39.3, oracle/C0 -84.2*.
  Manuscript + LEDGER updated (pairing-level framing); frozen matrix untouched.
- Round C CLOSED at 11 blind waves: READY++ declared (no new fatal across
  final 3 consecutive waves; repeated majors resolved/bounded; no headline
  needs unpublished evidence). Current PDF 35111287 (12pp, 0/0/0).
  Open: human gates (abstract/annotators/declarations/sign-off) +
  controller-swap experiment (approved, runs next).
