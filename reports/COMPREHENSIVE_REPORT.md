# Comprehensive status report — Paper 2 (V3) @ ECIR 2027

Date: 2026-09-18. Goal: frozen V3 information experiment → semantic transfer
analysis → sequential-decision evaluation → 12-page ECIR manuscript →
adversarial red-team to submission-ready. Nothing paid; free tier only
(Kaggle GPU + local CPU); frozen stack, no new technology.

## 1. DONE — benchmark development (closed 2026-09-14)

- Chronology V3.0 (pools, 8 nodes) → competition-ratio diagnosis → 16-node
  amendment → V3.1 regime-agnostic detour (saturated 0.98, dropped by
  measurement) → FINAL 64-node build. Test qrels unseen throughout
  (kernels hard-abort on test.tsv). Record: `reports/v3/benchmark_chronology.md`,
  `configs/v3/FINAL_FREEZE.md` (SHA manifest, all SHAs still match).
- Retrieval permanently frozen: in-repo BM25 → Qwen3-Embedding-0.6B
  instructed dense → RRF k=120 → Qwen3-Reranker-0.6B top-30 (+ Random and
  Oracle brackets). Dev ordering preserved on test
  (0.1150/0.1754/0.1817/0.1829 nDCG@10). No further sweeps.
  Record: `configs/v3/RETRIEVAL_FREEZE_FINAL.md`.
- Integrity tests: 16/16 (`test_v3full.py` + `test_v3proto.py`).

## 2. DONE — belief matrix + freeze

- Factorial: {bm25, dense, hybrid, rerank, random, oracle-relevant,
  oracle-factual, 7 interventions} × {C0, C1, C3} × {Qwen3-8B-AWQ,
  Qwen3-14B-AWQ} × {k=3, k=5} × 200 queries.
- Frontier amendment (execution impossibility, not results): GPT-4o blocked
  (`credit_balance_exhausted` after 3,424/19,600 fills; cache preserved under
  `results/v3main_superseded/`); harness/zen models rejected (no script-callable
  endpoint → fails freeze requirements); OpenRouter flagships rejected
  (free-tier rate limits need 20–390 days); Phi-4 14B probed and REJECTED
  (ignores both JSON schemas + nondeterministic); Qwen3-14B-AWQ selected
  (official quant, probe passed) — Q6 reframed as within-family scale check.
  Record: `configs/v3/MODEL_AMENDMENT_PHI4.md`.
- 4 AWQ shards + 4 Qwen14 shards via Kaggle runner (seal guards, prompt-SHA
  asserts, per-shard manifests; src_snap byte-identical). Cross-session
  vLLM nondeterminism found + handled: first-wins merge + 36-key audit log;
  sensitivity bounded (≤0.0021 Brier, 0.0 accuracy).
- Assembly: 2×12,600 rows, ZERO cache writes (mtime proof), freeze gate
  0 errors → `belief_manifest.json` (SHA-256). No simulation before freeze.

## 3. DONE — semantic transfer FIRST (Q1–Q7, query-level, saved before econ)

- 33-test module green; IR + semantic tables; paired bootstrap + Holm;
  nondeterminism-sensitivity helper with exact replication gate.
- Findings: real retrieval 0.28–0.44 acc (C0 0.40–0.43; oracle-C3 0.96+);
  C1<C0 in 17/20 cells; C3≈C1 on real retrieval; 14B≈8B (34/42 null);
  hit−miss +0.36–0.42; held-out-higher flag (mix suspected, descriptive).
  Outputs: `results/v3main/semantic/` + `SEMANTIC_REPORT.md`.

## 4. DONE — sequential decision evaluation

- 108,000 episodes (105 arms × 200 queries × 5 paired seeds 60000–60004 +
  NoInfo/Perfect/Hindsight rungs); balanced + empirical J; crossed
  bootstrap CIs (B=5,000) + cluster-CI honesty check; SIVR vs PerfectBelief
  AND matched OracleFactual (user: both); 4-gap decomposition.
- Headline: real retrieval harms vs NoInfo (C1 −35…−109 Holm-robust;
  C0/C3-rerank suggestive); oracle helps (+38…+178); rel→fact 0.00;
  control gap +456.8 dominates. Extension: dev-tuned constant (0,0,1) →
  test J 1881.13, +102 over NoInfo (SIVR 0.474), still above every real arm.
- Gym transfer (secondary, 107k eps): NoInfo wins everywhere incl. vs
  PerfectBelief (−50.7); rankings flip — kept secondary with artifact disclosed.
- Sim-selective no-op verification (C3 already implements the gate).
- Outputs: `results/v3main/sim/`, `gym/`, `sim_selective/`, reports.

## 5. DONE — delivery package (18 artifacts)

Ledger, belief manifest, retrieval/semantic/R×C/utility/decomposition/
held-out/cross-model/cost tables, Holm tables (sim 71/105 balanced,
75/105 empirical reject), stats, 4 figures PNG+PDF (byte-reproducible),
threats audit + human-validation protocol, hypothesis assessments,
conclusion, adversarial critique, SUBMIT recommendation.
Location: `results/v3main/delivery/` (+`figures/`).

## 6. DONE — red-team protocol + 11 blind waves + manuscript

- `docs/REDTEAM_PROTOCOL.md` (25 phases + sim/retrieval modules + stop
  conditions + run-twice rule); `redteam/paper2-ecir2027/` (12 files:
  MAP, CONTRIBUTION, SELF_OVERLAP, CLAIM_LEDGER.csv, PRESENTATION_QC,
  VENUE_FIT, STRONGEST_REJECTION, ACCEPTANCE_CASE, FAIRNESS_CHECK,
  REPRO_AUDIT, gates + mapped evidence).
- Fresh-env rebuild: PASS (54/54 files + 99 tests byte-identical;
  fonttools pinned after stale-baseline false alarm, honestly recorded).
- 12-page LNCS manuscript rebuilt for the V3 story (scripted tables, no
  hand numbers; 0 errors/overfull/undefined; anonymous; 12 content pages):
  `paper2_submission/manuscript_lncs/main.pdf` (sha256 `35111287…`).
- 11 blind review waves → all fixes applied (scoping, Holm stars,
  dual priors, cluster honesty, LP-vs-MILP correction, tuned-family Holm,
  Q1/Q4/Q6 fixes, 30+ verified cites) → **READY++ declared** in
  `redteam/.../ROUND_C.md` (no new fatal across final 3 consecutive waves).

## 7. LEFT — human gates (all owned by you)

| Gate | Due | Status |
|---|---|---|
| ECIR 2027 abstract registration (V3 story; draft exists) | Sep 21 | OPEN |
| Blind human qrel validation | pre-camera-ready | OPEN — no annotators available; disclosed as limitation |
| Author declarations (conflicts, concurrent submission, AI use) | at submission | OPEN |
| Final sign-off + submission | Oct 5 (full paper) | OPEN |
| Copy-editing (content freeze in force) | Oct 3–5 | OPEN |

## 8. LEFT — approved experiment not yet run

- **Controller-swap robustness** (your explicit approval): second fixed
  controller class on frozen beliefs/headline arms. Design agreed; not
  started. This is the top repeated reviewer demand and the one result
  that could force rescoping if harm vanishes. Recommend running before
  submission if GPU/CPU budget allows (~single-digit dollars equivalent
  on free tier; ~hours of Kaggle/local time).

## 9. LEFT — repo hygiene + known issues

- **Nothing since the freeze is committed.** All V3 work (src/*v3*,
  tools, tests, configs/v3, data/, runs/, manuscript rewrite, redteam/,
  docs/REDTEAM_PROTOCOL.md) is untracked; manuscript + requirements +
  .gitignore are modified-uncommitted. Recommend a commit once you approve
  (I did not commit: no instruction to do so; secrets check required first).
- Pre-existing failure (NOT mine, NOT blocking): `tests/test_phase9.py`
  collection error + 1 failure in `test_repairs_2026.py::TestEstimandAndDesign`
  — both `ModuleNotFoundError: gym_invmgmt` (package not installed in this
  env; legacy v1/v2 gym path I never touched). Scoped suites: 98/99 pass.
- `requirements.txt` gained one pin (fonttools==4.51.0, uncommitted).
- Optional (declined with recorded reason, do not reopen without new
  motive): n≥8 rankers, real-corpus/human replication, new LLM family,
  nested CV, noisy oracle, full-τ sweep, k-sweep, per-cell CI printing,
  20–30-seed rerun, paid APIs.

## 10. Bottom line

Evidence: complete and frozen. Manuscript: submission-ready per internal
READY++ (conditional on human gates). Strongest remaining risk is the one
you already approved addressing: whether harm survives a second controller.
If the swap preserves it, submit as is. If harm vanishes, the headline
narrows to controller-specific (honest outcome either way — the protocol
requires reporting whichever the experiment produces).
