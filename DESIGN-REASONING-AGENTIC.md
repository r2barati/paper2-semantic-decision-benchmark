# DESIGN: EXP-P2-REASONING-AGENTIC-v1

New isolated track. No frozen Paper-2 artifact is modified, reinterpreted, or overwritten.
All reusable components are consumed read-only (paths + pins in §2).

Scientific question: holding warning, environment, simulator, action space, seeds, and
evaluation protocol fixed, does additional reasoning or agentic information processing
improve semantic belief quality and realized downstream operational utility?

## 1. Causal ladder (five arms — incorporates Correction 1)

| Arm | Name | Inference procedure |
|-----|------|---------------------|
| A0 | Direct | warning → 1 LLM call, probabilities directly |
| A1 | Structured deliberation | warning → 1 LLM call, forced evidence/hypothesis/uncertainty fields + probabilities |
| A2 | Deliberation + verification | A1 output → exactly 1 verification call → revised probabilities, stop |
| R0 | Static RAG (control) | warning → 1 frozen retrieval (verbatim query, top-k) → 1 synthesis call |
| A3 | Agentic RAG | warning → hypotheses → retrieval 1 → inspect → ≤1 reformulated query → retrieval 2 → synthesize |

Headline contrasts (paired, same warnings/seeds):

* **A1−A0** = value of reasoning structure
* **A2−A1** = value of verification
* **R0−A2** = value of retrieval access
* **A3−R0** = value of agency (second adaptive query + loop)

Secondary: A2−A0, A3−A0, A3−A2 (reported, not headlined).
Reference arms retained: NoInfo (`src/interpreter.py:547`), RuleBased (`:578`),
PerfectSemantic (`:557`), plus constant/shuffled controls per `src/notext_controls.py`.

Terminology (Correction 5): A1/A2 are **structured deliberation / self-verification**,
not native hidden-CoT reasoning — Qwen3 runs with `enable_thinking=False`, so only
explicitly requested structured fields exist. No hidden chain-of-thought is exposed
or scored.

## 2. Frozen reuse (read-only)

* Simulator: `src/env.py:636` `InventoryEnv`, profit rule `:741-804`
  (rev 10.0 / fixed 5.0 + var 2.0 / hold 1.0 / stockout 8.0), horizon 40.
* Controller: `src/env.py:388` `CausalOptimizer.decide(state)`; warning-release
  protocol `src/experiment_phase5.py:114-145` (prior until t=12, belief after).
* Warnings: `src/confirmation_templates.py` (36, 12/12/12 per regime, disjoint from
  train-18). Prompt DEV uses a DEV subset only (see §7).
* Prior: `REGIME_PRIOR {normal:0.35, supplier_delay:0.35, demand_surge:0.30}`
  (`src/events.py:186`); regime params `:195-216`; `P5_WARNING_TIME=12`.
* Retrieval corpus: `data/v3/corpus.jsonl` — loader uses `d["text"]` ONLY.
  `metadata.kind` is never read. `queries.jsonl` is NOT used as evidence queries;
  its `metadata.true_regime/split` is never read (Correction 2 ground rule).
* Metrics/stats patterns: `src/metrics.py` (`compute_episode_metrics:40`,
  `signed_sivr:115`, `weighted_benchmark_return:227`, `crossed_bootstrap_ci:285`);
  `src/semantic_analysis_v3main.py` (Brier/NLL/acc `:358`, ECE `:429`,
  `paired_bootstrap_ci:482`, Holm `:557`, rank corr `:572`).
* Kaggle pattern: `kaggle_kernel/p2_awq_beliefs.py:94-109` (vLLM AWQ server),
  `scripts/kaggle_compute.py` JOBS, `tools/merge_awq_cache.py` first-wins merge.

## 3. Exact prompts (DEV; SHA-frozen before test — see §7)

Global decoding: `temperature=0`, `top_p=1.0`, `max_tokens=512`,
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
Output schema (all arms): `{"p_normal": f, "p_supplier_delay": f, "p_demand_surge": f}`,
floats in [0,1], sum to 1 within 1e-6. Prior is stated identically in every prompt.

**A0_DIRECT_SYS:** "You are an operational risk analyst. Given a textual supplier or
market warning, output regime probabilities directly. Respond ONLY with valid JSON:
`{"p_normal": <float 0-1>, "p_supplier_delay": <float 0-1>, "p_demand_surge": <float 0-1>}`.
Benchmark prior: normal 0.35, supplier_delay 0.35, demand_surge 0.30. The three
probabilities must sum to 1.0. No explanation, no retrieval, no revision."

**A1_DELIB_SYS:** "You are an operational risk analyst. Follow this fixed procedure:
(1) extract factual claims; (2) list which claims support each regime
(normal/supplier_delay/demand_surge); (3) list contradictory or ambiguous evidence;
(4) assign regime probabilities; (5) report. Respond ONLY with valid JSON:
`{"factual_claims": [...], "support": {"normal": [...], "supplier_delay": [...],
"demand_surge": [...]}, "contradictions": [...],
"probabilities": {"p_normal": <float>, "p_supplier_delay": <float>,
"p_demand_surge": <float>}}`. Prior: 0.35/0.35/0.30. Probabilities sum to 1.0."
(The `probabilities` sub-object is the scored belief; the rest is the stored
structured representation, saved separately.)

**A2_VERIFY_SYS (call 2):** "You are a verification analyst. Given the warning and
an initial structured belief, (1) re-inspect the warning; (2) list evidence
potentially inconsistent with the initial conclusion; (3) revise probabilities
exactly once. Respond ONLY with valid JSON: `{"inconsistent_evidence": [...],
"revised_probabilities": {"p_normal": <float>, "p_supplier_delay": <float>,
"p_demand_surge": <float>}}`. Prior: 0.35/0.35/0.30. Sum to 1.0. This is the sole
permitted revision."

**R0_SYNTH_SYS:** "You are an operational risk analyst. Given a warning and
frozen retrieved evidence (ranked excerpts), synthesize and output regime
probabilities. Respond ONLY with valid JSON:
`{"p_normal": <float>, "p_supplier_delay": <float>, "p_demand_surge": <float>}`.
Prior: 0.35/0.35/0.30. Sum to 1.0. Do not issue further queries."

**A3_HYPO_SYS (call 1):** "You are an operational risk analyst with a retrieval
tool. Given a warning, (1) state hypotheses per regime; (2) write ONE first
retrieval query (verbatim warning text is acceptable). Respond ONLY with valid JSON:
`{"hypotheses": {"normal": [...], "supplier_delay": [...], "demand_surge": [...]},
"query_1": "<string>"}`."

**A3_SYNTH_SYS (call 2, after ≤2 retrievals):** "Given the warning, your hypotheses,
and the retrieved evidence from at most two queries (optionally: one reformulated
query `query_2`), synthesize final regime probabilities. Respond ONLY with valid
JSON: `{"query_2_used": <true|false>, "query_2": "<string or empty>",
"used_evidence": ["<doc ids>"],
"probabilities": {"p_normal": <float>, "p_supplier_delay": <float>,
"p_demand_surge": <float>}}`. Prior: 0.35/0.35/0.30. Sum to 1.0. Stop."

## 4. Live frozen retriever (Correction 2 — preregistered, no switching)

Preregistered retriever for R0 AND A3 (identical code path):

* **Algorithm:** exact in-repo `src/retrieval.py:59` `BM25(k1=1.5, b=0.75)`,
  `tokenize()` `:51-52`, RSJ idf floored at 0 `:82-85`. CPU-only.
* **Corpus:** `data/v3/corpus.jsonl`, `text` field only. `metadata` ignored.
* **Queries:** R0 query = warning text verbatim (1 call). A3 `query_1` from model,
  optional `query_2` = single reformulation (≤2 calls total). No precomputed TREC
  rankings (they exist only for frozen V3 query IDs, not generated queries).
* **Depth:** top-k=3 evidence docs per final synthesis (V3 headline k).
* **Why not the full V3 pipeline:** Qwen3-Embedding-0.6B + Reranker-0.6B rerun for
  arbitrary generated queries inside the belief kernel doubles GPU memory/time and
  risks version drift; BM25-only is the auditable, deterministic, frozen-code
  choice. Recorded before any outcome is seen; switching retrievers after results
  is a protocol violation requiring a new versioned experiment.
* **Seal:** kernel hard-aborts if `qrels/test.tsv`, `qrels/dev.tsv`,
  `evidence_labels.parquet`, or `queries.jsonl:metadata` is readable in inputs;
  loader asserts only `corpus.jsonl` + warning texts present.

Known limitation (predeclared): controlled warnings (R-node/LT language) vs V3
corpus (S/H-node/W-window language) are entity-mismatched → conservative bias
against R0/A3 (hurts, never inflates). Recorded as a limitation, not patched
post hoc.

## 5. Dead-zone analysis (Correction 3)

Theory: `D(s) = {b : π(b;s) = π(b0;s)}` with `b0` = NoInfo prior belief.

* **(a) Local dead-zone diagnostic (warning time):** replay `CausalOptimizer.decide(s_12)`
  with arm belief vs NoInfo belief at the warning-release state `s_12`; equality
  (tol 1e-9, scalar orders) flags "locally dead". Cheap, reported per warning.
* **(b) Rigorous pathwise test (headline mechanism claim):** for fixed simulator
  seed ω, compare full action sequences `a_t(arm)` vs `a_t(baseline)` at EVERY
  decision point t=0..39. If equal ∀t, then by induction states, pipeline
  arrivals, demand draws (seed-fixed, belief-independent), and cumulative profit
  are identical ⇒ ΔJ = 0 exactly (up to float replay tol 1e-9). Demand draws
  depend only on ω, never on b — this is what makes the induction valid.
* Mechanism cells per (warning × seed × contrast): (1) sem↑+actionΔ+util↑;
  (2) sem↑+no actionΔ; (3) sem↑+harmful actionΔ; (4) sem↓+unchanged; (5) sem↓+harmful.
  Test: mean Δprofit ≈ 0 inside pathwise-D vs ≠ 0 outside.

## 6. Inference-once + statistics (Correction 4)

* **Inference unit = warning.** One belief per (warning × arm), cached by
  `sha256(arm || prompt_sha || warning_text)` in `beliefs.parquet`. Replayed
  across ALL paired simulator seeds. Never re-infer per seed (avoids
  pseudo-replication and saves GPU).
* **Semantic metrics** (per warning, vs true regime): Brier, log loss (clipped
  1e-9), accuracy, ECE-10bin, entropy of belief; retrieval diagnostics for R0/A3
  (n calls, doc ids). No test-label feedback into prompts.
* **Operational metrics** (per warning × seed via frozen controller+sim):
  total profit, Δ vs NoInfo, Δ vs A0, SIVR (`signed_sivr`), FP/FN economic cost
  (pattern `experiment_phase5_5.py:125`), per-regime value, seed-level win rate.
* **Uncertainty:** crossed warning × seed bootstrap (resample warnings and seeds;
  reuse `crossed_bootstrap_ci` pattern, B=5000, seed 42) for operational Δ;
  paired bootstrap for semantic Δ; Holm within contrast family; Cohen's d
  (new code, in-track only — no such helper exists in `src/`); semantic↔operational
  rank correlation to test monotonicity.

## 7. Split / leakage firewall

* DEV: confirmation-template DEV subset (e.g. 6 warnings: 2/regime, mixed
  ambiguity) + simulator DEV seeds (new disjoint range, e.g. 62100+). Prompt
  development DEV-only.
* TEST (sealed until freeze): remaining confirmation templates + disjoint seed
  range. Never: tune on test profit, inspect test outcomes to pick prompts,
  alter retrieval after test exposure, or select procedures on the sealed bank.
* Freeze gate: `config_frozen.json` + `prompt_hashes.json` (SHA256 of each prompt
  in §3 + retriever pin `BM25/k1=1.5/b=0.75/top-k=3`) written to
  `results/reasoning_agentic/` and appended to `SCIENTIFIC_LEDGER.md` BEFORE the
  full run. Kaggle dataset thereafter read-only.

## 8. Model + memory (fixed across A0–A3/R0)

`Qwen/Qwen3-8B-AWQ @4da05a8e`, vLLM `--quantization awq --max-model-len 4096
--gpu-memory-utilization 0.90`, temp 0, `enable_thinking=False`. Estimate: ~4.5 GB
(4-bit weights) + ~3–5 GB KV/activations → fits 16 GB Kaggle GPU (P100/T4);
sim/metrics/stats on CPU. Substitution ladder (recorded BEFORE inspecting results):
reduce ctx → next-capable fitting model. No paid APIs.

Fairness lock logged per call: prompt tokens, completion tokens, retrieval calls,
reasoning stages, latency; identical base weights/temp/warning/prior/schema/
controller/sim/seeds/horizon/costs across arms.

## 9. Kaggle jobs

* Smoke (this PR): `kaggle_kernel/p2_reasoning_smoke.py`, gpu=True, dataset
  `kaggle/inputs_reasoning/` (`corpus.jsonl` + `src/` snapshot + `shard.json`),
  outputs `cache_reasoning_smoke.zip + smoke_manifest.json`. Checkpoint per call.
  Orchestrator registration block is specified in `experiments/reasoning_agentic/README.md`
  (additive; `scripts/kaggle_compute.py` untouched by this change).
* Full (after PASS): `p2_reasoning_full.py`, same pattern, sharded by warning.

## 10. Gates (Correction 5 — tightened)

Smoke (tiny n ⇒ zero tolerance): **zero parse/schema failures** (any failure =
FAIL); probs sum to 1 ±1e-6; retrieval budgets exact (A0/A1/A2 = 0 calls,
R0 = 1, A3 ≤ 2); seal abort verified; controller parity (identical optimizer
params); simulator reproducibility (|Δprofit| ≤ 1e-9 same-seed replay);
checkpoint kill+resume lossless. Full-run tolerance: <1% parse failures with
predeclared fallback (prior belief, logged as `fallback=true`; aborts if ≥1%).

## 11. Stop conditions

Stop + report (new versioned experiment for any fix) if: frozen edits required;
split boundaries unclear; corpus exposes labels; controller differs; common-model
load fails; smoke parse failure > 0; seed pairing unmaintainable; retriever swap
tempting post-result.

## 12. First-action checklist status

1. Repo inspected (§2). 2. Reusable components identified (§2).
3. This design doc. 4. Exact prompts (§3). 5. Model + memory (§8).
6. Split/leakage firewall (§7 + loader asserts). 7. Smoke Kaggle job
(created in this change). 8. Smoke executed (local harness gates + GPU LLM
gate on Kaggle). 9. PASS/FAIL reported (see smoke report on run).
