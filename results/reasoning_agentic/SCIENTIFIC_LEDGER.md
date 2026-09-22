# Scientific ledger — EXP-P2-REASONING-AGENTIC-v1.1 (track-local)

## INVALIDATED: faulty DEV pilot pass (belief-blind controller coupling)

- **Signature:** `experiments/reasoning_agentic/run_simulation.py` passed the string
  literal `controller="optimizer"` instead of the frozen constant
  `CONTROLLER_OPTIMIZER="CausalOptimizer"`, routing every arm through the
  belief-blind base-stock heuristic. Symptom: all 8 arms, all warnings, all seeds
  produced bit-identical profits.
- **Scope:** first DEV pilot execution only. No files from that pass are preserved
  (outputs were overwritten by the rerun); its only record is this entry plus
  `results/reasoning_agentic/dev_pilot/DEV_PILOT_REPORT.md` (protocol-correction note).
- **Status:** INVALIDATED. Do not cite any number from the faulty pass.
- **Resolution:** corrected to import and pass the frozen constant (code SHA
  `905b768f…`); post-fix 1440-episode rerun is the frozen v1.1 DEV evidence;
  local smoke + GPU-smoke replay re-verified after the fix.

## FROZEN v1.1 (prospective-evaluation freeze)

- **Design:** `DESIGN-REASONING-AGENTIC.md`. Ladder A0→A1→A2→R0→A3; headline
  contrasts A1−A0 / A2−A1 / R0−A2 / A3−R0; reference arms NoInfo / RuleBased /
  PerfectSemantic.
- **Frozen config:** `results/reasoning_agentic/config_frozen.json` (final seeds
  62300–62329; DEV seeds 62100–62114; all six prompt hashes; model
  Qwen3-8B-AWQ@4da05a8e; BM25 k1=1.5/b=0.75/top-k=3; budgets 0/0/0/1/≤2;
  controller constant; simulator parameters; crossed bootstrap B=5000 seed 42).
- **Final bank:** FINAL_BANK_V2, 27 warnings, SHA256
  `27f1307d…d426979`, status prospectively authored and evaluation-untouched
  (see `FINAL_BANK_V2.spec.md`). Plaintexts sealed in local-only
  `results/reasoning_agentic/FINAL_BANK_V2.json` (never committed).
- **DEV evidence (frozen, PASS):** `dev_pilot/` (1440 episodes, gates A–C/E/G PASS,
  D+F WARN) and `smoke_gpu/` (11 gates PASS). GPU beliefs reused verbatim;
  84 DEV LLM calls, 0 failures.
- **Kernel:** `kaggle_kernel/p2_reasoning_smoke.py` SHA
  `63c238e9…` (identical since the dev2 push; vLLM 0.11.0, transformers 4.57.6,
  `enable_thinking=False`).

## Frozen reporting constraints

- The 24 TEST_SEAL warnings are track-held-out material only, never a pristine
  external test.
- A3: report observed query-2 frequency; no claim of routine iterative retrieval
  unless the sealed run shows it (DEV: 0/12).
- Weak/null R0/A3 outcomes are admissible under the predeclared warning/corpus
  entity mismatch; they are findings, not defects.
- Any post-tag change to prompts, arms, retriever, controller, simulator, seeds,
  metrics, gates, or banks requires a new versioned experiment.
