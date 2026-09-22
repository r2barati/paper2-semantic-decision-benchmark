# Paper 2 architecture

## Research question

When does retrieved text improve sequential decisions? The pursued answer is
negative and specific: not reliably — whether it does is a property of the
pipeline (selector × consumer × controller × environment), not of ranking
alone. Relevance-based evaluation cannot distinguish the cases.

## Pipeline (all stages after retrieval fixed)

```text
query + report feed → retrieval → evidence → interpreter → belief
  → controller → action → simulator → realised return
```

- Warning/text: `src/events.py` (11 + 18 templates),
  `src/confirmation_templates.py` (36 held-out), `src/capacity_drop_templates.py`
  (16 supply), `data/v3/corpus.jsonl` (200Q/4000D 64-node), per-episode pools
  (`src/warning_corpus.py`).
- Interpretation/retrieval/reasoning: `src/interpreter.py` (NoInfo/Rule/TFIDF/LLM/
  Perfect), `src/retrieval.py:rank` (none/random/bm25/tfidf/dense/oracle),
  V3 consumers `src/consumers_v3.py` (C0 rule / C1 naive-RAG / C3
  provenance-aware tau 0.5), reasoning arms `experiments/reasoning_agentic/`
  (A0/A1/A2/R0/A3, design-only, smoke FAILED).
- Belief: regime probability vector, sum-to-1 ±1e-6; constant/shuffled/argmax
  controls (`src/notext_controls.py`).
- Controller (fixed): `src/env.py:CausalOptimizer`, `src/gym_adapter.py:
  BeliefAdaptiveController`, `src/gym_adapter9.py:SupplyBeliefAdaptiveController`.
  No co-optimisation (declared limitation).
- Simulator: `src/env.py:InventoryEnv` (single-SKU lost-sales, H40) vs
  `gym-invmgmt==0.2.1 Serial-v0` (L[0,4,4], mu10×2, T30) vs capacity-drop
  divergent network (mu50, C90→30).
- Outcome: profit/reward + fill-rate (`env.S/D`, not `R/D`), Brier/log-loss/ECE,
  OIV/SIVR signed, crossed bootstrap B=5000 + Holm, win-rates.

## Frozen vs extension boundary

- **Preregistered/frozen V3 matrix:** `configs/v3/FINAL_FREEZE.md` (2026-09-14,
  corpus/query/qrel/split SHAs, prompts `C1 66e6890b`/`C3 5966cadd`, retrieval
  `BM25→Qwen-Emb→RRF120→Rerank-top30`, tuning SPENT) → `runs/v3main/*_full.trec`
  → `results/v3main/belief_manifest.json` (2×12600, 0 errors, freeze gate) →
  `results/v3main/sim/` (108k) + `gym/` (107k) → `delivery/LEDGER.md §§1–7`,
  Q1–Q7 + P1–P3 + 105-arm Holm. Test seal: `splits.json` dev40/test160.
- **Extensions (never pool into V3 headlines):** `delivery/LEDGER.md §§8–24` —
  D0, G2A-0/1/FULL, VHAT/VHAT2, Controller-B, selective, actions/selection, map,
  ladder, TrackA, Agentick. Each has its own DESIGN + seeds + Holm family and
  explicit `frozen matrix untouched` label. See `docs/experiment_registry.md`.

## Vocabulary

- `DEV` (tuning/selection, e.g. dev40, seeds 2500s/3200s/4200s/61000s),
  `TEST` (single-touch evaluation, e.g. test160, seeds 2000s/3100s/4000s/60000s).
- `confirmatory` (frozen-before-observed, Holm-controlled) vs `exploratory`
  (descriptive, null-retained, or post-hoc) vs `design-only`.
- `OIV` = oracle-reference minus NoInfo, same controller. `SIVR` = signed
  secondary normalization; undefined near-zero OIV, diagnostic-only if
  negative; values >1 mean imperfect belief beats reference under a
  misspecified controller (`docs/METRICS.md`, `src/metrics.py:signed_sivr`).

## Module map (logical; `src/` stays flat until post-ECIR)

- core: `env.py`, `events.py`, `metrics.py`, `gym_adapter.py`, `gym_adapter9.py`
- semantics: `interpreter.py`, `classical_baseline.py`, `confirmation_templates.py`,
  `capacity_drop_templates.py`, `notext_controls.py`, `offline_artifacts.py`
- retrieval: `retrieval.py`, `warning_corpus.py`, `corpus_v3*.py`, `qrels_v31*.py`,
  `consumers_v3.py`, `arms_v3.py`
- simulation: `experiment_phase*.py`, `experiment_retrieval.py`,
  `experiment_v3proto.py`, `beliefs_v3main.py`, `sim_eval_v3main.py`
- evaluation: `retrieval_analysis.py`, `semantic_analysis_v3main.py`,
  `cross_event_synthesis.py`, `results/correction_audit/recompute_corrections.py`,
  `results/publication/generate_*.py`, `tools/verify_release.py`,
  `tools/generate_accounting.py`, `tools/rebuild_offline_artifacts.py`
