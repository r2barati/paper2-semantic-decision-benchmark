# EXP-P2-REASONING-AGENTIC-v1 — runner notes

Canonical design: `/DESIGN-REASONING-AGENTIC.md` (repo root).

## Layout

* `config.yaml` — frozen config (model, retriever pin, budgets, seeds, tolerances)
* `prompts.py` — exact frozen prompt texts + `prompt_hashes()`
* `arms.py` — belief schema (sums to 1 ±1e-6), budgets `A0/A1/A2=0, R0=1, A3≤2`
* `retriever.py` — live frozen BM25 (`src/retrieval.py:BM25`, k1=1.5, b=0.75),
  text-only, seal aborts on `test.tsv/dev.tsv/evidence_labels.parquet/queries.jsonl`
* `agent.py` — `run_R0` / `run_A3` orchestration (≤2 retrieval calls, then stop)
* `run_inference.py` — ONE belief per (warning × arm), cached, replayed across seeds
* `run_simulation.py` — frozen `CausalOptimizer` + `InventoryEnv` replay; full action
  sequences for the pathwise dead-zone test
* `evaluate.py` — Brier / log loss / accuracy / ECE / entropy
* `stats.py` — headline contrasts `A1−A0, A2−A1, R0−A2, A3−R0`, crossed bootstrap

## Kaggle registration (additive — do NOT edit existing JOBS entries)

Append to `scripts/kaggle_compute.py` JOBS (after smoke PASS, before full run):

```python
"reasoning-smoke": {
    "slug": "paper2-reasoning-smoke",
    "title": "Paper2 reasoning smoke",
    "script": "kaggle_kernel/p2_reasoning_smoke.py",
    "gpu": True,
    "internet": False,
    "dataset_slug": "paper2-reasoning-inputs",
    "dataset_title": "paper2-reasoning-inputs",
    "dataset_dir": "kaggle/inputs_reasoning",
    "outputs": ["cache_reasoning_smoke.zip", "smoke_manifest.json"],
    "dest_dir": "runs/reasoning_agentic",
    "verify": "reasoningsmoke",
},
```

Dataset `kaggle/inputs_reasoning/`: `corpus.jsonl` (copy of `data/v3/corpus.jsonl`),
`src/` snapshot (read-only), `shard.json` (`{"warnings": [...6 DEV ids...]}`).
NO qrels / evidence_labels / queries.jsonl in the dataset — the kernel seal-aborts
if any appear.

## Local smoke (CPU, no GPU/LLM)

`python3 scripts/smoke_reasoning_local.py` — deterministic stub LLM (valid JSON),
verifies: schema/sum-to-1, budgets, seal, controller parity, simulator
reproducibility (|Δ|≤1e-9), checkpoint reload. Requires ZERO parse failures.
