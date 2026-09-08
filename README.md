# Consequence-Aware Evaluation of Evidence Selection

An executable benchmark that scores information access by the sequential
decisions it produces, not by upstream relevance alone.

## Research question

When does retrieved text improve sequential decisions?

The benchmark makes the whole path executable and holds every stage after
retrieval fixed, so differences in realised utility are attributable to which
evidence was selected:

```
query + report feed -> retrieval -> evidence -> interpreter -> belief
                    -> controller -> action -> simulator -> realised return
```

## Why controls are the point

Positive effects against an uninformed prior are easy to obtain and do not
establish that reading text helps. This release ships the controls that test
that claim honestly:

| Control | What it rules out |
|---|---|
| Constant-belief endpoints | the effect is just a better controller operating point |
| No-text operating point tuned on **development** seeds | the same, with tuning allowed |
| Shuffled text | the effect comes from the belief distribution's shape, not the content |
| Label-only (argmax) | the effect is attributable to probability quality |
| No-retrieval | retrieving anything at all beats retrieving nothing |
| Oracle evidence selector | how much headroom evidence selection actually has |
| Fold-ensemble (uncalibrated) | "calibration" is not confounded with ensembling |

## Key Metrics

- **OIV (Oracle Information Value):** Reward difference between Oracle-Belief Reference and NoInfo under the same fixed controller
- **SIVR (Semantic Information Value Recovery):** Signed secondary normalization of raw reward delta; undefined near zero OIV and diagnostic-only when OIV is negative
- **Semantic Regret:** Operational value lost due to imperfect interpretation

## Experiments

| Name | What it varies | Environment |
|---|---|---|
| **R** (retrieval) | the retrieval system, under matched evidence budgets | controlled inventory, 3 regimes |
| **C** (controlled) | the interpreter, with warning released at period 12 | controlled inventory, 3 regimes, 40 periods |
| **M** (multi-echelon) | the interpreter, on held-out wording | multi-echelon demand surge |
| **S** (supply-side) | the interpreter, on a different event mechanism | multi-echelon capacity drop |
| **B** (boundary) | one environment factor at a time | multi-echelon, 6 variants |

Exact seed, template, source and episode counts are in `docs/ACCOUNTING.md`,
generated from the saved episode files rather than written by hand.

## Interpreters

| Sensor | Type | Requires API |
|--------|------|:---:|
| NoInfo | Prior probabilities only | No |
| RuleBased | Keyword matching | No |
| TFIDF_Raw | TF-IDF + LogReg | No |
| TFIDF_Calibrated | TF-IDF + LogReg + Isotonic | No |
| gpt-4o | OpenAI GPT-4o | Yes* |
| OracleSemantic | Ground truth | No |
| HindsightOracle | Optimal (Experiment A only) | No |

*LLM predictions are cached; reproduction is possible without API access.

## Key findings

1. **Retrieval quality and decision utility can rank systems differently.** With
   the calibrated interpreter every practical retrieval system is worse than
   retrieving nothing, while the oracle evidence selector is clearly better;
   Kendall's tau between the nDCG and reward orderings is +0.33 at k=3.
2. **The gap has a mechanism.** Non-relevant documents are not interchangeable:
   a current report about a different site moves the controller the wrong way,
   an off-topic memo only wastes a slot.
3. **Whether the dissociation appears depends on the consumer.** Under the
   rule-based interpreter the two orderings agree exactly.
4. **A text-free control is a demanding baseline.** A constant belief tuned only
   on development seeds beats most interpreters in every transfer environment,
   and in one of them the shuffled-text control matches the genuine interpreter.
5. **Calibration is not an isolated intervention**, and probability quality is
   not always the operative variable: the label-only ablation of the raw
   classifier outperforms both calibrated arms.

## Quick Start

```bash
# 1. Install. requirements.txt includes the vendored simulator's own
#    dependencies (networkx, PyYAML), which it needs but does not resolve.
pip install -r requirements.lock

# 2. Reconstruct the offline artifacts that are not tracked: cached semantic
#    responses (from a tracked manifest, checksum-verified) and classifier
#    checkpoints (retrained deterministically). No API key, no network.
python3 -m tools.rebuild_offline_artifacts
python3 -m tools.rebuild_offline_artifacts --verify

# 3. Tests (offline)
python3 -m pytest tests/ -q

# 4. Verify the release end to end: artifacts, stored-belief simulation replay,
#    and regeneration of every table in the paper.
python3 -m tools.verify_release --all

# 5. Replay published results, or run an experiment
python3 -m benchmark.run --experiment gym --sensor tfidf_raw
python3 -m benchmark.run --experiment retrieval --sensor bm25
python3 -m src.experiment_retrieval

# 6. Live LLM (optional; everything above works without it)
echo "OPENAI_API_KEY=<key>" > .env
python3 -m benchmark.run --experiment controlled --sensor gpt4o --live
```

Regenerating the dense-retrieval embeddings is optional; they are frozen in the
repository. To rebuild them: `pip install -r requirements-retrieval.txt && python3 -m tools.build_retrieval_embeddings`.

## Project Structure

```
├── src/                          # Source code
│   ├── events.py                 # Regime definitions, warning templates
│   ├── env.py                    # Inventory simulator
│   ├── interpreter.py            # Semantic interpreters
│   ├── metrics.py                # SIVR, regret, decomposition
│   ├── classical_baseline.py     # TF-IDF + LogReg
│   ├── gym_adapter.py            # Paper-1 Gymnasium adapter
│   ├── confirmation_templates.py # Held-out templates
│   └── experiment_phase*.py      # Experiment runners
├── tests/                        # deterministic benchmark and correction tests
├── results/                      # Frozen experimental results
│   ├── phase5/                   # Controlled benchmark (original)
│   ├── phase5_5/                 # Real-LLM evaluation
│   ├── phase6/                   # Confirmation/robustness
│   ├── phase7_classical_baseline/# TF-IDF + LogReg baseline
│   ├── phase8_gym_replication/   # Paper-1 Gym (exploratory)
│   ├── phase8b_gym_confirmation/ # Paper-1 Gym (confirmation)
│   └── publication/              # Manuscript-ready tables
├── docs/                         # Formal definitions, design docs
├── paper_sections/               # Manuscript source material
├── benchmark/                    # Reproducible runner
│   ├── retrieval/                # Experiment R + frozen dense embeddings
│   └── frozen_llm_outputs/       # canonical LLM responses + provenance manifest
├── tools/                        # artifact rebuild, release verification, accounting
├── .llm_cache/                   # runtime LLM cache (not tracked)
├── BENCHMARK_VERSION             # Version freeze
├── SCIENTIFIC_LEDGER.md          # Authoritative numbers
├── REPRODUCIBILITY.md            # Reproduction guide
├── LICENSE / NOTICE              # MIT; third-party notices preserved
└── docs/ACCOUNTING.md            # generated experiment counts
```

## Limitations

- Synthetic report feed and synthetic inventory environments
- Relevance is defined by construction (entity and recency), not by assessors
- Experiment R's pools are generated per seed, so it has no language axis
- Dense retrieval is one general-purpose sentence encoder with frozen
  embeddings, not a trained dense retrieval system
- Single SKU, finite regime space
- Limited LLM coverage; provenance labels are reconstructed, not certified
- Fixed controller (not co-optimised with the interpreter)
- Evidence is consumed by concatenation; belief-level pooling is not evaluated

## Citation

```
@inproceedings{consequence_aware_evidence_2027,
  title  = {When Does Retrieved Text Improve Sequential Decisions?
            Consequence-Aware Evaluation of Evidence Selection},
  year   = {2027},
  note   = {Under review}
}
```

## License

MIT (`LICENSE`). Third-party components keep their own licenses; see `NOTICE`.

## Reproducibility and external artifacts

The repository contains source code, tests, frozen configuration, manuscript
material, and selected frozen result summaries. Local environments, API
credentials, LLM caches, raw response data, and large generated outputs are
not upload candidates. The Phase 1 SHA-256 manifest for external artifacts is
maintained outside this repository until an artifact-storage location is
approved. Reproduction should use the pinned `requirements.lock`, offline
tests, and explicitly documented input/output manifests; live LLM calls are
optional and must use credentials supplied through the environment only.
