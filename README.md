# When Relevance Does Not Transfer

An executable benchmark showing that **retrieval utility is
consumer-dependent**: the same ranking helps one downstream decision-maker and
harms another, and relevance-based evaluation cannot tell the two cases apart.

## Research question

When does retrieved text improve sequential decisions? The answer here is
negative and specific: not reliably, and whether it does is a property of the
pipeline rather than of the ranking.

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
| **R** (retrieval) — *the contribution* | the retrieval system and the downstream consumer, under matched evidence budgets | controlled inventory, 3 regimes |
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

1. **Every ranking system helps one consumer and harms another.** At k=3 the
   retriever × consumer interaction changes sign for all four rankers,
   magnitude +49.9 to +89.2 reward units.
2. **The evidence is there.** An oracle relevance selector improves *both*
   consumers (+60.6 and +69.0). The failure is mismatch, not scarcity.
3. **Relevance-based evaluation detects this for one consumer only.**
   Rank-reversal rate 0.00 for the rule-based consumer, 0.33 for the calibrated
   one, with no upstream signal separating them.
4. **The gradient is shared; the level is not.** Episode-level correlation
   between ranking quality and reward is near-identical for both consumers
   (+0.372, +0.382) — which is why a relevance-only evaluation misses the gap.
5. **Supporting diagnosis:** text-free, shuffled-text and label-only controls
   show that belief accuracy and probabilistic sophistication do not imply
   operational value, and a perfect-semantic belief reference is not an upper
   bound on realised return.

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
