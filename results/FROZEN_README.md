# FROZEN results — publication surface (do not edit)

Authoritative counts and numbers:

- `docs/ACCOUNTING.md` (generated from saved episode files; corrects Phase 7
  6→5 sensors, Phase 6 50→20 seeds, horizon 30→40, 4320 pairs→720 worlds)
- `SCIENTIFIC_LEDGER.md` (regenerated, B=5000, floor 1/(B+1), Holm, crossed
  bootstrap, signed SIVR, balanced `J_w` primary + prior-weighted secondary)

Frozen directories (representative manifests in parentheses):

- `results/phase7_classical_baseline/` = Experiment C (`experiment_manifest.json`,
  `split_manifest.json`, 8640 eps)
- `results/phase8b_gym_confirmation/` = Experiment M (`experiment_manifest.json`,
  `seed_manifest.json` 3100–3129, `linguistic_split_manifest.json`,
  `notext_tuning.json`)
- `results/phase9a_capacity_confirmation/` = Experiment S
  (`experiment_manifest.json`; clean linguistic claim only from
  `results/correction_audit/phase9a_test_templates_only.csv`)
- `results/phase9b_robustness/` = Experiment B, descriptive OFAT
  (`experiment_manifest.json` top-level; per-variant CSVs only)
- `results/retrieval/` = Experiment R (`retrieval_summary.csv`,
  `retrieval_episodes.csv`, `rank_agreement.csv/.md`, `frozen_embeddings.json`;
  no manifest — most-cited gap, see `docs/results_manifest_spec.md`)
- `results/v3main/` frozen matrix: `belief_manifest.json` (2×12600 rows,
  prompt SHAs `C1 66e6890b`/`C3 5966cadd`, data SHAs),
  `semantic/` (q1–q7), `sim/` (108k eps, 105-arm Holm), `gym/` (107k eps),
  `delivery/LEDGER.md` + `FINAL_ASSESSMENT.md`
- `results/frozen_llm_outputs/manifest.jsonl` (row provenance; many
  `model_id:unknown` — reconstructed, not certified)
- `results/correction_audit/` (`recompute_corrections.py`,
  `sivr_recomputed.csv`, `estimand_audit.csv`, `raw_reward_effects.csv`)
- `results/publication/` (generators `generate_*.py`; `corrected_*.md` current)

V3 extensions (`LEDGER.md §§8–24`: D0, G2A/G2FULL, VHAT/VHAT2, Controller-B,
selective, actions/selection, map, ladder, TrackA, Agentick) are **not** part
of the frozen matrix; each has its own DESIGN + seeds + Holm family. See
`docs/experiment_registry.md` and `docs/claim_evidence_ledger.md`.
