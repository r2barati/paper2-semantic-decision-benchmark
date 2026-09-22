# FINAL_BANK_V2 specification (committed; no texts, no IDs)

- **Status:** prospectively authored and evaluation-untouched. NOT globally
  untouched: authored after DEV behavior was observed (retrieval mismatch, A3
  zero second-query rate), from a rubric frozen before authoring with zero
  system/model feedback during authoring.
- **Schema version:** 1. Entries: `{template_id, ambiguity_level, regime, text}`.
- **Design:** 27 = 3 regimes (normal / supplier_delay / demand_surge) ×
  3 ambiguity levels (clear / moderate / vague) × 3 variants.
- **ID prefix:** `fx_` (fresh; disjoint from `delay_/surge_/normal_/cd_/cs_/cn_`).
- **Rubric:** delay = lead-time deterioration ~period 20; surge = demand increase
  ~period 20; normal = no disruption. Clear = specific parameters; moderate =
  partial/hedged; vague = general concern, no parameters. No regime label
  tokens in text.
- **Sealed artifact (local-only, never committed):**
  `results/reasoning_agentic/FINAL_BANK_V2.json`
- **SHA256:** `27f1307dba25ef3fcbc5c4f8cb823c98df293671be1bc25738b328144d426979`
- **Validation:** `tests/test_final_reasoning_bank.py` (skip-if-absent locally;
  skips on fresh clones): schema, 3-per-cell balance, ID + verbatim-text
  disjointness from train-18 / confirmation-36 / capacity-16 banks. Passed
  2/2 at authoring time.
- **Staging rule for the final run:** expose only `{template_id, text}`;
  labels stay in the local sealed file for scoring.
