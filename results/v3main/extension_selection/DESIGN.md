# Stage A — FROZEN DESIGN (2026-09-18, pre-execution; user adjustments incorporated)

Scope: analysis-only + CPU-only action-logging replay. No new LLM calls.
No frozen-matrix edits. Outputs: results/v3main/extension_selection/ +
extension_actions/ + LEDGER entries.

## A0 — action-logging replay (CPU-only, frozen beliefs/controller/seeds)

Subset (locked): systems {bm25,dense,hybrid,rerank,random,oracle-relevant,
oracle-factual} × consumers {C0(none),C1(8B,14B),C3(8B,14B)} × k=3 × test
queries (160) × seeds 60000-60004. New tool replays _run_p5_episode on
frozen belief rows and records per-episode 40-period order vectors.
VERIFICATION GATE (pre-registered): replayed profits must equal frozen
episodes.parquet profits exactly (fp tolerance 1e-9) for ALL replayed
items; any mismatch → STOP, debug protocol, no A2.
NoInfo rung reused from frozen episodes (no replay needed).

## Action proxy (locked, mechanical, per user adjustment — no post-hoc tuning)

Per episode: order vector o (40 periods). Per (arm, query):
ADis = mean_seed[(1/40) Σ_t |o_arm − o_NoInfo|] (order units),
ADisRate = fraction of (seed, period) with |Δo| > 0.5.
NoInfo comparator: frozen noinfo rung, same query+seed. Both reported;
ADis primary, ADisRate secondary.

## A1 — selectors (locked)

Strata: (C0,none), (C1,8B), (C1,14B), (C3,8B), (C3,14B). Within stratum,
select among {bm25,dense,hybrid,rerank} (random/oracle not selectable;
documented):
- S-nDCG: argmax dev nDCG@10 (ir_dev means).
- S-Brier: min dev mean Brier (semantic_all split==dev).
- S-J: argmax dev J, where dev J = plain mean episode profit over
  dev-query episodes per arm (documented deviation: headline test J uses
  balanced weights; dev selection uses plain means as the selection
  protocol, not a headline).
Test regret per stratum from frozen headline J (utility_by_arm_balanced,
test-only): R = J(test-best retriever in stratum) − J(selected).

## A3 — H1/H2 (locked rules; H1 primary contrast = nDCG-selected vs J-selected)

- H1a (PRIMARY, decides): ΔR = mean_s[R(S-nDCG,s) − R(S-J,s)] over 5 strata;
  CI via paired test-query bootstrap (resample 160 test queries w/repl,
  all seeds, plain-mean-profit metric, B=5000, seed 70000). Holds iff
  two-sided 95% CI lower bound > 0.
- H1b (calibration): R(S-nDCG) pooled vs query-split-half null on dev
  selection (500 dev half-splits → same select→regret protocol → null
  distribution of pooled R; holds iff observed > 95th pct).
- Brier secondary: R(S-Brier) vs same null; Holm over {H1b-nDCG, H1b-Brier}.
- H2: system×consumer interaction F on test episode profits (4 retrievers ×
  C0/C1/C3, 8B model primary, k=3) vs permutation null (shuffle system
  labels within query, 5000 perms; exact p). 14B as sensitivity.
- DECISION RULE (amended pre-execution per owner): upgrade proceeds to a
  B-authorization request iff H1a CI excludes 0. Otherwise the upgrade
  collapses to a methods-note and V3 stands. H1b/H2/Brier reported
  regardless (no rescue analyses).

## A2 — edge table (locked)

Per (query, system[k=3], consumer, model): nDCG@10 (ir_test), evidence_hit,
accuracy/Brier/ECE (semantic_all test), ADis/ADisRate (A0 replay),
ΔJ vs NoInfo (utility_by_query test). Sign-agreement + rank-transfer at
each interface: retrieval→belief, belief→action (via ADis), action→utility.
Scope: 7 systems above (interventions are B3 territory, excluded).
