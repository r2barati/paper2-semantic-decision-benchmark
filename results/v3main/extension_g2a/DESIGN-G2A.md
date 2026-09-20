# G2A-0 — FROZEN DESIGN (2026-09-20, pre-execution; 2A zero-LLM stage)

Scope: document-VoI WITHOUT new LLM calls. Two ingredients:
(a) C0/C3 leave-one-out re-computation on rerank k=3 (removal-only =
cache hits by construction); (b) analysis-only contrasts of the frozen
`intervention-*` arms (oracle-factual base). C1 LOO needs new calls and is
NOT in 2A-0 (it is the 2A-1 dev pilot, conditional on the pilot-signal rule
below). No Agentick. No 2B/2C work.
Outputs: `results/v3main/extension_g2a/` + `tools/g2a_*.py` + LEDGER entry.

## 1. Estimands (utility space, D0 boundary inherited)

Per query `q`, full evidence set `S_q` (|S|=3), consumer `c`, model `m`:

```
VoI(doc_i, q) = J(b(S_q)) − J(b(S_q ∖ {doc_i}))
```

`J(b)` = mean_seed episode profit replaying frozen belief `b` through the
frozen `CausalOptimizer` + `InventoryEnv` on paired seeds `60000–60004`
(same protocol as A0). Reported as Δ vs NoInfo via per-query
`delta_vs_noinfo` (frozen `utility_by_query` for full sets; fresh replay for
LOO sets, same seeds). Pooled over queries; paired test-query bootstrap
`B=5000`, seed `72000`.

* C3 LOO is a VALID document-causal contrast: each remaining doc's judgment
  is the identical cached per-doc LLM output (`prompt_sha 5966…`,
  `sha(model||prompt||own_node+text)`); only the frozen deterministic
  aggregation input changes. Verified prototype: full-set re-aggregation
  reproduces a frozen row to 6 decimals.
* C0 LOO is likewise valid: `consume_c0` is a deterministic rule over
  concatenated text, recomputed locally.
* C1 LOO would change the LLM input itself → NOT valid by replay → excluded.
* Frozen interventions (`drop-decisive`, `stale-swap`, `inject-contra`,
  `wrong-entity-swap`, `irrelevant-inject`, `duplicate-top`,
  `order-reverse`) are reported as precedent contrasts on the oracle-factual
  base (`J(arm) − J(oracle-factual)`, `J(arm) − J(NoInfo)`), never as
  substitutes for rerank document-VoI.

## 2. Scope (frozen)

Systems: `rerank` headline only (`k=3`). Consumers/models: `(C0,none)`,
`(C3,8B)`, `(C3,14B)` — C1 strata have no valid zero-LLM LOO and wait for
2A-1. Queries: all 200 (dev-40 for the pilot-signal decision, test-160 for
the once-only verdict). Ablations per set: 3 leave-one-out (positions
0,1,2). Swap-in of unseen pairs (= cache miss) is excluded; misses are
counted and reported, expected 0.

## 3. Gates (frozen, in order)

* **G1 — full-set reproduction:** re-aggregated/recomputed full 3-doc beliefs
  must match frozen parquet rows (`p_*` within `1e-9`, abstain exact) for
  100% of in-scope rows. Any mismatch → STOP, no LOO rows trusted.
* **G2 — cache completeness:** every LOO subset doc must hit cache.
  Miss count reported; any miss → that row excluded (never a new call).
* **G3 — replay profit match:** replayed full-set profits must match frozen
  `episodes.parquet` within `1e-9` (A0 gate rerun on the LOO subset keys).
* **Pilot-signal rule for 2A-1 (decided on dev-40):** propose the 240-call
  C1 dev pilot iff on dev either (i) pooled mean |VoI| per left-out position
  has a 95% bootstrap CI excluding 0 in ≥1 stratum, or (ii) max−min
  per-position pooled VoI exceeds the pooled bootstrap noise width. Else
  STOP: D0 existence-proof stands, no new LLM spent.

## 4. Test discipline (binding)

Dev-40 carries the go/no-go decision. Test-160 is evaluated ONCE for the
frozen LOO verdict under the granted fresh-single-touch authorization for
new artifacts (D0 verdict never reopened, no re-tuning: there are no tuned
thresholds in 2A-0 — per-position VoI is descriptive + CIs).

## 5. Non-goals

No new LLM calls; no C1 ablation; no k=5; no 4-system expansion; no 2B
proxy; no 2C comparison; no Agentick; no manuscript prose.
