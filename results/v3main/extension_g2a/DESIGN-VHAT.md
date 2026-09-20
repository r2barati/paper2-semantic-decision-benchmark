# VHAT — FROZEN DESIGN (2026-09-20, pre-execution; Phase 0+1)

## 0. Novelty boundaries (binding; separates the three contributions)

* **Scientific (EARNED, may be claimed):** operational document value is
  consumer-dependent with qualitatively different regimes (C0 positive,
  C3 ~=0, C1 mixed signed; permutation p<=0.0002); useful information can
  dissipate at identifiable interfaces (belief->action). No universality
  claim: "in this controlled inventory benchmark".
* **Model (TO EARN, conditional):** consumer-conditioned Operational-VoI
  `Vhat(e,b,c)` with conformal transmit rule — described ONLY as
  "outperforms preregistered relevance-, LLM-utility-, and always-transmit
  baselines on a fresh sealed population" if the confirmation holds. The
  word SOTA is BANNED unless published methods are compared under comparable
  conditions (they are not in this plan).
* **Experimental (TO EARN, conditional):** sealed fresh-split win + one
  external (Agentick, conditional) confirmation.
* **Borrowed, cited, never claimed:** task-oriented/semantic communications
  theory, Value-of-Information/OR, gain/dead-zones, conformal/selective
  prediction. Related-work wall (RRPO, RADEG, Active-RAG triggering +
  threshold-transfer failure, IDCR, UDCG) is addressed in manuscript, not
  re-litigated here.

## 1. Model (frozen form)

```
Vhat(e,b,c) = dBhat(e,c) x Shat(b,c)
```

* `dBhat(e,c)`: predicted belief shift (L1) induced by evidence `e` under
  consumer `c`, trained on LOO L1 labels.
* `Shat(b,c)`: predicted decision sensitivity (J-units per unit belief
  shift), trained on LOO (VoI, L1) pairs; the frozen controller FD is an
  input feature, not the whole head.
* Consumer conditioning: consumer one-hot (+ model id) is a mandatory input
  to both heads; a consumer-blind ablation is reported as a secondary.
* Transmit/drop per evidence position: DROP `e` iff `LCB(Vhat(e)) <= 0`,
  else keep. Fresh-bank evaluation replays kept-subset beliefs (C1 subsets
  pre-generated as new calls; C3 re-aggregated free).

## 2. Conformal validity (frozen; the tightened rule)

* Split-conformal on CLUSTER-DISJOINT data: 20% of training clusters
  (~13 of the 64-cluster dev+test union; cross-split duplicates collapse
  30+63), stratified by consumer, reserved as calibration and NEVER
  used to fit the final predictor (fit on remaining 80%).
* Nonconformity: cluster-mean residuals `s = V - Vhat` over calibration
  clusters. `LCB(e) = Vhat(e) - Q_{1-alpha}(s)`.
* **alpha = 0.1 frozen** (90% lower bound). Coverage is claimed ONLY under
  cluster exchangeability between calibration and fresh-bank clusters, stated
  as an assumption with the duplicate-context caveat recorded.

## 3. Training labels (frozen)

All 3000 inventory LOO VoI labels (dev 600 + test 2400; 5 strata x 3
positions) as training/calibration ONLY. Test-160 never confirms again.
Hyperparameters by cluster-CV within training labels. Fresh bank is the sole
confirmatory population.

## 4. Fresh-bank belief matrix (frozen; prices the calls BEFORE spending)

Per fresh query (N finalized by power rule below): rerank k=3 full + all 3
LOO subsets for C1 x {8B,14B} as new calls (N x 4 x 2 = 8N C1 calls); C3
full as new calls (2N; C3 LOO re-aggregated free); C0 local. At N=60:
480 C1 + 120 C3-full (+360 C3 per-doc) calls plus encode/rerank rerun.

## 5. Power rule (frozen; executed on historical labels BEFORE generating)

Using ONLY the 3000 training labels: simulate trials drawing N clusters with
replacement from the 93-cluster pool, inject true effect Delta, cluster-
bootstrap CI exclusion rate = power. Grid Delta in {10,15,25,40}, N in
{60,90}. **Bank N=60 iff power(N=60, Delta=25) >= 0.8, else N=90** (recompute;
if still <0.8, report required N up to cap 120). Delta=25 is just above the
smallest replicated per-position effect (+21.9, 14B/drop1 test). N is
finalized and committed before any fresh outcome exists.
EXECUTED 2026-09-20 (`tools/vhat_power.py`, `vhat_power.json`): training
population is 64 union clusters; power saturates (N=60: 0.998 at Delta=10,
1.0 at Delta>=15) → **bank N=60**. Caveat recorded: the simulation injects a
homogeneous shift; heterogeneous real effects widen per-cell CIs, which the
Holm secondaries will show honestly.

## 6. Confirmation (frozen; single touch, on the fresh bank only)

Vhat-gated pooled dJ vs NoInfo CI>0 (cluster bootstrap) AND beats max of the
frozen upstream baselines {nDCG-selected, LLM-utility-selected,
always-transmit} with CI>0 on the difference; per-stratum Holm secondaries.
Fail = oracle-to-deployable gap stands; no re-tuning, no second peek.
(c)-holds is the ONLY path to Agentick.

## 7. Non-goals

No fresh data touched in Phase 0+1 (no generator run, no bank outcomes, no
human labels). No SOTA language. No Agentick work. V3 + all prior verdicts
stand untouched regardless of outcome.
