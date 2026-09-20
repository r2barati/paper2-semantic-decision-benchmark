# V3 main experiment ledger (frozen; regenerated outputs only)

Every number below is recomputed from frozen artifacts by pinned code; no
value is hand-entered. Estimands: balanced J (uniform regime weights) for
utility; query-level paired bootstrap CIs (B=5,000, floor p<0.0002) for
effects; trec_eval nDCG for IR. Provenance chain:
corpus/qrels (`FINAL_FREEZE.md` SHAs) → trecs (`runs/v3main`, verified) →
beliefs (2×12,600 rows, `belief_manifest.json`, 0 gate errors) → semantic
(`results/v3main/semantic/`) → episodes (108,000, seeds 60000–60004) →
utility (`results/v3main/sim/`) → gym corroboration (`results/v3main/gym/`).

## 1. Retrieval (test n=160)

bm25 0.1150 / dense 0.1754 / hybrid-k120 0.1817 / rerank-top30 0.1829
(nDCG@10; R@10 0.1365/0.2062/0.2167/0.1938; R@20 0.2604/0.3771/0.4094/0.3708).
Dev ordering preserved; rerank−hybrid shrinks +0.024→+0.0012 out of sample.

## 2. Semantic accuracy (test, k=3)

Real retrieval: 0.28–0.44 all consumers/models (C0 reference 0.40–0.43 on the
same evidence). Oracle-factual C3: 0.9625–0.9688. C1 < C0 on identical
retrieved sets (bm25/8B: 0.275 vs 0.406). C3−C1 ≈ 0 on real retrieval
(+0.03–0.07, headline Holm-n.s.), +0.25–0.26 on oracle, −0.16–0.22 on random
(C3 abstains: 94%/56%). 14B ≈ 8B (34/42 Q6 CIs cross zero, max |Δ| 0.144).
Brier/NLL/ECE move with accuracy; C1 NLL inflates (up to 6.8) from confident
errors; ECE 0.46–0.78 real, ≤0.11 oracle-C3/8B.

## 3. Utility (balanced J primary, empirical J appendix; refs 1779.04 / 1994.40 / 2451.20)

Stars (∗) = Holm-0.05 rejection over the 105-arm family (71/105 balanced,
75/105 empirical); "ns" = Holm-n.s. even where the unadjusted CI excludes
zero. Text-cluster bootstrap CIs (64 text clusters) run 1.08× wider at the
median; every starred claim below holds under both. Inference is to new
queries/seeds within this bank only (single family; effective N ≈ 137).
k=3 Δ vs NoInfo (balanced): bm25 −11.8/−109.4/−97.3/−48.9/−60.7;
dense −9.4/−97.8/−96.7/−30.2/−55.7; hybrid −14.1/−89.3/−86.0/−48.6/−60.6;
rerank −37.3/−42.8/−38.6/−21.4/−16.9; random −32.0/−88.5/−89.5/−13.5/+11.0;
oracle +38.2/+75.9/+70.9/+178.4/+164.8 (cols C0/C1-8B/C1-14B/C3-8B/C3-14B).
Holm survivors among these: all C1 arms, all C3 LLM arms except rerank-C3
(Holm-n.s.), oracle LLM arms; Holm-n.s.: all C0 arms (incl. rerank −37.3 and
oracle +38.2), dense-C3-8B, random-C3-8B. k=5 same pattern (rerank C0 −4.7
n.s.; LLM arms still <0, Holm-robust).
Under empirical weights (0.2/0.4/0.4): C1 harm (−105…−130) and oracle-LLM
help (+82…+207) persist Holm-robust; neutral arms (rerank-C0/C3, bm25-C0,
hybrid-C0, random-C0) flip to ≈0/positive and stay Holm-n.s. Universal
quantifiers are therefore withdrawn: the scoped claim is C1-harm +
oracle-LLM-value, robust under both priors.
SIVR(Perfect): real −0.04…−0.51, oracle +0.18…+0.83. SIVR(OracleFactual):
real −0.08…−1.44. All VALID denominators.

## 4. Gaps (k=3): retrieval <0; relevance→factual 0.00; interpretation
C1∼140 → C3∼37–51; control +456.8 dominates all.

## 5. Transfer answers

Better IR → better semantics for LLM consumers (Q1 rho 0.8–1.0, direction
only: exact permutation p = 0.083–0.92, n=4 — significance never claimed),
flat for C0; consumer rankings stable (Q2 rho ≥ 0.8); interaction is C0-vs-LLM
(headline Holm: 3/10 reject — C0→C1/8B, C0→C3/8B, C0→C3/14B; C1→C3 and Q5/Q6
parity diffs Holm-n.s., i.e. nulls retained, not effects proven);
hit−miss +0.36–0.42 acc (Q4, retrieval-systems-only artifact); held-out entities score higher, mix suspected (Q7);
cache nondeterminism moves headline effects by ≤0.0021 Brier, 0.0 accuracy.

## 6. Gym (secondary)

NoInfo wins everywhere incl. vs PerfectBelief (−50.7): corroborates
NoInfo-robustness only. Rankings flip (rerank best→worst): environment
dependence, kept secondary.

## 7. What did not survive its controls

(a) Any claim that real retrieval helps utility — sign is negative.
(b) Any claim that C3 rescues real retrieval — remedy, not rescue.
(c) Any claim of 14B > 8B — parity. (d) Relevance→factual refinement —
zero downstream. (e) Held-out generalization — wrong-signed, descriptive.
What survived: oracle-evidence value, hit-quality dependence, C0-vs-LLM
interaction, control-gap dominance, NoInfo robustness across two worlds.

## 8. Extension: tuned no-text comparator (labeled, outside the freeze)

NoText_Tuned = constant belief (0.0, 0.0, 1.0), selected by grid search
(step 0.1, 66 triples) on DEV queries only (n=40) x tuning seeds 61000–61004
(dev_J 1885.6; benchmark prior ranked 19th). Evaluated on TEST x confirmation
seeds: J = 1881.13, Δ vs NoInfo +102.09 [+69.44, +138.24], p < 0.0002,
SIVR(Perfect) = 0.474 VALID. All real-retrieval arms sit below it too
(best real arm ≈ 1762 vs 1881). The harm finding survives a tuned comparator.
Extension files: `extension_notext_tuned*.json/parquet` (never in the frozen
matrix; freeze gate untouched).

## 9. Extension: second controller class (labeled, outside the freeze)

Belief-adaptive base-stock policy (`src/controller_basestock.py`, new file;
frozen code untouched) on identical frozen beliefs, paired seeds
60000–60004, same warning protocol, headline arms k=3 test-only
(`results/v3main/sim_controllerB/`, Holm over 9 extension arms, 8/9 reject).
NoInfo 1414.10; PerfectBelief 1502.8 (+88.7); tuned constant 1374.8 (−39.3
vs NoInfo — loses here). Preserved: rerank/C1 −55.0∗, rerank/C3 −94.8∗,
oracle/C1 +107.3∗, oracle/C3 +100.1∗. Reversed: BM25/C1 +54.5∗ (was −103.5∗),
tuned constant (was +102.1∗), oracle/C0 −84.2∗. Harm is pairing-dependent
(retriever × consumer × controller), not a universal retrieval property and
not a pure controller artifact (oracle gains + perfect helps under both
classes). This is the finding the red-team controller-swap demand asked for,
reported whichever way it came out.

## 10. Extension: selection-validity upgrade probe, Stage A (labeled, outside the freeze)

Motivation: test whether metric-based retriever selection creates
consequential downstream selection regret (stronger novelty claim) before
spending on new environments/controllers. Pre-committed design
(results/v3main/extension_selection/DESIGN.md): dev selectors (nDCG/Brier/J)
-> freeze -> sealed-test regret; H1a primary (nDCG-vs-J, paired bootstrap
CI must exclude 0); H1b calibration vs query-split-half null; H2
retriever x consumer interaction vs permutation null.
A0 action-logging replay (tools/run_actions_v3main.py, 8 shards, CPU-only,
frozen beliefs/controller/seeds): 33,600/33,600 profits match frozen
episodes.parquet EXACTLY (max diff 0.0) -> gate passed; per-episode
40-period order vectors in results/v3main/extension_actions/actions.parquet.
Action proxy (locked in advance): ADis = mean_seed mean_t |o_arm - o_NoInfo|,
ADisRate = fraction |Δo|>0.5.
Outcome: S-nDCG/S-J select rerank in ALL 5 strata; test-best is rerank in
all LLM strata (R=0) and dense in C0 (R=35.58 for both selectors).
H1a dR = 0.0 CI [0.0, 0.0] -> FAILS. H1b fails (7.12 < 28.17; Brier 0 <
10.08). H2 HOLDS (F=6.55/7.37, perm p=0.0002): pairing matters beyond
chance. A2 chain dissipates per interface (ret->bel rho~0.35; bel->act ~0;
act->util ~0). Per the locked decision rule the upgrade collapses to this
methods-note; V3 stands unchanged. No Stage B requested.
Files: results/v3main/extension_selection/{DESIGN.md,A1_selections.csv,
A1_H1b.json,A2_edges.parquet,A2_actions.parquet,A2_interfaces.json,
A3_verdict.json}, tools/run_actions_v3main.py, tools/a1_selectors.py, tools/a2_edges.py, tools/a3_tests.py. Frozen matrix untouched.

## 11. Extension D0: Gate-1 belief-VoI (labeled, outside the freeze)

Motivation: Stage-A showed value dissipates at bel→act and selectors agree
(H1a dR=0). D0 tests one decision-aware interface: a belief gate transmitting
the posterior iff expected action-value effect is positive, else NoInfo.
Frozen design: results/v3main/extension_d0/DESIGN-D0.md — VoI in utility
space (ADis mediator only), belief-VoI vs document-VoI boundary, Head-B
primary with C(b)=E[J(posterior)−J(NoInfo prior)], Brier feature (confidence
constant for C0/C1; evidence_hit degenerate 0 on dev), grid t in
{0.10..0.50}, all choices on dev-40, test-160 touched once.
Tune (dev-only, tools/d0_tune_gate.py): S-J=rerank all 5 strata; frozen
t={C0:0.2, C1-8B:0.3, C1-14B:0.1, C3-8B:0.3, C3-14B:0.3}; dev gated means
+63..+119 (vs arm means −7..+33).
Eval (single test touch, tools/d0_eval_gate.py, B=5000 seed 71000):
PRIMARY HOLDS — pooled Head-B gated ΔJ=+77.3 [62.8,92.0]; arm-gate
+0.03 [−22.0,22.2] fails. Per-stratum gated +62..+97, all Holm p=0.0.
Transmit rates 35–52% test (dev 40–57%). Always-transmit S-J pooled ≈−4.3
plain-mean, so the gate beats both proxies (nDCG/J agree: rerank everywhere).
Mechanistic read as pre-registered: transmission, not acquisition, was the
bottleneck. Caveats (binding): Brier uses true-regime labels — this is an
oracle-quality gate proving transmissible belief-VoI exists, NOT a deployable
policy and NOT document-causal; plain-mean selection metric deviates from
balanced headlines (as in Stage-A, documented). Per the locked rule, Gate-2
(document-VoI with new ablation beliefs) is now scientifically justified and
may be proposed — not executed here. V3 frozen matrix untouched.
Files: results/v3main/extension_d0/{DESIGN-D0.md,d0_tune.json,d0_verdict.json},
tools/d0_tune_gate.py, tools/d0_eval_gate.py.
