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

## 12. Extension G2A-0: zero-LLM document-VoI (labeled, outside the freeze)

Motivation: D0 proved recoverable belief-VoI with an oracle gate; G2A-0 asks
whether individual passages carry causal value, without spending LLM calls.
Frozen design: results/v3main/extension_g2a/DESIGN-G2A.md — VoI(doc_i) =
J(full)−J(minus-i) on rerank k=3 x {C0, C3-8B, C3-14B}; C1 excluded (concat
prompt changes = new calls -> 2A-1).
Gates all passed: G1 full-set re-aggregation matches 600/600 frozen rows
EXACTLY (max diff 0.0, abstain 0 mismatch); G2 0 cache misses (removal-only =
hits by construction); G3 replayed full-set profits match frozen episodes
3000/3000 EXACTLY. 1800 LOO belief rows x 5 paired seeds = 9000 LOO + 3000
full episodes, frozen controller/protocol.
Outcome (tools/g2a_contrast.py, B=5000 seed 72000): C0 shows positive
position-dependent document-VoI (test drop0 +9.06 [0.40,18.28], drop2 +22.58
[11.35,33.82], CIs exclude 0) — passages carry value for the rule consumer.
C3 (both models) LOO VoI ~= 0, all CIs cross 0 — the provenance-weighted
aggregation is redundant/robust at k=3; belief differences from one passage
do not move actions (fine-grained D0 dissipation). Frozen interventions
(oracle-factual base, precedent only): all removals/swap-ins degrade
(drop-decisive −16..−48, wrong-entity −26..−70, contra −14..−45,
irrelevant −7..−51, stale −7..−31); duplicate-top ~= 0 for LLM arms;
order-reverse exactly 0.0 for C0/C3 (order-free aggregation, as designed)
but −10.8/−37.4 for C1 — concat consumption IS order-sensitive.
Pilot-signal rule (dev-40): FIRES via rule (i) (C0 drop0/drop1 + C3-8B
drop2-negative CIs exclude 0; rule (ii) spread 11.82 < noise 28.74 fails).
Per the frozen rule, the 240-call C1 dev pilot (2A-1) is PROPOSED, not
executed: its value is specifically C1 concat passage-causality (order
sensitivity already hints it exists), since C3 document-VoI is flat. V3
frozen matrix untouched.
Files: results/v3main/extension_g2a/{DESIGN-G2A.md,loo_beliefs.parquet,
loo_episodes.parquet,g2a_gates.json,g2a_verify.json,g2a_verdict.json},
tools/{g2a_loo,g2a_replay,g2a_contrast}.py.

## 13. Extension G2A-1: C1 LOO dev pilot, 240 calls (labeled, outside the freeze)

Question (frozen DESIGN-G2A1.md): does removing individual passages causally
change C1 belief/action/utility enough to justify full document-VoI work?
Scope: dev-40 x rerank k=3 x C1 x {8B,14B} x LOO{drop0,1,2} = 240 calls.
Kernels paper2-v3-loo-pilot-{8b,14b} (pinned models/revisions, frozen vLLM
flags/patch/prompts, dev-only inputs, seal guard passed in-kernel); fetched +
verified (manifest pins, src snap byte-identical, 120 invocations/model,
0 fails, key-recompute D=90/model).
Pilot subtlety (benchmark property, not a defect): dev-40 holds duplicate
query texts with identical rerank sets (e.g. v3-q013/v3-q077); the C1 user
string carries no entity_node, so 120 invocations = 90 distinct contexts
(30 in-run cache hits, same dedup the frozen matrix has). All 180 payloads
carry the usual extra estimates that frozen consume_c1 ignores. Installed
180 files with 0 overwrites; parsed under byte-identical C1 normalization.
Replay (dev x 5 paired seeds): G3-analog gate 400/400 exact. Levels per
(model x position), B=5000: belief L1 shifts 0.13-0.33 (all CI>0), ADis
0.17-0.40 (all CI>0). J-VoI: 14B/drop1 +37.67 [+6.68,+73.67] (rank-2 passage
destroys value — misleading evidence) and 8B/drop2 -16.38 [-34.01,-1.02]
(rank-3 passage carries value); other 4 cells cross 0.
Continuation gate: FIRES (2 cells meet (a) J-VoI excl-0 AND (b) ADis>0).
Per the frozen rule this PROPOSES (not executes) full C1 document-VoI
(test-160, 960 calls) + deployable criticality proxy. Test stayed sealed;
D0/2A-0 verdicts unreopened. V3 frozen matrix untouched.
Files: extension_g2a/{DESIGN-G2A1.md,loo_sets_frozen.json,c1_loo_beliefs.
parquet,c1_loo_episodes.parquet,g2a1_verify.json,g2a1_verdict.json},
tools/{g2a1_assemble,g2a1_replay,g2a1_verdict}.py,
kaggle_kernel/p2_loo_{8b,14b}.py, runner JOBS loo-pilot-{8b,14b}.

## 14. Extension G2-FULL: C1 test document-VoI + deployable proxy (labeled, outside the freeze)

Design (frozen DESIGN-G2FULL.md BEFORE any test work): cluster unit
sha(text+docset) (63 test clusters), (a) Holm-6, (b) within-query consumer
permutation, (c) frozen-gate pooled dJ CI>0 AND beats max{nDCG,S-Brier,S-J,
always-transmit} upstream baselines. Proxy frozen on dev BEFORE test calls
(g2full_proxy.json: agreeXmodel_L1>=0.6, dev gated +75.5, committed 81b4cb4).
Test: 960 C1 calls (480/model, 0 fails, D=189 distinct contexts/model:
63 clusters x 3), 378 payloads (102 new + identical-dups per model, 3
divergent-dup first-wins patched with logged file), 0 overwrites otherwise.
G3-analog 1600/1600 exact. Single Stage-4 outcomes touch.
BUG DISCLOSED AND FIXED: first stage4 run mixed dev rows into the (b) pivot
(2A-0 replay covers dev+test), yielding NaN maxmin with p=0.0; fixed by
test-only filter + rerun before any interpretation. No thresholds changed.
Outcome: (a) HOLDS — 14B/drop1 +21.94 [+10.43,+34.11] Holm 0.0; 14B/drop2
+12.81 [+2.32,+23.91] Holm 0.078; dev-predicted cells replicate (dev +37.7/
-16.4 -> test +21.9/-13.0). (b) HOLDS at drop1 (p=0.0002) and drop2 (p=0.0),
not drop0 (p=0.21). Regime table: C0 positive (+6..+23), C3/8B ~=0
(-6..+0.6), C1/8B negative (-4..-13), C1/14B positive (+5..+22) — the
qualitatively different value regimes, now formal. Adopted wording:
"document-level operational value is consumer-dependent", not "determined".
(c) FAILS — frozen gate transmits 8/160 test queries (disagreement
distribution shifted vs dev); pooled -0.61 [-9.85,+6.95]; diff-vs-best CIs
cross 0 in both strata (numerically +17.8 each, correctly unclaimed).
Per the frozen rule the finding is the ORACLE-TO-DEPLOYABLE GAP: Brier-gate
(D0, +77) and disagreement-gate (dev, +75) prove transmissible value exists,
but the label-free proxy does not transfer its threshold. NO AGENTICK.
V3 frozen matrix untouched.
Files: extension_g2a/{DESIGN-G2FULL.md,g2full_proxy.json,loo_sets_test_
frozen.json,c1_loo_test_beliefs.parquet,c1_loo_test_episodes.parquet,
g2f_verify.json,g2full_stage4.json,test_install_divergences_8b.json},
tools/{g2f_proxy_screen,g2f_assemble_test,g2f_replay_test,g2f_stage4}.py,
kaggle_kernel/p2_loo_test_{8b,14b}.py, runner loo-test-{8b,14b}.

## 15. Extension VHAT: historical feasibility gate -> STOP, no bank (labeled, outside the freeze)

Sequencing (DESIGN-VHAT §§8-11, frozen before execution): train/freeze Vhat
on the 3000 historical LOO labels FIRST; build the 60-query fresh bank ONLY
if the preregistered gate passes. Vhat(e,b,c) = dBhat(e,c) x Shat(b,c)
(GBM heads, fixed config, consumer one-hot, deployability budget: own-belief
+ embeddings/ranks + controller-FD only; cross-model/consumer + labels
forbidden). Conformal: split cluster-disjoint 80/20, alpha=0.1 frozen.
Power (historical only): 64 union clusters; N=60 saturates -> bank N=60.
Gate (all OOF/held-out, deterministic seeds): (i) PASS, barely — pooled
Spearman CI [0.079,0.168] excludes 0; per-stratum 4/5 positive but C1-14B at
+0.002 and C1-8B at -0.036. (ii) FAIL — calibration coverage 0.672 vs band
[0.85,0.95]; mechanism diagnosed: row residual SD (74.5) is 5x cluster-mean
residual SD (15.0), and coverage is heteroscedastic across strata (C0 0.86 /
C1-14B 0.85 in-band; C1-8B 0.71; C3-8B 0.54 / C3-14B 0.48) — a pooled
cluster-mean quantile cannot cover row-level heavy-tailed VoI. (iii) FAIL —
OOF single-minimum-drop gating pooled diff CI [-5.66,+4.80] includes 0;
3/5 strata and 1/3 positions positive.
BUG DISCLOSED AND FIXED: the first (iii) computation used FINAL-fit
predictions instead of OOF (pooled CI [1.5,11.1], plus an impossible
-4.51/-4.51 doublet across C0 and C1-14B); caught by an independent
replicate, fixed to true OOF, rerun deterministic. The corrected numbers
fail (iii) on the main clause, which strengthens rather than weakens the
stop decision — reported here so the fix is auditable.
BUILD_BANK = false. Per the frozen rule: NO fresh bank, NO new LLM calls,
NO Agentick. The deployable-model path ends here; the paper keeps the
diagnostic + consumer-dependent-regimes + oracle-to-deployable-gap
contributions (Stages A/D0/G2A/G2-FULL(a)(b)). V3 frozen matrix untouched.
Files: extension_g2a/{DESIGN-VHAT.md,vhat_power.json,vhat_oof.parquet,
vhat_heads.pkl,vhat_model.json,vhat_gate.json}, tools/{vhat_power,
vhat_train}.py.

## 16. Workstream 1: dead-zone theory frozen (no computation yet)

Froze results/v3main/theory/DEAD_ZONES.md: action-equivalence set D(s) as
the zero-value object (corrected: basis chambers give affine sensitive
regions, not zero value); zero-value theorem with induction proof over the
frozen warning-release protocol + determinism lemma; no-order polytope as
exact special case; boundary taxonomy (LT-rounding walls enumerable modulo
bankers rounding; demand-RHS walls per-(t,state); censoring kinks; pipeline/
horizon indexation); §5 regime predictions (C3 inside chambers, C1 crossing
walls, C0 positive-sensitive); §6 empirical-map spec (decide()-only sweeps,
overlays; success-or-bound). Tripartite structure: dead zones + sensitive
regions + discontinuity boundaries.
Adopted corrections for later workstreams (recorded here, specified in
their design docs): (a) retrieval rung is "UDCG-inspired@3", never UDCG@3 —
frozen C3 judgments define machine utility, not UDCG's annotation; (b)
Agentick fairness lock — one fixed representation-agnostic policy adapter
across ascii/language/state_dict, oracle for ONS normalization only.
V3 frozen matrix untouched.

## 17. Workstream 1 map: FROZEN H1/H2/H3 ALL FAIL (labeled, outside the freeze)

Sweep (tools/map_sweep.py): 15 NoInfo trajectories x 8 periods = 120 states
x 861-node lattice = 103k decide() calls, CPU-only, zero LLM. Harvest gate
15/15 exact. Map is alive (prior-order>0 in 87.5% states; ~14 distinct
actions/state) — not degenerate.
Frozen verdict (tools/map_overlay.py, MAP_SPEC.md, query bootstrap):
H1 dead_C3=0.800 vs dead_C1=0.808, diff CI [-0.0118,-0.0051] (excludes 0 in
the WRONG direction); H2 cross_C1=0.146 vs cross_C3=0.168, diff CI
[-0.031,-0.013] (wrong direction); H3 pos_frac_C0=0.172, CI [0.164,0.180]
(<< 0.5). All three FAIL as frozen.
Mechanism diagnosed (exploratory, on frozen sweep tables — verdict stands):
(a) period-drowning — grid dead-fraction is 1.00 at t=0..15 and t=30, 0.01
at t=20, 0.35 at t=25; pooling over all 120 states buries decision-active
periods; (b) at active states (t=20,25) dead fractions are C0 0.25 / C1
0.23 / C3 0.19-0.20 — same ordering, so drowning is not the whole story;
(c) action magnitudes |order(post)-order(prior)| at active states are
near-identical across consumers (means 2.06-2.53, P90 5.0-6.2) — neither
dead-fractions nor response magnitudes separate C3-flat from C1-mixed.
POST-HOC refined hypothesis (labeled as such, NOT tested): D(s) determines
WHETHER value can flow (gating), while sign/magnitude needs belief-truth
alignment x sensitivity — C3 gated regardless of correctness (flat), C1
transmitted with correctness-dependent sign (mixed), C0 transmitted +
directionally correct (positive). Requires its own preregistered test
(e.g. correctness-conditioned response analysis); the map tables are frozen
for it. V3 frozen matrix untouched.
Files: theory/{MAP_SPEC.md,map_states.parquet,map_orders.parquet,
map_verify.json,map_verdict.json}, tools/{map_sweep,map_overlay}.py.

## 18. Workstream 2: five-rung ladder (frozen LADDER_SPEC.md; zero new calls)

R1 nDCG@3 (recomputed) -> R2 UDCG-inspired@3 (frozen C3 judgments, signed
gain + discount + ideal norm; 8B primary) -> R3 Brier (frozen) -> R4
rerank positional VoI (REUSED frozen verdicts; per-system VoI beyond rerank
does not exist — stated limitation) -> R5 test dJ (frozen). R2 coverage
gate: 0 misses, 0 IDCG<=0 cases. Rankings over {bm25,dense,hybrid,rerank},
query bootstrap B=2000, strong reversals (CI-excl-0 both rungs, opposite
signs) vs weak (point flips).
Findings (theory/ladder_tables.json + ladder_verdict.json): ZERO strong
reversals anywhere — no CI-confirmed sign flip between adjacent rungs.
Levels diverge while ranks mostly hold: R1 spreads 0.068-0.114 but R2
saturates 0.974-0.998 (LLM judges qrel-irrelevant docs highly useful);
R2 annotator-dependence is itself severe (8B-vs-14B machine-utility rho
0.139). Within C1, R3 and R5 rankings are IDENTICAL
(rerank>hybrid>dense>bm25; tau35 CI [0.667,1.0]) — better retrieval orders
beliefs orders J monotonically; consumer-dependence lives BETWEEN consumers
(C1 all-negative J vs C0 mixed vs C3 rerank-positive), not within C1's
ladder. C0 shows the sharpest rung tension (R2-vs-R3 tau CI [-1.0,-0.333];
R3 rank dense-first vs R5 hybrid-first, weak flips only). C3 R3 nearly flat
(0.26-0.29, ranks shuffle) while R5 separates rerank (+7/+12) from the rest
(-26..-67) — belief quality does not resolve what J separates. R5 here are
plain-mean dJ (metric-consistent across rungs; balanced headlines differ,
referenced). Interpretive lens per spec: transmission/sensitivity x
alignment x sequential value (exploratory, labeled). V3 frozen matrix
untouched.
Files: theory/{LADDER_SPEC.md,ladder_tables.json,ladder_verdict.json},
tools/ladder.py.

## 19. Agentick spike: STOP on ceiling (GoToGoal-only, frozen AGENTICK_SPIKE.md)

Fixed adapter (egocentric greedy + fixed fallback; info['valid_actions']
only) x {ascii,language,state_dict} x easy/dense x seeds 0-4 + random floor
+ oracle normalization. ONS = (ret-random)/(oracle-random) per installed
scoring. Env: isolated py3.12 venv, agentick @github main 2026-09-20,
CPU-only (~40 episodes); env never committed.
Outcome: adapter return 1.0 in ALL modes x seeds (parse 100% everywhere);
oracle 1.0; random 0.6. ONS = 1.0 everywhere -> max-min = 0, H-primary
FAILS by stop rule. Mechanism: CEILING, not refutation — greedy goal
approach is optimal on open-room GoToGoal-easy from every representation,
so the spike is UNINFORMATIVE about the interface principle (cannot
distinguish no-effect from too-easy-task/adapter). No SokobanPush/
SequenceMemory; bounded external generality stands as: not yet tested
beyond a trivial task. A harder spike (medium difficulty and/or planning/
memory tasks where greedy is suboptimal) needs its own frozen design —
proposed, not executed. V3 frozen matrix untouched.
Files: theory/{AGENTICK_SPIKE.md,agentick_spike_raw.json,
agentick_spike_verdict.json}, tools/agentick_spike.py.

## 20. Agentick hard spike: STOP on floor-adjacent null (SokobanPush-only, frozen AGENTICK_HARD.md)

Fixed egocentric push adapter (push-if-aligned/reposition/approach/fixed
fallback; valid_actions only) x {ascii,language,state_dict} x easy/dense x
seeds 0-4 (all oracle-solved, no replacements) + random floor + oracle
normalization. Same EGO-mapping bug class as the GoToGoal script (missing
behind/here keys) caught pre-run by probe and fixed with a frozen tie-break
amendment (no episodes recorded before the fix).
Outcome: adapter ONS 0.2 in ALL modes (1/5 seeds each; different seeds:
ascii/state_dict seed0, language seed2); random 0/15; parse 100%
everywhere. Max-min ONS = 0 -> H-primary FAILS by stop rule. Mechanism:
FLOOR-ADJACENT, mirroring the GoToGuide ceiling from below — greedy
egocentric pushing rarely solves even single-box Sokoban regardless of
representation, so the spike is UNINFORMATIVE (weak adapter + n=5), not a
refutation of the interface principle. Per the frozen rule: stop external
validation ENTIRELY; no SequenceMemory. Agentick remains a bounded
null/ceiling result (GoToGoal ceiling + Sokoban floor). The external-
validity leg of the A* package is therefore carried, if at all, by future
work with a competent fixed policy — proposed, not executed. V3 untouched.
Files: theory/{AGENTICK_HARD.md,agentick_hard_raw.json,
agentick_hard_verdict.json}, tools/agentick_hard.py.

## 21. Two-track freeze: Step-0 autopsy + DESIGN-TRACKA + DESIGN-VHAT2 (no computation)

Step0 (STEP0_AUTOPSY.md): F1 residual dominance (98% after stratum+position;
query largest marginal 8.9%); F2 zero-inflation split (C0 71% exact-0 vs C3
~0%; tails carry 94% SS); F3 heteroscedastic conformal failure (row SD 74.5
vs cluster-mean 15.0; per-stratum Q needs 24-49 vs pooled 18.7); F4
threshold collapse (dev-90th -> test-95th; 97% non-selective drops);
F5 cross-stratum independence (same evidence, near-independent labels;
sign flips). Pooled-Q failure described as structural ON THESE DATA (no
general impossibility language).
Track A (DESIGN-TRACKA.md): internal-robustness matrix (3 confirmatory
contrast families w/ frozen interaction estimands, rest descriptive) +
Agentick dose-response (authored BFS, forbidden-reads checklist, k in
{full,3,1,0}, monotonic test, planner gate >=80% over 10 seeds, modest
causal-interface claim) + protocol packaging. Zero new LLM.
Track B (DESIGN-VHAT2.md): "cluster-conformal Vhat2 repair" (Mondrian
considered-and-rejected with n=13 math); factored GBM + position one-hots
+ Shat-on-dBhat; corrected one-sided cluster-max chain
(S_c=max(Vhat-V), finite-sample quantile, alpha=0.1); B2 = development-only
(coverage >=0.85, selectivity band, gated-J CI). Untouched bank is the only
confirmatory path. Bidirectional data/claim firewall frozen.
Files: theory/{STEP0_AUTOPSY,DESIGN-TRACKA,DESIGN-VHAT2}.md.

## 22. Track A Kaggle pilot VERIFIED + integrity finding + Track B2 outcome

(a) Kaggle CPU-sim pilot (BeliefBaseStock x bm25/rerank x C1-8B x 10 test
queries x 5 seeds = 100 eps) verified end-to-end after 5 kernel versions:
flat dataset attach (basename matching + package reconstruct), exact pinned
deps (numpy 1.26.4/pandas 2.3.3/pyarrow 21.0.0/scipy 1.13.1/sklearn 1.6.1;
stock resolver keeps numpy>=2 and stale sys.modules shadows reinstalls --
both fixed), spawn-safe top-level worker, self-describing-record hash
circularity fixed. Profit gate vs fresh local HEAD re-run (100 eps):
n=100 max_abs=4.55e-13 max_rel=3.66e-16 violations=0 (frozen tol 1e-6 abs +
1e-9 rel, never loosened). Cross-machine CPU-sim path PROVEN; scale-up
authorized on this pattern with per-shard 5% local-rerun sample gates.
GPU quota untouched (gpu=False throughout).
(b) INTEGRITY FINDING (pre-existing, frozen file untouched):
results/v3main/sim_controllerB/episodesB.parquet (committed at a3e2652)
appends reruns with disagreeing profits for identical keys (4296/7200 key-
groups differ by up to 621). Published utilityB.json groupby-means therefore
mix the runs. Flagged for the manuscript threats section; no frozen file
altered. Pilot gate deliberately compares against a fresh HEAD re-run, not
this file.
(c) Track B2 (vhat2, local): (i) PASS (pooled rho CI [0.094,0.191], 4/5
strata); (ii) PASS-vacuous (coverage 1.0 via max-residual Q=278.97);
(iii) FAIL (selectivity 1.0 outside [0.05,0.95]; pooled diff CI
[-6.685,3.453]). BUILD_BANK_PROPOSAL false. Per frozen DESIGN-VHAT2: no
untouched-bank confirmation is proposed; claim ceiling stays development-
only. The separated clauses worked as designed (validity pass did not mask
the vacuous-bound + no-utility failures).
(d) Agentick Kaggle probe: COMPATIBLE (git present, pinned source install,
GoToGoal x 3 modes + oracle run, 0.29 min). Dose-response offload viable;
kept local until pilot pattern proven (now proven).

## 23. Track A Kaggle scale-up VERIFIED (6 shards, 24k eps, CPU-only, GPU quota untouched)

Base-stock x {dense, hybrid, random, inject-contra, stale-swap, drop-decisive}
x 5 strata x k=3 x 160 test queries x 5 seeds (4000 eps/shard) via template-
generated kernels (tools/gen_tracka_shards.py) + per-shard JOBS entries +
seeded-5% local-rerun sample gates (seed 12345, min 50 rows, frozen tol).
All 6 shards: structural + manifest + identifier-coverage checks pass;
sample gates 200 rows each, max_abs ~1e-12, 0 violations everywhere.
Debugged en route (recorded, no tolerance touched): dataset version
propagation race (canary re-push), C0 twin rows across belief files
(mirrors load_beliefs_dedup; pilot unaffected, C1-only). Max 5 concurrent
CPU sessions observed (b6 queued behind) — the "maximum safe" bound.
Pilot (100/100 exact) + scale (1200/1200 sampled exact) = cross-machine
CPU-sim identity PROVEN at ~25k episodes.
