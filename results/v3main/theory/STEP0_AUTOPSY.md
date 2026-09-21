# STEP 0 — Vhat failure autopsy (FROZEN record, 2026-09-20)

Shared input to both tracks under the bidirectional firewall. All numbers
from non-mutating reads of vhat_oof.parquet (3000 rows), vhat_gate.json,
vhat_model.json (Q=18.71), g2full_proxy.json, g2full_stage4.json. Gate
outcome: (i) PASS barely (pooled rho CI [0.079,0.168]; C1-14B +0.002,
C1-8B -0.036); (ii) FAIL coverage 0.672 vs [0.85,0.95]; (iii) FAIL pooled
diff CI [-5.657,4.801], 2-3/5 strata. BUILD_BANK false.

## F1. Variance: residual dominates everything (data-noise-limit)

Var(VoI)=5577 (SD 74.7, mean 2.4, median 0). Eta-squared: stratum 0.016,
consumer 0.008, model 0.009, position 0.0006, cluster 0.042, query 0.089
(largest marginal). After stratum+position means, 98% remains. Cluster-mean
SD (15.1) is 5x smaller than instance SD — averaging hides task difficulty.
Stratum x position means show interaction only (C0 peak drop2 +20.6,
C1-14B peak drop1 +25.1, C1-8B all negative, C3 all ~=0).

## F2. Zero-inflation regime split (structural)

34.7% of labels exactly 0; L1==0 on 30.2% (all VoI==0, model predicts mean
3.04 there — bias). frac(VoI==0): C0 0.708 / C1 ~0.44 / C3 ~0.01. Tails
carry variance: |VoI|>=50 is 30% rows but 94% SS. C3 is dense low-ceiling
noise (all 6 cells cross 0); signed-VoI ceiling ~= 0 there.

## F3. Heteroscedastic conformal failure (structural)

Row resid SD 74.5 vs cluster-mean 15.0. Per-stratum coverage of pooled Q:
C0 0.86 / C1-14B 0.85 / C1-8B 0.71 / C3 0.48-0.54. Needed per-stratum Q0.9:
24.7 / 49.4 / 26.2 / 29.5 / 27.5 vs pooled 18.7. A single pooled
cluster-mean quantile cannot cover row-level heavy-tailed VoI. (Describes
the frozen pooled-Q construction's structural failure ON THESE DATA only;
no general impossibility claimed.)

## F4. Threshold-collapse mechanism (design-threshold-choice)

Frozen agreeXmodel_L1>=0.6 sat at dev 90th pct but test ~95th+ pct;
transmit 12.5% -> 5.0% (8/160). Other observables do NOT shift
(entropy/margin/maxProb/priorL1 medians identical) — fragility is tail
choice, and Vhat's own features are distributionally stable (dev rho 0.136
vs test 0.118). Single-min-drop with huge Q dropped 97% of queries:
non-selective gating whose stratum signs cancel (pooled CI crosses 0).

## F5. Cross-stratum independence (structural)

Same query+position VoI correlations: C1 pair 0.33, C3 pair 0.57,
cross-consumer 0.05-0.25. Same evidence, different consumer ->
near-independent labels; C1-14B mean +14.6 vs C1-8B -6.7. One-hot pooled
GBM can shift intercepts, not sign-flipped mappings (OLS slopes include a
wrong-signed one). corr(L1,|VoI|): C0 0.97 / C1 0.83-0.87 / C3 0.38-0.39
(magnitude predictable except C3); signed corr(L1,VoI) flips sign by
stratum. Even perfect dBhat leaves Shat sign noise.

## Use constraints (firewall)

Both tracks may cite this autopsy. Neither track may re-tune on it beyond
their frozen designs: Track A uses it for interpretation only; Track B's
v2 skeleton (DESIGN-VHAT2.md) is the SOLE licensed response, itself frozen
before execution. Post-freeze observations -> future-work only.
