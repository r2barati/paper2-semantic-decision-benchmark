# V3 sequential-utility results (frozen beliefs → frozen controller → controlled sim)

Source: `results/v3main/sim/` (driver `tools/run_sim_v3main.py`). 108,000
episodes: 105 belief arms × 200 queries × 5 paired fresh seeds (60000–60004,
disjoint from 2000s/3100s/5000s ranges) + NoInfo/PerfectBelief/HindsightOracle
rungs. Estimand: balanced weighted return J (regime → main → query → seed,
uniform regime weights). Paired Δ vs NoInfo with crossed bootstrap CIs
(B=5,000, floor p = 1/5001 ≈ 0.0002; p at floor reported as <0.0002).
SIVR vs two references: PerfectBelief (sivr_*) and matched OracleFactual
(sivrF_*), signed guard throughout (all denominators positive → all VALID).

Reference returns: J(NoInfo) = 1779.04, J(PerfectBelief) = 1994.40
(OIV = +215.36), J(HindsightOracle) = 2451.20.

## Headline: real retrieval hurts, oracle evidence helps

Every LLM-consumer real-retrieval arm sits BELOW NoInfo with Holm-robust
margins (C1: −35 to −109; C3: −30 to −61 where significant); C0 comparisons
are directionally negative but Holm-n.s. Random is also ≤ NoInfo except the
tiny isolated 14B-C3-k3 (+11.0, Holm-robust but de-emphasized: sole positive
among 30 real cells, mechanistically bare).
Oracle evidence helps substantially in all LLM consumers (+71 to +178,
Holm-robust; oracle-C0 +38 Holm-n.s.).

## Utility table (k=3; Δ vs NoInfo, 95% CI)

Stars = Holm-0.05 rejection (105-arm family); "ns" = Holm-n.s. Text-cluster
CIs agree in direction throughout (median 1.08× wider).

| system | C0 | C1/8B | C1/14B | C3/8B | C3/14B |
|---|---|---|---|---|---|
| bm25 | -11.8ns | -109.4∗ | -97.3∗ | -48.9∗ | -60.7∗ |
| dense | -9.4ns | -97.8∗ | -96.7∗ | -30.2ns | -55.7∗ |
| hybrid | -14.1ns | -89.3∗ | -86.0∗ | -48.6∗ | -60.6∗ |
| rerank | -37.3ns | -42.8∗ | -38.6∗ | -21.4ns | -16.9ns |
| random | -32.0ns | -88.5∗ | -89.5∗ | -13.5ns | +11.0∗ |
| oracle | +38.2ns | +75.9∗ | +70.9∗ | +178.4∗ | +164.8∗ |

Only LLM-consumer arms (plus the tiny isolated random/C3/14B +11.0, kept
out of headlines) carry Holm-robust utility conclusions; every C0 comparison
is suggestive at best. k=5 rerank: C1 arms ∗ (−34.7/−43.8), C0/C3 ns.

Two orderings hold simultaneously: (a) among real retrievers, better IR is
LESS harmful (rerank > hybrid > dense ≈ bm25 for LLM consumers — but still
harmful or neutral); (b) only oracle evidence crosses into positive value.
The controller is not robust to retrieved evidence — it is actively
misled by it, while the tuned no-text prior (NoInfo = 1779) is a strong
operating point.

## SIVR (both references)

Vs PerfectBelief: real arms −0.04 to −0.51; oracle arms +0.18 (C0) to
+0.83 (C3/8B). Vs matched OracleFactual: real arms −0.08 to −1.44
(C1/bm25/8B recovers nothing of the factual-evidence value and then some —
ratio < −1 means moving away from the reference by more than the full OIV).
All statuses VALID (positive denominators); negative values are the finding.

## Four-gap decomposition (k=3 headline)

| arm | retrieval | relevance→factual | interpretation | control |
|---|---|---|---|---|
| C0 | -37.29 | 0.00 | +177.13 | +456.80 |
| C1/8B | -42.84 | 0.00 | +139.42 | +456.80 |
| C3/8B | -21.41 | 0.00 | +36.94 | +456.80 |
| C1/14B | -38.62 | 0.00 | +144.43 | +456.80 |
| C3/14B | -16.91 | 0.00 | +50.54 | +456.80 |

- retrieval_gap < 0: actual retrieval destroys value vs no retrieval.
- relevance→factual = 0.00: the factual filter adds nothing downstream —
  both oracle sets saturate their consumer identically (J 1817 C0 → 1957 C3).
- interpretation_gap shrinks C1 (∼140) → C3 (∼37–51): C3 closes most of the
  consumer gap on oracle evidence.
- downstream_control_gap (+456.8) dominates everything: even perfect beliefs
  leave more than twice the perfect-semantics OIV on the table — the
  controller, not the information, is the largest loss source.

## Reading (reported, not tuned-for)

The frozen experiment produced outcome class "retrieval harms; oracle helps;
control dominates": evidence consumption is not the main bottleneck — the
evidence itself (real retrieval) is harmful under this controller, while the
controller leaves the largest gains unrealized even with perfect semantics.
C3 over C1 is a partial remedy (−21 vs −43), not a rescue.
