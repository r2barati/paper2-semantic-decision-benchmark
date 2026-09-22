# Claim–evidence ledger

One row per headline claim: number + CI/p → result file → generator →
inputs/seeds → stats → gate → integrity flag → citable now? Sources:
`SCIENTIFIC_LEDGER.md` (R/C/M/S; B5000, floor 1/(B+1)=0.0002, Holm, crossed
bootstrap, signed SIVR, balanced `J_w` primary) and
`results/v3main/delivery/LEDGER.md` (V3; 105-arm Holm, B5000/10000).

## Core benchmark claims

| Claim | Number | Result file | Generator | Inputs / seeds | Stats | Gate / flag | Citable? |
|---|---|---|---|---|---|---|---|
| R: same ranking flips sign by consumer (k3, all four rankers) | Rule +11.6/+14.9/+25.8/+48.4 vs Cal −49.8/−74.3/−61.1/−1.5; interaction +49.9–+89.2 | `results/retrieval/retrieval_summary.csv:8-13,26-31` | `src/experiment_retrieval.py` + `retrieval_analysis.py:interaction` (B2000, no Holm) | per-episode pools; 5000–5029; 3240 eps | Paired deltas; interaction has no CI/p | Post-hoc design; synthetic relevance; superseded as headline in `main.tex` | Exploratory mismatch only |
| R: oracle helps both (evidence exists) | +60.6 [+48,+73] / +69.0 [+40,+96], p0.0005 | Same, oracle rows | Same | Same | CIs exclude 0 | Same limits | Exploratory positive control |
| R: rank-reversal 0.00 vs 0.33 | Rule 0/6 tau+1.00; Cal 2/6 tau+0.33 | `results/retrieval/rank_agreement.csv:3,6` | `retrieval_analysis.py:rank_reversal_rate` | n=4 systems / 6 pairs | Deterministic, no uncertainty | Same limits | Exploratory only |
| C: Rule held-out value | +205.74 [+161,+245] p<0.0002 Holm0.0022 SIVR 0.779 | `SCIENTIFIC_LEDGER.md:20` ← `results/phase7_classical_baseline/aggregate_sivr.csv` | `src/experiment_phase7.py` + `generate_scientific_ledger.py` | train18→test36; 2000–2019; 8640 eps | Crossed B5000 Holm | Held-out; fixed-controller/synthetic/3-family limits | Yes, with limits |
| C: Argmax 0.773 > Cal 0.338 | +204.26 [+85,+275] Holm0.0088 vs +89.35 [+13,+159] Holm0.035 | `SCIENTIFIC_LEDGER.md:21,24` | Same | Same | CI width 190; Brier/logloss disagree | Post-hoc ablation | Hypothesis only |
| M: Rule transfer | +100.97 [+95,+106] Holm0.0022 SIVR 1.198 | `SCIENTIFIC_LEDGER.md:38` ← `results/phase8b_gym_confirmation/aggregate_sivr.csv` | `src/experiment_phase8b.py` + ledger | 24 held-out; 3100–3129 disjoint; 8640 eps | Crossed B5000 Holm, both estimands | SIVR>1 = reference disclaimer | Yes, with limits |
| M: Const/Tuned 1.10 beats Oracle; gpt-4o 0.797 (prior n.s.) | +92.70 Holm0.0022; 4o +67.12 Holm0.008 balanced, +22.25 Holm0.12 prior n.s. | `SCIENTIFIC_LEDGER.md:39-42` | Same | dev 3200–3229 (disjoint) | Prior-dependence disclosed | Post-hoc arms; 50 unknown cache rows | Yes as comparator + limits |
| S: Const dominance | +391.53 [+387,+396] Holm0.0022 SIVR 3.473 | `SCIENTIFIC_LEDGER.md:57-58` | `src/experiment_phase9a.py` + ledger | 16 templates; 4000–4029 | Holm-robust | Text-free unaffected by leakage | Yes as comparator; falsifies broad transfer |
| B: 3 hold / 3 fail | hold +97–+116 VALID; long/lost/short OIV −3/−22/−183 † | `results/phase9b_robustness/cross_variant_comparison.csv`, `corrected_phase9b_boundary.md` | `src/experiment_phase9b.py` + `recompute_corrections.py` | 10 seeds 4100–4109; 7680 eps | Points only; descriptive OFAT | † + signed guard required | Descriptive only |

## Stress-tested claims (adversarial audit → verification verdict)

| Overstated form | Verified verdict | Why |
|---|---|---|
| V3 second headline-scale contribution | DISPROVED as stated | Scale executed (108k sim + 107k gym + 105-arm Holm) but human-qrel gate open per `FINAL_ASSESSMENT` + `annotation/` absent. Headline-scale executed analysis with submission-conditional gate. |
| V3 scale exceeds R in weight | DISPROVED | V3 broader (LLM consumers, ladder, gym, Holm); R cleaner (deterministic, no 1.9% nondeterminism, no model-amendment deviation). Scope > R; unconditional citability not > R. |
| C+M jointly rival R | DISPROVED as confirmatory rivalry | C-Rule + M-Rule/Const Holm-robust and held-out, but degraded arms + controls post-hoc (`39e311e`); synthetic + fixed-controller + `IN REPAIR` limits. Rival as diagnosis, not headline. |
| G2FULL supports decision-aware evaluation | NARROWED | (a) +21.94 Holm0.0 and (b) p0.0002 support consumer-dependent doc-VoI (proper prereg `81b4cb4→4f22378`, dev→test replication). (c) −0.61 FAILS + proxy FAILS + VHAT no-bank = deployable method fails. Existence, not method. File: `extension_g2a/g2full_stage4.json` (p=0.0 invalid; CIs valid). |
| D0 +77.3 usable | NARROWED | CI [62.8,92.0] survives as oracle-gated existence-proof (dev-tune→single-touch). p=0.0 invalid; plain-mean deviates balanced; Brier undeployable. Not a policy. File: `extension_d0/d0_verdict.json`. |
| Controller-B reversals | DISPROVED for numbers | `LEDGER §22b`: 4296/7200 groups differ ≤621, means mix runs. All `utilityB.json` deltas/CIs/p unusable until rebuild. Qualitative pairing-dependence = hypothesis only. |
| Agentick dose real result | NOT YET PROVEN | Verdict 7 lines lacks CIs/numbers/seeds; raw suggests monotone; env unpinned `@main`. `COMPUTED BUT NOT YET VALIDATED`. File: `theory/agentick_dose_verdict.json` + `agentick_dose_raw.json`. |
| Dead-zone refutation publication-ready | DISPROVED as confirmatory | Theorem proved conditional; H1–H3 same-commit spec with tight wrong-direction CIs = genuine but prereg-weak bounded nulls. Refined hypothesis design-only. Files: `theory/DEAD_ZONES.md`, `map_verdict.json`. |
| TrackA robustness evidence | DISPROVED | 24.1k gates PASS proves scale-up + cross-machine identity only. Zero A1/A2/A3 verdicts. Robustness claims design-only. Files: `runs/v3main_tracka/manifest_tracka_*.json`. |
| V3 Q1–Q7 completed not exploratory | DISPROVED in part | Q2/Q3-cells/Q4-artifact/Q5-pattern/sim-headlines confirmatory-with-gate; Q1 directional (n=4 Holm-n.s.), Q6 parity (nulls retained, Qwen-only), Q7 wrong-signed descriptive remain exploratory per `FINAL_ASSESSMENT`. |
| Human-qrel only IR-validity | DISPROVED | Qrels feed HIT (Q4), R1/R2 ladder, S-nDCG selection, retrieval table. Synthetic-by-construction relevance + R2 saturation 0.998 + rho 0.139 condition utility contrasts. Gate blocks submission, not just IR appendix. |

## Safe / exploratory / unusable split (pointer)

- **Safe now (with limits):** R exploratory mismatch + oracle control; C Rule + Cal direction; M Rule/Const/Oracle/4o (balanced); S Const comparator; V3 retrieval test, Q2, Q3 cells, Q4 artifact, Q5 pattern, sim C1-harm + oracle-value, gaps; G2A-0 positional + C3 flat; G2A-1 dev FIRE; G2FULL (a)(b) + (c) as null; D0 pooled existence; dead-zone theorem; A0 fidelity; TrackA infra.
- **Exploratory (retain, never confirmatory):** R reversal/corr; S-test-only; B map; V3 Q1/Q6/Q7, gym secondary, n.s. cells; selective; A2 dissipation; ladder; VHAT/VHAT2 nulls; spikes bounds.
- **Not usable:** S all-template linguistic claim (train-contaminated); Controller-B numbers; p=0.0 entries; B ratios without †; TrackA robustness / VHAT bank / Gate-2 / human n300 (design-only/blocked); reasoning A0–A3 (smoke FAIL 0/6); Phase5/5.5/6 original tables, 8A ranking, v1.0 tables (superseded numbers; phenomena live in corrected tables).
