# DEV pilot report — EXP-P2-REASONING-AGENTIC-v1

Diagnostic only: no arm selection, tuning, or performance claims.

## Operational means

            arm  mean_profit  delta_vs_NoInfo  delta_vs_A0     sivr sivr_status  win_frac_vs_NoInfo  win_frac_vs_A0  fp_n  fn_n  fp_cost  fn_cost
             A0  1727.698333        50.813889     0.000000 0.192587       VALID            0.677778        0.000000     0    30      0.0   2346.0
             A1  1702.642222        25.757778   -25.056111 0.097623       VALID            0.238889        0.344444     0    75      0.0      0.0
             A2  1710.545556        33.661111   -17.152778 0.127577       VALID            0.383333        0.205556     0    45      0.0      0.0
             R0  1734.112222        57.227778     6.413889 0.216896       VALID            0.683333        0.450000     0    30      0.0      0.0
             A3  1718.652500        41.768056    -9.045833 0.158303       VALID            0.622222        0.366667    15    15    112.0   1162.6
         NoInfo  1676.884444         0.000000   -50.813889 0.000000       VALID            0.000000        0.238889     0   120      0.0      0.0
      RuleBased  1876.372540       199.488095   148.674206 0.756069       VALID            0.700000        0.716667     0     0      0.0      0.0
PerfectSemantic  1940.733333       263.848889   213.035000 1.000000       VALID            0.977778        0.794444     0     0      0.0      0.0

## Headline contrasts (diagnostic)

contrast  dBrier_mean  dBrier_improve_mean                                     dBrier_ci  dProfit_mean                        dProfit_crossed_ci  dProfit_win_frac  dProfit_cohens_d  dProfit_normal  dProfit_supplier_delay  dProfit_demand_surge
   A1-A0     0.203333            -0.203333  [-0.37499999999999994, -0.00623958333333337]    -25.056111    [-75.6590833333332, 26.35494444444446]          0.344444         -0.261049      -50.513333               24.961667            -49.616667
   A2-A1    -0.044167             0.044167 [-1.850371707708594e-17, 0.11333333333333329]      7.903333 [-18.530513888888844, 39.019541666666704]          0.188889          0.144994        0.000000              -29.570000             53.280000
   R0-A2    -0.142500             0.142500   [0.0025000000000000183, 0.2991666666666667]     23.566667 [-10.505527777777809, 60.977902777777736]          0.550000          0.333700       60.103333               28.190000            -17.593333
   A3-R0     0.011462            -0.011462          [-0.1005315625, 0.06604406249999996]    -15.459722  [-39.36812361111104, 7.2954999999999695]          0.127778         -0.296309      -16.725833                0.000000            -29.653333

## Dead-zone cells

cell
sem_down_harmful             255
sem_down_unchanged           225
sem_up_act_change_util_up    154
sem_up_harmful_act            56
sem_up_no_act_change          30

Theorem gate G (pathwise-equal => |dProfit|<=1e-9): PASS

## Agency (A3 vs R0)

               q2_used  action_changed
warning_id                            
cd_clear_1       False           False
cd_moderate_1    False           False
cd_moderate_2    False           False
cd_vague_1       False           False
cn_clear_1       False            True
cn_moderate_1    False            True
cn_moderate_2    False            True
cn_vague_1       False           False
cs_clear_1       False            True
cs_moderate_1    False            True
cs_moderate_2    False            True
cs_vague_1       False            True

## Variance

contrast  warning_var   seed_var  interaction_var  identical_belief_frac  identical_action_frac
   A1-A0  8101.686670 369.016850      1390.465955               0.166667               0.166667
   A2-A1  2857.436117  37.734730       300.078356               0.666667               0.666667
   R0-A2  4008.753883 247.606706      1054.010348               0.166667               0.166667
   A3-R0  1480.976223 254.188747      1112.230856               0.250000               0.416667

## Viability gates (PASS/WARN/FAIL)
- A. Parse reliability: PASS (0 failures / 84 GPU calls; <1% target).
- B. Arm non-degeneracy: PASS (identical-belief fraction 0.17 A1-A0 / 0.67 A2-A1 / 0.17 R0-A2 / 0.25 A3-R0; verification often confirms, which is itself informative).
- C. Controller sensitivity: PASS (action-change fraction 0.33-0.83 across contrasts; boundaries crossed often enough to study).
- D. Agent activity: WARN (query_2 invoked 0/12 warnings; second-query machinery verified in harness, but A3-R0 on DEV measures query-1 formulation only).
- E. Retrieval validity: PASS (72/72 retrieved docs from frozen corpus, text hashes match).
- F. Statistical resolution: WARN (DEV-12 crossed CIs span ~±35-50 profit units; final freeze should scale warnings up to resolve arm-level deltas of this magnitude).
- G. Dead-zone theorem: PASS (0/720 pathwise-identical pairs violate |dProfit|<=1e-9).

## Protocol correction during pilot (versioned, pre-freeze)
`experiments/reasoning_agentic/run_simulation.py` passed `controller="optimizer"`
instead of the frozen `CONTROLLER_OPTIMIZER="CausalOptimizer"`, silently routing
all arms through the belief-blind heuristic (first pilot run: all profits
identical). Caught by the pilot's own sensitivity checks; fixed to import and
pass the frozen constant. No frozen file touched; no GPU re-inference needed
(beliefs unchanged); local smoke + GPU-smoke replay re-verified after the fix.
This is exactly the class of defect the DEV pilot exists to catch, and it was
caught before any freeze.

## Interpretation (no selection, no claims)
Semantic and operational deltas dissociate on DEV (e.g. A1-A0: Brier improves
-0.20 while profit deltas scatter across dead-zone cells). Weak R0/A3 vs
expectations is admissible per the predeclared warning/corpus mismatch.
