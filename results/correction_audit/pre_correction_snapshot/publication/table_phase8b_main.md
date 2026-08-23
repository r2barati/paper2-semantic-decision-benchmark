# Table: Phase-8B — Paper-1 Gymnasium Confirmation Results

## Main Results

| Sensor | AggSIVR | Mean Reward | Fill Rate | Belief Acc | Brier | Hier. 95% CI (vs NoInfo) |
|--------|--------:|------------:|----------:|-----------:|------:|--------------------------|
| RuleBased | 1.198 | 568.7 | 0.947 | 0.792 | 0.1615 | [54.3, 148.1] sig |
| OracleSemantic | 1.000 | 552.0 | 0.920 | 1.000 | 0.0000 |  |
| TFIDF_LogReg_Calibrated | 0.841 | 538.6 | 0.911 | 0.958 | 0.0463 | [25.8, 117.8] sig |
| gpt-4o | 0.797 | 534.9 | 0.906 | 0.958 | 0.0332 | [18.4, 118.4] sig |
| TFIDF_LogReg_Raw | 0.575 | 516.2 | 0.901 | 0.917 | 0.1812 | [26.7, 70.9] sig |
| NoInfo | 0.000 | 467.7 | 0.858 | 0.500 | 0.2900 |  |

## Regime Breakdown

| Regime | Sensor | Mean Reward | Fill Rate | SIVR |
|--------|--------|------------:|----------:|-----:|
| demand_surge | RuleBased | 664.3 | 0.899 | 1.000 |
| demand_surge | OracleSemantic | 664.3 | 0.899 | 1.000 |
| demand_surge | gpt-4o | 629.1 | 0.870 | 0.837 |
| demand_surge | TFIDF_LogReg_Calibrated | 622.0 | 0.863 | 0.803 |
| demand_surge | TFIDF_LogReg_Raw | 550.2 | 0.801 | 0.471 |
| demand_surge | NoInfo | 449.8 | 0.717 | 0.000 |
| normal | NoInfo | 485.7 | 0.999 | 0.000 |
| normal | TFIDF_LogReg_Raw | 482.2 | 1.000 | -0.167 |
| normal | RuleBased | 473.1 | 0.995 | -0.490 |
| normal | TFIDF_LogReg_Calibrated | 455.2 | 0.960 | -0.502 |
| normal | gpt-4o | 440.6 | 0.942 | -0.836 |
| normal | OracleSemantic | 439.6 | 0.941 | -0.867 |

## Win Rates vs NoInfo

| Sensor | Win Rate | Tie Rate | Pairs |
|--------|---------:|---------:|------:|
| RuleBased | 0.517 | 0.000 | 720 |
| TFIDF_LogReg_Raw | 0.517 | 0.000 | 720 |
| TFIDF_LogReg_Calibrated | 0.603 | 0.000 | 720 |
| gpt-4o | 0.493 | 0.000 | 720 |
| OracleSemantic | 0.533 | 0.000 | 720 |
