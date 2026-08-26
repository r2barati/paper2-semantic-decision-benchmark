# Why Paper 2 is not Zhao et al. (2021)

Zhao et al., *Calibrating Predictions to Decisions: A Novel Approach to Multi-Class Calibration* (NeurIPS 2021), is a serious direct predecessor. It defines decision calibration so that predicted and true class distributions are indistinguishable to a bounded downstream decision-maker, proves an efficient recalibration result under a bounded-action restriction, and evaluates it on static image-classification decisions.

| Dimension | Zhao et al. | Paper 2 |
|---|---|---|
| Prediction object | Class-probability vector | Belief over an operational disruption/regime from warning text |
| Calibration object | Distribution indistinguishability for downstream decision-makers | Empirical comparison of raw/calibrated beliefs after a fixed interpreter/controller interface |
| Decision maker | Bounded static action set and loss | Inventory controller acting repeatedly over a finite horizon |
| Dynamics | No state-transition environment | Inventory state transitions, lead times, backlog/shortage and holding consequences |
| Horizon | Static decision | Sequential trajectory utility |
| Costs | General decision losses | Asymmetric operational costs induced by the simulator/controller |
| Shift protocol | Image datasets and model predictions | Held-out warning templates, multi-echelon transfer, supply-side event and boundary variants |
| Semantics | Class labels | Textual event descriptions and regime beliefs |
| Theory | Decision-calibration definition, characterization, and efficient recalibration | No new calibration theorem; empirical mechanism and evaluation protocol |
| Main question | Can calibration be made decision-relevant efficiently? | When does semantic interpretation quality translate into sequential operational value? |

## Exact contribution boundary

Paper 2 does not introduce a stronger calibration definition, a recalibration algorithm, or a general sequential decision-calibration guarantee. Zhao et al. therefore limits any novelty claim about “decision-aware calibration.” Paper 2's remaining contribution is narrower: in a controlled sequential operational interface, it measures how classifier accuracy, probability calibration, controller behavior, event mechanism, and trajectory utility can dissociate, and it tests that dissociation under held-out wording and a richer inventory system. The result is empirical and framework-specific, not a replacement for Zhao et al.'s theory.

## What Paper 2 teaches that Zhao et al. does not

It reports evidence about a repeated-action controller and state-changing environment: a calibrated belief can produce higher trajectory value despite lower classification accuracy, the same semantic resources can have different value under different controllers, and a negative normalized value can arise when the perfect-semantic reference itself is mismatched to the environment. These are bounded observations from the frozen benchmark, not claims that the NeurIPS calibration theory is insufficient.

