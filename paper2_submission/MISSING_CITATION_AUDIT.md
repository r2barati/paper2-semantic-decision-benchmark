# Missing-citation audit

Audit date: 2026-08-26. This is a focused audit, not a count-maximization exercise. Candidate works were classified by whether the manuscript needs them to support a comparison, not by age or prestige.

| Candidate neighborhood / primary work | Status | Reason |
|---|---|---|
| Zhao et al., *Calibrating Predictions to Decisions* (NeurIPS 2021) | MUST_CITE / present | Direct decision-calibration threat. |
| Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017) | MUST_CITE / present | Standard probabilistic calibration reference. |
| DeGroot & Fienberg, *The Comparison and Evaluation of Forecasters* (1983) | MUST_CITE / present | Foundational forecast evaluation. |
| Donti et al., task-based end-to-end learning (NeurIPS 2017) | MUST_CITE / present | Direct predict-through-downstream-task comparison. |
| Elmachtoub & Grigas, Smart Predict-then-Optimize (Management Science 2022) | MUST_CITE / present | Direct prescriptive-analytics comparison. |
| Mandi et al., decision-focused learning through learning-to-rank (ICML 2022) | MUST_CITE / present | Direct decision-focused learning comparison. |
| Farahmand et al., value-aware model learning (AISTATS 2017) | MUST_CITE / present | Closest learned-dynamics value-aware reference. |
| Lambert et al., objective mismatch in MBRL (L4DC 2020) | MUST_CITE / present | Direct prediction/control mismatch reference. |
| Hubbs et al., OR-Gym (arXiv 2020) | MUST_CITE / present | Benchmark lineage. |
| Li et al., OptiGuide / LLMs for Supply Chain Optimization (arXiv 2023) | USEFUL / present | Nearby LLM-plus-operations system; not a semantic sensing comparison. |
| Voelcker et al., calibrated value-aware model learning (ICML 2025) | USEFUL | Recent dynamics/calibration threat; not required for the paper's classifier-calibration claim. |
| Wilder et al., Melding the Data-Decisions Pipeline (NeurIPS 2019) | USEFUL | Decision-focused learning predecessor; add only if space permits. |
| Amos & Kolter, OptNet (ICML 2017) | BACKGROUND | Differentiable optimization lineage; not used by this method. |
| Mulamba et al., contrastive learning for decision-focused prediction (2021) | USEFUL | Direct decision-focused method family; covered conceptually by Mandi. |
| Kotary et al., end-to-end constrained optimization learning (2021) | BACKGROUND | Broad survey/lineage; no manuscript claim requires it. |
| Kuleshov et al., accurate uncertainties for deep learning (ICML 2018) | BACKGROUND | Calibration-adjacent but not needed for the reported TF-IDF result. |
| Nixon et al., measuring calibration in deep learning (CVPR 2019) | NOT_NEEDED | Vision-specific metric discussion does not support a manuscript comparison. |
| Ovadia et al., can you trust your model's uncertainty? (NeurIPS 2019) | NOT_NEEDED | Shifted uncertainty, not the paper's calibration-to-utility claim. |
| Calibration-under-shift follow-ups | USEFUL | Relevant adjacent evidence, but no current claim depends on it. |
| Value-of-information foundations | BACKGROUND | The manuscript states the relation without a theorem or historical priority claim. |
| Rubin, potential outcomes (1987) | BACKGROUND | Not needed for the fixed-controller descriptive estimand. |
| Recent supply-chain LLM reviews (2024--2026) | NOT_NEEDED | Reviews do not support a specific manuscript claim and would overstate coverage. |

## Result

No new bibliography entry is required. The existing direct neighborhood is covered with verified primary sources, and the manuscript's claim boundaries do not require citation inflation. The former approximate Quan entry was removed and replaced by the exact Li et al. OptiGuide preprint, which is a useful but secondary comparison.

