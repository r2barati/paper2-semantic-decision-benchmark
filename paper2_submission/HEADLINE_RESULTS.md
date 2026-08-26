# Headline results

## Main-text results

1. **Accuracy/value dissociation:** Phase 7 raw TF-IDF accuracy is 0.889 versus
   0.861 for calibrated TF-IDF, while Brier is 0.552 versus 0.240 and log-loss
   is 0.932 versus 0.397; SIVR is 0.250 versus 0.648.
2. **Held-out operational transfer:** Phase 8B uses 30 seeds, 24 held-out
   templates, and 4,320 episodes. Reward deltas versus NoInfo are +101.0,
   +48.5, +70.8, and +67.1 for RuleBased, raw TF-IDF, calibrated TF-IDF,
   and GPT-4o, respectively, with the frozen hierarchical intervals.
3. **Mechanistic controller dependence:** the historical Phase-4 matrix shows
   semantic value under the heuristic controller but near-zero value under the
   stronger CausalOptimizer. This is supporting evidence, not a universal
   controller theorem.
4. **Supply-side transfer:** Phase 9A uses a supplier-capacity-drop shock;
   RuleBased SIVR is 0.823 and positive reward effects are reported. TF-IDF
   values include training templates and are not clean linguistic OOD evidence.
5. **Boundary behavior:** Phase 9B is descriptive. Low/high noise variants
   are stable; long-lead and lost-sales variants produce negative oracle OIV
   because the fixed controller is mismatched.

## Secondary/appendix results

The rule-based baseline, GPT extraction behavior, threshold sensitivity, full
episode counts, phase history, and all SIVR edge cases belong in tables or the
appendix. The paper should not become a phase-by-phase leaderboard.
