# Future Work

## Experimental Extensions (not implemented)

- **Stochastic lead times:** Evaluate how semantic information value changes when lead times are themselves uncertain
- **Multi-SKU:** Test across product families with substitution effects
- **Real-world text:** Validate with actual supplier emails, news feeds, or regulatory filings
- **Additional LLMs:** Evaluate open-source models (Llama, Mistral), domain-specific models
- **Continuous disruption severity:** Move beyond binary regime classification to severity estimation
- **Co-optimized controller:** Design controllers that jointly optimize with the interpreter
- **Competing system designs:** Benchmark different architectural choices, not just interpreter quality
- **Adaptive controller:** Controller that adjusts strategy based on observed belief quality
- **Larger seed sweeps:** Tighter confidence intervals for small effect sizes
- **Web/news data:** Real-time semantic sensing from public information channels

## Theoretical Extensions

- **Information-theoretic bounds:** Formalize the relationship between mutual information and SIVR
- **Regret analysis:** Sample complexity bounds for semantic-value estimation
- **Calibration theory:** Formal conditions under which calibration improves SIVR
- **Controller sensitivity:** General conditions for SIVR > 1 (imperfect beliefs outperforming oracle)
