# NoInfo prior audit

| Phase | Code-executed NoInfo belief | Documentation/declared prior | Correction decision |
|---|---|---|---|
| Phase 6/7 | `REGIME_PRIOR = {normal:.35, supplier_delay:.35, demand_surge:.30}` | Historical documentation was inconsistent in places | Historical results retained; corrected audit uses balanced and this declared prior |
| Phase 8A/8B | `{normal:.70, demand_surge:.30}` from `get_no_info_probs` | 70/30 | Historical results retained; balanced and 70/30 estimates are both reported |
| Phase 9A/9B | `{normal:.50, supplier_capacity_drop:.50}` from `get_no_info_probs_9` | Code-executed uniform prior; some prose implied 70/30 | Historical results retain the uniform belief; corrected documentation now says so |

Changing a NoInfo belief changes controller actions and therefore requires new
episodes. No historical execution was silently changed and no new simulation
was launched for this correction. Where raw episodes suffice, the secondary
prior-weighted estimate is recomputed analytically.
