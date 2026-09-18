# V3 gym transfer (SECONDARY external validation, gym-invmgmt==0.2.1)

Source: `results/v3main/gym/` (driver `tools/run_gym_v3main.py`). 107,000
episodes, same 5 paired seeds. J(NoInfo) = 375.01. 2-class projection caveat
applies throughout (demand_surge → surge env; normal/supplier_delay → normal
env; supplier_delay belief mass unread by the controller).

## Result: NoInfo wins everywhere — including against PerfectBelief

Every arm, including PerfectBelief (−50.7, CI [−61.8, −41.0]), sits below
NoInfo (all p < 0.0002). The gym base-stock controller over-orders on any
surge belief, so the gym world cannot discriminate information value at all:
it only confirms that the no-text prior is a robust operating point — the
same qualitative NoInfo-strength seen in the controlled sim.

## Rankings do not transfer across environments (descriptive)

Controlled k=3 harm ranking (least→most): C0/rerank ≈ C3/rerank < dense <
hybrid < bm25 < C1/* (CIs overlap in places). Gym k=3 harm ranking:
random/C3/14B (+20.8, sole positive) > C1/* > oracle/C3 (−43) > dense >
bm25 > hybrid > rerank (−240). The best-controlled real system (rerank) is
the worst gym system. Environment dependence of system rankings is itself a
transfer finding: no retriever is universally safe.

## Verdict for the paper

Gym corroborates the primary result directionally (no retrieval system beats
the no-text operating point in either world) and adds ranking-instability
across environments. It is kept secondary: the projection limitation plus
the PerfectBelief<NoInfo artifact mean gym J-differences are not
information-value estimates.
