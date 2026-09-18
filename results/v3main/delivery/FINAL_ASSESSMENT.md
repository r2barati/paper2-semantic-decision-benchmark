# Final assessment: hypotheses, conclusion, critique, recommendation

## Preregistered hypothesis assessments (Q1–Q7 + paper-level)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| Q1 | Better retrieval → better semantic beliefs | Mixed | Direction for C1/C3 (rho 0.8–1.0, exact p 0.083–0.92, n=4: significance never claimed), flat for C0 |
| Q2 | Retriever ranking stable across C0/C1/C3 | Supported | rho 0.80–1.00, all p < 0.003 |
| Q3 | Retriever × Consumer interaction | Supported | Headline Holm 3/10 reject (C0→C1/8B, C0→C3 both models); C1→C3 ≈ 0 — interaction is C0-vs-LLM |
| Q4 | Evidence-hit explains semantic quality | Supported | +0.36–0.42 acc, all CIs exclude zero |
| Q5 | C3 recovers what C1 misses | Mixed | +0.25 on oracle, ≈0 on real retrieval, abstains on random |
| Q6 | Findings replicate 8B → 14B | Supported | 34/42 cells null (max |Δ| 0.144); direction mixed; parity, not superiority |
| Q7 | Generalization to held-out entities | Unsupported | Wrong-signed (held-out higher); mix effect suspected |
| P1 | Retrieval gains transfer to utility | Unsupported | Sign negative for every real arm |
| P2 | Transfer depends on consumer | Supported | C1 −109 vs C3 −21 vs C0 −12 (same evidence, bm25/8B) |
| P3 | Oracle decomposition ordered | Mixed | retrieval<0, rel→fact=0, interpretation shrinks, control dominates |

## Strongest defensible scientific conclusion

In a frozen sequential-decision benchmark where only the belief source
varies, real retrieval systems (BM25 → dense → hybrid → rerank, 0.115–0.183
nDCG) systematically reduce operating return relative to a tuned no-text
prior (Δ −9…−109, 71/105 Holm rejections), while oracle-selected evidence
from the same corpus adds up to +178. The loss is therefore mismatch, not
scarcity: the evidence the controller needs is extractable (oracle ladder),
retrieval ranking orders it poorly for control (rankings flip across
environments), and consumers cannot compensate (C3 ≈ C1 on real retrieval;
14B ≈ 8B). The largest remaining gap is downstream control (+456.8 even with
perfect semantics). A no-text operating point wins in two worlds
(controlled + gym) — including against perfect beliefs in gym.

## Adversarial ECIR-reviewer critique (and answers)

1. *"Negative utility just means your controller is bad."* — Agreed in part:
the control gap (+456.8) is the largest loss and is reported first-class.
But controller quality cannot explain the oracle ladder (same controller
gains +178 on oracle evidence) or the R×C interaction sign pattern. The
claim is scoped to this controller class, stated in threats.
2. *"Only 160 test queries / 4 systems for rank claims."* — Q1 is flagged
small-n by design (n=4 systems); the query-level CIs (n=160) carry the
weight, and Holm is applied. More queries would strengthen, not change.
3. *"Qwen-only; no frontier lab model."* — True and disclosed: GPT-4o was
blocked on billing, Phi-4 failed the schema gate; the 8B×14B check is
within-family. The core findings do not depend on model identity (C0 alone
shows the harm pattern without any LLM).
4. *"Simulated inventory, not real operations."* — True; gym corroboration
is provided with its own artifact disclosed. The contribution is the
evaluation pattern (retrieval→belief→control with frozen interfaces),
reusable wherever a controller and corpus exist.
5. *"Held-out result undermines generalization claims."* — We make none;
the wrong-signed held-out gap is reported as a limitation with a suspected
cause, and generalization is listed as future work requiring new banks.

## Recommendation: SUBMIT (ECIR, short/negative-result track if available)

The result is surprising, robust (frozen pipeline, both SIVR references,
sensitivity-bounded, Holm-controlled), and actionable (relevance-only
retrieval evaluation misses control-relevant quality; no-text priors deserve
benchmark status). Conditions before submission: blind human qrel validation
gate (protocol ready, unexecuted), one full independent rerun of
`tools/verify_release`-equivalent for v3 (fetch-gate recomputation already
covers kernels; rerun the analysis drivers from a clean checkout), and a
prose pass scoping every claim exactly as this ledger does. No further
tuning or data collection is required — or permitted — under the freeze.
