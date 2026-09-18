# Strongest rejection (300–500 words, as an ECIR expert would write it)

This paper reports that BM25/dense/hybrid/reranked evidence (nDCG 0.115–0.183
on a synthetic 64-text feed with constructed, non-human qrels) reduces
operating return versus reading nothing in one lost-sales inventory simulator
with one receding-horizon controller class and two Qwen models, while oracle
evidence helps. That is exactly what eRAG (weak relevance→utility
correlation), GroGU (model-specific grounding utility), and Cuconasu/Yoran
(distractors hurt; stronger retrievers retrieve harder negatives) already
established for single-step RAG — here repeated with a weaker retriever set
(rerank–hybrid edge +0.001 out of sample) and relabeled as sequential. The
"six-node ladder" repackages SPO-style decision-regret (Elmachtoub & Grigas)
and Donti's inventory task-loss — which already showed prediction≠decision
in this very domain while *learning* through the task — with a signed
skill-score the authors admit is bookkeeping. The interaction (3/10 cells,
all rule-vs-LLM) is a level shift between a deterministic rule and LLMs, not
evidence about retrieval; C1-vs-C3 and all parity nulls are retained. The
no-text comparator that "beats every retriever" is an always-surge constant
exploiting an 18:1 payoff asymmetry, losing under normal-heavy priors by the
authors' own table. No human relevance judgments, no new language family
(64 texts, effective N≈137), Qwen-only beliefs, no new ranker, no user
model. Under a Bayes-optimal policy information cannot hurt; negative value
here indicts the fixed controller, and the +456.8 control gap (2× the
perfect-semantics value) concedes exactly that. This is a disciplined
controller-bug report with an honesty appendix, not an IR contribution at
the ECIR full-paper bar. Reject; encourage a short paper once validated on
human qrels and a second domain.

## Why it does not carry (evidence, not rhetoric)

1. Priority claims fail on scope: eRAG/GroGU/Cuconasu/Yoran are single-step
   QA/generation; none has a frozen sequential controller, persisting state,
   asymmetric operating costs, or paired ΔJ with crossed world×language
   randomisation. The conjunction (sequential + frozen + paired + ladder) is
   unaddressed by every cited work, not merely untitled by them.
2. "Controller bug" fails against the oracle ladder: the same fixed
   controller gains +178 on oracle evidence and the tuned constant gains
   +102 — a broken controller cannot produce evidence-conditioned gains of
   that size with Holm-robust CIs. Misspecification explains the level, not
   the evidence-conditioned differences, which is precisely the paper's
   estimand (value of information *to this controller class*, stated).
3. "Weak retriever" fails against the design: rerank is the best available
   ranker in this bank and still neutral/negative while oracle helps;
   absolute nDCG is disclosed, not hidden, with an explicit no-comparison
   rule. A stronger retriever is a different bank, i.e. future work, not a
   flaw in the measured divergence.
4. Single-domain/family/models and no human qrels are conceded limitations
   with bounded consequences (C0 needs no qrels/LLM; interaction replicates
   across models; bank scoping explicit). They lower significance; they do
   not invalidate the measured pattern.
