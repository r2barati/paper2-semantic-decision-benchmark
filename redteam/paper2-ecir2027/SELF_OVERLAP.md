# Self-overlap audit: v1 vs v2 vs V3

Platform reused across all three: inventory simulator lineage, retrieval
machinery, bootstrap/Crossed-CI statistics, LNCS manuscript skeleton. What
is new in V3 (this submission) vs each predecessor:

- vs v1 (TMLR-tracked *When Does Semantic Information Improve Sequential
  Decisions?*, frozen in `v1/`): v1 asked whether semantic information helps
  and answered affirmatively from estimates later shown to depend on an
  information-boundary defect (NoInfo planner saw the true disruption),
  a missing inventory term, and FIFO timing faults (see `AUDIT_RESPONSE_2026.md`
  findings 2–4). V3's headline is the reverse pattern under repaired
  mechanics with a tuned comparator v1 lacked. No result is shared; v1
  numbers are superseded, preserved only under
  `results/pre_correction_archive_2026/`.
- vs v2 (ECIR-repair lineage, Experiment R + transfer environments M/S,
  frozen in `v2/`): v2 established consumer-dependent utility with
  RuleBased/calibrated consumers and TF-IDF/GPT-4o interpreters. V3 replaces
  the corpus (64-node report feed), the consumers (C0/C1/C3), the models
  (Qwen3-8B/14B AWQ, pinned), the ladder (six nodes + hindsight), the
  comparator (dev-tuned constant), and the statistics (dual priors, cluster
  bootstrap, headline Holm). No v2 episode, belief, or table enters V3.
- Shared infrastructure (simulator core, metric definitions, runner
  patterns) is method, not contribution; V3's contribution (ladder
  instrument + measured pattern + attribution) has no overlap with any
  published claim of v1/v2.

Separation verdict: clearly separable. The risk this audit guards against
(same platform ⇒ same paper) does not materialize: the scientific claims
are disjoint and in places reversed.
