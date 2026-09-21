# DESIGN-TRACKA — FROZEN (2026-09-20)

Track A = internal robustness + one informative external test. Zero new
LLM calls. Claim ceiling: robustness/generalization of the consumer-
dependence finding; "external validity" reserved for the Agentick leg;
no universality language.

## A1. Inventory robustness matrix (CPU-only, cached beliefs)

Cells: consumers {C0, C1-8B/14B, C3-8B/14B} (frozen beliefs) x controllers
{frozen LP, base-stock} x regimes {normal, delay, surge} x evidence
{NoInfo, oracle-factual k=3, rerank k=3, inject-contra, stale-swap,
drop-decisive} (+ Phase-9A capacity-drop and no-text controls BY REFERENCE,
no new runs). Sim recompute behind require_frozen_beliefs (SHA match).
ONLY three confirmatory families (rest descriptive):
 (i) consumer VoI-ordering rank concordance across controllers (Spearman +
     within-query permutation);
 (ii) controller difference in transmission rates (paired crossing/ADis-rate
     difference, CI excludes 0);
 (iii) consumer x regime interaction F vs within-query permutation null
     (H2 machinery).
Paired seeds 60000-60004; cluster-aware (query-cluster) bootstrap CIs;
Holm across the three families. Any fail -> recorded bound.

## A2. Agentick dose-response (authored planner, no privilege)

Authored BFS shortest push-plan planner (bounded depth/node caps frozen at
build: depth<=40, nodes<=20000), reading ONLY permitted obs + valid_actions
per the frozen forbidden-reads checklist (no api.bfs/grid/agent/task_config
internals; oracle retained solely for ONS normalization).
Conditions per task: full parsed state vs radius-k entity masking with
k in {full, 3, 1, 0} (SokobanPush-v0; SequenceMemory iff build-time check
confirms oracle + dense/sparse coverage, else Sokoban only), same planner,
same seeds. Primary: paired J_full - J_k with CIs + monotonic-degradation
test (ordered paired CIs; NO linear slope on unequally spaced k).
Cross-format (ascii/state_dict full-parse vs language partial-parse) is
SECONDARY (level shifts = parser competence; slopes = information value).
PLANNER COMPETENCE GATE (frozen, first): on full-information parsed state,
>=80% of oracle-solved instances over 10 seeds (next-integer replacement
until 10 oracle-solved; recorded). Below 80% -> STOP, third
uninformative-consumer outcome, no masked episodes counted. Modest claim
only: causal value of information available through the observation
interface. CPU-only, isolated venv, never committed.

## A3. Protocol packaging (NeurIPS E&D shape)

EVAL-PROTOCOL.md (sealed splits, paired seeds, exact-replay gates, LOO
interventions, cluster bootstrap, Holm, permutation, oracle/deployable
separation, preregistered stopping with precedents A / D0-Gate1 / VHAT) +
scripts index + threats audit. ECIR submission untouched; new manuscript
dir only at manuscript phase (not this track).

## Non-goals / firewall

No new LLM; no new consumers/models/evidence beyond cached; no ORAgentBench;
no manuscript prose. Track A results may not determine Track B features,
thresholds, subgroups, or stopping (and vice versa); shared input is the
frozen Step-0 autopsy only. V3 + all prior verdicts stand untouched.
