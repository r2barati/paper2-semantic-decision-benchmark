# AGENTICK SPIKE — FROZEN DESIGN (2026-09-20, pre-execution; GoToGoal only)

Purpose (binding): test whether INTERFACE-dependent consequential value
exists outside retrieval/inventory at all — NOT recreate an nDCG failure,
NOT rank representations by presumed quality.

## 1. Intervention / fairness lock (frozen)

Same task/state/dynamics (GoToGoal-v0, difficulty easy, reward dense),
same 5 seeds, SAME fixed consumer/policy adapter; ONLY the observation
representation changes across {ascii, language, state_dict}. The adapter
is one deterministic program: parse mode-specific obs into
(agent_pos, goal_pos, walkable-set), then act by BFS-shortest-path
first-step (greedy goal approach); ties broken by fixed action order.
Parser code is mechanical translation written AFTER inspecting formats
(decision rule frozen here; parser recorded). Oracle runs exist ONLY for
ONS normalization, never as a tested consumer. A random policy runs as a
labeled floor reference (descriptive, not the tested consumer).

## 2. Outcome (frozen)

Primary: Oracle-Normalized Score per installed scoring code (read before
running; formula recorded in the verdict). Secondary: adapter parse-success
rate per mode (valid positions extracted?) as the diagnostic separator:
100%-everywhere + ONS differences => interface effect beyond parseability;
parse differences => parser-quality confound, reported as limitation.

## 3. Hypotheses (frozen; NO ordinal ladder)

Treat ascii/language/state_dict as ALTERNATIVE interfaces. No claim like
"language should be best" is preregistered, and no fidelity ordering is
assumed (none is defined).
* **H-primary:** representation/interface choice changes sequential reward
  (ONS) under otherwise fixed conditions. HOLDS iff max-min mode ONS
  paired bootstrap CI (seeds as pairs, B=2000) excludes 0.
* **H-secondary:** an upstream proxy rank-order does not perfectly order
  ONS. Proxy = parse-success rate; HOLDS (interesting) iff parse succeeds
  ~100% in all modes yet ONS differs, or the ONS order contradicts the
  compactness order with CIs excluding 0. Descriptive either way.
* **STOP RULE:** H-primary fails => STOP, report bounded external
  generality, no SokobanPush/SequenceMemory. Pass => propose the 2-task
  extension (not executed here).

## 4. Budget / non-goals (frozen)

GoToGoal only: 3 modes x 5 seeds x (adapter + random) + oracle
normalization runs (~40 episodes, CPU-only). No LLM/VLM agents, no
rgb_array, no SFT sets, no other tasks/difficulties, no manuscript prose.
Venv (py3.12 via uv) is environment, never committed — version pins recorded.
V3 frozen matrix untouched.
