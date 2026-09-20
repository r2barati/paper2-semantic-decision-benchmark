# AGENTICK HARD SPIKE — FROZEN DESIGN (2026-09-20; SokobanPush only)

Purpose (binding): one fair chance to test "information interface can
alter sequential value" where representation can plausibly matter
(irreversible pushes). GoToGoal spike stays a ceiling report; nothing
here reopens it.

## 1. Intervention / fairness lock (frozen)

Same task/state/dynamics (SokobanPush-v0, easy, dense), same 5 seeds,
SAME fixed consumer across {ascii, language, state_dict}. Adapter program:
parse obs -> egocentric (facing; box dir/dist; target dir/dist; wall-ahead
flag) -> frozen push-priority rule; reads info['valid_actions'] ONLY.
Parsers are mechanical translation (formats inspected pre-freeze); the
DECISION RULE below is identical for all modes:
 1. PUSH: box adjacent (dist 1) in side D AND target same side with
    greater dist -> move toward D.
 2. REPOSITION: box adjacent but unaligned -> move toward target side.
 3. APPROACH: otherwise move toward box side (diagonals: forward first;
    bare "behind": fixed move_left tie-break; "here": interact).
 4. Fallback (chosen move invalid): fixed order [move_up, move_right,
    move_down, move_left, interact, noop] filtered to valid_actions.
Oracle runs for ONS normalization only (mode-invariance verified pre-freeze:
1.0 all modes seed 0). Random floor descriptive (rng.integers(0,6) with
per-episode seed 9100+seed).

## 2. Seeds (frozen with replacement rule)

Seeds 0-4. REPLACEMENT RULE (task validation, not adapter tuning: ONS
normalization requires solvable seeds): if the oracle fails any seed,
replace with the next integer (5, 6, ...) until 5 oracle-solved seeds;
replacements recorded. Adapter outcomes never influence seed choice.

## 3. Outcome / hypotheses (frozen; no ordinal ladder)

Primary ONS per installed scoring: per-episode (ret − random)/(optimal −
random) clipped, random_baseline = mean over pooled random episodes,
optimal = mean oracle return; averaged. Secondary: parse-success rate per
mode (facing+box+target all present).
* **H-primary:** max-min mode ONS paired bootstrap CI (B=2000, seeds paired)
  excludes 0. **STOP RULE:** fail => stop external validation ENTIRELY;
  Agentick remains a bounded null/ceiling result; no SequenceMemory.
* **H-secondary:** parse 100% everywhere + ONS differences => interface
  effect beyond parseability; parse gaps => confound limitation, reported.
* **PASS:** H-primary holds => PROPOSE (not execute) SequenceMemory as
  second confirmation.

## 4. Budget / non-goals

SokobanPush only: 3 modes x 5 seeds x (adapter + random) + oracle runs
(~40 episodes CPU-only). No LLM/VLM agents, no rgb_array, no other tasks/
difficulties, no manuscript prose. Venv never committed. V3 untouched.
