# G2A-1 — FROZEN DESIGN (2026-09-20, pre-execution; C1 dev pilot, 240 calls)

Question (frozen): does removing individual passages causally change C1
belief/action/utility enough to justify full document-VoI work?

## 1. Scope (frozen)

Dev-40 queries × rerank k=3 × C1 × {Qwen3-8B-AWQ, Qwen3-14B-AWQ} ×
LOO{drop0,drop1,drop2} = 40×1×1×2×3 = **240 new C1 calls** (120/model).
No C3 new calls (2A-0 covered C3 by re-aggregation). No k=5, no other
systems, no swap-ins (unseen pairs would also be new calls — deferred with
the full proposal).

## 2. No-test-access (binding)

* Kernel inputs contain DEV-ONLY queries (`queries_dev40.jsonl`); test query
  text is never attached. Seal guard (no `dev.tsv`/`test.tsv`) retained.
  Kernel asserts every processed qid ∈ the embedded dev list.
* Local scripts assert `query_id ∈ dev_ids` on every new-belief row and
  refuse test IDs. The 240 payloads cannot produce test beliefs by
  construction (no test contexts exist).
* Test-160 stays sealed; D0/2A-0 test verdicts are never reopened.

## 3. Method (frozen; mirrors the frozen belief runs)

* Kernels `p2_loo_8b.py` / `p2_loo_14b.py`: pinned model+revision
  (8B `4da05a8e…`, 14B `31c69efc…`), vLLM flags identical to the frozen runs
  (awq, max-model-len 4096, mem 0.90/0.85), same thinking-disable transport
  patch (prompt text untouched, SHAs asserted in-kernel), `consume_c1` on
  2-doc subsets, identical cache key/file scheme (`{model,prompt_sha,raw,
  usage}`), src snap byte-identical, outputs `cache_loo_dev.zip` +
  `pilot_manifest.json`.
* Fetch→verify→install: manifest pins (model, revision, C1 SHA
  `66e6890b…`, tau 0.5, src snap SHAs, seal string), zip holds exactly 120
  files/model, 20-file random schema spot-check (precedent pattern), full
  local parse of all 240 payloads under frozen C1 normalization
  (probs/s, confidence rule — byte-identical formula), cache-key
  recomputation matches every filename, filename-collision assert (zero
  overwrites) before additive install into the live beliefs caches.
* Belief rows → sim replay on dev queries × paired seeds 60000–60004
  (frozen controller/protocol) + G3-analog gate (replayed full-set C1
  rerank dev profits vs frozen dev episodes, 1e-9).

## 4. Continuation gate (binding, decided on dev-40, B=5000 bootstrap)

Levels reported per (position × model) cell: belief change (L1 prob shift),
action change (ADis vs full-set orders), utility change
(J-VoI = dJ_full − dJ_loo).
**FIRE iff (a) per-position J-VoI CI excludes 0 in ≥1 of the 6 cells, AND
(b) the same cell shows ADis-vs-full mean CI > 0** (action survival;
semantic-only change without action/J → FAIL).
Fire → propose full C1 document-VoI (test-160, 960 calls) + deployable
criticality proxy. Fail → stop all document-level expansion and pivot to
the consumer-architecture story (C0 causal / C3 redundant / C1 insensitive).

## 5. Non-goals

No test access, no C3 calls, no k=5, no system expansion, no 2B/2C/2D work,
no manuscript prose. Either gate outcome is recorded in LEDGER and
committed, including a fail.
