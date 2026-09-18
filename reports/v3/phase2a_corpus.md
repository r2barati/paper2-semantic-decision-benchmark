# Phase 2A — full corpus freeze (200 queries / 3,413 docs / 3,881 pairs)

AMENDMENT (2026-09-14, rules v2 — validity defect, score-independent mechanism):
the 8-entity build put ~328 same-entity current docs per node in competition while
dense top-10 was 100% same-entity: within-entity ranking (where grades 2/3 live) was
near-chance (dev nDCG@10 ~0.015, Recall@100 0.26 vs Recall@500 0.99 — retrievable but
buried). Queries carry no aboutness signal BY DESIGN, so the fix is structural, not
informational: 16 entities (~12 queries/node, guard ≤220, measured 201), 4 held-out
(~50 queries preserved). First-build numbers below are superseded; current manifest
is authoritative. No query/topic content was added anywhere.

Frozen rules: configs/v3/dataset_full.yaml (written BEFORE generation).
Retrieval stack, instruction, prompts, metric: untouched (all Phase-1 freezes hold).

## Scale
- Queries: 200 (80 delay / 80 surge / 40 normal); dev 40 (16/16/8), test 160
- Entities: R1–R6 standard; R7/R8 fully held out (50 test queries, 0 dev)
- Variant holdout: bank idx mod 5 == 0 → test-only (dev pools exclude by construction)
- Grades: {3:400, 2:400, 1:400, 0:2681}; 100% freshly authored, zero V1/V2 verbatim
- IDs: v3-q001..200 / v3-d00001..; v3p-* quarantine enforced by test
- Manifest: data/v3/manifests/dataset_manifest.json (SHA-256 per file)

## Audit
- Independent text-only grader κ = **0.874** (n=3881). One instrument fix during audit:
  the grader's keyword lists covered only wave-1 vocabulary (κ=0.468); extended to
  wave-2 boilerplate/correction markers. Corpus text untouched — generator rules were
  frozen before measurement, and the grader never sees metadata.
- Leakage: no exact full-text dupes; dev has zero held-out entities/variants;
  TF-IDF-train exclusion by construction (no bank import anywhere in generator).
- Severity/timing metadata recorded per query (text-realized only; simulator frozen).

## Reproduce
`python3 -m pytest tests/test_v3full.py tests/test_v3proto.py -q` (16 tests).
Regenerate: `build_full_corpus(seed=20260914)` + `write_full(..., 'data/v3')` — byte-identical manifest expected.
