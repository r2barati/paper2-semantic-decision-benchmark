# Qrel validity: chronology + human-validation plan (FROZEN record)

## Chronology (keep both numbers; do not rewrite history)

1. Prototype first measurement: independent text-only grader κ = **0.823** (n=290).
2. Full build first measurement: κ = **0.468** (n=3881).
3. Diagnosis: the grader's keyword lists covered only wave-1 vocabulary; wave-2
   boilerplate (landscaping, furniture, duplication-correction texts) fell through
   to wrong grades. Corpus text predated the measurement in both runs.
4. Instrument fix (grader only, still text-only, still metadata-blind): extended
   keyword lists to wave-2 vocabulary. Re-measured: prototype κ = **0.881**,
   full κ = **0.874** (n=3881).
5. Generator/corpus text: UNCHANGED throughout. No qrel, grade, or document was
   edited in response to any κ number.

## Why this is defensible (and its limit)

- The grader is a measurement instrument, not the label source: labels come from
  the construction rules (entity/recency/aboutness), recorded before generation.
- The fix extended the instrument's vocabulary coverage; it cannot see metadata,
  so agreement gains reflect genuine text-label alignment, not leakage.
- LIMIT (stated openly): the grader shares the generator's authorship. It guards
  against construction bugs, not against shared blind spots.

## Required human validation (pre-submission gate, not optional)

- **Sample:** 300 query-document pairs, stratified by grade (75×0/1/2/3) and regime,
  drawn from TEST ONLY, blind to generator metadata, kind labels, and qrels.
- **Annotators:** 2 independent judges with the written grade rubric (entity/recency/
  aboutness only). Adjudicate disagreements by discussion; report raw + adjudicated κ.
- **Pass bar:** human–qrel κ ≥ 0.6 per grade AND no systematic grade-3 miss pattern.
  Failure → corpus revision + re-freeze (never silent relabeling).
- **Record:** annotator instructions, raw judgments, adjudication log, κ — all frozen
  in `data/v3/annotation/` before the main experiment analysis.
