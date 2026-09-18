# Presentation QC (citation-supports-sentence audit)

Automated check (2026-09-17, rerun on demand): 22 `\cite` keys used, 22
`.bib` entries defined, 0 undefined, 0 uncited. V2 remnants without textual
support (`cuer2026`, `hubbs2020orgym`, `puterman1994markov`) removed;
OptiGuide retained with a new supporting sentence (sensor-vs-explainer
distinction). Cuconasu/Yoran/ARES/RAGAS added with content sentences.

Content spot-checks (sentence → cited support):
- "relevance labels correlate weakly with downstream performance" →
  salemi2024erag ✓ (their reported weak correlation).
- "grounding utility differs per consuming model" → grogu2026 ✓.
- "distracting passages can hurt more than random ones" →
  cuconasu2024power + yoran2024making ✓ (distractor harm; random-fill gains).
- "trains through a stochastic program" → donti2017task ✓.
- "SPO loss scores predictions by induced decision error" →
  elmachtoub2022smart ✓.
- "translator and explainer rather than sensor" → li2023optiguide ✓
  (verified against the paper's documented system role).
- "component-wise RAG evaluation with uncertainty" →
  saadfalcon2023ares + es2023ragas ✓.
- Situational/task-based lineage → borlund2003concept ✓ (concept, not method
  — manuscript states borrowing is conceptual).
- nDCG/BM25/embeddings/calibration/VOI sentences → jarvelin2002cumulated,
  robertson1994okapi, reimers2019sentencebert, guo2017calibration,
  degroot1983comparison, niculescu2005predicting, howard1966information ✓.
- Simulator pin → gyminvmgmt entry carries version + sdist SHA-256 ✓
  (render verified intact in PDF build).

Table/figure self-containment: all six tables carry full captions (estimand,
n, CIs, Holm rule, references); figure captions state axes and n.
Terminology: NoInfo (prior) vs NoText_Tuned (extension) vs C0 (rule-based
consumer) used consistently; check passed via grep (3 distinct terms, no
conflation in tables).
