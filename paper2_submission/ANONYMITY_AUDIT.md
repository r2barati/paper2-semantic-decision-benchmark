# Anonymity audit

## Current package risks

- The source repository README and citation metadata contain author and
  institutional information.
- Paper-1 paths, GitHub URLs, and self-citations may identify the authors.
- Cached LLM metadata and environment files must not enter a supplement without
  review.

## TMLR submission actions

- Remove author names, affiliations, acknowledgments, funding identifiers, and
  identifying repository URLs from the anonymous PDF and supplement.
- Replace self-references with third-person wording where required.
- Strip absolute filesystem paths and PDF metadata.
- Keep provenance internally; do not delete historical records.
- Reinsert author-facing metadata only after acceptance or when venue policy
  allows it.

Current issue count in the future export: **not yet certified**. Human PDF and
supplement scanning is required before submission.
