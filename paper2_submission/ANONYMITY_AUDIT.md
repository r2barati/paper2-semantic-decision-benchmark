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

Machine scan of the current anonymous manuscript source found no author name,
affiliation, email, personal URL, absolute path, or repository identity.
Official TMLR template sample names/URLs are excluded from the proposed
supplement and are not Paper-2 identities. Current machine issue count:
**0 detected; human PDF/supplement confirmation remains required**.
