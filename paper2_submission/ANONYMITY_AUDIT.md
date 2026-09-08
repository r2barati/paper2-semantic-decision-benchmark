# Anonymity audit

Regenerated 7 September 2026. Run `python3 -m tools.build_anonymous_supplement`
to rebuild and re-check the reviewer artifact.

## What the builder does

Removes git history, credentials, editor state, private planning documents and
the superseded TMLR staging directory; rewrites author-machine absolute paths to
`<repo>`; removes author name, institutional email and personal repository URLs
from the vendored dependency's *packaging metadata*. It then re-scans the built
tree and fails the build if any identity string survives.

## Current result

```
copied 577 files, 5 redacted, 1174 skipped
ANONYMITY AUDIT PASSED
```

Independently verified: a case-insensitive `grep -ril` over the built tree finds
the author's name only in `third_party/gym-invmgmt-paper/LICENSE`, and finds no
author home-directory paths.

## A bug this audit previously had

The audit used the same word-boundary regexes as the redaction step, and
reported a **false pass** while shipping the author's name and address inside
the builder's own `REDACTIONS` table. A pattern of the form
`\b<Given> <Family>\b` does not match its own escaped source text: the
character immediately preceding the name there is the word character `b` from
the `\b` escape, so there is no word boundary at that position and the pattern
skips over its own literal.

Two fixes: the builder excludes itself from the artifact, and the audit now uses
case-insensitive plain substring search over **all** files rather than the
redaction regexes over text-suffixed files. An anonymity check must not share
its matching logic with the transformation it is checking.

## Residual risk requiring an author decision

`third_party/gym-invmgmt-paper/LICENSE` is an MIT notice that names a copyright
holder. It is retained verbatim: anonymity does not license removing a
third-party copyright notice, and doing so would be a licence violation.

That notice names one of the authors. A determined reviewer could follow it.
The manuscript itself does not help them --- it cites the simulator by its
published benchmark lineage rather than as "Paper-1" or by repository URL, and
`NOTICE` records the dependency without linking it to this submission's
authorship. But the residual link exists and **the authors must decide** how to
handle it. Options, in decreasing order of preference:

1. Depend on the upstream published package rather than a vendored copy of the
   authors' own fork, so the notice names an external project.
2. Disclose it in the submission's anonymity statement, if the venue permits.
3. Accept the risk.

This is a human decision about the authors' own prior work and cannot be made
from this repository.

## Not certified here

Author declarations, conflicts of interest, concurrent-submission status and the
AI-use statement are facts about people and about submissions elsewhere. No tool
in this repository can establish them.

---

<!-- previous content retained below -->

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

## This file is checked too

The audit scans every file in the built artifact, including this one. An earlier
draft of this document spelled the author's name out while explaining the bug
above, and the audit failed the build for it --- correctly. The names are
described rather than written here, and the builder excludes itself from the
artifact.
