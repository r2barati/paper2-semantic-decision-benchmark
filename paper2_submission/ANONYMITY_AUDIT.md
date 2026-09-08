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

## Residual risk: resolved

An earlier build retained `third_party/gym-invmgmt-paper/LICENSE`, an MIT notice
naming a copyright holder who is also an author. It was kept verbatim, because
anonymity does not license removing a third-party copyright notice, and that
left a residual link a determined reviewer could follow.

The link is now gone at its source rather than papered over. The simulator is a
**declared, pinned dependency** installed from its published release
(`gym-invmgmt==0.2.1`, sdist SHA-256 `a2a286cd95d9967b...`) and cited in third
person, instead of a vendored fork distributed inside the artifact. No copyright
notice was altered; the fork simply is not part of the submission.

The version change was not assumed to be safe. All 14,400 stored gym episodes
were replayed against the published release and reproduced their recorded
rewards with a maximum error of **0.0**, so no published number depends on which
of the two the reader installs.

The built artifact now contains no occurrence of the authors' name, email,
institutional domain or home-directory paths, verified by case-insensitive
substring scan over every file.

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
