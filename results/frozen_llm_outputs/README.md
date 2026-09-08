# Frozen semantic outputs

Offline semantic-output export used for the published results. Each cached
response has a manifest record giving the reconstructed model identifier,
template id, cache-key formula and hash, canonical parsed response, and a
and two checksums.

`collected_file_sha256` is the byte checksum of the artifact as originally
written; `canonical_sha256` is the checksum of the canonical
`json.dumps(payload, sort_keys=True)` form. Rebuilding from this manifest
reproduces the canonical bytes exactly. Original whitespace and key order
are not recoverable and carry no scientific content.

## Provenance

Model labels are **reconstructed** by re-deriving each cache key from every
known (model, template text) pair, not inferred from the filename prefix.
Entries with no matching pair are labelled `unknown`.

`cache_key_hash` is a hash of the model identifier and the warning text. It
is *not* a hash of the extraction prompt; the prompt surface is fingerprinted
separately in the `*_prompt_sha256` fields.

## What this export does and does not establish

It preserves the canonical parsed JSON the pipeline consumed, so results can
be reproduced offline. It does **not** preserve complete provider responses,
exact model snapshots, or collection timestamps, so it cannot be used to
claim independent cross-model reproducibility.

## Contents

- records: 213
- unresolved provenance: 50
- gpt-3.5-turbo: 49
- gpt-4o: 65
- gpt-4o-mini: 49
- unknown: 50
