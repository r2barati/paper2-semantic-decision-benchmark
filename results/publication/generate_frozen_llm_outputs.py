"""Materialize a self-contained, provenance-labelled semantic cache export.

Provenance is *reconstructed*, never assumed.  Cache filenames are the first
16 hex characters of a SHA-256 over a key that already contains the model
identifier and the warning text, so the true (model, text, key-family) triple
for a cache file can be recovered by enumerating every model and template the
repository knows about and re-deriving the key.

Two key families exist:

``regime``    ``sha256("regime_{model}||{text}")[:16]``      -- src/interpreter.py
``phase9``    ``sha256("{model}||{text}")[:16]``, file named ``phase9_<h>.json``
              -- src/experiment_phase9a.py, src/experiment_phase9b.py

The previous exporter labelled every non-``phase9_`` file ``gpt-4o-mini``,
which mislabelled all 36 GPT-4o and all 36 GPT-3.5-turbo confirmation entries.
Anything this script cannot match is now recorded as ``unknown`` rather than
being given a plausible-looking default.

The field formerly called ``prompt_hash`` was never a hash of the prompt: it
is a hash of model identifier and text.  It is now named ``cache_key_hash``
and the system prompt, schema and generation settings are hashed separately so
that a change in the extraction prompt is actually visible in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / ".llm_cache"
OUT = ROOT / "results" / "frozen_llm_outputs"

# Every model identifier that has ever been used to populate the cache.
KNOWN_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4",
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _known_texts() -> dict[str, str]:
    """Every warning text the repository can produce, keyed by template id."""
    texts: dict[str, str] = {}

    def absorb(templates, prefix=""):
        for tmpl in templates:
            if isinstance(tmpl, dict) and "text" in tmpl:
                tid = str(tmpl.get("template_id", f"{prefix}{len(texts)}"))
                texts[tid] = tmpl["text"]

    from src.events import WARNING_TEMPLATES, REGIME_WARNING_TEMPLATES
    from src.confirmation_templates import CONFIRMATION_TEMPLATES
    from src.capacity_drop_templates import ALL_CAPACITY_DROP_TEMPLATES

    absorb(WARNING_TEMPLATES, "warn_")
    absorb(REGIME_WARNING_TEMPLATES, "regime_")
    absorb(CONFIRMATION_TEMPLATES, "conf_")
    absorb(ALL_CAPACITY_DROP_TEMPLATES, "cap_")
    return texts


def build_provenance_index() -> dict[str, dict]:
    """Map every reconstructable cache filename to its true provenance."""
    texts = _known_texts()
    index: dict[str, dict] = {}
    for tid, text in texts.items():
        for model in KNOWN_MODELS:
            regime_h = _sha(f"regime_{model}||{text}")[:16]
            index.setdefault(f"{regime_h}.json", {
                "model_id": model,
                "template_id": tid,
                "key_family": "regime",
                "cache_key_hash": regime_h,
                "provenance": "reconstructed_from_cache_key",
            })
            plain_h = _sha(f"{model}||{text}")[:16]
            index.setdefault(f"phase9_{plain_h}.json", {
                "model_id": model,
                "template_id": tid,
                "key_family": "phase9",
                "cache_key_hash": plain_h,
                "provenance": "reconstructed_from_cache_key",
            })
            # The original single-event interpreter (src/interpreter.py
            # `_cache_path`) uses the same plain key with no filename prefix.
            index.setdefault(f"{plain_h}.json", {
                "model_id": model,
                "template_id": tid,
                "key_family": "event",
                "cache_key_hash": plain_h,
                "provenance": "reconstructed_from_cache_key",
            })
    return index


def prompt_fingerprints() -> dict[str, str]:
    """Hash the actual prompt surface separately from the cache key."""
    from src import interpreter as I

    parts = {}
    for name in ("EXTRACTION_PROMPT", "REGIME_EXTRACTION_PROMPT", "SYSTEM_PROMPT"):
        value = getattr(I, name, None)
        if isinstance(value, str):
            parts[f"{name.lower()}_sha256"] = _sha(value)
    return parts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = build_provenance_index()
    fingerprints = prompt_fingerprints()

    records = []
    unknown = 0
    for source in sorted(SOURCE.glob("*.json")):
        raw = source.read_bytes()
        payload = json.loads(raw)
        shutil.copyfile(source, OUT / source.name)

        info = index.get(source.name)
        if info is None:
            unknown += 1
            info = {
                "model_id": "unknown",
                "template_id": "unknown",
                "key_family": "phase9" if source.name.startswith("phase9_") else "regime",
                "cache_key_hash": source.stem.removeprefix("phase9_"),
                "provenance": "unresolved_no_matching_model_text_pair",
            }

        record = {
            "cache_key_hash": info["cache_key_hash"],
            "cache_key_formula": (
                'sha256("{model}||{text}")[:16]' if info["key_family"] == "phase9"
                else 'sha256("regime_{model}||{text}")[:16]'
            ),
            "key_family": info["key_family"],
            "model_id": info["model_id"],
            "model_snapshot": "unknown",
            "template_id": info["template_id"],
            "provenance": info["provenance"],
            "response_kind": "canonical_parsed_json",
            "probabilities": payload.get("regime_probabilities", payload),
            "raw_or_canonical_response": json.dumps(payload, sort_keys=True),
            "parsing_status": "cached_parsed",
            # Two distinct checksums, because they certify different things:
            #   collected_file_sha256  -- the bytes of the artifact as
            #     originally written by the collection run (provenance).
            #   canonical_sha256       -- sha256 over the canonical
            #     `json.dumps(payload, sort_keys=True)` serialization. This is
            #     the *reproducible* contract: rebuilding the cache from this
            #     manifest reproduces the canonical bytes exactly, while the
            #     original pretty-printed whitespace and key order are not
            #     recoverable and are not scientifically meaningful.
            "collected_file_sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
            "cache_checksum_sha256": hashlib.sha256(raw).hexdigest(),
            "source_cache_file": source.name,
        }
        record.update(fingerprints)
        records.append(record)

    (OUT / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    )

    by_model: dict[str, int] = {}
    for r in records:
        by_model[r["model_id"]] = by_model.get(r["model_id"], 0) + 1

    (OUT / "README.md").write_text(
        "# Frozen semantic outputs\n\n"
        "Offline semantic-output export used for the published results. Each cached\n"
        "response has a manifest record giving the reconstructed model identifier,\n"
        "template id, cache-key formula and hash, canonical parsed response, and a\n"
        "and two checksums.\n\n"
        "`collected_file_sha256` is the byte checksum of the artifact as originally\n"
        "written; `canonical_sha256` is the checksum of the canonical\n"
        "`json.dumps(payload, sort_keys=True)` form. Rebuilding from this manifest\n"
        "reproduces the canonical bytes exactly. Original whitespace and key order\n"
        "are not recoverable and carry no scientific content.\n\n"
        "## Provenance\n\n"
        "Model labels are **reconstructed** by re-deriving each cache key from every\n"
        "known (model, template text) pair, not inferred from the filename prefix.\n"
        "Entries with no matching pair are labelled `unknown`.\n\n"
        "`cache_key_hash` is a hash of the model identifier and the warning text. It\n"
        "is *not* a hash of the extraction prompt; the prompt surface is fingerprinted\n"
        "separately in the `*_prompt_sha256` fields.\n\n"
        "## What this export does and does not establish\n\n"
        "It preserves the canonical parsed JSON the pipeline consumed, so results can\n"
        "be reproduced offline. It does **not** preserve complete provider responses,\n"
        "exact model snapshots, or collection timestamps, so it cannot be used to\n"
        "claim independent cross-model reproducibility.\n\n"
        "## Contents\n\n"
        f"- records: {len(records)}\n"
        f"- unresolved provenance: {unknown}\n"
        + "".join(f"- {m}: {n}\n" for m, n in sorted(by_model.items()))
    )
    print(f"wrote {len(records)} records; unresolved={unknown}; by model={by_model}")


if __name__ == "__main__":
    main()
