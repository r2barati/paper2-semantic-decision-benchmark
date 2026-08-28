"""Materialize a self-contained, provenance-labelled semantic cache export."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".llm_cache"
OUT = ROOT / "results" / "frozen_llm_outputs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(SOURCE.glob("*.json")):
        raw = source.read_bytes()
        payload = json.loads(raw)
        cache_name = source.stem
        model_id = "gpt-4o" if cache_name.startswith("phase9_") else "gpt-4o-mini"
        target = OUT / source.name
        shutil.copyfile(source, target)
        records.append({
            "prompt_hash": cache_name.removeprefix("phase9_"),
            "model_id": model_id,
            "probabilities": payload.get("regime_probabilities", payload),
            "raw_or_canonical_response": json.dumps(payload, sort_keys=True),
            "parsing_status": "cached_parsed",
            "cache_checksum_sha256": hashlib.sha256(raw).hexdigest(),
            "source_cache_file": source.name,
        })
    (OUT / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    (OUT / "README.md").write_text(
        "# Frozen semantic outputs\n\n"
        "This directory is the offline semantic-output export used for published results. "
        "Each JSON response has a manifest record with prompt hash, model identifier, "
        "probabilities, parsing status, and SHA-256 checksum. The aggregate publication "
        "tables replay frozen episode CSVs and do not require an API key.\n"
    )


if __name__ == "__main__":
    main()
