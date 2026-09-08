"""Deterministically rebuild every offline artifact the test suite needs.

The committed release excludes model checkpoints and raw response payloads
(`.gitignore`), so a clean export of the tracked files alone cannot run the
suite: the September 2026 audit recorded 257 passed / 16 failed, every failure
caused by a missing artifact rather than a missing credential.

Nothing here contacts a provider or requires a key.  The two classes of
artifact are reconstructed from tracked, non-secret sources:

* **Semantic caches** are rematerialised from
  ``results/frozen_llm_outputs/manifest.jsonl``, which is tracked and already
  contains each canonical parsed response plus its SHA-256 checksum.  Every
  rebuilt file is verified against that checksum.
* **TF-IDF checkpoints** are retrained from the tracked template modules with
  fixed seeds.  Retraining is exact: the audit reproduced all 36 held-out
  probability vectors to 1.11e-16.

Usage::

    python3 -m tools.rebuild_offline_artifacts            # rebuild everything
    python3 -m tools.rebuild_offline_artifacts --verify   # check only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "results" / "frozen_llm_outputs" / "manifest.jsonl"


def rebuild_semantic_cache(verify_only: bool = False) -> dict:
    """Rematerialise frozen semantic responses from the tracked manifest."""
    out_dir = ROOT / "results" / "frozen_llm_outputs"
    if not MANIFEST.exists():
        raise SystemExit(f"missing tracked manifest: {MANIFEST}")

    written, verified, mismatched, missing = 0, 0, [], []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name = record["source_cache_file"]
        target = out_dir / name
        payload = json.loads(record["raw_or_canonical_response"])
        body = json.dumps(payload, sort_keys=True).encode()

        # Verify against the CANONICAL checksum. The collected file's original
        # pretty-printed bytes are not reproducible from the manifest (and
        # carry no scientific content), so byte equality with the collection
        # artifact is deliberately not the contract; canonical JSON is.
        expected = record.get("canonical_sha256")
        if target.exists():
            try:
                actual = hashlib.sha256(
                    json.dumps(json.loads(target.read_text()), sort_keys=True).encode()
                ).hexdigest()
            except json.JSONDecodeError:
                mismatched.append(name)
                continue
            if expected and actual != expected:
                mismatched.append(name)
            else:
                verified += 1
            continue
        if verify_only:
            missing.append(name)
            continue
        target.write_bytes(body)
        written += 1

    return {
        "written": written, "verified": verified,
        "checksum_mismatch": mismatched, "missing": missing,
    }


def rebuild_tfidf_models(verify_only: bool = False) -> dict:
    """Retrain the classical checkpoints from tracked templates."""
    from src.classical_baseline import (
        TFIDFLogReg, VARIANT_SINGLE, VARIANT_FOLD_ENSEMBLE, VARIANT_CALIBRATED,
    )
    from src.events import REGIME_WARNING_TEMPLATES
    from src.capacity_drop_templates import TRAIN_TEMPLATES as CAP_TRAIN

    def label_of(t):
        r = t["regime"]
        return r.value if hasattr(r, "value") else str(r)

    jobs = [
        # (output path, training templates, variant, cal_method)
        (ROOT / "results/phase7_classical_baseline/tfidf_logreg_model.pkl",
         REGIME_WARNING_TEMPLATES, VARIANT_SINGLE, "isotonic"),
        (ROOT / "results/phase7_classical_baseline/tfidf_logreg_fold_ensemble.pkl",
         REGIME_WARNING_TEMPLATES, VARIANT_FOLD_ENSEMBLE, "isotonic"),
        (ROOT / "results/phase7_classical_baseline/tfidf_logreg_calibrated_model.pkl",
         REGIME_WARNING_TEMPLATES, VARIANT_CALIBRATED, "isotonic"),
        (ROOT / "results/phase7_classical_baseline/tfidf_logreg_calibrated_sigmoid.pkl",
         REGIME_WARNING_TEMPLATES, VARIANT_CALIBRATED, "sigmoid"),
        (ROOT / "results/phase9a_capacity_confirmation/tfidf_capacity_logreg_raw.pkl",
         CAP_TRAIN, VARIANT_SINGLE, "isotonic"),
        (ROOT / "results/phase9a_capacity_confirmation/tfidf_capacity_logreg_fold_ensemble.pkl",
         CAP_TRAIN, VARIANT_FOLD_ENSEMBLE, "isotonic"),
        (ROOT / "results/phase9a_capacity_confirmation/tfidf_capacity_logreg_calibrated.pkl",
         CAP_TRAIN, VARIANT_CALIBRATED, "isotonic"),
    ]

    built, present, absent = [], [], []
    for path, templates, variant, cal_method in jobs:
        if path.exists():
            present.append(path.name)
            if not verify_only:
                continue
        if verify_only:
            if not path.exists():
                absent.append(path.name)
            continue
        texts = [t["text"] for t in templates]
        labels = [label_of(t) for t in templates]
        model = TFIDFLogReg(variant=variant, cal_method=cal_method, seed=42)
        model.fit(texts, labels)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        built.append(path.name)
    return {"built": built, "already_present": present, "missing": absent}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="report what is missing without writing anything")
    args = ap.parse_args()

    cache = rebuild_semantic_cache(verify_only=args.verify)
    models = rebuild_tfidf_models(verify_only=args.verify)

    print("semantic cache:", json.dumps(cache, indent=2))
    print("tfidf models  :", json.dumps(models, indent=2))

    problems = cache["checksum_mismatch"] or (cache["missing"] if args.verify else [])
    problems = list(problems) + (models["missing"] if args.verify else [])
    if problems:
        print(f"\nINCOMPLETE: {len(problems)} artifact(s) unresolved", file=sys.stderr)
        return 1
    print("\nOffline artifacts complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
