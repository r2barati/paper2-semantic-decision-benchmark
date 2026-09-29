"""Freeze v5 protocol and source hashes after local tests pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = [
    ROOT / "versions/sem2act-v5/protocol/confirmation.yaml",
    ROOT / "versions/sem2act-v5/experiments/signed_estimands.py",
    ROOT / "versions/sem2act-v5/experiments/selective_entropy.py",
    ROOT / "versions/sem2act-v5/experiments/controllers.py",
    ROOT / "versions/sem2act-v5/experiments/lockbox/generate_lockbox.py",
    ROOT / "versions/sem2act-v5/experiments/lockbox/provenance_validate.py",
]


def main() -> None:
    records = []
    for path in FILES:
        records.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })
    joined = "\n".join(f"{r['path']} {r['sha256']}" for r in records).encode()
    freeze = {
        "schema_version": 1,
        "experiment_id": "v5-confirmation-package",
        "status": "protocol-frozen",
        "frozen_utc": "2026-09-24",
        "files": records,
        "protocol_hash": hashlib.sha256(joined).hexdigest(),
        "signed_estimands": {
            "delta": "J_Rerank - J_NoInfo",
            "interaction": "[DeltaJ40 - DeltaJ1]_persistent - [DeltaJ40 - DeltaJ1]_reset",
            "primary_null": "H0: I_40_1 = 0",
        },
        "no_design_changes_after_freeze": True,
    }
    out = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(freeze, indent=2) + "\n")
    version = ROOT / "versions/sem2act-v5/VERSION.yaml"
    text = version.read_text()
    text = text.replace("protocol_hash: pending", f"protocol_hash: {freeze['protocol_hash']}")
    version.write_text(text)
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
