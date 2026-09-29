"""Create the immutable provenance manifest for the v5 Lightning backend.

The manifest hashes the prospective amendment, runtime lock, non-lockbox smoke
fixture, protocol freeze, and every execution adapter before any remote job is
allowed to produce results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/lightning_runtime_lock.json"
AMENDMENT = ROOT / "versions/sem2act-v5/amendments/lightning_backend_v1.yaml"
FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
PROTOCOL = ROOT / "versions/sem2act-v5/protocol/confirmation.yaml"
PROTOCOL_FREEZE = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
OUT = ROOT / "versions/sem2act-v5/manifests/lightning_runtime_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="validate the existing manifest without rewriting it")
    args = parser.parse_args()

    for path in (LOCK, AMENDMENT, FIXTURE, PROTOCOL, PROTOCOL_FREEZE):
        if not path.exists():
            raise SystemExit(f"required freeze input missing: {path}")

    lock = load_json(LOCK)
    protocol_freeze = load_json(PROTOCOL_FREEZE)
    if lock["protocol_hash"] != protocol_freeze["protocol_hash"]:
        raise SystemExit("runtime lock and protocol freeze hashes differ")
    if lock["protocol_hash"] != "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022":
        raise SystemExit("unexpected v5 protocol hash")

    source_paths = [ROOT / path for path in lock["source_contract"]]
    for path in source_paths:
        if not path.exists():
            raise SystemExit(f"source contract file missing: {path}")

    manifest = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-lightning-runtime-freeze-v1",
        "status": "frozen-before-result-bearing-execution",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": lock["protocol_hash"],
        "lock_id": lock["lock_id"],
        "lock_sha256": sha256(LOCK),
        "amendment_sha256": sha256(AMENDMENT),
        "smoke_fixture_sha256": sha256(FIXTURE),
        "protocol_sha256": sha256(PROTOCOL),
        "source_contract_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "smoke_fixture_policy": {
            "path": str(FIXTURE.relative_to(ROOT)),
            "is_lockbox_query": False,
            "must_be_hashed_before_model_execution": True,
        },
        "failure_policy": lock["failure_policy"],
        "no_result_dependent_runtime_changes": True,
    }

    if args.check:
        if not OUT.exists():
            raise SystemExit(f"missing freeze manifest: {OUT}")
        existing = load_json(OUT)
        for key in (
            "protocol_hash", "lock_id", "lock_sha256", "amendment_sha256",
            "smoke_fixture_sha256", "protocol_sha256", "source_contract_sha256",
        ):
            if existing.get(key) != manifest[key]:
                raise SystemExit(f"freeze manifest drift: {key}")
        print(json.dumps(existing, indent=2))
        return 0

    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
