"""Freeze or verify Moon's reranker-only execution contract; launches no job."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/moon_reranker_runtime_lock.json"
AMENDMENT = ROOT / "versions/sem2act-v5/amendments/moon_reranker_backend_v1.yaml"
FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
PROTOCOL = ROOT / "versions/sem2act-v5/protocol/confirmation.yaml"
PROTOCOL_FREEZE = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
OUT = ROOT / "versions/sem2act-v5/manifests/moon_reranker_runtime_freeze.json"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_manifest() -> dict:
    for path in (LOCK, AMENDMENT, FIXTURE, PROTOCOL, PROTOCOL_FREEZE):
        if not path.is_file():
            raise SystemExit(f"required Moon freeze input missing: {path}")
    lock = json.loads(LOCK.read_text())
    frozen_protocol = json.loads(PROTOCOL_FREEZE.read_text())
    if lock.get("protocol_hash") != PROTOCOL_HASH or frozen_protocol.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("Moon reranker protocol hash drift")
    if lock.get("lock_id") != "sem2act-v5-moon-reranker-runtime-v1":
        raise SystemExit("Moon reranker runtime lock id drift")
    platform = lock.get("platform", {})
    if (platform.get("output_quota_minimum_free_bytes") != 10_000_000
            or platform.get("output_quota_probe_bytes") != 10_000_000
            or platform.get("instruction_flag_check") != "every /proc/cpuinfo processor record"):
        raise SystemExit("Moon host/quota preflight policy drift")
    source_hashes = {}
    for relative in lock.get("source_contract", []):
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Moon source contract file missing: {relative}")
        source_hashes[relative] = sha(path)
    return {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-moon-reranker-runtime-freeze-v1",
        "status": "frozen-before-result-bearing-execution;host-and-parity-gated",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "lock_id": lock["lock_id"],
        "lock_sha256": sha(LOCK),
        "amendment_sha256": sha(AMENDMENT),
        "smoke_fixture_sha256": sha(FIXTURE),
        "protocol_sha256": sha(PROTOCOL),
        "source_contract_sha256": source_hashes,
        "failure_policy": lock["failure_policy"],
        "no_result_dependent_runtime_changes": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = expected_manifest()
    if args.check:
        if not OUT.is_file():
            raise SystemExit(f"Moon runtime freeze missing: {OUT}")
        actual = json.loads(OUT.read_text())
        for key, value in expected.items():
            if key != "frozen_utc" and actual.get(key) != value:
                raise SystemExit(f"Moon runtime freeze drift: {key}")
        print(json.dumps(actual, indent=2, sort_keys=True))
    else:
        OUT.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
