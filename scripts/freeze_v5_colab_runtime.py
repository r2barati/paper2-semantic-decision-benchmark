"""Freeze or verify the exact Colab execution contract; never launches a job."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/colab_runtime_lock.json"
AMENDMENT = ROOT / "versions/sem2act-v5/amendments/colab_backend_v1.yaml"
FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
PROTOCOL = ROOT / "versions/sem2act-v5/protocol/confirmation.yaml"
PROTOCOL_FREEZE = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
OUT = ROOT / "versions/sem2act-v5/manifests/colab_runtime_freeze.json"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> dict:
    for path in (LOCK, AMENDMENT, FIXTURE, PROTOCOL, PROTOCOL_FREEZE):
        if not path.is_file():
            raise SystemExit(f"required Colab freeze input missing: {path}")
    lock = json.loads(LOCK.read_text())
    protocol = json.loads(PROTOCOL_FREEZE.read_text())
    if lock.get("lock_id") != "sem2act-v5-colab-runtime-v1":
        raise SystemExit("Colab runtime lock id drift")
    if lock.get("protocol_hash") != PROTOCOL_HASH or protocol.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("Colab runtime protocol hash drift")
    runtime = lock.get("runtime", {})
    if (runtime.get("torch"), runtime.get("cuda"), runtime.get("gpu_min_vram_gib")) != (
        "2.11.0+cu128", "12.8", 14
    ):
        raise SystemExit("Colab Torch/CUDA/T4 memory policy drift")
    if runtime.get("python_policy") != "record-the-base-image-version;do-not-replace":
        raise SystemExit("Colab Python image policy drift")
    source_contract = {}
    for relative in lock.get("source_contract", []):
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Colab source contract file missing: {relative}")
        source_contract[relative] = sha(path)
    return {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-colab-runtime-freeze-v1",
        "status": "frozen-before-result-bearing-execution;input-gated",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "lock_id": lock["lock_id"],
        "lock_sha256": sha(LOCK),
        "amendment_sha256": sha(AMENDMENT),
        "smoke_fixture_sha256": sha(FIXTURE),
        "protocol_sha256": sha(PROTOCOL),
        "source_contract_sha256": source_contract,
        "failure_policy": lock["failure_policy"],
        "smoke_fixture_policy": {
            "path": str(FIXTURE.relative_to(ROOT)),
            "is_lockbox_query": False,
            "must_be_hashed_before_model_execution": True,
        },
        "no_result_dependent_runtime_changes": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = manifest()
    if args.check:
        if not OUT.is_file():
            raise SystemExit(f"Colab runtime freeze missing: {OUT}")
        actual = json.loads(OUT.read_text())
        for key, value in expected.items():
            if key == "frozen_utc":
                continue
            if actual.get(key) != value:
                raise SystemExit(f"Colab runtime freeze drift: {key}")
        print(json.dumps(actual, indent=2, sort_keys=True))
        return 0
    OUT.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(json.dumps(expected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
