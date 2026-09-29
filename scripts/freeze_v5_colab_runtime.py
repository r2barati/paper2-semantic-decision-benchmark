"""Create or verify the prospective Colab GPU runtime freeze manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/colab_runtime_lock_v1.json"
AMENDMENT = ROOT / "versions/sem2act-v5/amendments/colab_backend_v1.yaml"
FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
OUT = ROOT / "versions/sem2act-v5/manifests/colab_runtime_freeze_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected() -> dict:
    lock = json.loads(LOCK.read_text())
    if lock.get("protocol_hash") != "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022":
        raise SystemExit("Colab runtime lock protocol hash drift")
    if lock.get("runtime", {}).get("torch") != "2.11.0+cu128":
        raise SystemExit("Colab exact Torch runtime pin drift")
    sources = {}
    for rel in lock.get("source_contract", []):
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"Colab source contract missing: {rel}")
        sources[rel] = sha(path)
    return {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-colab-runtime-freeze-v1",
        "status": "prospective-before-result-bearing-execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": lock["protocol_hash"],
        "lock_id": lock["lock_id"],
        "lock_sha256": sha(LOCK),
        "amendment_sha256": sha(AMENDMENT),
        "fixture_sha256": sha(FIXTURE),
        "source_contract_sha256": sources,
        "runtime_policy": lock["runtime"],
        "result_bearing_execution_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record = expected()
    if args.check:
        if not OUT.exists():
            raise SystemExit(f"missing Colab runtime freeze: {OUT}")
        old = json.loads(OUT.read_text())
        for key, value in record.items():
            if key == "created_utc":
                continue
            if old.get(key) != value:
                raise SystemExit(f"Colab runtime freeze drift: {key}")
    else:
        OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(json.loads(OUT.read_text()) if args.freeze else record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
