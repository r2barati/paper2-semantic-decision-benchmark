"""Freeze and verify the prospective Kaggle execution environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/kaggle_runtime_lock.json"
AMENDMENT = ROOT / "versions/sem2act-v5/amendments/kaggle_backend_v3.yaml"
FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
PROTOCOL = ROOT / "versions/sem2act-v5/protocol/confirmation.yaml"
PROTOCOL_FREEZE = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
OUT = ROOT / "versions/sem2act-v5/manifests/kaggle_runtime_freeze.json"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def validate() -> dict:
    for path in (LOCK, AMENDMENT, FIXTURE, PROTOCOL, PROTOCOL_FREEZE):
        if not path.exists():
            raise SystemExit(f"required freeze input missing: {path}")
    lock = load(LOCK)
    protocol_freeze = load(PROTOCOL_FREEZE)
    if lock.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("Kaggle runtime lock protocol hash drift")
    if protocol_freeze.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("protocol freeze hash drift")
    if lock.get("status") != "prospective-before-result-bearing-execution":
        raise SystemExit("Kaggle runtime lock must remain prospective before results")
    if lock.get("platform", {}).get("provider") != "kaggle":
        raise SystemExit("Kaggle runtime lock provider drift")
    if lock.get("platform", {}).get("machine") != "NvidiaTeslaT4":
        raise SystemExit("Kaggle runtime lock machine drift")
    if lock.get("platform", {}).get("machine_shape_request") != "NvidiaTeslaT4":
        raise SystemExit("Kaggle runtime lock machine-shape request drift")
    if (lock.get("platform", {}).get("gpu_count") != 2
            or lock.get("platform", {}).get("visible_gpu_count") != 1):
        raise SystemExit("Kaggle allocation/visible-GPU policy drift")
    if lock.get("platform", {}).get("owner") != "rezabarati2":
        raise SystemExit("Kaggle runtime lock owner drift")
    if lock.get("runtime", {}).get("torch") != "2.10.0+cu128" or lock.get("runtime", {}).get("cuda") != "12.8":
        raise SystemExit("Kaggle base torch/CUDA policy drift")
    if lock.get("runtime", {}).get("package_install_policy") != "exact-pins-preserve-image-torch-cuda-tuple":
        raise SystemExit("Kaggle package policy must preserve base Torch/CUDA")
    for family, spec in lock.get("models", {}).items():
        for key in ("model_id", "revision", "tokenizer_revision", "dtype",
                    "backend", "device_map", "attention_backend",
                    "max_context_tokens", "batch_size"):
            if key not in spec:
                raise SystemExit(f"{family}: missing runtime field {key}")
        if spec["revision"] != spec["tokenizer_revision"]:
            raise SystemExit(f"{family}: model/tokenizer revision mismatch")
        if spec["dtype"] != "float16":
            raise SystemExit(f"{family}: dtype drift")
    source_contract = []
    for relative in lock.get("source_contract", []):
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"source contract file missing: {relative}")
        source_contract.append(path)
    return {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-kaggle-runtime-freeze-v3",
        "status": "prospective-before-result-bearing-execution",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "lock_id": lock["lock_id"],
        "lock_sha256": sha256(LOCK),
        "amendment_sha256": sha256(AMENDMENT),
        "smoke_fixture_sha256": sha256(FIXTURE),
        "protocol_sha256": sha256(PROTOCOL),
        "source_contract_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_contract
        },
        "smoke_fixture_policy": {
            "path": str(FIXTURE.relative_to(ROOT)),
            "is_lockbox_query": False,
            "must_be_hashed_before_model_execution": True,
        },
        "failure_policy": lock["failure_policy"],
        "no_result_dependent_runtime_changes": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = validate()
    if args.check:
        if not OUT.exists():
            raise SystemExit(f"missing freeze manifest: {OUT}")
        existing = load(OUT)
        for key in (
            "protocol_hash", "lock_id", "lock_sha256", "amendment_sha256",
            "smoke_fixture_sha256", "protocol_sha256", "source_contract_sha256",
        ):
            if existing.get(key) != manifest[key]:
                raise SystemExit(f"Kaggle freeze drift: {key}")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
