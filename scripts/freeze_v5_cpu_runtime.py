"""Freeze or verify the prospective Sem2Act-v5 CPU runtime.

The command intentionally refuses to freeze until all three official GGUF
artifacts, conversion provenance, hardware/thread settings, and the complete
non-lockbox canary result exist and pass. It never launches inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions/sem2act-v5/manifests/cpu_runtime_lock.json"
AMENDMENT = ROOT / "versions/sem2act-v5/amendments/cpu_backend_v1.yaml"
ARTIFACTS = ROOT / "versions/sem2act-v5/manifests/cpu_model_artifacts.json"
PROTOCOL = ROOT / "versions/sem2act-v5/protocol/confirmation.yaml"
PROTOCOL_FREEZE = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
FIXTURE = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
CANARY = ROOT / "versions/sem2act-v5/manifests/cpu_canary.json"
RERANKER_CANARY = ROOT / "versions/sem2act-v5/manifests/cpu_reranker_canary.json"
HOST = ROOT / "versions/sem2act-v5/manifests/cpu_host_preflight.json"
ENGINE_BUILD = ROOT / "versions/sem2act-v5/manifests/cpu_engine_build.json"
OUT = ROOT / "versions/sem2act-v5/manifests/cpu_runtime_freeze.json"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_inputs() -> dict:
    paths = (
        LOCK, AMENDMENT, ARTIFACTS, PROTOCOL, PROTOCOL_FREEZE, FIXTURE,
        CANARY, RERANKER_CANARY, HOST, ENGINE_BUILD,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("CPU freeze inputs missing: " + ", ".join(missing))
    lock = read(LOCK)
    if lock.get("status") != "prospective-before-cpu-preflight":
        raise SystemExit("CPU runtime lock must remain prospective until this freeze")
    if lock.get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("CPU freeze protocol hash drift")
    if read(PROTOCOL_FREEZE).get("protocol_hash") != PROTOCOL_HASH:
        raise SystemExit("protocol freeze hash drift")
    source_paths = {}
    for relative in lock.get("source_contract", []):
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"CPU source contract file missing: {relative}")
        source_paths[relative] = path
    artifacts = read(ARTIFACTS)
    if artifacts.get("status") != "complete-before-result-bearing-execution":
        raise SystemExit("CPU artifact manifest is not complete")
    for name in ("conversion_script_sha256", "quantizer_binary_sha256"):
        if not artifacts.get(name):
            raise SystemExit(f"missing CPU artifact provenance hash: {name}")
    for family, spec in artifacts.get("models", {}).items():
        artifact = Path(spec.get("path", ""))
        if not artifact.exists() or not spec.get("sha256"):
            raise SystemExit(f"{family}: missing GGUF path/hash")
        if sha(artifact) != spec["sha256"]:
            raise SystemExit(f"{family}: GGUF hash mismatch")
    engine = read(ENGINE_BUILD)
    if engine.get("status") != "pass-before-model-conversion" or engine.get("commit") != lock["engine"]["commit"]:
        raise SystemExit("CPU engine build provenance is not valid")
    if any(not item.get("sha256") for item in engine.get("binaries", {}).values()):
        raise SystemExit("CPU engine binary hashes are missing")
    host = read(HOST)
    if host.get("status") != "pass" or int(host.get("frozen_threads", 0)) < 1:
        raise SystemExit("CPU host preflight is not a passing frozen profile")
    if not host.get("llama_server") or not host.get("llama_server_sha256"):
        raise SystemExit("CPU host preflight lacks the pinned llama-server binary hash")
    canary = read(CANARY)
    reranker_canary = read(RERANKER_CANARY)
    if canary.get("status") != "pass":
        raise SystemExit("CPU consumer canary did not pass")
    if canary.get("deterministic_repeat_equality") is not True:
        raise SystemExit("CPU consumer canary did not pass deterministic repeat equality")
    if set(canary.get("models", {})) != {"qwen", "llama", "mistral"}:
        raise SystemExit("CPU consumer canary does not cover all three families")
    if any(
        spec.get("status") != "pass" or spec.get("schema_valid") is not True
        for spec in canary["models"].values()
    ):
        raise SystemExit("CPU consumer canary has a model failure")
    if reranker_canary.get("status") != "pass":
        raise SystemExit("CPU reranker canary did not pass")
    if reranker_canary.get("schema_valid") is not True:
        raise SystemExit("CPU reranker canary schema check failed")
    if reranker_canary.get("deterministic_repeat_equality") is not True:
        raise SystemExit("CPU reranker canary did not pass deterministic repeat equality")
    return {
        "lock": lock,
        "artifacts": artifacts,
        "host": host,
        "engine": engine,
        "canary": canary,
        "reranker_canary": reranker_canary,
        "source_paths": source_paths,
    }


def make_manifest(inputs: dict) -> dict:
    return {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-cpu-runtime-freeze-v1",
        "status": "frozen-before-result-bearing-execution",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": PROTOCOL_HASH,
        "lock_id": inputs["lock"]["lock_id"],
        "lock_sha256": sha(LOCK),
        "amendment_sha256": sha(AMENDMENT),
        "artifact_manifest_sha256": sha(ARTIFACTS),
        "host_preflight_sha256": sha(HOST),
        "engine_build_sha256": sha(ENGINE_BUILD),
        "canary_sha256": sha(CANARY),
        "reranker_canary_sha256": sha(RERANKER_CANARY),
        "smoke_fixture_sha256": sha(FIXTURE),
        "protocol_sha256": sha(PROTOCOL),
        "source_contract_sha256": {
            relative: sha(path) for relative, path in inputs["source_paths"].items()
        },
        "llama_cpp_commit": inputs["lock"]["engine"]["commit"],
        "quantization_type": inputs["lock"]["engine"]["quantization_type"],
        "model_artifact_sha256": {
            family: spec["sha256"]
            for family, spec in inputs["artifacts"]["models"].items()
        },
        "frozen_threads": inputs["host"]["frozen_threads"],
        "frozen_threads_batch": inputs["host"]["frozen_threads_batch"],
        "llama_server_sha256": inputs["host"]["llama_server_sha256"],
        "failure_policy": inputs["lock"]["failure_policy"],
        "no_result_dependent_runtime_changes": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.freeze and not args.check:
        parser.error("choose --freeze or --check")
    if args.check:
        if not OUT.exists():
            raise SystemExit(f"missing CPU runtime freeze: {OUT}")
        inputs = validate_inputs()
        expected = make_manifest(inputs)
        existing = read(OUT)
        for key, value in expected.items():
            if key == "frozen_utc":
                continue
            if existing.get(key) != value:
                raise SystemExit(f"CPU runtime freeze drift: {key}")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    inputs = validate_inputs()
    manifest = make_manifest(inputs)
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
