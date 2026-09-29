"""Fail-closed validation for the frozen v5 Lightning runtime contract.

This module contains no experiment logic.  It validates only the prospective
execution amendment, the immutable v5 protocol hash, and the exact model
runtime settings that a Lightning job is allowed to use.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any


PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
def configured_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


LOCK_REL = configured_path(
    "SEM2ACT_RUNTIME_LOCK", "versions/sem2act-v5/manifests/lightning_runtime_lock.json"
)
FREEZE_REL = configured_path(
    "SEM2ACT_RUNTIME_FREEZE", "versions/sem2act-v5/manifests/lightning_runtime_freeze.json"
)
FIXTURE_REL = Path("versions/sem2act-v5/fixtures/runtime_smoke_v1.json")
AMENDMENT_REL = Path("versions/sem2act-v5/amendments/lightning_backend_v1.yaml")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".git").exists() and (candidate / "versions").exists():
            return candidate
    raise RuntimeError("could not locate repository root")


ROOT = repo_root()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_lock(root: Path = ROOT) -> dict[str, Any]:
    path = root / LOCK_REL
    if not path.exists():
        raise RuntimeError(f"missing runtime lock: {path}")
    lock = read_json(path)
    if lock.get("status") != "frozen-before-result-bearing-execution":
        raise RuntimeError("runtime lock is not frozen")
    if lock.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("runtime lock protocol hash drift")
    runtime = lock.get("runtime", {})
    if lock.get("lock_id") == "sem2act-v5-colab-runtime-v1":
        if runtime.get("torch") != "2.11.0+cu128" or runtime.get("cuda") != "12.8":
            raise RuntimeError("Colab Torch/CUDA policy drift")
        if runtime.get("gpu_min_vram_gib") != 14:
            raise RuntimeError("Colab minimum VRAM policy drift")
        if runtime.get("python_policy") != "record-the-base-image-version;do-not-replace":
            raise RuntimeError("Colab Python image policy drift")
    return lock


def load_freeze_manifest(root: Path = ROOT) -> dict[str, Any]:
    path = root / FREEZE_REL
    if not path.exists():
        raise RuntimeError(f"missing runtime freeze manifest: {path}")
    manifest = read_json(path)
    lock = root / LOCK_REL
    lock_payload = read_json(lock)
    amendment = root / lock_payload["amendment"]
    fixture = root / FIXTURE_REL
    expected = {
        "lock_sha256": sha256(lock),
        "amendment_sha256": sha256(amendment),
        "smoke_fixture_sha256": sha256(fixture),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"runtime freeze manifest {key} mismatch")
    if manifest.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("runtime freeze manifest protocol hash drift")
    for relative, expected_sha in manifest.get("source_contract_sha256", {}).items():
        source = root / relative
        if not source.exists() or sha256(source) != expected_sha:
            raise RuntimeError(f"runtime freeze source contract mismatch: {relative}")
    return manifest


def load_fixture(root: Path = ROOT) -> dict[str, Any]:
    fixture = read_json(root / FIXTURE_REL)
    if fixture.get("fixture_id") != "sem2act-v5-runtime-smoke-v1":
        raise RuntimeError("unexpected smoke fixture id")
    query_id = fixture.get("query_id", "")
    if query_id.startswith("v5-lb-"):
        raise RuntimeError("smoke fixture is a lockbox query")
    if len(fixture.get("documents", [])) != 3:
        raise RuntimeError("smoke fixture must contain exactly three documents")
    return fixture


def validate_lock(root: Path = ROOT) -> dict[str, Any]:
    lock = load_lock(root)
    load_freeze_manifest(root)
    fixture = load_fixture(root)

    required_families = ("reranker", "qwen", "llama", "mistral")
    if tuple(lock.get("models", {})) != required_families:
        raise RuntimeError("runtime lock model family order or coverage drift")
    for family in required_families:
        spec = lock["models"][family]
        for key in ("model_id", "revision", "tokenizer_revision", "dtype",
                    "backend", "device_map", "attention_backend",
                    "max_context_tokens", "batch_size"):
            if key not in spec:
                raise RuntimeError(f"{family}: missing frozen runtime field {key}")
        if spec["revision"] != spec["tokenizer_revision"]:
            raise RuntimeError(f"{family}: model/tokenizer revision mismatch")
        if spec["dtype"] != "float16":
            raise RuntimeError(f"{family}: dtype drift")
        if family in ("llama", "mistral"):
            quant = spec.get("quantization", {})
            expected = {
                "method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
            }
            if quant != expected:
                raise RuntimeError(f"{family}: bitsandbytes configuration drift")
        if family == "qwen":
            if spec.get("quantization") != {"method": "awq"}:
                raise RuntimeError("qwen: AWQ configuration drift")
            if spec.get("attention_backend") != "TRITON_ATTN":
                raise RuntimeError("qwen: attention backend drift")
            if spec.get("max_num_seqs") != 1 or spec.get("max_num_batched_tokens") != 4096:
                raise RuntimeError("qwen: scheduler configuration drift")
        if family == "reranker" and spec.get("max_context_tokens") != 1024:
            raise RuntimeError("reranker: context configuration drift")

    if fixture["query_id"] in {
        f"v5-lb-q{i:04d}" for i in range(1, 241)
    }:
        raise RuntimeError("smoke fixture collides with the lockbox keyspace")
    return lock


def environment_snapshot() -> dict[str, Any]:
    """Return safe runtime diagnostics; never include credentials."""
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "lightning_job_id_present": bool(os.environ.get("LIGHTNING_JOB_ID")),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }
    try:
        import torch

        snapshot.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())],
            "device_total_memory_bytes": [
                int(torch.cuda.get_device_properties(i).total_memory)
                for i in range(torch.cuda.device_count())
            ],
        })
    except Exception as exc:  # pragma: no cover - exercised on remote image
        snapshot["torch_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def require_t4_environment() -> dict[str, Any]:
    snapshot = environment_snapshot()
    if not snapshot.get("cuda_available"):
        raise RuntimeError("CUDA is unavailable")
    if snapshot.get("device_count") != 1:
        raise RuntimeError(f"expected exactly one CUDA device: {snapshot}")
    name = snapshot["devices"][0].lower()
    if "t4" not in name:
        raise RuntimeError(f"expected an NVIDIA T4, got {snapshot['devices'][0]}")
    minimum_gib = load_lock().get("runtime", {}).get("gpu_min_vram_gib")
    if minimum_gib is not None:
        memory = snapshot["device_total_memory_bytes"][0] / 2**30
        if memory < float(minimum_gib):
            raise RuntimeError(f"expected at least {minimum_gib} GiB visible VRAM, got {memory:.2f} GiB")
    return snapshot


def package_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "bitsandbytes", "vllm", "openai", "pydantic", "yaml")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:
            versions[name] = None
    return versions
