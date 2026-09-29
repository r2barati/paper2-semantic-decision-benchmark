"""Shared fail-closed contracts for the v5 CPU execution branch.

This module contains no result data and never reads qrels.  It centralizes the
frozen prompt, keyspace, server, hashing, and CPU-only rules used by the
preflight, canary, shard planner, and result runners.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".git").exists() and (candidate / "versions").exists():
            return candidate
    raise RuntimeError("could not locate repository root")


ROOT = repo_root()
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
CPU_LOCK_REL = Path("versions/sem2act-v5/manifests/cpu_runtime_lock.json")
CPU_FREEZE_REL = Path("versions/sem2act-v5/manifests/cpu_runtime_freeze.json")
CPU_ARTIFACTS_REL = Path("versions/sem2act-v5/manifests/cpu_model_artifacts.json")
FIXTURE_REL = Path("versions/sem2act-v5/fixtures/runtime_smoke_v1.json")
GPU_CONSUMER_REL = Path("versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(relative: Path) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text())


def load_gpu_consumer_contract():
    path = ROOT / GPU_CONSUMER_REL
    spec = importlib.util.spec_from_file_location("sem2act_v5_gpu_consumer_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen consumer contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GPU_CONSUMER = load_gpu_consumer_contract()
PROMPT_C1 = GPU_CONSUMER.PROMPT_C1
PROMPT_C3_DOC = GPU_CONSUMER.PROMPT_C3_DOC
PROMPT_SHAS = dict(GPU_CONSUMER.EXPECTED_PROMPT_SHAS)
SYSTEMS_QWEN = ("bm25", "rerank", "oracle")
SYSTEMS_CROSS_FAMILY = ("rerank", "oracle")
CONSUMERS = ("C1", "C3")


def load_cpu_lock() -> dict[str, Any]:
    lock = read_json(CPU_LOCK_REL)
    if lock.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("CPU runtime lock protocol hash drift")
    if lock.get("platform", {}).get("accelerator") != "cpu_only":
        raise RuntimeError("CPU runtime lock is not CPU-only")
    if lock.get("platform", {}).get("cuda") is not False:
        raise RuntimeError("CPU runtime lock permits CUDA")
    if lock.get("platform", {}).get("metal") is not False:
        raise RuntimeError("CPU runtime lock permits Metal")
    if lock.get("engine", {}).get("quantization_type") != "Q4_K_M":
        raise RuntimeError("CPU quantization drift")
    if lock.get("engine", {}).get("commit") != "9f70b2cecd1a9a3f73ac525c47ca22a6ee9a7b69":
        raise RuntimeError("llama.cpp commit drift")
    for family in ("qwen", "llama", "mistral"):
        spec = lock.get("models", {}).get(family, {})
        expected = {
            "backend": "llama.cpp",
            "dtype": "ggml_q4_k_m",
            "bitsandbytes": "disabled",
            "device_map": "cpu",
            "max_context_tokens": 4096,
            "batch_size": 512,
            "ubatch_size": 512,
            "attention_backend": "llama.cpp_cpu",
        }
        for key, value in expected.items():
            if spec.get(key) != value:
                raise RuntimeError(f"{family}: CPU inference setting drift: {key}")
    reranker = lock.get("models", {}).get("reranker", {})
    for key, value in {"bitsandbytes": "disabled", "device_map": "cpu"}.items():
        if reranker.get(key) != value:
            raise RuntimeError(f"reranker: CPU inference setting drift: {key}")
    return lock


def load_cpu_freeze(require_frozen: bool = True) -> dict[str, Any]:
    path = ROOT / CPU_FREEZE_REL
    if not path.exists():
        raise RuntimeError(f"missing CPU runtime freeze: {path}")
    manifest = json.loads(path.read_text())
    if require_frozen and manifest.get("status") != "frozen-before-result-bearing-execution":
        raise RuntimeError("CPU runtime is not frozen")
    if manifest.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("CPU runtime freeze protocol hash drift")
    return manifest


def load_artifacts(require_complete: bool = False) -> dict[str, Any]:
    artifacts = read_json(CPU_ARTIFACTS_REL)
    if artifacts.get("engine_commit") != "9f70b2cecd1a9a3f73ac525c47ca22a6ee9a7b69":
        raise RuntimeError("artifact engine commit drift")
    if artifacts.get("quantization_type") != "Q4_K_M":
        raise RuntimeError("artifact quantization drift")
    for family, spec in artifacts.get("models", {}).items():
        if spec.get("revision") != spec.get("tokenizer_revision"):
            raise RuntimeError(f"{family}: model/tokenizer revision mismatch")
        if require_complete:
            path = spec.get("path")
            digest = spec.get("sha256")
            if not path or not digest:
                raise RuntimeError(f"{family}: GGUF artifact hash/path is missing")
            artifact_path = Path(path)
            if not artifact_path.exists():
                raise RuntimeError(f"{family}: GGUF artifact missing: {artifact_path}")
            if sha256_file(artifact_path) != digest:
                raise RuntimeError(f"{family}: GGUF artifact hash mismatch")
    if require_complete:
        for field in ("conversion_script_sha256", "quantizer_binary_sha256"):
            if not artifacts.get(field):
                raise RuntimeError(f"missing artifact provenance hash: {field}")
    return artifacts


def load_fixture() -> dict[str, Any]:
    fixture = read_json(FIXTURE_REL)
    if fixture.get("fixture_id") != "sem2act-v5-runtime-smoke-v1":
        raise RuntimeError("unexpected CPU smoke fixture")
    if str(fixture.get("query_id", "")).startswith("v5-lb-"):
        raise RuntimeError("CPU smoke fixture collides with lockbox")
    if len(fixture.get("documents", [])) != 3:
        raise RuntimeError("CPU smoke fixture must contain exactly three documents")
    return fixture


def cpu_environment() -> dict[str, Any]:
    """Capture safe host facts; never include credentials or model text."""
    physical = None
    logical = os.cpu_count() or 1
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True) or logical
        memory_bytes = int(psutil.virtual_memory().total)
    except Exception:
        memory_bytes = None
        if sys.platform == "darwin":
            try:
                memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            except Exception:
                pass
    if physical is None:
        physical = logical
    thread_count = max(1, min(int(physical), 8))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": platform.processor() or "unknown",
        "physical_cores": int(physical),
        "logical_cores": int(logical),
        "memory_bytes": memory_bytes,
        "memory_gib": None if memory_bytes is None else round(memory_bytes / 2**30, 3),
        "frozen_threads_default": thread_count,
        "cuda_visible_devices_before_override": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "metal_disabled": True,
        "cuda_disabled": True,
    }


def cpu_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["GGML_CUDA"] = "0"
    env["GGML_METAL"] = "0"
    env["GGML_VULKAN"] = "0"
    env["GGML_SYCL"] = "0"
    return env


def expected_systems(family: str) -> tuple[str, ...]:
    if family == "qwen":
        return SYSTEMS_QWEN
    if family in {"llama", "mistral"}:
        return SYSTEMS_CROSS_FAMILY
    raise ValueError(f"unknown CPU consumer family: {family}")


def call_keys_for_row(row: dict[str, Any], family: str) -> list[str]:
    keys: list[str] = []
    qid = row["query_id"]
    system = row["system"]
    if system not in expected_systems(family):
        return keys
    keys.append(f"{family}|{qid}|{system}|C1|")
    for doc in row.get("documents", []):
        keys.append(f"{family}|{qid}|{system}|C3|{doc['doc_id']}")
    return keys


def belief_key(row: dict[str, Any], family: str, consumer: str) -> str:
    return f"{family}|{row['query_id']}|{row['system']}|{consumer}"


def expected_call_keys(rows: Iterable[dict[str, Any]], family: str) -> list[str]:
    result: list[str] = []
    for row in sorted(rows, key=lambda item: (item["query_id"], item["system"])):
        result.extend(call_keys_for_row(row, family))
    return result


def expected_belief_keys(rows: Iterable[dict[str, Any]], family: str) -> list[str]:
    result: list[str] = []
    for row in sorted(rows, key=lambda item: (item["query_id"], item["system"])):
        if row["system"] in expected_systems(family):
            result.extend([
                belief_key(row, family, "C1"),
                belief_key(row, family, "C3"),
            ])
    return result


def server_command(binary: Path, model_path: Path, threads: int, threads_batch: int, port: int) -> list[str]:
    lock = load_cpu_lock()
    server = lock["server"]
    return [
        str(binary),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--n-gpu-layers", str(server["gpu_layers"]),
        "--ctx-size", str(server["context_tokens"]),
        "--batch-size", str(server["batch_size"]),
        "--ubatch-size", str(server["ubatch_size"]),
        "--threads", str(threads),
        "--threads-batch", str(threads_batch),
        "--parallel", str(server["parallel_slots"]),
        "--jinja",
        "--reasoning", server["reasoning"],
        "--reasoning-format", server["reasoning_format"],
        "--no-slots",
    ]


def post_chat(base_url: str, model: str, messages: list[dict[str, str]], timeout: int = 1800) -> dict[str, Any]:
    lock = load_cpu_lock()
    decoding = lock["decoding"]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "seed": decoding["seed"],
        "max_tokens": decoding["max_new_tokens"],
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {exc.code}: {body[:500]}") from exc


def response_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("llama-server response missing choices[0].message.content") from exc
    if not isinstance(content, str):
        raise ValueError("llama-server response content is not text")
    return content


def wait_for_server(base_url: str, process: subprocess.Popen[Any], timeout: int = 1800) -> dict[str, Any]:
    import time

    deadline = time.time() + timeout
    last_error = "not contacted"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            request = urllib.request.Request(base_url.rstrip("/") + "/health")
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return {"http_status": response.status}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"llama-server did not become healthy: {last_error}")


def model_server_id(base_url: str) -> str:
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/models")
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    try:
        return str(payload["data"][0]["id"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("llama-server returned no model id") from exc
