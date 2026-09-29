"""Non-lockbox CPU model canary for the v5 runtime.

The canary loads one official GGUF at a time, sends only the fixed smoke
fixture, validates C1/C3 output, and repeats each request to test exact
determinism.  It never reads the lockbox or qrels.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contract import (  # noqa: E402
    ROOT,
    PROMPT_C1,
    PROMPT_C3_DOC,
    cpu_subprocess_env,
    load_artifacts,
    load_cpu_lock,
    load_fixture,
    model_server_id,
    post_chat,
    response_content,
    server_command,
    sha256_bytes,
    wait_for_server,
)


def parse_and_validate(contract, consumer: str, raw: str) -> dict:
    parsed = contract.parse_json(raw)
    return contract.validate_c1(parsed) if consumer == "C1" else contract.validate_c3(parsed)


def user_for(fixture: dict, consumer: str, doc: dict | None = None) -> str:
    if consumer == "C1":
        return (
            f"Operator information need: {fixture['query_text']}\n\nEvidence:\n"
            + "\n\n".join(f"[DOC {item['doc_id']}] {item['text']}" for item in fixture["documents"])
        )
    assert doc is not None
    return f"Operator's own node: {fixture['entity_node']}\n\nEvidence document:\n{doc['text']}"


def peak_rss_mb() -> float | None:
    value = getattr(resource.getrusage(resource.RUSAGE_SELF), "ru_maxrss", None)
    if value is None:
        return None
    # macOS reports bytes; Linux reports KiB.
    return round(value / (1024 * 1024 if sys.platform == "darwin" else 1024), 3)


def one_model_canary(binary: Path, family: str, model_path: Path, threads: int, threads_batch: int,
                     fixture: dict, contract, port: int) -> dict:
    command = server_command(binary, model_path, threads, threads_batch, port)
    env = cpu_subprocess_env()
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    })
    log_path = ROOT / "versions/sem2act-v5/runtime" / f"cpu_canary_{family}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        health = wait_for_server(f"http://127.0.0.1:{port}", process, timeout=1800)
        model_name = model_server_id(f"http://127.0.0.1:{port}")
        calls = []
        prompts = [("C1", PROMPT_C1, user_for(fixture, "C1"))]
        prompts.extend(("C3", PROMPT_C3_DOC, user_for(fixture, "C3", doc)) for doc in fixture["documents"])
        for repeat in (1, 2):
            for consumer, prompt, user in prompts:
                response = post_chat(
                    f"http://127.0.0.1:{port}", model_name,
                    [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                    timeout=1800,
                )
                raw = response_content(response)
                parsed = parse_and_validate(contract, consumer, raw)
                calls.append({
                    "repeat": repeat,
                    "consumer": consumer,
                    "raw_sha256": sha256_bytes(raw.encode("utf-8")),
                    "raw": raw,
                    "parsed": parsed,
                })
        first = [item["raw"] for item in calls[:len(prompts)]]
        second = [item["raw"] for item in calls[len(prompts):]]
        return {
            "status": "pass",
            "family": family,
            "model_path": str(model_path),
            "server_command": command,
            "health": health,
            "schema_valid": True,
            "expected_calls": len(prompts) * 2,
            "actual_calls": len(calls),
            "deterministic_repeat_equality": first == second,
            "calls": calls,
            "wall_seconds": round(time.monotonic() - started, 3),
            "peak_rss_mb": peak_rss_mb(),
            "tokens_per_second": None,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--host-manifest", type=Path,
                        default=ROOT / "versions/sem2act-v5/manifests/cpu_host_preflight.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "versions/sem2act-v5/manifests/cpu_canary.json")
    parser.add_argument("--base-port", type=int, default=18765)
    args = parser.parse_args()
    output = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-cpu-canary-v1",
        "status": "failed",
        "fixture_id": "sem2act-v5-runtime-smoke-v1",
        "result_bearing_execution_started": False,
        "models": {},
    }
    try:
        lock = load_cpu_lock()
        fixture = load_fixture()
        artifacts = load_artifacts(require_complete=True)
        host = json.loads(args.host_manifest.read_text())
        threads = int(host["frozen_threads"])
        threads_batch = int(host["frozen_threads_batch"])
        if args.artifact_manifest and Path(args.artifact_manifest).resolve() != (ROOT / "versions/sem2act-v5/manifests/cpu_model_artifacts.json").resolve():
            raise RuntimeError("unexpected artifact manifest path")
        contract = __import__("importlib.util").util.spec_from_file_location(
            "sem2act_v5_cpu_consumer_contract", ROOT / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py"
        )
        module = __import__("importlib.util").util.module_from_spec(contract)
        contract.loader.exec_module(module)
        for offset, family in enumerate(("qwen", "llama", "mistral")):
            spec = artifacts["models"][family]
            output["models"][family] = one_model_canary(
                args.llama_server, family, Path(spec["path"]), threads, threads_batch,
                fixture, module, args.base_port + offset,
            )
        output["status"] = "pass"
        output["deterministic_repeat_equality"] = all(
            item.get("deterministic_repeat_equality") is True for item in output["models"].values()
        )
        if not output["deterministic_repeat_equality"]:
            raise RuntimeError("at least one CPU model failed repeat equality")
        output["runtime_lock_id"] = lock["lock_id"]
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
