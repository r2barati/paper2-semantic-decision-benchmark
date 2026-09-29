"""Frozen v5 Qwen3-8B primary consumer job.

This kernel uses the pinned AWQ checkpoint through a pinned vLLM server. It
receives only qrel-free evidence rows and source snapshots, writes every cache
raw response, and fails closed on schema, call-count, or coverage errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3-8B-AWQ"
EXPECTED_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
EXPECTED_PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}
SYSTEMS = ("bm25", "rerank", "oracle")


def runtime_spec() -> dict:
    path = os.environ.get("SEM2ACT_RUNTIME_LOCK")
    if not path:
        raise SystemExit("Lightning Qwen requires SEM2ACT_RUNTIME_LOCK")
    lock = json.loads(Path(path).read_text())
    spec = lock["models"]["qwen"]
    if spec["model_id"] != MODEL or spec["revision"] != EXPECTED_REVISION:
        raise SystemExit("Qwen runtime model pin drift")
    if spec["tokenizer_revision"] != EXPECTED_REVISION:
        raise SystemExit("Qwen tokenizer revision drift")
    return spec


def working_root() -> Path:
    path = Path(os.environ.get("SEM2ACT_OUTPUT_ROOT", "/kaggle/working"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_input(name: str) -> Path:
    matches = []
    roots = []
    if os.environ.get("SEM2ACT_INPUT_ROOT"):
        roots.append(os.environ["SEM2ACT_INPUT_ROOT"])
    roots.extend(("/kaggle/input", "/kaggle/working", "."))
    for root_name in roots:
        root = Path(root_name)
        if root.exists():
            matches.extend(p for p in root.rglob(name) if p.is_file())
    if not matches:
        raise FileNotFoundError(name)
    return sorted(set(matches), key=lambda p: str(p))[0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if any("qrel" in path.name.lower()
           for root_name in (os.environ.get("SEM2ACT_INPUT_ROOT", "/kaggle/input"), "/kaggle/working")
           for root, _, names in os.walk(root_name)
           for path in [Path(root) / name for name in names]):
        raise SystemExit("qrel-bearing input detected")
    spec_path = find_input("model_spec.json")
    spec = json.loads(spec_path.read_text())
    if spec.get("model_id") != MODEL or spec.get("revision") != EXPECTED_REVISION:
        raise SystemExit("Qwen model pin drift")
    if spec.get("protocol_hash") != spec.get("expected_protocol_hash"):
        raise SystemExit("protocol hash mismatch")
    input_path = find_input("evidence_inputs.jsonl")
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    if len(rows) != 720:
        raise SystemExit(f"expected 720 evidence rows, got {len(rows)}")
    keys = {(row["query_id"], row["system"]) for row in rows}
    expected = {
        (f"v5-lb-q{i:04d}", system)
        for i in range(1, 241) for system in SYSTEMS
    }
    if keys != expected or len(keys) != len(rows):
        raise SystemExit("Qwen evidence keyspace failed")
    if any(len(row.get("documents", [])) != 3 for row in rows):
        raise SystemExit("Qwen evidence depth drift")

    source_files = ("consumers_v3.py", "interpreter.py", "events.py")
    out_root = working_root()
    source_dir = out_root / "v5src" / "src"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "__init__.py").write_text("")
    for name in source_files:
        shutil.copy2(find_input(name), source_dir / name)
    sys.path.insert(0, str(source_dir.parent))
    import src.consumers_v3 as consumers

    if {
        "C1": consumers.prompt_sha(consumers.PROMPT_C1),
        "C3": consumers.prompt_sha(consumers.PROMPT_C3_DOC),
    } != EXPECTED_PROMPT_SHAS or consumers.ABSTAIN_TAU != 0.5:
        raise SystemExit("consumer prompt or aggregation pin drift")

    locked = runtime_spec()
    import torch
    before_torch_cuda = (torch.__version__, str(torch.version.cuda or ""))
    lock_path = os.environ.get("SEM2ACT_RUNTIME_LOCK")
    runtime_lock = json.loads(Path(lock_path).read_text())
    expected_torch_cuda = (
        runtime_lock["runtime"].get("torch"),
        runtime_lock["runtime"].get("cuda"),
    )
    if expected_torch_cuda[0] and before_torch_cuda != expected_torch_cuda:
        raise SystemExit(f"base Torch/CUDA mismatch: {before_torch_cuda}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "vllm==0.11.0", "openai==2.48.0", "pydantic==2.12.5",
         "transformers==4.57.6"],
        check=True,
    )
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import json,torch; print(json.dumps([torch.__version__, str(torch.version.cuda or '')]))"],
        check=True, capture_output=True, text=True,
    )
    after_torch_cuda = tuple(json.loads(torch_probe.stdout.strip().splitlines()[-1]))
    if after_torch_cuda != before_torch_cuda:
        raise SystemExit(f"package installation changed base Torch/CUDA: {before_torch_cuda} -> {after_torch_cuda}")
    os.environ["PAPER2_V3_CACHE_DIR"] = str(out_root / "cache_out")
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
    os.environ["LLM_API_KEY"] = "local"
    Path(os.environ["PAPER2_V3_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    server_env = os.environ.copy()
    server_env["VLLM_ATTENTION_BACKEND"] = locked["attention_backend"]
    server = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--revision", EXPECTED_REVISION,
         "--tokenizer", MODEL, "--tokenizer-revision", locked["tokenizer_revision"],
         "--dtype", locked["dtype"], "--quantization", "awq",
         "--max-model-len", str(locked["max_context_tokens"]),
         "--max-num-seqs", str(locked["max_num_seqs"]),
         "--max-num-batched-tokens", str(locked["max_num_batched_tokens"]),
         "--gpu-memory-utilization", str(locked["gpu_memory_utilization"]),
         "--enforce-eager", "--seed", "0", "--port", "8000"],
        env=server_env,
        stdout=open("/tmp/sem2act-v5-qwen-vllm.log", "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
                break
            except Exception:
                time.sleep(15)
        else:
            raise SystemExit("vLLM server did not become healthy")

        original_client = consumers._client

        def patched_client():
            client = original_client()
            original_create = client.chat.completions.create

            def create(*args, **kwargs):
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
                return original_create(*args, **kwargs)

            client.chat.completions.create = create
            return client

        consumers._client = patched_client
        belief_rows = []
        failures = []
        n_calls = 0
        for index, row in enumerate(
            sorted(rows, key=lambda item: (item["system"], item["query_id"])), 1
        ):
            docs = [
                {"doc_id": item["doc_id"], "text": item["text"]}
                for item in row["documents"]
            ]
            for consumer_name in ("C1", "C3"):
                try:
                    if consumer_name == "C1":
                        belief = consumers.consume_c1(
                            docs, row["query_text"], MODEL
                        )
                        n_calls += 1
                    else:
                        belief = consumers.consume_c3(
                            docs, row["query_text"], MODEL,
                            row["metadata"]["entity_node"]
                        )
                        n_calls += 3
                    belief_rows.append({
                        "query_id": row["query_id"],
                        "system": row["system"],
                        "consumer": consumer_name,
                        "model": MODEL,
                        "k": 3,
                        "true_regime": row["metadata"]["true_regime"],
                        "p_normal": belief.p_normal,
                        "p_supplier_delay": belief.p_supplier_delay,
                        "p_demand_surge": belief.p_demand_surge,
                        "abstain": bool(belief.abstain),
                        "confidence": belief.confidence,
                        "evidence_ids": list(belief.evidence_ids),
                    })
                except Exception as exc:
                    failures.append({
                        "query_id": row["query_id"],
                        "system": row["system"],
                        "consumer": consumer_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            if index % 20 == 0:
                print(f"{index}/720 rows calls={n_calls} failures={len(failures)}",
                      flush=True)

        cache_dir = Path(os.environ["PAPER2_V3_CACHE_DIR"])
        cache_files = sorted(cache_dir.glob("*.json"))
        raw_path = out_root / "raw_outputs.jsonl"
        with raw_path.open("w") as handle:
            for path in cache_files:
                payload = json.loads(path.read_text())
                payload["cache_file"] = path.name
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        beliefs_path = out_root / "beliefs.jsonl"
        with beliefs_path.open("w") as handle:
            for row in sorted(
                belief_rows,
                key=lambda item: (item["system"], item["consumer"], item["query_id"])
            ):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if n_calls != 2880:
            failures.append({"error": f"expected 2880 calls, got {n_calls}"})
        if len(cache_files) != 2880:
            failures.append({
                "error": f"expected 2880 cache files, got {len(cache_files)}"
            })
        if len(belief_rows) != 1440:
            failures.append({
                "error": f"expected 1440 belief rows, got {len(belief_rows)}"
            })
        expected_belief_keys = {
            (f"v5-lb-q{i:04d}", system, consumer)
            for i in range(1, 241)
            for system in SYSTEMS
            for consumer in ("C1", "C3")
        }
        actual_belief_keys = {
            (row["query_id"], row["system"], row["consumer"])
            for row in belief_rows
        }
        if actual_belief_keys != expected_belief_keys:
            failures.append({"error": "belief keyspace failed"})
        manifest = {
            "schema_version": 1,
            "experiment_id": "v5-primary-qwen-consumer",
            "status": "pass" if not failures else "failed",
            "model": MODEL,
            "revision": EXPECTED_REVISION,
            "systems": list(SYSTEMS),
            "consumers": ["C1", "C3"],
            "k": 3,
            "n_input_rows": len(rows),
            "expected_calls": 2880,
            "actual_calls": n_calls,
            "n_cache_files": len(cache_files),
            "n_fail": len(failures),
            "qrels_read": False,
            "prompt_shas": EXPECTED_PROMPT_SHAS,
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "enable_thinking": False,
            },
            "runtime_pins": {
                "vllm": "0.11.0",
                "openai": "2.48.0",
                "pydantic": "2.12.5",
            },
            "input_sha256": {
                "evidence_inputs.jsonl": sha(input_path),
                "model_spec.json": sha(spec_path),
            },
            "source_sha256": {
                name: sha(find_input(name)) for name in source_files
            },
            "outputs": {
                "raw_outputs": "raw_outputs.jsonl",
                "beliefs": "beliefs.jsonl",
                "failures": "failures.json",
            },
        }
        (out_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        (out_root / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        (out_root / "runtime_versions.json").write_text(json.dumps({
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "vllm": __import__("vllm").__version__,
            "transformers": __import__("transformers").__version__,
        }, indent=2) + "\n")
        print(json.dumps(manifest, indent=2), flush=True)
        if failures:
            raise SystemExit("Qwen consumer failed closed")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
