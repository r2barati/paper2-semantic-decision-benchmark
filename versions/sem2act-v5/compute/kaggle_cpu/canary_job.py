"""Private Kaggle CPU preflight for one Sem2Act-v5 model family.

The job receives only a model configuration and the fixed non-lockbox smoke
fixture. It downloads the pinned official checkpoint, builds the pinned
llama.cpp commit when needed, runs the CPU canary, and writes provenance. It
never receives Sem2Act lockbox queries, evidence inputs, qrels, or results.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ENGINE_COMMIT = "9f70b2cecd1a9a3f73ac525c47ca22a6ee9a7b69"
ENGINE_TAG = "b11194"
ENGINE_SOURCE_DATASET = "siavashsimin/sem2act-v5-llama-cpp-b11194-source"
ENGINE_SOURCE_DIR = "/kaggle/input/sem2act-v5-llama-cpp-b11194-source/llama-cpp-b11194/llama.cpp"
ENGINE_SOURCE_MANIFEST_PATH = "/kaggle/input/sem2act-v5-llama-cpp-b11194-source/engine_source_manifest.json"
ENGINE_ARCHIVE_PATH = "/kaggle/input/sem2act-v5-llama-cpp-b11194-source/llama-cpp-b11194.tar.gz"
ENGINE_ARCHIVE_SHA256 = "bc351a39f7a2c68f0e9827f86d0185c4ec7617be29493c4f8d349ecf41bbcbc6"
ENGINE_SOURCE_TREE_SHA256 = "1bbfb63c28543063eadf2d55b9e74f63866d4d7d3fbfe448bab4a2d6e08a84e8"
QUANTIZATION = "Q4_K_M"
FIXTURE_NAME = "runtime_smoke_v1.json"
PROMPT_C1 = (
    "You are an operational risk analyst. Given a textual supplier or market "
    "warning, classify the likely operational regime. Respond ONLY with valid "
    "JSON matching this exact schema:\n"
    '{"normal": <float 0-1>, "supplier_delay": <float 0-1>, '
    '"demand_surge": <float 0-1>, '
    '"estimated_lt_increase": <int>, "estimated_duration": <int>, '
    '"estimated_demand_multiplier": <float>}\n'
    "The three regime probabilities should sum approximately to 1.0.\n"
    "estimated_lt_increase: additional lead time periods if supplier_delay.\n"
    "estimated_duration: how many periods the event lasts.\n"
    "estimated_demand_multiplier: demand multiplier if demand_surge (1.0 if not).\n"
    "Do not include any other text."
)
PROMPT_C3 = (
    "You are an operational risk analyst. Given ONE evidence document and the "
    "operator's own node described below, extract structured evidence. Respond ONLY "
    "with valid JSON matching this exact schema:\n"
    '{"entity_match": <true|false>, "event": <"supplier_delay"|"demand_surge"|"normal"|"none">, '
    '"fresh": <true|false>, "stance": <"support"|"refute"|"na">, '
    '"confidence": <float 0-1>}\n'
    "entity_match: whether the document concerns the operator's own node. "
    "fresh: whether it describes the current window (not resolved/closed). "
    "stance: support/refute relative to its own event claim. "
    "Do not include any other text."
)
PROMPT_SHAS = {
    "C1": hashlib.sha256(PROMPT_C1.encode()).hexdigest()[:16],
    "C3": hashlib.sha256(PROMPT_C3.encode()).hexdigest()[:16],
}
EXPECTED_PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}

# The Kaggle script runtime may omit auxiliary package files. The packager
# replaces these placeholders with the exact qrel-free config and fixture.
EMBEDDED_CANARY_CONFIG = None
EMBEDDED_RUNTIME_FIXTURE = None
EMBEDDED_RUNTIME_FIXTURE_SHA256 = None


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_snapshot(path: Path) -> dict:
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or ".cache" in item.parts:
            continue
        files.append({"path": str(item.relative_to(path)), "sha256": sha(item), "bytes": item.stat().st_size})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"file_count": len(files), "files_sha256": digest, "files": files}


def json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def attached_model_dir(config: dict) -> Path | None:
    """Find the one pinned official model mounted by Kaggle model_sources."""
    source = config.get("model_source")
    if not source:
        return None
    input_root = Path(os.environ.get("SEM2ACT_MODEL_INPUT_ROOT", "/kaggle/input"))
    if not input_root.exists():
        raise RuntimeError("Kaggle model source root is missing")
    expected_type = source.get("model_type")
    expected_architecture = source.get("architecture")
    candidates = []
    for config_path in input_root.rglob("config.json"):
        try:
            model_config = json.loads(config_path.read_text())
        except Exception:
            continue
        architectures = model_config.get("architectures") or []
        if expected_type and model_config.get("model_type") != expected_type:
            continue
        if expected_architecture and expected_architecture not in architectures:
            continue
        model_dir = config_path.parent
        has_index = (model_dir / "model.safetensors.index.json").exists()
        has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
        if has_index or has_weights:
            candidates.append(model_dir)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one mounted official model for {source.get('source_ref')}, found {len(candidates)}"
        )
    return candidates[0]


def attached_patch_file(config: dict) -> Path:
    source = config.get("model_source") or {}
    input_root = Path(os.environ.get("SEM2ACT_MODEL_INPUT_ROOT", "/kaggle/input"))
    patch_name = source.get("patch_filename", "tokenizer_config.json")
    patch_slug = source.get("patch_dataset_slug")
    candidates = [
        item for item in input_root.rglob(patch_name)
        if patch_slug and patch_slug in str(item)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one pinned source patch, found {len(candidates)}")
    return candidates[0]


def assemble_model_source(config: dict, attached: Path) -> Path:
    source = config.get("model_source") or {}
    if source.get("type") != "verified_kaggle_dataset_assembly":
        return attached
    patch = attached_patch_file(config)
    destination = Path("/kaggle/working/hf_model_assembled")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(attached, destination)
    target = destination / source.get("patch_filename", "tokenizer_config.json")
    shutil.copy2(patch, target)
    return destination


def resolve_model_token(config: dict) -> str:
    """Use the private secret for gated models; public pins may be anonymous."""
    if config.get("model_source", {}).get("type") == "official_kaggle_model":
        return "official_kaggle_model_source"
    gated = config["model_id"].startswith("meta-llama/")
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        if gated:
            raise RuntimeError(f"HF_TOKEN Kaggle secret unavailable: {type(exc).__name__}") from exc
        os.environ.pop("HF_TOKEN", None)
        return "public_unauthenticated"
    if not token:
        if gated:
            raise RuntimeError("HF_TOKEN Kaggle secret is empty")
        os.environ.pop("HF_TOKEN", None)
        return "public_unauthenticated"
    os.environ["HF_TOKEN"] = token
    return "kaggle_secret"


def parse_json(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("response is not an object")
    return obj


def validate_c1(obj: dict) -> dict:
    required = {"normal", "supplier_delay", "demand_surge", "estimated_lt_increase", "estimated_duration", "estimated_demand_multiplier"}
    if set(obj) != required:
        raise ValueError("C1 schema keys differ")
    probabilities = [float(obj[key]) for key in ("normal", "supplier_delay", "demand_surge")]
    if any(not 0 <= value <= 1 for value in probabilities) or abs(sum(probabilities) - 1) > 0.05:
        raise ValueError("C1 probability schema invalid")
    for key in ("estimated_lt_increase", "estimated_duration"):
        value = int(obj[key])
        if value < 0 or float(obj[key]) != value:
            raise ValueError("C1 integer schema invalid")
    if float(obj["estimated_demand_multiplier"]) < 0:
        raise ValueError("C1 multiplier schema invalid")
    return obj


def validate_c3(obj: dict) -> dict:
    required = {"entity_match", "event", "fresh", "stance", "confidence"}
    if set(obj) != required:
        raise ValueError("C3 schema keys differ")
    if not isinstance(obj["entity_match"], bool) or not isinstance(obj["fresh"], bool):
        raise ValueError("C3 boolean schema invalid")
    if obj["event"] not in {"supplier_delay", "demand_surge", "normal", "none"}:
        raise ValueError("C3 event schema invalid")
    if obj["stance"] not in {"support", "refute", "na"}:
        raise ValueError("C3 stance schema invalid")
    if not 0 <= float(obj["confidence"]) <= 1:
        raise ValueError("C3 confidence schema invalid")
    return obj


def cpu_env(threads: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "GGML_CUDA": "0",
        "GGML_METAL": "0",
        "GGML_VULKAN": "0",
        "GGML_SYCL": "0",
    })
    if threads is not None:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(threads)
    return env


def source_tree_hash(source: Path) -> str:
    files = []
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        files.append({"path": "llama.cpp/" + str(item.relative_to(source)), "sha256": sha(item), "bytes": item.stat().st_size})
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_engine_source() -> tuple[Path, Path, Path]:
    """Resolve the pinned engine bundle across Kaggle mount-name variants."""
    direct = (Path(ENGINE_SOURCE_DIR), Path(ENGINE_ARCHIVE_PATH), Path(ENGINE_SOURCE_MANIFEST_PATH))
    if direct[0].exists() or direct[1].exists() or direct[2].exists():
        return direct
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for manifest_path in sorted(input_root.rglob("engine_source_manifest.json")):
            parent = manifest_path.parent
            source_dir = parent / "llama-cpp-b11194" / "llama.cpp"
            archive = parent / "llama-cpp-b11194.tar.gz"
            if source_dir.exists() or archive.exists():
                return source_dir, archive, manifest_path
    return direct


def build_engine(root: Path) -> tuple[Path, Path, dict]:
    source_dir, archive, source_manifest_path = resolve_engine_source()
    source_mode = "git_clone"
    archive_sha = None
    if source_dir.exists():
        if not source_manifest_path.exists():
            raise RuntimeError("pinned llama.cpp source manifest is missing")
        source_manifest = json.loads(source_manifest_path.read_text())
        if source_manifest.get("commit") != ENGINE_COMMIT or source_manifest.get("source_tree_sha256") != ENGINE_SOURCE_TREE_SHA256:
            raise RuntimeError("pinned llama.cpp source manifest drift")
        if source_tree_hash(source_dir) != ENGINE_SOURCE_TREE_SHA256:
            raise RuntimeError("pinned llama.cpp source tree hash mismatch")
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(source_dir, root)
        archive_sha = source_manifest.get("archive_sha256")
        source_mode = "private_kaggle_dataset_source_tree"
    elif archive.exists():
        archive_sha = sha(archive)
        if archive_sha != ENGINE_ARCHIVE_SHA256:
            raise RuntimeError("pinned llama.cpp source archive hash mismatch")
        staging = root.parent / "llama.cpp.source"
        if staging.exists():
            shutil.rmtree(staging)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                    raise RuntimeError("unsafe source archive member")
            bundle.extractall(staging)
        extracted = staging / "llama.cpp"
        if not (extracted / "CMakeLists.txt").exists():
            raise RuntimeError("pinned llama.cpp source archive is incomplete")
        if root.exists():
            shutil.rmtree(root)
        shutil.move(str(extracted), str(root))
        shutil.rmtree(staging, ignore_errors=True)
        source_mode = "private_kaggle_dataset_archive"
    else:
        if not root.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--branch", ENGINE_TAG,
                            "https://github.com/ggml-org/llama.cpp", str(root)], check=True)
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if head != ENGINE_COMMIT:
            subprocess.run(["git", "-C", str(root), "checkout", "--detach", ENGINE_COMMIT], check=True)
    head = ENGINE_COMMIT if source_mode.startswith("private_kaggle_dataset") else subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != ENGINE_COMMIT:
        raise RuntimeError("llama.cpp commit drift")
    build = root / "build"
    server = build / "bin" / "llama-server"
    quantizer = build / "bin" / "llama-quantize"
    if not server.exists() or not quantizer.exists():
        subprocess.run([
            "cmake", "-S", str(root), "-B", str(build),
            "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=ON", "-DGGML_CUDA=OFF",
            "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DGGML_SYCL=OFF",
            "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_TOOLS=ON",
        ], check=True)
        subprocess.run(["cmake", "--build", str(build), "--config", "Release",
                        "--target", "llama-server", "llama-quantize", "-j", "2"], check=True)
    if not server.exists() or not quantizer.exists():
        raise RuntimeError("pinned llama.cpp build did not produce required binaries")
    return server, quantizer, {
        "commit": head,
        "source_mode": source_mode,
        "source_archive_sha256": archive_sha,
        "server_sha256": sha(server),
        "quantizer_sha256": sha(quantizer),
        "build_flags": [
            "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=ON", "-DGGML_CUDA=OFF",
            "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DGGML_SYCL=OFF",
        ],
    }


def acquire_model(config: dict, destination: Path) -> Path:
    attached = attached_model_dir(config)
    if attached is not None:
        return assemble_model_source(config, attached)
    from huggingface_hub import snapshot_download
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config["model_id"], revision=config["revision"],
        local_dir=str(destination), token=os.environ.get("HF_TOKEN"),
    )
    return destination


def convert_model(config: dict, model_dir: Path, engine_root: Path, quantizer: Path, output: Path) -> dict:
    converter = engine_root / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise RuntimeError("convert_hf_to_gguf.py missing from pinned engine")
    f16 = output.with_suffix(".f16.gguf")
    subprocess.run([sys.executable, str(converter), str(model_dir), "--outfile", str(f16), "--outtype", "f16"], check=True)
    subprocess.run([str(quantizer), str(f16), str(output), QUANTIZATION], check=True)
    return {
        "conversion_script_sha256": sha(converter),
        "gguf_sha256": sha(output),
        "gguf_path": str(output),
        "quantizer_sha256": sha(quantizer),
    }


def post_json(base_url: str, model: str, messages: list[dict[str, str]]) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 256,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read().decode())


def wait_server(base_url: str, process: subprocess.Popen) -> None:
    deadline = time.time() + 1800
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("llama-server health timeout")


def server_id(base_url: str) -> str:
    with urllib.request.urlopen(base_url + "/v1/models", timeout=30) as response:
        return str(json.loads(response.read().decode())["data"][0]["id"])


def load_packaged_json(filename: str, embedded: dict | None) -> dict:
    path = Path(filename)
    if path.exists():
        return json.loads(path.read_text())
    if embedded is not None:
        return embedded
    raise FileNotFoundError(filename)


def packaged_fixture_sha256() -> str:
    path = Path(FIXTURE_NAME)
    if path.exists():
        return sha(path)
    if EMBEDDED_RUNTIME_FIXTURE_SHA256 is not None:
        return EMBEDDED_RUNTIME_FIXTURE_SHA256
    raise FileNotFoundError(FIXTURE_NAME)


def cleanup_large_preflight_files() -> None:
    """Keep only the small audit output in the Kaggle kernel result."""
    for path in (
        Path("/kaggle/working/hf_model"),
        Path("/kaggle/working/artifacts"),
        Path("/kaggle/working/llama.cpp"),
    ):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def consumer_canary(config: dict, server: Path, gguf: Path, fixture: dict, threads: int) -> dict:
    port = 18765
    command = [str(server), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
               "--n-gpu-layers", "0", "--ctx-size", "4096", "--batch-size", "512",
               "--ubatch-size", "512", "--threads", str(threads), "--threads-batch", str(threads),
               "--parallel", "1", "--jinja", "--reasoning", "off", "--reasoning-format", "none", "--no-slots"]
    log = Path("/kaggle/working/llama-server.log").open("w")
    process = subprocess.Popen(command, env=cpu_env(threads), stdout=log, stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{port}"
        wait_server(base, process)
        model_name = server_id(base)
        prompts = [("C1", PROMPT_C1, f"Operator information need: {fixture['query_text']}\n\nEvidence:\n" +
                    "\n\n".join(f"[DOC {doc['doc_id']}] {doc['text']}" for doc in fixture["documents"]))]
        prompts.extend(("C3", PROMPT_C3, f"Operator's own node: {fixture['entity_node']}\n\nEvidence document:\n{doc['text']}")
                       for doc in fixture["documents"])
        raw_outputs = []
        for repeat in (1, 2):
            for consumer, prompt, user in prompts:
                response = post_json(base, model_name, [{"role": "system", "content": prompt}, {"role": "user", "content": user}])
                raw = str(response["choices"][0]["message"]["content"])
                parsed = parse_json(raw)
                parsed = validate_c1(parsed) if consumer == "C1" else validate_c3(parsed)
                raw_outputs.append({"repeat": repeat, "consumer": consumer, "raw": raw,
                                    "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(), "parsed": parsed})
        n = len(prompts)
        return {
            "status": "pass",
            "schema_valid": True,
            "deterministic_repeat_equality": [x["raw"] for x in raw_outputs[:n]] == [x["raw"] for x in raw_outputs[n:]],
            "expected_calls": len(prompts) * 2,
            "actual_calls": len(raw_outputs),
            "raw_outputs": raw_outputs,
            "model_server_id": model_name,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def main() -> int:
    result = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-kaggle-cpu-canary-v1",
        "status": "failed",
        "result_bearing_execution_started": False,
        "qrels_read": False,
    }
    started = time.monotonic()
    try:
        config = load_packaged_json("canary_config.json", EMBEDDED_CANARY_CONFIG)
        fixture = load_packaged_json(FIXTURE_NAME, EMBEDDED_RUNTIME_FIXTURE)
        if fixture.get("query_id", "").startswith("v5-lb-"):
            raise RuntimeError("canary fixture collides with lockbox")
        if PROMPT_SHAS != EXPECTED_PROMPT_SHAS:
            raise RuntimeError(f"prompt hash drift: {PROMPT_SHAS}")
        credential_source = resolve_model_token(config)
        env = cpu_env()
        physical = os.cpu_count() or 1
        threads = max(1, min(physical, 8))
        os.environ.update(cpu_env(threads))
        engine_root = Path("/kaggle/working/llama.cpp")
        server, quantizer, build = build_engine(engine_root)
        model_source = dict(config.get("model_source") or {
            "type": "huggingface_snapshot",
            "model_id": config["model_id"],
            "revision": config["revision"],
        })
        result.update({
            "family": config["family"],
            "model_id": config["model_id"],
            "revision": config["revision"],
            "tokenizer_revision": config["tokenizer_revision"],
            "model_source": model_source,
            "runtime": {
                key: config[key]
                for key in (
                    "backend", "dtype", "bitsandbytes", "device", "device_map",
                    "max_context_tokens", "batch_size", "ubatch_size", "attention_backend",
                )
                if key in config
            },
            "engine": build,
            "threads": threads,
            "platform": {"python": platform.python_version(), "machine": platform.machine(), "physical_cores": physical},
            "fixture_sha256": packaged_fixture_sha256(),
            "credential_source": credential_source,
        })
        if config["kind"] == "consumer":
            model_dir = acquire_model(config, Path("/kaggle/working/hf_model"))
            gguf = Path("/kaggle/working/artifacts") / config["gguf_filename"]
            gguf.parent.mkdir(parents=True, exist_ok=True)
            source_snapshot = model_snapshot(model_dir)
            source_spec = config.get("model_source") or {}
            expected_source_hash = source_spec.get("content_sha256")
            if source_spec.get("type") == "official_kaggle_model" and not expected_source_hash:
                raise RuntimeError("official model source hash is missing; run and accept the source-only probe first")
            if expected_source_hash and source_snapshot["files_sha256"] != expected_source_hash:
                raise RuntimeError("mounted official model content hash differs from source-only probe")
            result["source_model_snapshot"] = source_snapshot
            result["model_source"]["resolved_model_path"] = str(model_dir)
            result["artifact"] = convert_model(config, model_dir, engine_root, quantizer, gguf)
            result["canary"] = consumer_canary(config, server, gguf, fixture, threads)
            result["source_script_sha256"] = sha(Path(__file__))
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_dir = acquire_model(config, Path("/kaggle/working/hf_model"))
            if torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                raise RuntimeError("reranker canary detected an accelerator")
            torch.set_num_threads(threads)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=config["revision"], padding_side="left", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, revision=config["revision"], trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to("cpu").eval()
            yes, no = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
            prefix = tokenizer.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
            suffix = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
            text = f"<Instruct>: Given an operational supply-chain information need, retrieve documents relevant to the described entity, event, and time context.\n<Query>: {fixture['query_text']}\n<Document>: {fixture['documents'][0]['text']}"
            def score():
                enc = tokenizer(text, truncation=True, max_length=1024-len(prefix)-len(suffix), return_attention_mask=False)
                batch = tokenizer.pad({"input_ids": [prefix + enc["input_ids"] + suffix]}, padding=True, return_tensors="pt")
                with torch.inference_mode():
                    logits = model(**batch).logits[:, -1, :]
                return float(torch.stack([logits[:, no], logits[:, yes]], dim=1).log_softmax(dim=1)[:, 1].exp()[0])
            first, second = score(), score()
            result["canary"] = {"status": "pass", "schema_valid": True, "deterministic_repeat_equality": first == second, "score": first, "expected_calls": 2, "actual_calls": 2}
            result["source_script_sha256"] = sha(Path(__file__))
            result["model_source"]["resolved_model_path"] = str(model_dir)
            result["model_snapshot"] = model_snapshot(model_dir)
        result["status"] = "pass"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    cleanup_large_preflight_files()
    Path("preflight_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" and result.get("canary", {}).get("deterministic_repeat_equality") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
