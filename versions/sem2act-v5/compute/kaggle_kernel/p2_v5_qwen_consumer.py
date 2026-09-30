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
PINNED_TRANSFORMERS = "4.57.6"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_RUNTIME_LOCK_SHA256 = "ce88bc5c2d61d30ecbf649fb09c65b2502e599c53a42732da1acec06b2808b10"
EXPECTED_SMOKE_FIXTURE_SHA256 = "9ab69722e0062fc38eb27721f24e9cd83ca7944655973cbbe85f079372ccd74b"
SYSTEMS = ("bm25", "rerank", "oracle")
CREDENTIAL_AMENDMENT_ID = "sem2act-v5-qwen-credential-amendment-1"
PUBLIC_MODEL = "Qwen/Qwen3-8B-AWQ"
PUBLIC_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
EXPECTED_PUBLIC_FILES = {
    "config.json": "7457674d8044143cd4159e47deecf28fd6698a9826569094b60d7bead8f351ee",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


def find_input(name: str) -> Path:
    matches = []
    for root_name in ("/kaggle/input", "/kaggle/working", "."):
        root = Path(root_name)
        if root.exists():
            matches.extend(p for p in root.rglob(name) if p.is_file())
    if not matches:
        raise FileNotFoundError(name)
    return sorted(set(matches), key=lambda p: str(p))[0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _anonymous_get(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "sem2act-v5-qwen-consumer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def verify_public_revision_anonymous() -> None:
    """Pre-execution anonymous identity check for the exact pinned public revision.

    No credentials are sent. Fails closed unless the pinned revision resolves
    anonymously, is public, and the model/tokenizer file identities match.
    """
    if MODEL != PUBLIC_MODEL or EXPECTED_REVISION != PUBLIC_REVISION:
        raise SystemExit(
            "public credential fallback is restricted to the pinned public model/revision")
    try:
        payload = json.loads(_anonymous_get(
            f"https://huggingface.co/api/models/{MODEL}/revision/{EXPECTED_REVISION}",
            30).decode())
    except Exception as exc:
        raise SystemExit(f"anonymous revision check failed: {type(exc).__name__}")
    if payload.get("sha") != EXPECTED_REVISION:
        raise SystemExit("pinned revision sha mismatch (public check)")
    if payload.get("private") is True:
        raise SystemExit("pinned model is private; credentials required")
    for fname, want in EXPECTED_PUBLIC_FILES.items():
        try:
            data = _anonymous_get(
                f"https://huggingface.co/{MODEL}/resolve/{EXPECTED_REVISION}/{fname}", 60)
        except Exception as exc:
            raise SystemExit(f"anonymous file check failed for {fname}: {type(exc).__name__}")
        if hashlib.sha256(data).hexdigest() != want:
            raise SystemExit(f"{fname} identity drift (public check)")


def resolve_credential() -> str:
    """Secret-first credential resolution; returns the credential_source string.

    Tries the HF_TOKEN Kaggle secret with bounded retries for host boot.
    If the UserSecrets channel is unavailable (observed as OSError, e.g.
    ConnectionError), falls back to public_unauthenticated ONLY for the exact
    pinned, verified-public model/revision, after anonymous revision plus
    model/tokenizer identity verification. Fails closed on empty secrets,
    non-channel errors, any gated/private model, and any identity drift.
    No token is ever written to datasets, files, logs, or the repo.
    """
    token = None
    last_exc = None
    for _attempt in range(8):
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(15)
    if token:
        os.environ["HF_TOKEN"] = token
        return "hf_token_kaggle_secret"
    if token is not None and not token:
        raise SystemExit("HF_TOKEN Kaggle secret is empty")
    if last_exc is None:
        raise SystemExit("HF_TOKEN Kaggle secret is unavailable")
    if not isinstance(last_exc, OSError):
        raise SystemExit(f"HF_TOKEN Kaggle secret error: {type(last_exc).__name__}")
    verify_public_revision_anonymous()
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    return "public_unauthenticated"


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    credential_source = resolve_credential()
    if any("qrel" in path.name.lower()
           for root_name in ("/kaggle/input", "/kaggle/working")
           for root, _, names in os.walk(root_name)
           for path in [Path(root) / name for name in names]):
        raise SystemExit("qrel-bearing input detected")
    spec_path = find_input("model_spec.json")
    spec = json.loads(spec_path.read_text())
    if spec.get("model_id") != MODEL or spec.get("revision") != EXPECTED_REVISION:
        raise SystemExit("Qwen model pin drift")
    if spec.get("protocol_hash") != spec.get("expected_protocol_hash"):
        raise SystemExit("protocol hash mismatch")
    if spec.get("runtime_lock_sha256") != EXPECTED_RUNTIME_LOCK_SHA256:
        raise SystemExit("runtime lock hash mismatch")
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
    source_dir = Path("/kaggle/working/v5src/src")
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

    import torch
    before = (torch.__version__, str(torch.version.cuda or ""))
    if before != (EXPECTED_TORCH, EXPECTED_CUDA):
        raise SystemExit(f"Kaggle image Torch/CUDA mismatch: {before}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "vllm==0.11.0", "openai==2.48.0", "pydantic==2.12.5",
         f"transformers=={PINNED_TRANSFORMERS}"],
        check=True,
    )
    probe = subprocess.run(
        [sys.executable, "-c", "import json,torch; print(json.dumps([torch.__version__, str(torch.version.cuda or '')]))"],
        capture_output=True, text=True, check=True,
    )
    after = tuple(json.loads(probe.stdout.strip().splitlines()[-1]))
    if after != before:
        raise SystemExit(f"package installation changed Kaggle Torch/CUDA: {before} -> {after}")
    os.environ["PAPER2_V3_CACHE_DIR"] = "/kaggle/working/cache_out"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
    os.environ["LLM_API_KEY"] = "local"
    Path(os.environ["PAPER2_V3_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--revision", EXPECTED_REVISION,
         "--quantization", "awq", "--max-model-len", "4096",
         "--gpu-memory-utilization", "0.85", "--port", "8000"],
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
        smoke_path = find_input("runtime_smoke_v1.json")
        if sha(smoke_path) != EXPECTED_SMOKE_FIXTURE_SHA256:
            raise SystemExit("Qwen smoke fixture hash drift")
        smoke = json.loads(smoke_path.read_text())
        smoke_client = consumers._client()
        smoke_users = [
            (consumers.PROMPT_C1,
             f"Operator information need: {smoke['query_text']}\n\nEvidence:\n" +
             "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in smoke["documents"])),
            (consumers.PROMPT_C3_DOC,
             f"Operator's own node: {smoke['entity_node']}\n\nEvidence document:\n{smoke['documents'][0]['text']}"),
        ]
        for prompt, user in smoke_users:
            response = smoke_client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=256,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user}],
            )
            raw = response.choices[0].message.content or ""
            parsed = consumers._parse_json(raw)
            required = (
                {"normal", "supplier_delay", "demand_surge",
                 "estimated_lt_increase", "estimated_duration",
                 "estimated_demand_multiplier"}
                if prompt == consumers.PROMPT_C1 else
                {"entity_match", "event", "fresh", "stance", "confidence"}
            )
            if not isinstance(parsed, dict) or set(parsed) != required:
                raise SystemExit("Qwen smoke schema failure")
        smoke_manifest = {
            "status": "pass", "fixture_sha256": EXPECTED_SMOKE_FIXTURE_SHA256,
            "schema": "C1/C3 JSON objects", "n_calls": 2,
        }
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
        raw_path = Path("/kaggle/working/raw_outputs.jsonl")
        with raw_path.open("w") as handle:
            for path in cache_files:
                payload = json.loads(path.read_text())
                payload["cache_file"] = path.name
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        beliefs_path = Path("/kaggle/working/beliefs.jsonl")
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
            "runtime_lock_sha256": EXPECTED_RUNTIME_LOCK_SHA256,
            "smoke": smoke_manifest,
            "systems": list(SYSTEMS),
            "consumers": ["C1", "C3"],
            "k": 3,
            "n_input_rows": len(rows),
            "expected_calls": 2880,
            "actual_calls": n_calls,
            "n_cache_files": len(cache_files),
            "n_fail": len(failures),
            "qrels_read": False,
            "credential_source": credential_source,
            "credential_amendment": CREDENTIAL_AMENDMENT_ID,
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
                "transformers": PINNED_TRANSFORMERS,
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
        Path("/kaggle/working/run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        Path("/kaggle/working/failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        Path("/kaggle/working/runtime_versions.json").write_text(json.dumps({
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
