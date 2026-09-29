"""Fixed, non-lockbox GPU smoke tests for the v5 Lightning runtime.

The fixture is deliberately outside the 240-query lockbox.  These tests only
prove model loading, tokenization, CUDA execution, and response schemas; their
outputs are never included in the scientific analysis.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from runtime_contract import (
    FIXTURE_REL,
    FREEZE_REL,
    LOCK_REL,
    ROOT,
    environment_snapshot,
    load_fixture,
    load_lock,
    package_versions,
    require_t4_environment,
    sha256,
    validate_lock,
)


KERNEL_DIR = ROOT / "versions/sem2act-v5/compute/kaggle_kernel"
sys.path.insert(0, str(KERNEL_DIR))
import p2_v5_consumer as consumer_schema  # noqa: E402


def output_root() -> Path:
    path = Path(os.environ.get(
        "SEM2ACT_OUTPUT_ROOT",
        str(ROOT / "versions/sem2act-v5/runtime/lightning_smoke"),
    ))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_result(family: str, result: dict[str, Any]) -> None:
    path = output_root() / f"{family}_smoke_manifest.json"
    path.write_text(json.dumps(result, indent=2) + "\n")


def provenance_fields(lock: dict[str, Any]) -> dict[str, Any]:
    """Attach the frozen contract identity to every smoke result."""
    return {
        "protocol_hash": lock["protocol_hash"],
        "runtime_lock_sha256": sha256(ROOT / LOCK_REL),
        "runtime_freeze_manifest_sha256": sha256(ROOT / FREEZE_REL),
        "smoke_fixture_id": load_fixture()["fixture_id"],
    }


def fixture_users(fixture: dict[str, Any]) -> tuple[str, list[str]]:
    c1 = (
        f"Operator information need: {fixture['query_text']}\n\nEvidence:\n"
        + "\n\n".join(
            f"[DOC {doc['doc_id']}] {doc['text']}"
            for doc in fixture["documents"]
        )
    )
    c3 = [
        f"Operator's own node: {fixture['entity_node']}\n\n"
        f"Evidence document:\n{doc['text']}"
        for doc in fixture["documents"]
    ]
    return c1, c3


def parse_json(raw: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("smoke output is not a JSON object")
    return value


def validate_c1(obj: dict[str, Any]) -> None:
    expected = {
        "normal", "supplier_delay", "demand_surge",
        "estimated_lt_increase", "estimated_duration",
        "estimated_demand_multiplier",
    }
    if set(obj) != expected:
        raise ValueError(f"C1 schema mismatch: {sorted(obj)}")
    probs = [float(obj[name]) for name in ("normal", "supplier_delay", "demand_surge")]
    if any(not 0 <= value <= 1 for value in probs):
        raise ValueError("C1 probability outside [0, 1]")


def validate_c3(obj: dict[str, Any]) -> None:
    expected = {"entity_match", "event", "fresh", "stance", "confidence"}
    if set(obj) != expected:
        raise ValueError(f"C3 schema mismatch: {sorted(obj)}")


def canary() -> dict[str, Any]:
    lock = load_lock()
    snapshot = require_t4_environment()
    import torch

    x = torch.ones((512, 512), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()):
        raise RuntimeError("CUDA canary produced non-finite output")
    result = {
        "schema_version": 1,
        "stage": "cuda_canary",
        "status": "pass",
        "environment": snapshot,
        "fixture_sha256": sha256(ROOT / FIXTURE_REL),
        **provenance_fields(lock),
    }
    write_result("cuda", result)
    return result


def load_transformer(family: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    lock = load_lock()
    spec = lock["models"][family]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"],
        revision=spec["tokenizer_revision"],
        use_fast=True,
        padding_side="left",
        trust_remote_code=family == "reranker",
    )
    quant = spec["quantization"]
    kwargs: dict[str, Any] = {
        "revision": spec["revision"],
        "torch_dtype": torch.float16,
        "device_map": spec["device_map"],
        "attn_implementation": spec["attention_backend"],
    }
    if quant["method"] == "bitsandbytes":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quant["load_in_4bit"],
            bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], **kwargs
    ).eval()
    return tokenizer, model


def transformer_smoke(family: str) -> dict[str, Any]:
    import torch

    fixture = load_fixture()
    lock = load_lock()
    spec = lock["models"][family]
    tokenizer, model = load_transformer(family)
    c1_user, c3_users = fixture_users(fixture)
    messages = [
        {"role": "system", "content": consumer_schema.PROMPT_C1},
        {"role": "user", "content": c1_user},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if encoded.shape[-1] > spec["max_input_tokens"]:
        raise RuntimeError("smoke fixture exceeds frozen consumer input context")
    device = torch.device("cuda:0")
    encoded = encoded.to(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    with torch.inference_mode():
        generated = model.generate(
            encoded,
            max_new_tokens=spec["max_new_tokens"],
            do_sample=spec["do_sample"],
            num_beams=spec["num_beams"],
            use_cache=spec["use_cache"],
            pad_token_id=tokenizer.eos_token_id,
        )
    raw_c1 = tokenizer.decode(
        generated[0, encoded.shape[-1]:], skip_special_tokens=False
    ).strip()
    parsed_c1 = parse_json(raw_c1)
    validate_c1(parsed_c1)

    c3_messages = [
        {"role": "system", "content": consumer_schema.PROMPT_C3_DOC},
        {"role": "user", "content": c3_users[0]},
    ]
    c3_encoded = tokenizer.apply_chat_template(
        c3_messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    if c3_encoded.shape[-1] > spec["max_input_tokens"]:
        raise RuntimeError("C3 smoke fixture exceeds frozen consumer input context")
    with torch.inference_mode():
        c3_generated = model.generate(
            c3_encoded,
            max_new_tokens=spec["max_new_tokens"],
            do_sample=spec["do_sample"],
            num_beams=spec["num_beams"],
            use_cache=spec["use_cache"],
            pad_token_id=tokenizer.eos_token_id,
        )
    raw_c3 = tokenizer.decode(
        c3_generated[0, c3_encoded.shape[-1]:], skip_special_tokens=False
    ).strip()
    parsed_c3 = parse_json(raw_c3)
    validate_c3(parsed_c3)
    result = {
        "schema_version": 1,
        "stage": f"{family}_model_load_and_smoke",
        "status": "pass",
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "tokenizer_revision": spec["tokenizer_revision"],
        "fixture_sha256": sha256(ROOT / FIXTURE_REL),
        "raw_outputs": {"C1": raw_c1, "C3": raw_c3},
        "parsed_schema_keys": {
            "C1": sorted(parsed_c1), "C3": sorted(parsed_c3)
        },
        "environment": environment_snapshot(),
        "package_versions": package_versions(),
        **provenance_fields(lock),
    }
    write_result(family, result)
    return result


def reranker_smoke() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fixture = load_fixture()
    lock = load_lock()
    spec = lock["models"]["reranker"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["tokenizer_revision"],
        padding_side="left", use_fast=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=spec["revision"],
        torch_dtype=torch.float16, device_map=spec["device_map"],
        attn_implementation=spec["attention_backend"],
        trust_remote_code=True,
    ).eval()
    prefix = tokenizer.encode(
        '<|im_start|>system\nJudge whether the Document meets the requirements based '
        'on the Query and the Instruct provided. Note that the answer can only be '
        '"yes" or "no".<|im_end|>\n<|im_start|>user\n',
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    texts = [
        f"<Instruct>: Given an operational supply-chain information need, retrieve documents relevant to the described entity, event, and time context.\n<Query>: {fixture['query_text']}\n<Document>: {doc['text']}"
        for doc in fixture["documents"]
    ]
    encoded = tokenizer(
        texts, padding=False, truncation="longest_first",
        max_length=spec["max_context_tokens"] - len(prefix) - len(suffix),
        return_attention_mask=False,
    )
    ids = [prefix + item + suffix for item in encoded["input_ids"]]
    batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**batch).logits[:, -1, :]
    yes = tokenizer.convert_tokens_to_ids("yes")
    no = tokenizer.convert_tokens_to_ids("no")
    score = torch.stack([logits[:, no], logits[:, yes]], dim=1).log_softmax(dim=1)[:, 1].exp()
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError("reranker smoke produced non-finite score")
    result = {
        "schema_version": 1,
        "stage": "reranker_model_load_and_smoke",
        "status": "pass",
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "tokenizer_revision": spec["tokenizer_revision"],
        "fixture_sha256": sha256(ROOT / FIXTURE_REL),
        "n_pairs": len(texts),
        "finite_scores": True,
        "environment": environment_snapshot(),
        "package_versions": package_versions(),
        **provenance_fields(lock),
    }
    write_result("reranker", result)
    return result


def qwen_smoke() -> dict[str, Any]:
    """Run one fixed C1 and C3 request through the frozen local vLLM server."""
    import openai

    fixture = load_fixture()
    lock = load_lock()
    spec = lock["models"]["qwen"]
    port = "18000"
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = spec["attention_backend"]
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", spec["model_id"], "--revision", spec["revision"],
        "--tokenizer", spec["model_id"], "--tokenizer-revision", spec["tokenizer_revision"],
        "--dtype", spec["dtype"], "--quantization", "awq",
        "--max-model-len", str(spec["max_context_tokens"]),
        "--max-num-seqs", str(spec["max_num_seqs"]),
        "--max-num-batched-tokens", str(spec["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(spec["gpu_memory_utilization"]),
        "--enforce-eager", "--seed", "0", "--port", port,
    ]
    log_path = output_root() / "qwen_vllm_smoke.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("vLLM smoke server did not become healthy")
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local")
        c1_user, c3_users = fixture_users(fixture)
        responses = {}
        for name, system, user in (
            ("C1", consumer_schema.PROMPT_C1, c1_user),
            ("C3", consumer_schema.PROMPT_C3_DOC, c3_users[0]),
        ):
            response = client.chat.completions.create(
                model=spec["model_id"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = response.choices[0].message.content or ""
            parsed = parse_json(raw)
            (validate_c1 if name == "C1" else validate_c3)(parsed)
            responses[name] = {"raw": raw, "keys": sorted(parsed)}
        result = {
            "schema_version": 1,
            "stage": "qwen_model_load_and_smoke",
            "status": "pass",
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "tokenizer_revision": spec["tokenizer_revision"],
            "fixture_sha256": sha256(ROOT / FIXTURE_REL),
            "responses": responses,
            "environment": environment_snapshot(),
            "package_versions": package_versions(),
            **provenance_fields(lock),
        }
        write_result("qwen", result)
        return result
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


def run(stage: str) -> dict[str, Any]:
    validate_lock()
    if stage == "cuda_canary":
        return canary()
    if stage == "reranker":
        return reranker_smoke()
    if stage == "qwen":
        return qwen_smoke()
    if stage in ("llama", "mistral"):
        return transformer_smoke(stage)
    raise ValueError(f"unknown smoke stage: {stage}")
