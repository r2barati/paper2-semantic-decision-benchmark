"""Deterministic v5 cross-family consumer kernel."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

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
PROMPT_C3_DOC = (
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
    "C3": hashlib.sha256(PROMPT_C3_DOC.encode()).hexdigest()[:16],
}
EXPECTED_PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}
REGIMES = ("normal", "supplier_delay", "demand_surge")
SYSTEMS = ("rerank", "oracle")
PINNED_TRANSFORMERS = "4.57.6"
PINNED_BITSANDBYTES = "0.48.1"
EXPECTED_RUNTIME_LOCK_SHA256 = "df2618f13337b37d59ae4e21cda4eb3423569190869db27614b651f07234ca6b"
EXPECTED_SMOKE_FIXTURE_SHA256 = "9ab69722e0062fc38eb27721f24e9cd83ca7944655973cbbe85f079372ccd74b"


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


def require_hf_secret() -> None:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise SystemExit(f"HF_TOKEN Kaggle secret is unavailable: {type(exc).__name__}")
    if not token:
        raise SystemExit("HF_TOKEN Kaggle secret is empty")
    import os
    os.environ["HF_TOKEN"] = token


def install_runtime_pins() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
         "--index-url", "https://download.pytorch.org/whl/cu126", "torch==2.7.1+cu126"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
         f"transformers=={PINNED_TRANSFORMERS}",
         f"bitsandbytes=={PINNED_BITSANDBYTES}"],
        check=True,
    )


def parse_json(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text.split("\n", 1)[1].rsplit(fence, 1)[0].strip()
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
    required = {
        "normal", "supplier_delay", "demand_surge",
        "estimated_lt_increase", "estimated_duration",
        "estimated_demand_multiplier",
    }
    if set(obj) != required:
        raise ValueError("C1 schema keys differ from frozen schema")
    probabilities = {key: float(obj[key]) for key in REGIMES}
    if any(not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("C1 probability out of range")
    if abs(sum(probabilities.values()) - 1) > 0.05:
        raise ValueError("C1 probabilities do not sum within tolerance")
    for key in ("estimated_lt_increase", "estimated_duration"):
        value = int(obj[key])
        if value < 0 or float(obj[key]) != value:
            raise ValueError("C1 integer field invalid")
    if float(obj["estimated_demand_multiplier"]) < 0:
        raise ValueError("C1 multiplier invalid")
    return obj


def validate_c3(obj: dict) -> dict:
    required = {"entity_match", "event", "fresh", "stance", "confidence"}
    if set(obj) != required:
        raise ValueError("C3 schema keys differ from frozen schema")
    if not isinstance(obj["entity_match"], bool) or not isinstance(obj["fresh"], bool):
        raise ValueError("C3 boolean field invalid")
    if obj["event"] not in {"supplier_delay", "demand_surge", "normal", "none"}:
        raise ValueError("C3 event invalid")
    if obj["stance"] not in {"support", "refute", "na"}:
        raise ValueError("C3 stance invalid")
    if not 0 <= float(obj["confidence"]) <= 1:
        raise ValueError("C3 confidence invalid")
    return obj

def aggregate_c1(obj: dict, doc_ids: list[str]) -> dict:
    total = sum(float(obj[key]) for key in REGIMES)
    return {
        "p_normal": float(obj["normal"]) / total,
        "p_supplier_delay": float(obj["supplier_delay"]) / total,
        "p_demand_surge": float(obj["demand_surge"]) / total,
        "abstain": False,
        "confidence": min(1.0, total) if total <= 1.5 else 0.5,
        "evidence_ids": doc_ids,
    }


def aggregate_c3(items: list[tuple[str, dict]], doc_ids: list[str]) -> dict:
    import math
    agg = {key: 0.0 for key in REGIMES}
    support = 0.0
    for _, item in items:
        weight = float(item["confidence"])
        weight *= 1.0 if item["entity_match"] else 0.2
        weight *= 1.0 if item["fresh"] else 0.2
        if item["event"] in agg and item["stance"] == "support":
            agg[item["event"]] += weight
            support = max(support, weight)
        elif item["event"] in agg and item["stance"] == "refute":
            agg[item["event"]] -= 0.5 * weight
    if support < 0.5:
        return {
            "p_normal": 1 / 3,
            "p_supplier_delay": 1 / 3,
            "p_demand_surge": 1 / 3,
            "abstain": True,
            "confidence": support,
            "evidence_ids": doc_ids,
        }
    prior = {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30}
    scores = {key: agg[key] + 0.5 * prior[key] for key in REGIMES}
    maximum = max(scores.values())
    exponentials = {key: math.exp(value - maximum) for key, value in scores.items()}
    total = sum(exponentials.values())
    return {
        "p_normal": exponentials["normal"] / total,
        "p_supplier_delay": exponentials["supplier_delay"] / total,
        "p_demand_surge": exponentials["demand_surge"] / total,
        "abstain": False,
        "confidence": min(1.0, support),
        "evidence_ids": doc_ids,
    }


def main() -> int:
    require_hf_secret()
    install_runtime_pins()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if (platform.python_version_tuple()[:2] != ("3", "12")
            or torch.__version__ != "2.7.1+cu126"
            or not str(torch.version.cuda or "").startswith("12.6")
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "t4" not in torch.cuda.get_device_name(0).lower()):
        raise SystemExit("v5 consumer runtime/device differs from the exact Kaggle lock")
    if PROMPT_SHAS != EXPECTED_PROMPT_SHAS:
        raise SystemExit(f"prompt hash drift: {PROMPT_SHAS}")

    input_path = find_input("evidence_inputs.jsonl")
    spec_path = find_input("model_spec.json")
    spec = json.loads(spec_path.read_text())
    model_id, revision = spec.get("model_id"), spec.get("revision")
    allowed = {
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }
    if model_id not in allowed or not revision or revision.startswith("resolved_"):
        raise SystemExit("model spec is not an immutable official checkpoint")
    if spec.get("protocol_hash") != spec.get("expected_protocol_hash"):
        raise SystemExit("model spec protocol hash mismatch")
    if spec.get("runtime_lock_sha256") != EXPECTED_RUNTIME_LOCK_SHA256:
        raise SystemExit("runtime lock hash mismatch")
    if any("qrel" in path.name.lower() for path in (input_path, spec_path)):
        raise SystemExit("qrel-bearing input detected")

    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    if len(rows) != 720:
        raise SystemExit(f"expected 720 input rows, got {len(rows)}")
    selected = [row for row in rows if row.get("system") in SYSTEMS]
    if len(selected) != 480:
        raise SystemExit(f"expected 480 selected rows, got {len(selected)}")
    keys = [(row["query_id"], row["system"]) for row in selected]
    expected = {
        (f"v5-lb-q{i:04d}", system)
        for i in range(1, 241) for system in SYSTEMS
    }
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise SystemExit("consumer keyspace is not exact")
    if any(len(row.get("documents", [])) != 3 for row in selected):
        raise SystemExit("consumer evidence depth drift")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, padding_side="left", use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, quantization_config=quantization,
        torch_dtype=torch.float16, device_map="auto"
    ).eval()
    device = next(model.parameters()).device
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    out_dir = Path("/kaggle/working")
    raw_handle = (out_dir / "raw_outputs.jsonl").open("w")
    belief_rows, failures = [], []
    call_id = 0

    def generate(prompt: str, user: str) -> str:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                input_ids, max_new_tokens=256, do_sample=False, num_beams=1,
                use_cache=True, pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(
            output[0, input_ids.shape[-1]:], skip_special_tokens=False
        ).strip()

    smoke_path = find_input("runtime_smoke_v1.json")
    if sha(smoke_path) != EXPECTED_SMOKE_FIXTURE_SHA256:
        raise SystemExit("consumer smoke fixture hash drift")
    smoke = json.loads(smoke_path.read_text())

    def smoke_call(prompt, user, validator):
        raw = generate(prompt, user)
        validator(parse_json(raw))

    smoke_c1_user = (
        f"Operator information need: {smoke['query_text']}\n\nEvidence:\n" +
        "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in smoke["documents"])
    )
    smoke_c3_user = (
        f"Operator's own node: {smoke['entity_node']}\n\n"
        f"Evidence document:\n{smoke['documents'][0]['text']}"
    )
    smoke_call(PROMPT_C1, smoke_c1_user, validate_c1)
    smoke_call(PROMPT_C3_DOC, smoke_c3_user, validate_c3)
    smoke_manifest = {
        "status": "pass", "fixture_sha256": EXPECTED_SMOKE_FIXTURE_SHA256,
        "schema": "C1/C3 JSON objects", "n_calls": 2,
    }

    for row in sorted(selected, key=lambda item: (item["system"], item["query_id"])):
        qid, system = row["query_id"], row["system"]
        docs = row["documents"]
        doc_ids = [doc["doc_id"] for doc in docs]
        c1_user = (
            f"Operator information need: {row['query_text']}\n\nEvidence:\n"
            + "\n\n".join(f"[DOC {doc['doc_id']}] {doc['text']}" for doc in docs)
        )
        c3_users = [
            f"Operator's own node: {row['metadata']['entity_node']}\n\n"
            f"Evidence document:\n{doc['text']}"
            for doc in docs
        ]
        for consumer, prompt, users in (
            ("C1", PROMPT_C1, [c1_user]),
            ("C3", PROMPT_C3_DOC, c3_users),
        ):
            parsed_items = []
            for doc_index, user in enumerate(users):
                call_id += 1
                raw, record = "", {
                    "call_id": call_id, "query_id": qid, "system": system,
                    "consumer": consumer,
                    "doc_id": None if consumer == "C1" else doc_ids[doc_index],
                    "model": model_id, "revision": revision,
                    "prompt_sha": PROMPT_SHAS[consumer], "temperature": 0.0,
                    "valid": False,
                }
                try:
                    raw = generate(prompt, user)
                    parsed = parse_json(raw)
                    parsed = validate_c1(parsed) if consumer == "C1" else validate_c3(parsed)
                    record.update({"raw": raw, "parsed": parsed, "valid": True})
                    parsed_items.append((record["doc_id"] or "", parsed))
                except Exception as exc:
                    record.update({
                        "raw": raw, "error": f"{type(exc).__name__}: {exc}"
                    })
                    failures.append(record)
                raw_handle.write(json.dumps(record, sort_keys=True) + "\n")
                raw_handle.flush()
            if consumer == "C1" and parsed_items:
                belief_rows.append({
                    "query_id": qid, "system": system, "consumer": "C1",
                    "model": model_id, "k": 3,
                    "true_regime": row["metadata"]["true_regime"],
                    **aggregate_c1(parsed_items[0][1], doc_ids),
                })
            elif consumer == "C3" and len(parsed_items) == 3:
                belief_rows.append({
                    "query_id": qid, "system": system, "consumer": "C3",
                    "model": model_id, "k": 3,
                    "true_regime": row["metadata"]["true_regime"],
                    **aggregate_c3(parsed_items, doc_ids),
                })

    raw_handle.close()
    if len(belief_rows) != 960:
        failures.append({"error": f"expected 960 beliefs, got {len(belief_rows)}"})
    if call_id != 1920:
        failures.append({"error": f"expected 1920 calls, got {call_id}"})

    beliefs_path = out_dir / "beliefs.jsonl"
    with beliefs_path.open("w") as handle:
        for row in sorted(
            belief_rows,
            key=lambda item: (item["system"], item["consumer"], item["query_id"])
        ):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1, "experiment_id": "v5-cross-family-consumer",
        "status": "pass" if not failures else "failed",
        "model": model_id, "revision": revision,
        "runtime_lock_sha256": EXPECTED_RUNTIME_LOCK_SHA256,
        "smoke": smoke_manifest,
        "systems": list(SYSTEMS), "consumers": ["C1", "C3"], "k": 3,
        "n_input_rows": len(rows), "n_selected_rows": len(selected),
        "expected_calls": 1920, "actual_calls": call_id,
        "n_fail": len(failures), "qrels_read": False,
        "prompt_shas": PROMPT_SHAS,
        "decoding": {
            "temperature": 0.0, "top_p": 1.0, "do_sample": False,
            "num_beams": 1, "seed": 0
        },
        "quantization": {
            "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True
        },
        "runtime_pins": {
            "transformers": PINNED_TRANSFORMERS,
            "bitsandbytes": PINNED_BITSANDBYTES,
        },
        "input_sha256": {
            "evidence_inputs.jsonl": sha(input_path),
            "model_spec.json": sha(spec_path)
        },
        "outputs": {
            "raw_outputs": "raw_outputs.jsonl", "beliefs": "beliefs.jsonl"
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (out_dir / "runtime_versions.json").write_text(json.dumps({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "bitsandbytes": __import__("bitsandbytes").__version__,
        "cuda": torch.version.cuda,
    }, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if failures:
        raise SystemExit("consumer run failed closed; raw outputs were frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
