"""Run the fixed non-lockbox reranker CPU canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_DIR = ROOT / "versions/sem2act-v5/compute/cpu"
sys.path.insert(0, str(CPU_DIR))
from contract import cpu_subprocess_env, load_cpu_lock, load_fixture, sha256_file  # noqa: E402


EXPECTED_FIXTURE_SHA = "9ab69722e0062fc38eb27721f24e9cd83ca7944655973cbbe85f079372ccd74b"


def model_snapshot(path: Path) -> dict:
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or ".cache" in item.parts:
            continue
        files.append({"path": str(item.relative_to(path)), "sha256": sha256_file(item), "bytes": item.stat().st_size})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"file_count": len(files), "files_sha256": digest, "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host-manifest", type=Path, default=ROOT / "versions/sem2act-v5/manifests/cpu_host_preflight.json")
    parser.add_argument("--output", type=Path, default=ROOT / "versions/sem2act-v5/manifests/cpu_reranker_canary.json")
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-cpu-reranker-canary-v1",
        "status": "failed",
        "result_bearing_execution_started": False,
    }
    try:
        lock = load_cpu_lock()
        fixture_path = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
        if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != EXPECTED_FIXTURE_SHA:
            raise RuntimeError("CPU smoke fixture hash drift")
        fixture = load_fixture()
        host = json.loads(args.host_manifest.read_text())
        threads = int(host["frozen_threads"])
        os.environ.update(cpu_subprocess_env())
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            raise RuntimeError("reranker canary detected an accelerator")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        spec = lock["models"]["reranker"]
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, revision=spec["revision"], padding_side="left", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, revision=spec["revision"], trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to("cpu").eval()
        yes = tokenizer.convert_tokens_to_ids("yes")
        no = tokenizer.convert_tokens_to_ids("no")
        prefix = tokenizer.encode(
            '<|im_start|>system\nJudge whether the Document meets the requirements based '
            'on the Query and the Instruct provided. Note that the answer can only be '
            '"yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False
        )
        suffix = tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
        )
        text = f"<Instruct>: Given an operational supply-chain information need, retrieve documents relevant to the described entity, event, and time context.\n<Query>: {fixture['query_text']}\n<Document>: {fixture['documents'][0]['text']}"

        def score() -> float:
            encoded = tokenizer(text, truncation=True, max_length=1024 - len(prefix) - len(suffix), return_attention_mask=False)
            batch = tokenizer.pad({"input_ids": [prefix + encoded["input_ids"] + suffix]}, padding=True, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**batch).logits[:, -1, :]
            return float(torch.stack([logits[:, no], logits[:, yes]], dim=1).log_softmax(dim=1)[:, 1].exp()[0])
        started = time.monotonic()
        first = score()
        second = score()
        if not (0.0 <= first <= 1.0 and first == second):
            raise RuntimeError("reranker canary repeat mismatch or invalid score")
        result.update({
            "status": "pass",
            "schema_valid": True,
            "deterministic_repeat_equality": True,
            "fixture_sha256": EXPECTED_FIXTURE_SHA,
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "dtype": "float32",
            "device": "cpu",
            "score": first,
            "wall_seconds": round(time.monotonic() - started, 3),
            "threads": threads,
            "source_script_sha256": sha256_file(Path(__file__)),
            "model_snapshot": model_snapshot(args.model_path),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
