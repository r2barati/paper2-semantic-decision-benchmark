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
MOON_LOCK = ROOT / "versions/sem2act-v5/manifests/moon_cpu_reranker_runtime_v1.json"
sys.path.insert(0, str(CPU_DIR))
from contract import cpu_subprocess_env, load_cpu_lock, load_fixture, sha256_file  # noqa: E402


EXPECTED_FIXTURE_SHA = "9ab69722e0062fc38eb27721f24e9cd83ca7944655973cbbe85f079372ccd74b"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


def resolve_model_path(value: Path) -> Path:
    if value.exists():
        return value.resolve()
    if str(value) != MODEL_ID:
        raise RuntimeError(f"model path does not exist and is not the pinned model id: {value}")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, token=True))


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
        moon_lock = json.loads(MOON_LOCK.read_text())
        moon_lock_sha = sha256_file(MOON_LOCK)
        if (host.get("status") != "pass"
                or host.get("protocol_hash") != lock["protocol_hash"]
                or host.get("runtime_lock_sha256") != moon_lock_sha
                or host.get("torch") != "2.7.1+cpu"
                or host.get("transformers") != "4.57.6"
                or host.get("torch_num_threads") != 16
                or host.get("torch_num_interop_threads") != 1
                or host.get("torch_cuda_available") is not False
                or host.get("torch_cuda_runtime") is not None):
            raise RuntimeError("Moon host preflight is blocked or does not match the committed runtime lock")
        if (moon_lock.get("status") != "prospective-before-moon-host-preflight"
                or moon_lock.get("protocol_hash") != lock["protocol_hash"]
                or moon_lock.get("runtime", {}).get("threads") != 16
                or moon_lock.get("model", {}).get("revision") != MODEL_REVISION):
            raise RuntimeError("Moon reranker lock status/protocol/model/thread mismatch")
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
        model_path = resolve_model_path(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, revision=spec["revision"], padding_side="left", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=spec["revision"], trust_remote_code=True,
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
        def score(document: str) -> float:
            text = (
                "<Instruct>: Given an operational supply-chain information need, "
                "retrieve documents relevant to the described entity, event, and time context.\n"
                f"<Query>: {fixture['query_text']}\n<Document>: {document}"
            )
            encoded = tokenizer(text, truncation=True, max_length=1024 - len(prefix) - len(suffix), return_attention_mask=False)
            batch = tokenizer.pad({"input_ids": [prefix + encoded["input_ids"] + suffix]}, padding=True, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**batch).logits[:, -1, :]
            return float(torch.stack([logits[:, no], logits[:, yes]], dim=1).log_softmax(dim=1)[:, 1].exp()[0])

        started = time.monotonic()
        fixture_rows = fixture["documents"]
        scores_by_threads = {}
        rankings_by_threads = {}
        for thread_count in (4, threads):
            torch.set_num_threads(thread_count)
            first_scores = [score(row["text"]) for row in fixture_rows]
            second_scores = [score(row["text"]) for row in fixture_rows]
            if first_scores != second_scores or any(not 0.0 <= value <= 1.0 for value in first_scores):
                raise RuntimeError(f"reranker canary repeat mismatch or invalid score at {thread_count} threads")
            scores_by_threads[str(thread_count)] = first_scores
            rankings_by_threads[str(thread_count)] = [
                row["doc_id"] for _, row in sorted(
                    zip(first_scores, fixture_rows),
                    key=lambda pair: (-pair[0], pair[1]["doc_id"]),
                )
            ]
        if rankings_by_threads["4"] != rankings_by_threads[str(threads)]:
            raise RuntimeError("reranker fixture ranking changes between the frozen 4-thread and proposed Moon thread settings")
        torch.set_num_threads(threads)
        target_scores = scores_by_threads[str(threads)]
        baseline_scores = scores_by_threads["4"]
        max_abs_score_diff = max(abs(a - b) for a, b in zip(target_scores, baseline_scores))
        result.update({
            "status": "pass",
            "schema_valid": True,
            "deterministic_repeat_equality": True,
            "thread_comparison": {
                "fixture_document_ids": [row["doc_id"] for row in fixture_rows],
                "baseline_threads": 4,
                "target_threads": threads,
                "ranking_equal": True,
                "ranking": rankings_by_threads[str(threads)],
                "baseline_scores": baseline_scores,
                "target_scores": target_scores,
                "max_abs_score_diff": max_abs_score_diff,
            },
            "fixture_sha256": EXPECTED_FIXTURE_SHA,
            "protocol_hash": lock["protocol_hash"],
            "runtime_lock_sha256": moon_lock_sha,
            "host_preflight_sha256": sha256_file(args.host_manifest),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "dtype": "float32",
            "device": "cpu",
            "score": target_scores[0],
            "wall_seconds": round(time.monotonic() - started, 3),
            "threads": threads,
            "source_script_sha256": sha256_file(Path(__file__)),
            "model_snapshot": model_snapshot(model_path),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
