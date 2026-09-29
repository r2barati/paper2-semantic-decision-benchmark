"""Run one frozen CPU reranker shard with the exact v5 reranker revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CPU_DIR = ROOT / "versions/sem2act-v5/compute/cpu"
sys.path.insert(0, str(CPU_DIR))
from contract import (  # noqa: E402
    PROMPT_SHAS,
    cpu_subprocess_env,
    load_cpu_freeze,
    load_cpu_lock,
    sha256_bytes,
    sha256_file,
)

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = (
    "Given an operational supply-chain information need, retrieve documents "
    "relevant to the described entity, event, and time context."
)
TOPN = 50
MAXLEN = 1024
MOON_LOCK_PATH = ROOT / "versions/sem2act-v5/manifests/moon_cpu_reranker_runtime_v1.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any("qrel" in json.dumps(row, sort_keys=True).lower() for row in rows):
        raise RuntimeError(f"qrel token detected in {path}")
    return rows


def resolve_model_path(value: Path) -> Path:
    if value.exists():
        return value.resolve()
    if str(value) != MODEL_ID:
        raise RuntimeError(f"model path does not exist and is not the pinned model id: {value}")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, token=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--moon-host-manifest", type=Path)
    parser.add_argument("--moon-canary-manifest", type=Path)
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    lock = load_cpu_lock()
    moon_mode = bool(args.moon_host_manifest or args.moon_canary_manifest)
    canary_manifest = {}
    if moon_mode:
        if not args.moon_host_manifest or not args.moon_canary_manifest:
            raise SystemExit("Moon mode requires both a passing host preflight and reranker canary")
        moon_lock = json.loads(MOON_LOCK_PATH.read_text())
        host_manifest = json.loads(args.moon_host_manifest.read_text())
        canary_manifest = json.loads(args.moon_canary_manifest.read_text())
        moon_lock_sha = sha256_file(MOON_LOCK_PATH)
        if moon_lock.get("status") != "prospective-before-moon-host-preflight" or moon_lock.get("protocol_hash") != lock["protocol_hash"]:
            raise SystemExit("Moon CPU reranker lock status/protocol mismatch")
        if (moon_lock.get("runtime", {}).get("threads") != 16
                or moon_lock.get("runtime", {}).get("threads_batch") != 16
                or moon_lock.get("model", {}).get("revision") != REVISION):
            raise SystemExit("Moon CPU reranker model/thread settings drift")
        if host_manifest.get("status") != "pass" or host_manifest.get("runtime_lock_sha256") != moon_lock_sha:
            raise SystemExit("Moon host preflight does not match the committed runtime lock")
        if canary_manifest.get("status") != "pass" or canary_manifest.get("deterministic_repeat_equality") is not True:
            raise SystemExit("Moon fixed-fixture reranker canary did not pass exact repeat equality")
        if (canary_manifest.get("runtime_lock_sha256") != moon_lock_sha
                or canary_manifest.get("host_preflight_sha256") != sha256_file(args.moon_host_manifest)):
            raise SystemExit("Moon canary is not bound to this host/runtime preflight")
        if canary_manifest.get("fixture_sha256") != sha256_file(ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"):
            raise SystemExit("Moon canary fixture hash mismatch")
        freeze = {
            "frozen_threads": int(host_manifest.get("torch_num_threads", 0)),
            "frozen_threads_batch": int(host_manifest.get("torch_num_threads", 0)),
            "runtime_lock_sha256": moon_lock_sha,
            "host_preflight_sha256": sha256_file(args.moon_host_manifest),
            "canary_sha256": sha256_file(args.moon_canary_manifest),
        }
        if freeze["frozen_threads"] != 16:
            raise SystemExit("Moon pilot thread count is not the approved 16")
    else:
        freeze = load_cpu_freeze(require_frozen=True)
    if lock["models"]["reranker"]["model_id"] != MODEL_ID or lock["models"]["reranker"]["revision"] != REVISION:
        raise SystemExit("reranker model pin drift")
    shard = json.loads(args.shard.read_text())
    if shard.get("stage") != "reranker":
        raise SystemExit("not a reranker shard")
    if moon_mode and (
            shard.get("manifest_id") != "sem2act-v5-cpu-reranker-shard-000"
            or shard.get("query_ids") != [f"v5-lb-q{i:04d}" for i in range(1, 21)]
            or shard.get("candidate_depth") != 50
            or shard.get("expected_queries") != 20):
        raise SystemExit("Moon pilot is not the exact frozen first 20-query shard")
    queries = load_jsonl(args.queries)
    corpus = {row["_id"]: row["text"] for row in load_jsonl(args.corpus)}
    candidates = {row["query_id"]: row["doc_ids"] for row in load_jsonl(args.candidates)}
    for key, path in (("queries_sha256", args.queries), ("corpus_sha256", args.corpus), ("candidates_sha256", args.candidates)):
        if shard.get(key) != sha256_file(path):
            raise SystemExit(f"pilot input hash differs from fixed shard manifest: {key}")
    selected = [row for row in queries if row["_id"] in set(shard["query_ids"])]
    if {row["_id"] for row in selected} != set(shard["query_ids"]):
        raise SystemExit("reranker shard query coverage mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "rankings.jsonl"
    accepted: dict[str, dict] = {}
    if accepted_path.exists():
        for line in accepted_path.read_text().splitlines():
            row = json.loads(line)
            key = row["key"]
            if key in accepted:
                raise SystemExit(f"duplicate accepted reranker key: {key}")
            accepted[key] = row
    missing = [row for row in sorted(selected, key=lambda item: item["_id"])
               if f"reranker|{row['_id']}" not in accepted]
    started = time.monotonic()
    failures = []
    if missing:
        if args.model_path is None:
            raise SystemExit("--model-path is required for missing reranker keys")
        model_path = resolve_model_path(args.model_path)
        # Imports and model loading happen only after the frozen result gate.
        os.environ.update(cpu_subprocess_env())
        os.environ["OMP_NUM_THREADS"] = str(freeze["frozen_threads"])
        os.environ["MKL_NUM_THREADS"] = str(freeze["frozen_threads"])
        os.environ["OPENBLAS_NUM_THREADS"] = str(freeze["frozen_threads"])
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if moon_mode and (torch.__version__ != "2.7.1+cpu"
                          or getattr(torch.version, "cuda", None) is not None
                          or __import__("transformers").__version__ != "4.57.6"):
            raise SystemExit("Moon CPU runtime differs from the committed package lock")

        if torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            raise SystemExit("CPU reranker detected an accelerator")
        torch.set_num_threads(int(freeze["frozen_threads"]))
        torch.set_num_interop_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, revision=REVISION, padding_side="left", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=REVISION, trust_remote_code=True,
            torch_dtype=torch.float32, device_map={"": "cpu"}
        ).eval()
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

        def score(query: str, doc: str) -> float:
            text = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
            encoded = tokenizer(text, truncation=True, max_length=MAXLEN - len(prefix) - len(suffix), return_attention_mask=False)
            ids = prefix + encoded["input_ids"] + suffix
            batch = tokenizer.pad({"input_ids": [ids]}, padding=True, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**batch).logits[:, -1, :]
            return float(torch.stack([logits[:, no], logits[:, yes]], dim=1).log_softmax(dim=1)[:, 1].exp()[0])

        with accepted_path.open("a") as handle:
            for row in missing:
                qid = row["_id"]
                scored = [(score(row["text"], corpus[doc_id]), doc_id) for doc_id in candidates[qid]]
                ranking = [doc_id for _, doc_id in sorted(scored, key=lambda pair: (-pair[0], pair[1]))]
                record = {
                    "key": f"reranker|{qid}",
                    "query_id": qid,
                    "ranking": ranking,
                    "scores_sha256": sha256_bytes(json.dumps(scored, sort_keys=True).encode()),
                    "ranking_sha256": sha256_bytes(json.dumps(ranking, separators=(",", ":")).encode()),
                    "valid": len(ranking) == TOPN and len(set(ranking)) == TOPN,
                }
                if not record["valid"]:
                    failures.append({"key": record["key"], "error": "ranking schema failure"})
                else:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    accepted[record["key"]] = record
    expected = set(shard["expected_query_keys"])
    if set(accepted) != expected:
        failures.append({"error": "reranker shard keyspace failed", "missing": sorted(expected - set(accepted))})
    failures_path = args.output_dir / "failures.json"
    failures_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")
    rankings_path = args.output_dir / "rankings.jsonl"
    canary_model_snapshot = canary_manifest.get("model_snapshot") if moon_mode else None
    manifest = {
        "schema_version": 1,
        "manifest_id": shard["manifest_id"],
        "status": "pass" if not failures else "failed",
        "protocol_hash": lock["protocol_hash"],
        "runtime_freeze_sha256": None if moon_mode else sha256_file(ROOT / "versions/sem2act-v5/manifests/cpu_runtime_freeze.json"),
        "runtime_lock_sha256": freeze.get("runtime_lock_sha256"),
        "host_preflight_sha256": freeze.get("host_preflight_sha256"),
        "canary_sha256": freeze.get("canary_sha256"),
        "shard_manifest_sha256": sha256_file(args.shard),
        "input_sha256": {
            "queries.jsonl": sha256_file(args.queries),
            "corpus.jsonl": sha256_file(args.corpus),
            "rerank_candidates.jsonl": sha256_file(args.candidates),
        },
        "pilot_only": moon_mode,
        "threads": freeze.get("frozen_threads"),
        "torch": "2.7.1+cpu" if moon_mode else None,
        "transformers": "4.57.6" if moon_mode else None,
        "model_snapshot_sha256": None if not canary_model_snapshot else sha256_bytes(json.dumps(canary_model_snapshot, sort_keys=True, separators=(",", ":")).encode()),
        "account_identity": os.environ.get("TMU_USER", "").strip(),
        "execution_sha": os.environ.get("SEM2ACT_HANDOFF_SHA"),
        "stage": "reranker",
        "expected_queries": len(expected),
        "accepted_queries": len(accepted),
        "n_fail": len(failures),
        "n_pairs": len(accepted) * TOPN,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "dtype": "float32",
        "device": "cpu",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "qrels_read": False,
        "seed": None,
        "seed_applicability": "none; deterministic model.eval inference",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "output_sha256": {
            "rankings.jsonl": sha256_file(rankings_path),
            "failures.json": sha256_file(failures_path),
        },
        "source_script_sha256": sha256_file(Path(__file__)),
        "source_contract_sha256": sha256_file(CPU_DIR / "contract.py"),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
