"""Run one frozen CPU reranker shard with the exact v5 reranker revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any("qrel" in json.dumps(row, sort_keys=True).lower() for row in rows):
        raise RuntimeError(f"qrel token detected in {path}")
    return rows


def validate_moon_parity(payload: dict) -> None:
    profiles = payload.get("profiles", {})
    reference = profiles.get("4", {})
    candidate = profiles.get("16", {})
    scores4 = reference.get("scores", [])
    scores16 = candidate.get("scores", [])
    delta = payload.get("max_absolute_score_delta", float("inf"))
    if (payload.get("status") != "pass"
            or payload.get("exact_repeat_equality") is not True
            or payload.get("ranking_equal") is not True
            or payload.get("authorized_threads") != 16
            or payload.get("score_tolerance") != 1e-6
            or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta > 1e-6
            or reference.get("repeat_equal") is not True
            or candidate.get("repeat_equal") is not True
            or not isinstance(scores4, list) or len(scores4) != 3
            or not isinstance(scores16, list) or len(scores16) != 3
            or any(not isinstance(score, (int, float)) or not math.isfinite(score)
                   for score in scores4 + scores16)
            or reference.get("ranking") != candidate.get("ranking")
            or max((abs(a - b) for a, b in zip(scores4, scores16)), default=float("inf")) > 1e-6):
        raise RuntimeError("Moon 4-vs-16 parity evidence is incomplete or failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--moon-pilot-manifest", type=Path,
                        help="required for the separately frozen Moon reranker-only route")
    parser.add_argument("--moon-query-id",
                        help="score one complete authorized query in its own Moon subprocess")
    args = parser.parse_args()
    moon_runtime = None
    if args.moon_pilot_manifest:
        moon_runtime = json.loads(args.moon_pilot_manifest.read_text())
        if moon_runtime.get("status") != "authorized-after-gates":
            raise SystemExit("Moon pilot authorization manifest did not pass its gates")
        if moon_runtime.get("protocol_hash") != "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022":
            raise SystemExit("Moon pilot protocol hash drift")
        if moon_runtime.get("thread_count") != 16 or moon_runtime.get("parity_status") != "pass":
            raise SystemExit("Moon pilot 16-thread parity authorization is missing")
        if (moon_runtime.get("subprocess_query_chunk_size") != 1
                or moon_runtime.get("candidate_depth") != TOPN
                or moon_runtime.get("resume_policy") != "one-shot-empty-output-only"
                or not args.moon_query_id
                or args.moon_query_id not in moon_runtime.get("authorized_query_ids", [])):
            raise SystemExit("Moon authorization does not permit this one-query subprocess")
        if (moon_runtime.get("subprocess_query_chunk_size") != 1
                or moon_runtime.get("candidate_depth") != TOPN
                or moon_runtime.get("resume_policy") != "one-shot-empty-output-only"
                or not args.moon_query_id
                or args.moon_query_id not in moon_runtime.get("authorized_query_ids", [])):
            raise SystemExit("Moon authorization does not permit this one-query subprocess")
        lock_path = ROOT / "versions/sem2act-v5/manifests/moon_reranker_runtime_lock.json"
        freeze_path = ROOT / "versions/sem2act-v5/manifests/moon_reranker_runtime_freeze.json"
        if moon_runtime.get("runtime_lock_sha256") != sha256_file(lock_path):
            raise SystemExit("Moon pilot runtime lock hash mismatch")
        if moon_runtime.get("runtime_freeze_sha256") != sha256_file(freeze_path):
            raise SystemExit("Moon pilot runtime freeze hash mismatch")
        if moon_runtime.get("pilot_shard_sha256") != sha256_file(args.shard):
            raise SystemExit("Moon pilot shard hash mismatch")
        for name, path in (
            ("queries.jsonl", args.queries), ("corpus.jsonl", args.corpus),
            ("rerank_candidates.jsonl", args.candidates),
        ):
            if moon_runtime.get("input_files", {}).get(name) != sha256_file(path):
                raise SystemExit(f"Moon pilot input hash mismatch: {name}")
        for field in ("host_preflight", "thread_canary"):
            record = moon_runtime.get(field, {})
            path = Path(record.get("path", ""))
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise SystemExit(f"Moon {field} evidence path/hash mismatch")
            payload = json.loads(path.read_text())
            if payload.get("status") != "pass":
                raise SystemExit(f"Moon {field} did not pass")
            if payload.get("execution_sha") != moon_runtime.get("execution_sha"):
                raise SystemExit(f"Moon {field} execution SHA mismatch")
        canary_record = json.loads(Path(moon_runtime["thread_canary"]["path"]).read_text())
        try:
            validate_moon_parity(canary_record)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        freeze = {"frozen_threads": 16, "frozen_threads_batch": 16}
        runtime_freeze_sha256 = moon_runtime["runtime_freeze_sha256"]
    else:
        if args.moon_query_id:
            raise SystemExit("--moon-query-id is only valid for the authorized Moon route")
        freeze = load_cpu_freeze(require_frozen=True)
        runtime_freeze_sha256 = sha256_file(ROOT / "versions/sem2act-v5/manifests/cpu_runtime_freeze.json")
    lock = load_cpu_lock()
    if lock["models"]["reranker"]["model_id"] != MODEL_ID or lock["models"]["reranker"]["revision"] != REVISION:
        raise SystemExit("reranker model pin drift")
    shard = json.loads(args.shard.read_text())
    if shard.get("stage") != "reranker":
        raise SystemExit("not a reranker shard")
    if moon_runtime and shard.get("resume_policy") != "one-shot-empty-output-only":
        raise SystemExit("Moon shard must use the one-shot empty-output policy")
    if moon_runtime and shard.get("resume_policy") != "one-shot-empty-output-only":
        raise SystemExit("Moon shard must use the one-shot empty-output policy")
    for name, path in (
        ("queries_sha256", args.queries), ("corpus_sha256", args.corpus),
        ("candidates_sha256", args.candidates),
    ):
        if shard.get(name) != sha256_file(path):
            raise SystemExit(f"planned reranker shard input hash mismatch: {name}")
    queries = load_jsonl(args.queries)
    corpus = {row["_id"]: row["text"] for row in load_jsonl(args.corpus)}
    candidates = {row["query_id"]: row["doc_ids"] for row in load_jsonl(args.candidates)}
    selected = [row for row in queries if row["_id"] in set(shard["query_ids"])]
    if {row["_id"] for row in selected} != set(shard["query_ids"]):
        raise SystemExit("reranker shard query coverage mismatch")
    if args.moon_query_id:
        if args.moon_query_id not in shard["query_ids"]:
            raise SystemExit("Moon subprocess query is outside the authorized pilot shard")
        selected = [row for row in selected if row["_id"] == args.moon_query_id]
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
    if args.moon_query_id and accepted:
        raise SystemExit("Moon query subprocess output directory must start empty")
    missing = [row for row in sorted(selected, key=lambda item: item["_id"])
               if f"reranker|{row['_id']}" not in accepted]
    started = time.monotonic()
    failures = []
    if missing:
        if args.model_path is None:
            raise SystemExit("--model-path is required for missing reranker keys")
        # Imports and model loading happen only after the frozen result gate.
        os.environ.update(cpu_subprocess_env())
        os.environ["OMP_NUM_THREADS"] = str(freeze["frozen_threads"])
        os.environ["MKL_NUM_THREADS"] = str(freeze["frozen_threads"])
        os.environ["OPENBLAS_NUM_THREADS"] = str(freeze["frozen_threads"])
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            raise SystemExit("CPU reranker detected an accelerator")
        torch.set_num_threads(int(freeze["frozen_threads"]))
        torch.set_num_interop_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, revision=REVISION, padding_side="left", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, revision=REVISION, trust_remote_code=True,
            torch_dtype=torch.float32, device_map={"": "cpu"}
        ).eval()
        if moon_runtime:
            (args.output_dir / "result_bearing_started.json").write_text(json.dumps({
                "schema_version": 1,
                "status": "started",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "execution_sha": moon_runtime["execution_sha"],
                "protocol_hash": lock["protocol_hash"],
                "runtime_lock_sha256": moon_runtime["runtime_lock_sha256"],
                "runtime_freeze_sha256": moon_runtime["runtime_freeze_sha256"],
                "shard_sha256": sha256_file(args.shard),
                "thread_count": int(freeze["frozen_threads"]),
                "chunk_query_id": args.moon_query_id,
                "candidate_count": len(candidates[args.moon_query_id]),
                "qrels_read": False,
            }, indent=2, sort_keys=True) + "\n")
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
                expected_docs = candidates[qid]
                record = {
                    "key": f"reranker|{qid}",
                    "query_id": qid,
                    "ranking": ranking,
                    "scores_sha256": sha256_bytes(json.dumps(scored, sort_keys=True).encode()),
                    "ranking_sha256": sha256_bytes(json.dumps(ranking, separators=(",", ":")).encode()),
                    "valid": (len(ranking) == TOPN and len(set(ranking)) == TOPN
                              and len(expected_docs) == TOPN
                              and len(set(expected_docs)) == TOPN
                              and set(ranking) == set(expected_docs)),
                }
                if not record["valid"]:
                    failures.append({"key": record["key"], "error": "ranking schema failure"})
                else:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    accepted[record["key"]] = record
    expected = ({f"reranker|{args.moon_query_id}"} if args.moon_query_id
                else set(shard["expected_query_keys"]))
    if set(accepted) != expected:
        failures.append({"error": "reranker shard keyspace failed", "missing": sorted(expected - set(accepted))})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_soft, cpu_hard = resource.getrlimit(resource.RLIMIT_CPU)
    expected_query_ids = [args.moon_query_id] if args.moon_query_id else shard["query_ids"]
    manifest = {
        "schema_version": 1,
        "manifest_id": shard["manifest_id"],
        "status": "pass" if not failures else "failed",
        "protocol_hash": lock["protocol_hash"],
        "runtime_profile": "moon-reranker-only-v1" if moon_runtime else "cpu-consumer-v1",
        "runtime_freeze_sha256": runtime_freeze_sha256,
        "runtime_lock_sha256": (
            moon_runtime["runtime_lock_sha256"] if moon_runtime
            else sha256_file(ROOT / "versions/sem2act-v5/manifests/cpu_runtime_lock.json")
        ),
        "execution_sha": moon_runtime.get("execution_sha") if moon_runtime else None,
        "host_preflight_sha256": moon_runtime["host_preflight"]["sha256"] if moon_runtime else None,
        "thread_canary_sha256": moon_runtime["thread_canary"]["sha256"] if moon_runtime else None,
        "stage": "reranker",
        "expected_queries": len(expected),
        "expected_query_ids": expected_query_ids,
        "expected_pairs": sum(len(candidates[qid]) for qid in expected_query_ids),
        "accepted_queries": len(accepted),
        "accepted_pairs": sum(
            len(accepted[f"reranker|{qid}"].get("ranking", []))
            for qid in expected_query_ids
            if f"reranker|{qid}" in accepted
        ),
        "n_fail": len(failures),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "dtype": "float32",
        "device": "cpu",
        "thread_count": int(freeze["frozen_threads"]),
        "inter_op_threads": 1,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "process_cpu_seconds": round(usage.ru_utime + usage.ru_stime, 3),
        "process_cpu_limit_soft_seconds": None if cpu_soft == resource.RLIM_INFINITY else cpu_soft,
        "process_cpu_limit_hard_seconds": None if cpu_hard == resource.RLIM_INFINITY else cpu_hard,
        "chunk_query_id": args.moon_query_id,
        "model_staging_manifest_sha256": (
            moon_runtime.get("model_staging_manifest_sha256") if moon_runtime else None
        ),
        "result_bearing_execution_started": bool(moon_runtime and missing),
        "qrels_read": False,
    }
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
