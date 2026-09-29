"""v5 qrel-free reranker.

Reads only queries, corpus, and BM25 candidate IDs. No qrels are attached or
read. The scoring recipe is the frozen Qwen3-Reranker-0.6B yes/no-logit method.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = (
    "Given an operational supply-chain information need, retrieve documents "
    "relevant to the described entity, event, and time context."
)
TOPN = 50
MAXLEN = 1024
PINNED_TRANSFORMERS = "4.57.6"
PINNED_KAGGLE_TORCH = "2.7.1+cu126"
PINNED_COLAB_TORCH = "2.11.0+cu128"
KAGGLE_RUNTIME_LOCK_SHA256 = "df2618f13337b37d59ae4e21cda4eb3423569190869db27614b651f07234ca6b"
COLAB_RUNTIME_LOCK_SHA256 = "b40454497dd6a365aa1531b691eb515bab06a156e1a158d774af92d9cb4c3c4b"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
EXPECTED_SMOKE_FIXTURE_SHA256 = "9ab69722e0062fc38eb27721f24e9cd83ca7944655973cbbe85f079372ccd74b"
EXPECTED_SMOKE_CANONICAL_SHA256 = "f90caec2e59315135fffb4b032fb0afceb7b9ca6cc164aaafabd26b548ba1004"
DEFAULT_PILOT_LIMIT = None
SMOKE_FIXTURE = {
    "schema_version": 1,
    "fixture_id": "sem2act-v5-runtime-smoke-v1",
    "purpose": "non-result-bearing model load, tokenization, schema, and CUDA smoke",
    "query_id": "runtime-smoke-q0000",
    "query_text": "Assess the current operational risk for the Northwind distribution node during this planning window.",
    "entity_node": "Northwind distribution node",
    "documents": [
        {"doc_id": "runtime-smoke-d0001", "text": "Northwind supplier bulletin: inbound replenishment is delayed by four periods because of a current port disruption."},
        {"doc_id": "runtime-smoke-d0002", "text": "Northwind operations note: customer orders remain within the normal seasonal range for this planning window."},
        {"doc_id": "runtime-smoke-d0003", "text": "Northwind market brief: a current promotion is associated with a possible demand increase at nearby outlets."},
    ],
    "expected": {
        "reranker": "finite_yes_no_score",
        "C1": ["normal", "supplier_delay", "demand_surge", "estimated_lt_increase", "estimated_duration", "estimated_demand_multiplier"],
        "C3": ["entity_match", "event", "fresh", "stance", "confidence"],
    },
}


def require_hf_secret(backend):
    if backend == "colab":
        if not os.environ.get("HF_TOKEN"):
            raise SystemExit("HF_TOKEN is unavailable from the Muse/Colab secret store")
        return
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise SystemExit(f"HF_TOKEN Kaggle secret is unavailable: {type(exc).__name__}")
    if not token:
        raise SystemExit("HF_TOKEN Kaggle secret is empty")
    os.environ["HF_TOKEN"] = token


def install_runtime_pins(backend):
    if backend == "kaggle":
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "--index-url", "https://download.pytorch.org/whl/cu126",
            f"torch=={PINNED_KAGGLE_TORCH}",
        ], check=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            f"transformers=={PINNED_TRANSFORMERS}",
        ], check=True)


def find_input(name, input_root=None):
    roots = [input_root] if input_root else ["/kaggle/input", ".", "inputs", "/kaggle/working/inputs"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(name)


def strip_transport(text):
    return text.strip()


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("kaggle", "colab"), default=os.environ.get("SEM2ACT_BACKEND", "kaggle"))
    parser.add_argument("--pilot-limit", type=int, default=DEFAULT_PILOT_LIMIT, choices=(20,))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--runtime-lock")
    args = parser.parse_args()
    backend = args.backend
    started_utc = datetime.now(timezone.utc).isoformat()
    account_identity = os.environ.get("SEM2ACT_ACCOUNT_ID", "").strip()
    if backend == "colab" and not account_identity:
        raise SystemExit("set SEM2ACT_ACCOUNT_ID to the non-secret Colab account/profile alias")
    expected_lock_sha = KAGGLE_RUNTIME_LOCK_SHA256 if backend == "kaggle" else COLAB_RUNTIME_LOCK_SHA256
    if backend == "colab":
        if not args.runtime_lock:
            raise SystemExit("Colab requires --runtime-lock from the exact checkout")
        if sha(args.runtime_lock) != expected_lock_sha:
            raise SystemExit("Colab runtime-lock hash mismatch")
    install_runtime_pins(backend)
    require_hf_secret(backend)

    import torch
    import transformers

    expected_torch = PINNED_KAGGLE_TORCH if backend == "kaggle" else PINNED_COLAB_TORCH
    expected_cuda = "12.6" if backend == "kaggle" else "12.8"
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Python runtime drift: expected 3.12, got {platform.python_version()}")
    if torch.__version__ != expected_torch or not str(torch.version.cuda or "").startswith(expected_cuda):
        raise SystemExit("torch/CUDA runtime differs from the explicit backend lock")
    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise SystemExit(f"transformers pin drift: {transformers.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if "t4" not in device_name.lower():
        raise SystemExit(f"expected one Tesla T4, got {device_name}")
    device_memory = torch.cuda.get_device_properties(0).total_memory
    if device_memory < 14 * 1024**3:
        raise SystemExit("T4 device exposes less than 14 GiB of memory")
    runtime = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda, "transformers": transformers.__version__,
        "device_count": torch.cuda.device_count(), "device_name": device_name,
        "device_memory_bytes": device_memory,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "cuda_driver_report": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:2000],
    }

    if backend == "kaggle" and any("qrel" in name.lower() for name in os.listdir("/kaggle/input")):
        raise SystemExit("qrel-bearing input detected")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, padding_side="left", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.float16, device_map={"": "cuda:0"}
    ).eval()

    yes = tokenizer.convert_tokens_to_ids("yes")
    no = tokenizer.convert_tokens_to_ids("no")
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

    @torch.no_grad()
    def score_batch(pairs):
        texts = [
            f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
            for query, doc in pairs
        ]
        enc = tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=MAXLEN - len(prefix) - len(suffix),
        )
        ids = [prefix + item + suffix for item in enc["input_ids"]]
        batch = tokenizer.pad(
            {"input_ids": ids}, padding=True, return_tensors="pt"
        ).to(model.device)
        logits = model(**batch).logits[:, -1, :]
        return torch.stack([logits[:, no], logits[:, yes]], dim=1).log_softmax(
            dim=1
        )[:, 1].exp().tolist()

    if hashlib.sha256(json.dumps(SMOKE_FIXTURE, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != EXPECTED_SMOKE_CANONICAL_SHA256:
        raise SystemExit("embedded non-lockbox fixture content drift")
    smoke = SMOKE_FIXTURE
    smoke_scores = score_batch([(smoke["query_text"], smoke["documents"][0]["text"])])
    if len(smoke_scores) != 1 or not math.isfinite(smoke_scores[0]):
        raise SystemExit("reranker smoke schema/execution failure")
    smoke_manifest = {
        "status": "pass", "fixture_sha256": EXPECTED_SMOKE_FIXTURE_SHA256,
        "schema": "one finite reranker score", "n_calls": 1,
    }

    output_root = args.output_root or ("/kaggle/working" if backend == "kaggle" else "/content/sem2act-v5/pilot")
    os.makedirs(output_root, exist_ok=True)
    if args.smoke_only:
        canary = {
            "schema_version": 1, "status": "pass", "stage": "reranker-smoke-only",
            "protocol_hash": PROTOCOL_HASH, "runtime_lock_sha256": expected_lock_sha,
            "backend": backend, "model": MODEL_ID, "revision": REVISION,
            "fixture_sha256": EXPECTED_SMOKE_FIXTURE_SHA256, "runtime": runtime,
            "account_identity": account_identity or "rezabarati2",
            "qrels_read": False, "result_bearing_execution_started": False,
        }
        with open(os.path.join(output_root, "canary_manifest.json"), "w") as handle:
            json.dump(canary, handle, indent=2, sort_keys=True)
        print(json.dumps(canary, indent=2, sort_keys=True), flush=True)
        return 0

    corpus_path = find_input("corpus.jsonl", args.input_root)
    queries_path = find_input("queries.jsonl", args.input_root)
    candidates_path = find_input("rerank_candidates.jsonl", args.input_root)
    docs = {x["_id"]: x["text"] for x in map(json.loads, open(corpus_path))}
    queries = [json.loads(x) for x in open(queries_path)]
    candidates = [json.loads(x) for x in open(candidates_path)]
    if len(queries) != 240 or len(candidates) != 240:
        raise SystemExit("expected exactly 240 queries and candidate rows")
    candidate_map = {x["query_id"]: x["doc_ids"] for x in candidates}
    if set(candidate_map) != {q["_id"] for q in queries} or any(len(candidate_map[q["_id"]]) != TOPN for q in queries):
        raise SystemExit("candidate depth/keyspace drift")
    queries = sorted(queries, key=lambda row: row["_id"])
    pilot = args.pilot_limit == 20
    selected = queries[:20] if pilot else queries
    expected_pilot = [f"v5-lb-q{i:04d}" for i in range(1, 21)]
    if pilot and [q["_id"] for q in selected] != expected_pilot:
        raise SystemExit("pilot query keyspace differs from the fixed first shard")

    output = {}
    for index, query in enumerate(selected, start=1):
        qid = query["_id"]
        cand = candidate_map[qid]
        scores = []
        for start in range(0, len(cand), 32):
            scores.extend(score_batch([(query["text"], docs[did])
                                       for did in cand[start:start + 32]]))
        output[qid] = [
            did for _, did in sorted(zip(scores, cand), key=lambda pair: (-pair[0], pair[1]))
        ]
        print(f"{index}/{len(selected)} {qid}", flush=True)

    output_name = "pilot_rerank.trec" if pilot else "rerank.trec"
    with open(os.path.join(output_root, output_name), "w") as handle:
        for qid, ranking in output.items():
            for rank, did in enumerate(ranking, start=1):
                handle.write(f"{qid} Q0 {did} {rank} {1000-rank} v5-rerank\n")

    input_hashes = {
        name: sha(find_input(name, args.input_root))
        for name in ("queries.jsonl", "corpus.jsonl", "rerank_candidates.jsonl")
    }
    runtime_versions_path = os.path.join(output_root, "runtime_versions.json")
    with open(runtime_versions_path, "w") as handle:
        json.dump(runtime, handle, indent=2, sort_keys=True)

    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-lockbox-rerank-pilot" if pilot else "v5-lockbox-rerank",
        "pilot_only": pilot,
        "pilot_query_ids": list(output) if pilot else None,
        "model": MODEL_ID,
        "revision": REVISION,
        "runtime_lock_sha256": expected_lock_sha,
        "protocol_hash": PROTOCOL_HASH,
        "backend": backend,
        "account_identity": account_identity or "rezabarati2",
        "smoke": smoke_manifest,
        "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        "candidate_depth": TOPN,
        "n_queries": len(output),
        "n_pairs": len(output) * TOPN,
        "qrels_read": False,
        "input_sha256": input_hashes,
        "n_fail": 0,
        "seed": None,
        "seed_applicability": "none; deterministic model.eval inference",
        "runtime": runtime,
        "execution_sha": os.environ.get("SEM2ACT_HANDOFF_SHA"),
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "output_sha256": {
            output_name: sha(os.path.join(output_root, output_name)),
            "runtime_versions.json": sha(runtime_versions_path),
        },
    }
    with open(os.path.join(output_root, "run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
