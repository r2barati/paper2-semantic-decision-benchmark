"""v5 qrel-free reranker pilot on Kaggle GPU: 20 queries, 1,000 candidate scores.

This is the smallest authorized result step for the frozen V5 retrieval
protocol.  It is the identical scoring recipe, model revision, dtype, context
length, batch size, and candidate order as the full 240-query kernel
``p2_v5_rerank.py``; the only difference is the query range and the added
provenance gates required before any lockbox pair is scored.

No qrels are attached, read, or requested.  Nothing in this kernel writes
outside /kaggle/working.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
AMENDMENT_ID = "sem2act-v5-kaggle-reranker-pilot-v1"
LOCK_ID = "sem2act-v5-kaggle-reranker-pilot-v1"
EXPECTED_PILOT_LOCK_SHA256 = "6c144ac7d1e36ba2d68914466b054aab1dd9e73e52127411d0a652680a3db34c"
PARENT_LOCK_SHA256 = "ce88bc5c2d61d30ecbf649fb09c65b2502e599c53a42732da1acec06b2808b10"

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
DTYPE = "float16"
MAXLEN = 1024
BATCH_SIZE = 32
TOPN = 50
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_PYTHON = "3.12"

INSTRUCTION = (
    "Given an operational supply-chain information need, retrieve documents "
    "relevant to the described entity, event, and time context."
)
INSTRUCTION_SHA256 = "27878869801ce25ce6722edf80f57169fc72ee528863f82e9a4a942e755f0ea0"

SMOKE_FIXTURE_SHA256 = "9ab69722e0062fc38eb27721f24e9cd83ca7944655973cbbe85f079372ccd74b"
# The frozen ``sem2act-v5-rerank-inputs`` dataset v1 carries exactly three
# result-bearing files and no smoke fixture, so the non-result-bearing smoke
# gate is embedded here and pinned by hash instead of being read from disk.
SMOKE_FIXTURE_B64 = (
    "ewogICJzY2hlbWFfdmVyc2lvbiI6IDEsCiAgImZpeHR1cmVfaWQiOiAic2VtMmFjdC12NS1ydW50"
    "aW1lLXNtb2tlLXYxIiwKICAicHVycG9zZSI6ICJub24tcmVzdWx0LWJlYXJpbmcgbW9kZWwgbG9h"
    "ZCwgdG9rZW5pemF0aW9uLCBzY2hlbWEsIGFuZCBDVURBIHNtb2tlIiwKICAicXVlcnlfaWQiOiAi"
    "cnVudGltZS1zbW9rZS1xMDAwMCIsCiAgInF1ZXJ5X3RleHQiOiAiQXNzZXNzIHRoZSBjdXJyZW50"
    "IG9wZXJhdGlvbmFsIHJpc2sgZm9yIHRoZSBOb3J0aHdpbmQgZGlzdHJpYnV0aW9uIG5vZGUgZHVy"
    "aW5nIHRoaXMgcGxhbm5pbmcgd2luZG93LiIsCiAgImVudGl0eV9ub2RlIjogIk5vcnRod2luZCBk"
    "aXN0cmlidXRpb24gbm9kZSIsCiAgImRvY3VtZW50cyI6IFsKICAgIHsKICAgICAgImRvY19pZCI6"
    "ICJydW50aW1lLXNtb2tlLWQwMDAxIiwKICAgICAgInRleHQiOiAiTm9ydGh3aW5kIHN1cHBsaWVy"
    "IGJ1bGxldGluOiBpbmJvdW5kIHJlcGxlbmlzaG1lbnQgaXMgZGVsYXllZCBieSBmb3VyIHBlcmlv"
    "ZHMgYmVjYXVzZSBvZiBhIGN1cnJlbnQgcG9ydCBkaXNydXB0aW9uLiIKICAgIH0sCiAgICB7CiAg"
    "ICAgICJkb2NfaWQiOiAicnVudGltZS1zbW9rZS1kMDAwMiIsCiAgICAgICJ0ZXh0IjogIk5vcnRo"
    "d2luZCBvcGVyYXRpb25zIG5vdGU6IGN1c3RvbWVyIG9yZGVycyByZW1haW4gd2l0aGluIHRoZSBu"
    "b3JtYWwgc2Vhc29uYWwgcmFuZ2UgZm9yIHRoaXMgcGxhbm5pbmcgd2luZG93LiIKICAgIH0sCiAg"
    "ICB7CiAgICAgICJkb2NfaWQiOiAicnVudGltZS1zbW9rZS1kMDAwMyIsCiAgICAgICJ0ZXh0Ijog"
    "Ik5vcnRod2luZCBtYXJrZXQgYnJpZWY6IGEgY3VycmVudCBwcm9tb3Rpb24gaXMgYXNzb2NpYXRl"
    "ZCB3aXRoIGEgcG9zc2libGUgZGVtYW5kIGluY3JlYXNlIGF0IG5lYXJieSBvdXRsZXRzLiIKICAg"
    "IH0KICBdLAogICJleHBlY3RlZCI6IHsKICAgICJyZXJhbmtlciI6ICJmaW5pdGVfeWVzX25vX3Nj"
    "b3JlIiwKICAgICJDMSI6IFsibm9ybWFsIiwgInN1cHBsaWVyX2RlbGF5IiwgImRlbWFuZF9zdXJn"
    "ZSIsICJlc3RpbWF0ZWRfbHRfaW5jcmVhc2UiLCAiZXN0aW1hdGVkX2R1cmF0aW9uIiwgImVzdGlt"
    "YXRlZF9kZW1hbmRfbXVsdGlwbGllciJdLAogICAgIkMzIjogWyJlbnRpdHlfbWF0Y2giLCAiZXZl"
    "bnQiLCAiZnJlc2giLCAic3RhbmNlIiwgImNvbmZpZGVuY2UiXQogIH0KfQo="
)
MODEL_SNAPSHOT_FILE_COUNT = 12
MODEL_SNAPSHOT_FILES_SHA256 = "5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353"
MODEL_SNAPSHOT_FILES = (
    ("1_LogitScore/config.json", "73e3156450564d8a98b7e47bcf5aace0f29600828b51937da545571e84db3ff3", 57),
    ("chat_template.jinja", "6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d", 741),
    ("config.json", "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0", 727),
    ("config_sentence_transformers.json", "6a153d6696f78fd588c1c728967f0b773ea869d3c6028f151ce71ebe49140762", 325),
    ("generation_config.json", "81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db", 214),
    ("merges.txt", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", 1671853),
    ("model.safetensors", "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b", 1191588280),
    ("modules.json", "6f13b6b4a89e577b591b2077bca40c67c26541a6740a8809267cb474f90806a9", 280),
    ("sentence_bert_config.json", "3234ebd224d492cbe8d55d5ec80a3f408451c4db3005bafb64fe1c51c763e01e", 362),
    ("tokenizer.json", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", 11422654),
    ("tokenizer_config.json", "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0", 9706),
    ("vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", 2776833),
)

INPUT_SHA256 = {
    "corpus.jsonl": "6cab24bc7420635674856f35345cd4c5c30a23d2d0ee243a79b69d36bc8ab0e1",
    "queries.jsonl": "c457d93ea195d82a738219ad152382380d89b4b5e0c3fabde5cf4d6d33434986",
    "rerank_candidates.jsonl": "11dcac88ad210447ac16f78decbb5dd09776648e14b083aa3d89f67c8b701803",
}

PILOT_QUERY_IDS = tuple(f"v5-lb-q{index:04d}" for index in range(1, 21))
PILOT_QUERIES = 20
PILOT_EXPECTED_SCORES = 1000
WORKING = Path("/kaggle/working")
MODEL_DIR = WORKING / "model"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hf_secret() -> None:
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise SystemExit(f"HF_TOKEN Kaggle secret is unavailable: {type(exc).__name__}") from exc
    if not token:
        raise SystemExit("HF_TOKEN Kaggle secret is empty")
    os.environ["HF_TOKEN"] = token


def install_runtime_pins() -> None:
    import torch

    before = (torch.__version__, str(torch.version.cuda or ""))
    if before != (EXPECTED_TORCH, EXPECTED_CUDA):
        raise SystemExit(f"Kaggle image Torch/CUDA mismatch: {before}")
    if ".".join(str(sys.version_info)[:2]) != EXPECTED_PYTHON:
        raise SystemExit(f"Kaggle image Python drift: {sys.version_info[:2]}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
         f"transformers=={EXPECTED_TRANSFORMERS}"],
        check=True,
    )
    probe = subprocess.run(
        [sys.executable, "-c",
         "import json,torch;print(json.dumps([torch.__version__, str(torch.version.cuda or '')]))"],
        capture_output=True, text=True, check=True,
    )
    after = tuple(json.loads(probe.stdout.strip().splitlines()[-1]))
    if after != before:
        raise SystemExit(f"package installation changed Kaggle Torch/CUDA: {before} -> {after}")
    import transformers

    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise SystemExit(f"transformers pin drift: {transformers.__version__}")


def require_single_t4() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        listing = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        raise SystemExit(f"cannot verify allocated Kaggle GPUs: {type(exc).__name__}: {exc}") from exc
    physical = [line for line in listing.splitlines() if line.startswith("GPU ")]
    if len(physical) != 2 or any("t4" not in line.lower() for line in physical):
        raise SystemExit(f"expected exactly two allocated T4 GPUs, got: {listing}")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    if "t4" not in name.lower():
        raise SystemExit(f"expected Nvidia T4, got {name}")
    return {"device_name": name, "physical_device_count": len(physical), "nvidia_smi": listing}


def assert_no_qrels() -> list[str]:
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        raise SystemExit("no Kaggle input dataset is attached")
    names = sorted(entry.name for entry in input_root.iterdir())
    offending = [name for name in names if "qrel" in name.lower() or "judgment" in name.lower()]
    if offending:
        raise SystemExit(f"qrel-bearing input detected: {offending}")
    for path in input_root.rglob("*"):
        if path.is_file() and ("qrel" in path.name.lower() or "judgment" in path.name.lower()):
            raise SystemExit(f"qrel-bearing file detected: {path.name}")
    return names


def find_input(name: str) -> Path:
    for root in (Path("/kaggle/input"), WORKING, Path("."), Path("/kaggle/working/inputs")):
        if not root.exists():
            continue
        for dirpath, _, files in os.walk(root):
            if name in files:
                return Path(dirpath) / name
    raise SystemExit(f"required input file is not attached: {name}")


def load_smoke_fixture() -> dict:
    raw = base64.b64decode(SMOKE_FIXTURE_B64)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SMOKE_FIXTURE_SHA256:
        raise SystemExit(f"embedded smoke fixture hash drift: {digest}")
    return json.loads(raw.decode())


def verify_inputs() -> dict:
    resolved = {}
    for name, expected in INPUT_SHA256.items():
        path = find_input(name)
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"input hash drift for {name}: {actual} != {expected}")
        resolved[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    smoke = load_smoke_fixture()
    resolved["runtime_smoke_v1.json"] = {"source": "embedded", "sha256": SMOKE_FIXTURE_SHA256}
    return resolved, smoke


def stage_exact_model() -> dict:
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        raise SystemExit(f"refusing to stage into a non-empty model directory: {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODEL_DIR / ".cache")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(MODEL_DIR),
        allow_patterns=[name for name, _, _ in MODEL_SNAPSHOT_FILES],
    )
    actual = []
    for item in sorted(MODEL_DIR.rglob("*")):
        if not item.is_file() or ".cache" in item.parts:
            continue
        actual.append({"path": item.relative_to(MODEL_DIR).as_posix(), "sha256": sha256_file(item),
                       "bytes": item.stat().st_size})
    expected = [
        {"path": name, "sha256": digest, "bytes": size}
        for name, digest, size in MODEL_SNAPSHOT_FILES
    ]
    if actual != expected:
        for entry in actual:
            print(f"staged: {entry}", flush=True)
        raise SystemExit(
            "staged model files differ from the frozen 12-file accepted snapshot"
        )
    aggregate = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if aggregate != MODEL_SNAPSHOT_FILES_SHA256:
        raise SystemExit(f"model snapshot aggregate drift: {aggregate}")
    if len(actual) != MODEL_SNAPSHOT_FILE_COUNT:
        raise SystemExit("model snapshot file count drift")
    return {"file_count": len(actual), "files_sha256": aggregate, "files": actual}


def load_pilot_queries() -> tuple[dict, dict]:
    docs = {}
    with open(find_input("corpus.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            docs[row["_id"]] = row["text"]
    queries = {}
    with open(find_input("queries.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[row["_id"]] = row["text"]
    candidates = {}
    with open(find_input("rerank_candidates.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates[row["query_id"]] = row["doc_ids"]
    if len(queries) != 240 or len(candidates) != 240:
        raise SystemExit(f"input keyspace drift: {len(queries)} queries / {len(candidates)} candidate rows")
    if any(len(candidates[qid]) != TOPN for qid in queries):
        raise SystemExit("candidate depth drift")
    missing = [qid for qid in PILOT_QUERY_IDS if qid not in queries or qid not in candidates]
    if missing:
        raise SystemExit(f"pilot queries absent from the frozen keyspace: {missing}")
    return {qid: queries[qid] for qid in PILOT_QUERY_IDS}, {
        qid: candidates[qid] for qid in PILOT_QUERY_IDS
    }


def main() -> int:
    if hashlib.sha256(INSTRUCTION.encode()).hexdigest() != INSTRUCTION_SHA256:
        raise SystemExit("frozen reranker instruction drift")
    input_names = assert_no_qrels()
    device = require_single_t4()
    require_hf_secret()
    install_runtime_pins()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    inputs, smoke = verify_inputs()
    snapshot = stage_exact_model()

    queries, candidates = load_pilot_queries()
    candidate_set = {doc_id for ids in candidates.values() for doc_id in ids}
    docs = {}
    with open(find_input("corpus.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["_id"] in candidate_set:
                    docs[row["_id"]] = row["text"]
    missing_docs = sorted(candidate_set - set(docs))
    if missing_docs:
        raise SystemExit(
            f"corpus is missing {len(missing_docs)} frozen candidates, first: {missing_docs[:3]}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), padding_side="left", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float16,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
        trust_remote_code=True,
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
        batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(model.device)
        logits = model(**batch).logits[:, -1, :]
        return torch.stack([logits[:, no], logits[:, yes]], dim=1).log_softmax(dim=1)[:, 1].exp().tolist()

    smoke_score = score_batch([(smoke["query_text"], smoke["documents"][0]["text"])])
    if len(smoke_score) != 1 or not math.isfinite(smoke_score[0]):
        raise SystemExit("non-lockbox reranker smoke produced a non-finite score")
    smoke_manifest = {
        "status": "pass",
        "fixture_sha256": SMOKE_FIXTURE_SHA256,
        "fixture_source": "embedded",
        "schema": "one finite reranker score",
        "n_calls": 1,
    }

    rows = []
    scores_written = 0
    for position, qid in enumerate(PILOT_QUERY_IDS, start=1):
        candidate_ids = candidates[qid]
        scores = []
        for start in range(0, len(candidate_ids), BATCH_SIZE):
            window = candidate_ids[start:start + BATCH_SIZE]
            scores.extend(score_batch([(queries[qid], docs[did]) for did in window]))
        if len(scores) != TOPN or any(not math.isfinite(value) for value in scores):
            raise SystemExit(f"pilot scoring failure for {qid}")
        scores_written += len(scores)
        ranking = [did for _, did in sorted(zip(scores, candidate_ids), key=lambda pair: (-pair[0], pair[1]))]
        if len(set(ranking)) != TOPN or set(ranking) != set(candidate_ids):
            raise SystemExit(f"pilot ranking is not a permutation of the frozen candidates for {qid}")
        for rank, did in enumerate(ranking, start=1):
            rows.append(f"{qid} Q0 {did} {rank} {1000 - rank} v5-rerank-pilot")
        print(f"{position}/{PILOT_QUERIES} {qid}", flush=True)

    if scores_written != PILOT_EXPECTED_SCORES or len(rows) != PILOT_EXPECTED_SCORES:
        raise SystemExit(f"pilot coverage failure: {scores_written} scores / {len(rows)} rows")
    if sorted({row.split()[0] for row in rows}) != sorted(PILOT_QUERY_IDS):
        raise SystemExit("pilot query coverage failure")

    (WORKING / "rerank.trec").write_text("\n".join(rows) + "\n")
    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-reranker-pilot",
        "stage": "reranker",
        "shard_index": 0,
        "status": "pass",
        "protocol_hash": PROTOCOL_HASH,
        "amendment_id": AMENDMENT_ID,
        "lock_id": LOCK_ID,
        "runtime_lock_sha256": EXPECTED_PILOT_LOCK_SHA256,
        "parent_runtime_lock_sha256": PARENT_LOCK_SHA256,
        "model": MODEL_ID,
        "revision": REVISION,
        "model_snapshot": snapshot,
        "dtype": DTYPE,
        "attention_backend": "eager",
        "device": device["device_name"],
        "physical_device_count": device["physical_device_count"],
        "nvidia_smi": device["nvidia_smi"],
        "instruction_sha256": INSTRUCTION_SHA256,
        "candidate_depth": TOPN,
        "batch_size": BATCH_SIZE,
        "max_context_tokens": MAXLEN,
        "query_ids": list(PILOT_QUERY_IDS),
        "n_queries": len(PILOT_QUERY_IDS),
        "n_scores": scores_written,
        "n_ranking_rows": len(rows),
        "n_fail": 0,
        "qrels_read": False,
        "attached_input_datasets": input_names,
        "input_sha256": {name: entry["sha256"] for name, entry in inputs.items()},
        "smoke": smoke_manifest,
        "rerank_trec_sha256": hashlib.sha256((WORKING / "rerank.trec").read_bytes()).hexdigest(),
        "stop_point": "stop-after-pilot-000;return-provenance-to-codex",
    }
    (WORKING / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (WORKING / "runtime_versions.json").write_text(json.dumps({
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "pinned_transformers": EXPECTED_TRANSFORMERS,
        "cuda": str(torch.version.cuda or ""),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
