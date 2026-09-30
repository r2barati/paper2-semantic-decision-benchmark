"""v5 qrel-free reranker.

Reads only queries, corpus, and BM25 candidate IDs. No qrels are attached or
read. The scoring recipe is the frozen Qwen3-Reranker-0.6B yes/no-logit method.
"""

import hashlib
import json
import os
import subprocess
import sys

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = (
    "Given an operational supply-chain information need, retrieve documents "
    "relevant to the described entity, event, and time context."
)
TOPN = 50
MAXLEN = 1024
PINNED_TRANSFORMERS = "4.57.6"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_PYTHON = "3.12"
EXPECTED_RUNTIME_LOCK_SHA256 = "ce88bc5c2d61d30ecbf649fb09c65b2502e599c53a42732da1acec06b2808b10"
EXPECTED_SMOKE_FIXTURE_SHA256 = "9ab69722e0062fc38eb27721f24e9cd83ca7944655973cbbe85f079372ccd74b"


def resolve_model_token() -> str:
    """Use the Kaggle secret when it is readable; ungated pins may load anonymously.

    Mirrors the CPU canary policy and the accepted pilot. A transport-level
    failure from the secrets backend must not hard-fail an ungated pin, but
    must still hard-fail a gated one so a gated model is never silently
    fetched without credentials.
    """
    gated = MODEL_ID.startswith("meta-llama/")
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        if gated:
            raise SystemExit(f"HF_TOKEN Kaggle secret unavailable: {type(exc).__name__}") from exc
        os.environ.pop("HF_TOKEN", None)
        return "public_unauthenticated"
    if not token:
        if gated:
            raise SystemExit("HF_TOKEN Kaggle secret is empty")
        os.environ.pop("HF_TOKEN", None)
        return "public_unauthenticated"
    os.environ["HF_TOKEN"] = token
    return "kaggle_secret"


def install_runtime_pins():
    import torch
    before = (torch.__version__, str(torch.version.cuda or ""))
    if before != (EXPECTED_TORCH, EXPECTED_CUDA):
        raise SystemExit(f"Kaggle image Torch/CUDA mismatch: {before}")
    if ".".join(str(part) for part in sys.version_info[:2]) != EXPECTED_PYTHON:
        raise SystemExit(f"Kaggle image Python drift: {sys.version_info[:2]}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
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


def find_input(name):
    for root in ("/kaggle/input", ".", "inputs", "/kaggle/working/inputs"):
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(name)


def strip_transport(text):
    return text.strip()


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if any("qrel" in name.lower() for name in os.listdir("/kaggle/input")):
        raise SystemExit("qrel-bearing input detected")
    if not __import__("torch").cuda.is_available():
        raise SystemExit("v5 reranker requires CUDA")

    credential_source = resolve_model_token()
    install_runtime_pins()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/tmp/hf-cache"
    os.environ["HF_HUB_CACHE"] = "/tmp/hf-cache/hub"

    docs = {
        x["_id"]: x["text"]
        for x in map(json.loads, open(find_input("corpus.jsonl")))
    }
    queries = [json.loads(x) for x in open(find_input("queries.jsonl"))]
    candidates = [json.loads(x) for x in open(find_input("rerank_candidates.jsonl"))]
    if len(queries) != 240 or len(candidates) != 240:
        raise SystemExit("expected exactly 240 queries and candidate rows")
    candidate_map = {x["query_id"]: x["doc_ids"] for x in candidates}
    if any(len(candidate_map[q["_id"]]) != TOPN for q in queries):
        raise SystemExit("candidate depth drift")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, padding_side="left", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.float16, attn_implementation="eager",
        device_map={"": "cuda:0"},
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

    smoke_path = find_input("runtime_smoke_v1.json")
    if sha(smoke_path) != EXPECTED_SMOKE_FIXTURE_SHA256:
        raise SystemExit("reranker smoke fixture hash drift")
    smoke = json.loads(open(smoke_path).read())
    smoke_scores = score_batch([(smoke["query_text"], smoke["documents"][0]["text"])])
    if len(smoke_scores) != 1 or not __import__("math").isfinite(smoke_scores[0]):
        raise SystemExit("reranker smoke schema/execution failure")
    smoke_manifest = {
        "status": "pass", "fixture_sha256": EXPECTED_SMOKE_FIXTURE_SHA256,
        "schema": "one finite reranker score", "n_calls": 1,
    }

    output = {}
    for index, query in enumerate(queries, start=1):
        qid = query["_id"]
        cand = candidate_map[qid]
        scores = []
        for start in range(0, len(cand), 32):
            scores.extend(score_batch([(query["text"], docs[did])
                                       for did in cand[start:start + 32]]))
        output[qid] = [
            did for _, did in sorted(zip(scores, cand), key=lambda pair: (-pair[0], pair[1]))
        ]
        print(f"{index}/{len(queries)} {qid}", flush=True)

    with open("/kaggle/working/rerank.trec", "w") as handle:
        for qid, ranking in output.items():
            for rank, did in enumerate(ranking, start=1):
                handle.write(f"{qid} Q0 {did} {rank} {1000-rank} v5-rerank\n")

    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-lockbox-rerank",
        "model": MODEL_ID,
        "revision": REVISION,
        "runtime_lock_sha256": EXPECTED_RUNTIME_LOCK_SHA256,
        "credential_source": credential_source,
        "attention_backend": "eager",
        "smoke": smoke_manifest,
        "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        "candidate_depth": TOPN,
        "n_queries": len(output),
        "qrels_read": False,
        "input_sha256": {
            name: sha(find_input(name))
            for name in ("queries.jsonl", "corpus.jsonl", "rerank_candidates.jsonl")
        },
        "n_fail": 0,
    }
    with open("/kaggle/working/run_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    with open("/kaggle/working/runtime_versions.json", "w") as handle:
        json.dump({
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "pinned_transformers": PINNED_TRANSFORMERS,
        }, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
