"""v5 qrel-free reranker.

Reads only queries, corpus, and BM25 candidate IDs. No qrels are attached or
read. The scoring recipe is the frozen Qwen3-Reranker-0.6B yes/no-logit method.
"""

import hashlib
import json
import os
import sys

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = (
    "Given an operational supply-chain information need, retrieve documents "
    "relevant to the described entity, event, and time context."
)
TOPN = 50
MAXLEN = 1024


def runtime_spec():
    """Read the prospective backend lock when running under Lightning."""
    path = os.environ.get("SEM2ACT_RUNTIME_LOCK")
    if not path:
        raise SystemExit("Lightning reranker requires SEM2ACT_RUNTIME_LOCK")
    lock = json.loads(open(path).read())
    spec = lock["models"]["reranker"]
    if spec["model_id"] != MODEL_ID or spec["revision"] != REVISION:
        raise SystemExit("reranker runtime model pin drift")
    if spec["tokenizer_revision"] != REVISION:
        raise SystemExit("reranker tokenizer revision drift")
    if spec["dtype"] != "float16" or spec["batch_size"] != 32:
        raise SystemExit("reranker runtime dtype or batch drift")
    return spec


def working_root():
    path = os.environ.get("SEM2ACT_OUTPUT_ROOT", "/kaggle/working")
    os.makedirs(path, exist_ok=True)
    return path


def find_input(name):
    roots = []
    if os.environ.get("SEM2ACT_INPUT_ROOT"):
        roots.append(os.environ["SEM2ACT_INPUT_ROOT"])
    roots.extend(("/kaggle/input", ".", "inputs", "/kaggle/working/inputs"))
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(name)


def strip_transport(text):
    return text.strip()


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    locked = runtime_spec()
    max_length = locked["max_context_tokens"]
    batch_size = locked["batch_size"]
    input_root = os.environ.get("SEM2ACT_INPUT_ROOT", "/kaggle/input")
    for dirpath, _, names in os.walk(input_root):
        if any("qrel" in name.lower() for name in names):
            raise SystemExit("qrel-bearing input detected")
    if not __import__("torch").cuda.is_available():
        raise SystemExit("v5 reranker requires CUDA")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        MODEL_ID, revision=REVISION, padding_side="left", use_fast=True,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=locked["device_map"],
        attn_implementation=locked["attention_backend"]
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
            max_length=max_length - len(prefix) - len(suffix),
        )
        ids = [prefix + item + suffix for item in enc["input_ids"]]
        batch = tokenizer.pad(
            {"input_ids": ids}, padding=True, return_tensors="pt"
        ).to(model.device)
        logits = model(**batch).logits[:, -1, :]
        return torch.stack([logits[:, no], logits[:, yes]], dim=1).log_softmax(
            dim=1
        )[:, 1].exp().tolist()

    output = {}
    for index, query in enumerate(queries, start=1):
        qid = query["_id"]
        cand = candidate_map[qid]
        scores = []
        for start in range(0, len(cand), batch_size):
            scores.extend(score_batch([(query["text"], docs[did])
                                       for did in cand[start:start + 32]]))
        output[qid] = [
            did for _, did in sorted(zip(scores, cand), key=lambda pair: (-pair[0], pair[1]))
        ]
        print(f"{index}/{len(queries)} {qid}", flush=True)

    out_root = working_root()
    with open(os.path.join(out_root, "rerank.trec"), "w") as handle:
        for qid, ranking in output.items():
            for rank, did in enumerate(ranking, start=1):
                handle.write(f"{qid} Q0 {did} {rank} {1000-rank} v5-rerank\n")

    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-lockbox-rerank",
        "model": MODEL_ID,
        "revision": REVISION,
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
    with open(os.path.join(out_root, "run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(out_root, "runtime_versions.json"), "w") as handle:
        json.dump({
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        }, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
