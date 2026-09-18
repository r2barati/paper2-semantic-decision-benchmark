"""Paper2 V3 main runs: full-corpus RRF-k120 + rerank-top30 for ALL 200 queries.

Frozen pipeline (configs/v3/RETRIEVAL_FREEZE_FINAL.md). Vectors recomputed here
from the pinned embedding (corpus changed since 2B). NO qrels attached at all --
this kernel ranks only; all metrics are computed locally at analysis time.
"""

import json
import os
import subprocess
import sys


def _need(mod, min_version=None):
    try:
        m = __import__("importlib").import_module(mod)
        v = getattr(m, "__version__", "0")
        if min_version and tuple(int(x) for x in v.split(".")[:2]) < min_version:
            return True
        print(f"{mod} {v} present, keeping", flush=True)
        return False
    except ImportError:
        return True


if _need("sentence_transformers"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "sentence-transformers==5.1.2"], check=True)
if _need("transformers", (4, 51)):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "transformers==4.57.6"], check=True)

import numpy as np
import torch

EMB = "Qwen/Qwen3-Embedding-0.6B"
EMB_REV = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RR = "Qwen/Qwen3-Reranker-0.6B"
RR_REV = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
RRF_K = 120
TOPN = 30


def find_input(name):
    for root in ("/kaggle/input", ".", "inputs", "/kaggle/working/inputs"):
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(name)


def main():
    print("cuda:", torch.cuda.is_available(), flush=True)
    for root in ("/kaggle/input",):
        for dirpath, _, files in os.walk(root):
            assert "dev.tsv" not in files and "test.tsv" not in files, \
                "QRELS ATTACHED -- main runs rank only; aborting"
    docs = [json.loads(l) for l in open(find_input("corpus.jsonl"))]
    queries = [json.loads(l) for l in open(find_input("queries.jsonl"))]
    instr = open(find_input("qwen_instruction.txt")).read().strip()
    print(f"docs={len(docs)} queries={len(queries)}", flush=True)

    from sentence_transformers import SentenceTransformer
    try:
        m = SentenceTransformer(EMB, revision=EMB_REV,
                                model_kwargs={"dtype": torch.float16})
        m.encode(["warmup"], show_progress_bar=False)
        print("gpu encode ok", flush=True)
    except Exception as e:
        print(f"gpu unusable ({type(e).__name__}), CPU fallback", flush=True)
        m = SentenceTransformer(EMB, revision=EMB_REV, device="cpu")
    E = np.asarray(m.encode([d["text"] for d in docs], batch_size=64,
                            show_progress_bar=False, normalize_embeddings=True),
                   dtype="float32")
    Q = np.asarray(m.encode([f"Instruct: {instr}\nQuery: {q['text']}" for q in queries],
                            batch_size=64, show_progress_bar=False,
                            normalize_embeddings=True), dtype="float32")
    np.savez_compressed("/kaggle/working/qwen3emb_main_vectors.npz",
                        doc_ids=np.array([d["_id"] for d in docs]),
                        qids=np.array([q["_id"] for q in queries]),
                        doc_emb=E, q_emb=Q)
    S = Q @ E.T
    dids = [d["_id"] for d in docs]
    dense_rank = {}
    for i, q in enumerate(queries):
        dense_rank[q["_id"]] = sorted(dids, key=lambda d: (-S[i, dids.index(d)], d))
    with open("/kaggle/working/dense_full.trec", "w") as f:
        for q in queries:
            for r, did in enumerate(dense_rank[q["_id"]], start=1):
                f.write(f"{q['_id']} Q0 {did} {r} {100000 - r} dense\n")

    bm25 = {}
    with open(find_input("bm25_full.trec")) as f:
        for line in f:
            qid, _, did, _, _, _ = line.split()
            bm25.setdefault(qid, []).append(did)

    def rrf(rankings, k):
        from collections import defaultdict
        sc = defaultdict(float)
        for ranking in rankings:
            for i, did in enumerate(ranking, start=1):
                sc[did] += 1.0 / (k + i)
        return sorted(sc, key=lambda d: (-sc[d], d))

    hyb = {q["_id"]: rrf([bm25[q["_id"]], dense_rank[q["_id"]]], RRF_K) for q in queries}
    with open("/kaggle/working/hybrid_k120_full.trec", "w") as f:
        for q in queries:
            for r, did in enumerate(hyb[q["_id"]], start=1):
                f.write(f"{q['_id']} Q0 {did} {r} {100000 - r} hybrid\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(RR, revision=RR_REV,
                                              padding_side="left", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            RR, revision=RR_REV, trust_remote_code=True,
            dtype=torch.float16, device_map="auto").eval()
        model(torch.tensor([[1, 2, 3]], device=model.device))
        print("gpu rerank ok", flush=True)
    except Exception as e:
        print(f"gpu unusable ({type(e).__name__}), CPU fallback", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            RR, revision=RR_REV, trust_remote_code=True,
            dtype=torch.float32).to("cpu").eval()
    t_true = tokenizer.convert_tokens_to_ids("yes")
    t_false = tokenizer.convert_tokens_to_ids("no")
    pre = tokenizer.encode(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
        "\"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
    suf = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                           add_special_tokens=False)
    txt = {d["_id"]: d["text"] for d in docs}

    @torch.no_grad()
    def score_batch(pairs):
        texts = [f"<Instruct>: {instr}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
        enc = tokenizer(texts, padding=False, truncation="longest_first",
                        return_attention_mask=False,
                        max_length=1024 - len(pre) - len(suf))
        ids = [pre + e + suf for e in enc["input_ids"]]
        batch = tokenizer.pad({"input_ids": ids}, padding=True,
                              return_tensors="pt").to(model.device)
        logits = model(**batch).logits[:, -1, :]
        t = logits[:, t_true]
        fl = logits[:, t_false]
        return torch.stack([fl, t], dim=1).log_softmax(dim=1)[:, 1].exp().tolist()

    out = {}
    for qi, q in enumerate(queries):
        cand = hyb[q["_id"]][:TOPN]
        scores = []
        for i in range(0, len(cand), 32):
            scores.extend(score_batch([(q["text"], txt[c]) for c in cand[i:i + 32]]))
        top = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
        out[q["_id"]] = top + [d for d in hyb[q["_id"]] if d not in top]
        if (qi + 1) % 20 == 0:
            print(f"{qi + 1}/{len(queries)}", flush=True)
    with open("/kaggle/working/rerank_full.trec", "w") as f:
        for qid, r in out.items():
            for i, did in enumerate(r, start=1):
                f.write(f"{qid} Q0 {did} {i} {100000 - i} rerank\n")
    with open("/kaggle/working/run_manifest.json", "w") as f:
        json.dump({"emb": EMB, "emb_rev": EMB_REV, "rr": RR, "rr_rev": RR_REV,
                   "rrf_k": RRF_K, "topn": TOPN, "n_queries": len(queries),
                   "n_docs": len(docs)}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
