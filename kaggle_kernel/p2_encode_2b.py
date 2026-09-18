"""Paper2 V3 Phase-2B: encode full corpus + dev-only fusion/depth selection (GPU).

Frozen: embedding pin+revision, instruction text, recipe, candidate grids
(configs/v3/tuning_scope.yaml). DEV QRELS ONLY are attached -- test qrels never
leave the local machine, so test nDCG cannot be peeked at during tuning.

Outputs (/kaggle/working): qwen3emb_full_vectors.npz, dense_full.trec,
hybrid_k{30,60,120}_dev.trec, rerank_top{n}_dev.trec (n=30/50/100),
dev_selection.json (nDCG@10 per candidate), run_manifest.json.
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
RRF_KS = [30, 60, 120]
DEPTHS = [30, 50, 100]


def find_input(name):
    for root in ("/kaggle/input", ".", "inputs", "/kaggle/working/inputs"):
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(name)


def _dcg(gains, k):
    import math
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg(rel, ranking, k=10):
    gains = [rel.get(d, 0) for d in ranking]
    ideal = sorted(rel.values(), reverse=True)
    d = _dcg(gains, k)
    i = _dcg(ideal, k)
    return d / i if i else 0.0


def main():
    print("cuda:", torch.cuda.is_available(), flush=True)
    docs = [json.loads(l) for l in open(find_input("corpus.jsonl"))]
    queries = [json.loads(l) for l in open(find_input("queries.jsonl"))]
    instr = open(find_input("qwen_instruction.txt")).read().strip()
    dev_rel = {}
    for line in open(find_input("dev.tsv")).read().splitlines()[1:]:
        qid, did, g = line.split("\t")
        dev_rel.setdefault(qid, {})[did] = int(g)
    dev_ids = sorted(dev_rel)
    print(f"docs={len(docs)} queries={len(queries)} dev={len(dev_ids)}", flush=True)
    # hard guard: test qrels must not be attached
    for root in ("/kaggle/input",):
        for dirpath, _, files in os.walk(root):
            assert "test.tsv" not in files, "TEST QRELS ATTACHED -- aborting"

    from sentence_transformers import SentenceTransformer
    try:
        m = SentenceTransformer(EMB, revision=EMB_REV,
                                model_kwargs={"dtype": torch.float16})
        m.encode(["warmup"], show_progress_bar=False)
        print("gpu encode ok", flush=True)
    except Exception as e:
        print(f"gpu unusable ({type(e).__name__}), CPU fallback", flush=True)
        m = SentenceTransformer(EMB, revision=EMB_REV, device="cpu")
    E = m.encode([d["text"] for d in docs], batch_size=64,
                 show_progress_bar=False, normalize_embeddings=True)
    Q = m.encode([f"Instruct: {instr}\nQuery: {q['text']}" for q in queries],
                 batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    E = np.asarray(E, dtype="float32")
    Q = np.asarray(Q, dtype="float32")
    np.savez_compressed("/kaggle/working/qwen3emb_full_vectors.npz",
                        doc_ids=np.array([d["_id"] for d in docs]),
                        qids=np.array([q["_id"] for q in queries]),
                        doc_emb=E, q_emb=Q)
    S = Q @ E.T
    dids = [d["_id"] for d in docs]
    qidx = {q["_id"]: i for i, q in enumerate(queries)}
    dense_rank = {q["_id"]: sorted(dids, key=lambda d, i=qidx[q["_id"]]: (-S[i, dids.index(d)], d))
                  for q in queries}
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

    selection = {}
    for k in RRF_KS:
        hyb = {qid: rrf([bm25[qid], dense_rank[qid]], k) for qid in dev_ids}
        mean = sum(ndcg(dev_rel[qid], hyb[qid]) for qid in dev_ids) / len(dev_ids)
        selection[f"hybrid_rrf{k}"] = round(mean, 4)
        with open(f"/kaggle/working/hybrid_k{k}_dev.trec", "w") as f:
            for qid in dev_ids:
                for r, did in enumerate(hyb[qid], start=1):
                    f.write(f"{qid} Q0 {did} {r} {100000 - r} hybrid\n")
    print("RRF dev:", selection, flush=True)

    # rerank depths on dev (best-RRF hybrid as candidate source is itself a dev
    # decision: use RRF60 fixed as the candidate generator, vary only depth)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(RR, revision=RR_REV,
                                              padding_side="left", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            RR, revision=RR_REV, trust_remote_code=True,
            dtype=torch.float16, device_map="auto").eval()
        model(torch.tensor([[1, 2, 3]], device=model.device))
        print("gpu warmup ok", flush=True)
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
    instr_frozen = instr  # the ONE predeclared domain instruction

    @torch.no_grad()
    def score_batch(pairs):
        texts = [f"<Instruct>: {instr_frozen}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
        enc = tokenizer(texts, padding=False, truncation="longest_first",
                        return_attention_mask=False, max_length=1024 - len(pre) - len(suf))
        ids = [pre + e + suf for e in enc["input_ids"]]
        batch = tokenizer.pad({"input_ids": ids}, padding=True,
                              return_tensors="pt").to(model.device)
        logits = model(**batch).logits[:, -1, :]
        t = logits[:, t_true]
        fl = logits[:, t_false]
        return torch.stack([fl, t], dim=1).log_softmax(dim=1)[:, 1].exp().tolist()

    base60 = {}
    with open("/kaggle/working/hybrid_k60_dev.trec") as f:
        for line in f:
            qid, _, did, _, _, _ = line.split()
            base60.setdefault(qid, []).append(did)
    for depth in DEPTHS:
        out = {}
        for q in queries:
            if q["_id"] not in dev_ids:
                continue
            cand = base60[q["_id"]][:depth]
            scores = []
            for i in range(0, len(cand), 32):
                scores.extend(score_batch([(q["text"], txt[c]) for c in cand[i:i + 32]]))
            top = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
            out[q["_id"]] = top + [d for d in base60[q["_id"]] if d not in top]
        mean = sum(ndcg(dev_rel[qid], out[qid]) for qid in dev_ids) / len(dev_ids)
        selection[f"rerank_top{depth}"] = round(mean, 4)
        with open(f"/kaggle/working/rerank_top{depth}_dev.trec", "w") as f:
            for qid in dev_ids:
                for r, did in enumerate(out[qid], start=1):
                    f.write(f"{qid} Q0 {did} {r} {100000 - r} rerank\n")
        print(f"depth {depth}: {selection[f'rerank_top{depth}']}", flush=True)

    with open("/kaggle/working/dev_selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    with open("/kaggle/working/run_manifest.json", "w") as f:
        json.dump({"emb": EMB, "emb_rev": EMB_REV, "rr": RR, "rr_rev": RR_REV,
                   "rrf_grid": RRF_KS, "depth_grid": DEPTHS,
                   "dev_queries": len(dev_ids), "dev_selection": selection}, f, indent=2)
    print("SELECTION:", selection, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
