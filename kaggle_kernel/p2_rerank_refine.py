"""Paper2 V3 refinement: rerank depths {30,50,100} on hybrid_k120 candidates (dev only).

Confirms the depth selection on the FROZEN k_rrf=120 base (the Phase-2B sweep ran
depths on a k60 base; k120 won the RRF grid by 0.1831 vs 0.1808). Vectors travel
with the dataset (no re-encode). Test qrels are not attached (hard abort if seen).
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


if _need("transformers", (4, 51)):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "transformers==4.57.6"], check=True)

import numpy as np
import torch

RR = "Qwen/Qwen3-Reranker-0.6B"
RR_REV = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
DEPTHS = [30, 50, 100]
RRF_K = 120


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
    for root in ("/kaggle/input",):
        for dirpath, _, files in os.walk(root):
            assert "test.tsv" not in files, "TEST QRELS ATTACHED -- aborting"
    dev_rel = {}
    for line in open(find_input("dev.tsv")).read().splitlines()[1:]:
        qid, did, g = line.split("\t")
        dev_rel.setdefault(qid, {})[did] = int(g)
    dev_ids = sorted(dev_rel)
    docs = {}
    with open(find_input("corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            docs[d["_id"]] = d["text"]
    queries = [json.loads(l) for l in open(find_input("queries.jsonl"))]
    z = np.load(find_input("qwen3emb_full_vectors.npz"))
    dids = [str(x) for x in z["doc_ids"]]
    qidx = {str(x): i for i, x in enumerate(z["qids"])}
    S = z["q_emb"] @ z["doc_emb"].T

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
    with open(find_input("qwen_instruction.txt")) as f:
        instr = f.read().strip()

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

    qtext = {q["_id"]: q["text"] for q in queries}
    base = {}
    for qid in dev_ids:
        dense = sorted(dids, key=lambda d: (-S[qidx[qid], dids.index(d)], d))
        base[qid] = rrf([bm25[qid], dense], RRF_K)
    selection = {}
    for depth in DEPTHS:
        out = {}
        for qi, qid in enumerate(dev_ids):
            cand = base[qid][:depth]
            scores = []
            for i in range(0, len(cand), 32):
                scores.extend(score_batch([(qtext[qid], docs[c]) for c in cand[i:i + 32]]))
            top = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
            out[qid] = top + [d for d in base[qid] if d not in top]
            print(f"depth {depth}: {qi + 1}/{len(dev_ids)}", flush=True)
        mean = sum(ndcg(dev_rel[qid], out[qid]) for qid in dev_ids) / len(dev_ids)
        selection[f"rerank_k120_top{depth}"] = round(mean, 4)
        with open(f"/kaggle/working/rerank_k120_top{depth}_dev.trec", "w") as f:
            for q in dev_ids:
                for r, did in enumerate(out[q], start=1):
                    f.write(f"{q} Q0 {did} {r} {100000 - r} rerank\n")
    with open("/kaggle/working/dev_selection_refine.json", "w") as f:
        json.dump(selection, f, indent=2)
    with open("/kaggle/working/run_manifest.json", "w") as f:
        json.dump({"rr": RR, "rr_rev": RR_REV, "rrf_k": RRF_K, "depths": DEPTHS,
                   "instruction": instr, "dev_selection": selection}, f, indent=2)
    print("SELECTION:", selection, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
