"""Paper2 V3 Phase-1C reranker (Kaggle GPU kernel, script type).

Reranks hybrid-instruct top-50 with Qwen3-Reranker-0.6B at the FROZEN pin,
using the DOCUMENTED causal-LM yes/no-logit recipe with the ONE predeclared
domain instruction. Reads frozen inputs from the attached dataset
(paper2-v3proto-inputs); writes trec + qlevel json + run manifest to
/kaggle/working (kernel outputs).

Nothing here is tunable: instruction text, model revision, top-50 depth,
recipe, and scoring are all frozen. See configs/v3/ in the source repo.
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
        print(f"{mod} {v} present, keeping (no torch churn)", flush=True)
        return False
    except ImportError:
        return True


# NEVER upgrade/downgrade torch or numpy: the image's GPU build must survive.
# Install missing pieces with --no-deps only.
if _need("transformers", (4, 51)):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "transformers==4.57.6"], check=True)
# NOTE: ir_measures is deliberately NOT installed here: its pytrec_eval backend
# is unavailable in this image and the fallback provider cannot score. The kernel
# reports stdlib graded metrics (function below, cross-checked against ir_measures
# 0.4.3 locally); cmd_fetch re-verifies with ir_measures before installing outputs.

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = ("Given an operational supply-chain information need, retrieve "
               "documents relevant to the described entity, event, and time context.")
MAXLEN = 1024
TOPN = 50


def find_input(name):
    for root in ("/kaggle/input", ".", "inputs", "/kaggle/working/inputs"):
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(f"{name} not found under /kaggle/input")


def _dcg(gains, k):
    # LINEAR gains, exactly matching trec_eval/ir_measures nDCG (verified to
    # 1e-9 against ir_measures 0.4.3 in tests/test_v3proto.py). Do NOT use
    # exponential (2^g-1) gains here: src/retrieval.py's hand formula does, and
    # the two differ (0.2649 vs 0.2565 on v3p-q01). V3's frozen metric is trec_eval's.
    import math
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def stdlib_metrics(qrels, ranking):
    """Graded nDCG@10 / R@10 / R@20 / RR / AP. Mirrors ir_measures definitions;
    cross-checked against ir_measures 0.4.3 in tests/test_v3proto.py."""
    import math
    rel = qrels
    n_rel = sum(1 for v in rel.values() if v > 0)
    gains = [rel.get(d, 0) for d in ranking]
    ideal = sorted(rel.values(), reverse=True)
    dcg = _dcg(gains, 10)
    idcg = _dcg(ideal, 10)
    rr = 0.0
    for i, d in enumerate(ranking, start=1):
        if rel.get(d, 0) > 0:
            rr = 1.0 / i
            break
    precisions = [sum(1 for d in ranking[:i + 1] if rel.get(d, 0) > 0) / (i + 1)
                  for i, d in enumerate(ranking) if rel.get(d, 0) > 0]
    ap = sum(precisions) / n_rel if n_rel else 0.0
    return {"nDCG@10": dcg / idcg if idcg else 0.0,
            "R@10": sum(1 for d in ranking[:10] if rel.get(d, 0) > 0) / n_rel if n_rel else 0.0,
            "R@20": sum(1 for d in ranking[:20] if rel.get(d, 0) > 0) / n_rel if n_rel else 0.0,
            "RR": rr, "AP": ap}


def main():
    print("cuda:", torch.cuda.is_available(), flush=True)
    qrels = {}
    for fn in ("dev.tsv", "test.tsv"):
        with open(find_input(fn)) as f:
            for line in f.read().splitlines()[1:]:
                qid, did, g = line.split("\t")
                qrels.setdefault(qid, {})[did] = int(g)
    docs, queries = {}, []
    with open(find_input("corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            docs[d["_id"]] = d["text"]
    with open(find_input("queries.jsonl")) as f:
        for line in f:
            queries.append(json.loads(line))
    hyb = {}
    with open(find_input("hybrid-real-instruct.trec")) as f:
        for line in f:
            qid, _, did, _, _, _ = line.split()
            hyb.setdefault(qid, []).append(did)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, padding_side="left", trust_remote_code=True)
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    dev_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    print(f"device: {dev_name} capability: {dev_cap}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.float16, device_map="auto").eval()
    # Card lottery: some assigned GPUs lack kernel images in the image torch
    # build (same failure ADIA hit). Warm up on GPU; fall back to CPU in-run.
    try:
        _ = model(torch.tensor([[1, 2, 3]], device=model.device))
        print("gpu warmup ok", flush=True)
    except Exception as e:
        print(f"gpu unusable ({type(e).__name__}), falling back to CPU", flush=True)
        model = model.to("cpu").float()
    t_true = tokenizer.convert_tokens_to_ids("yes")
    t_false = tokenizer.convert_tokens_to_ids("no")
    pre = tokenizer.encode(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
        "\"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
    suf = tokenizer.encode(
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False)

    @torch.no_grad()
    def score_batch(pairs):
        texts = [f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"
                 for q, d in pairs]
        enc = tokenizer(texts, padding=False, truncation="longest_first",
                        return_attention_mask=False,
                        max_length=MAXLEN - len(pre) - len(suf))
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
        for i in range(0, len(cand), 64):
            scores.extend(score_batch([(q["text"], docs[c]) for c in cand[i:i + 64]]))
        top = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
        out[q["_id"]] = top + [d for d in hyb[q["_id"]] if d not in top]
        print(f"{qi + 1}/{len(queries)} {q['_id']}", flush=True)

    with open("/kaggle/working/rerank-qwen3-instruct.trec", "w") as f:
        for qid, r in out.items():
            for i, did in enumerate(r, start=1):
                f.write(f"{qid} Q0 {did} {i} {1000 - i} rerank-qwen3-instruct\n")

    per_q, agg_acc = {}, {}
    for q in queries:
        m = stdlib_metrics(qrels[q["_id"]], out[q["_id"]])
        per_q[q["_id"]] = round(m["nDCG@10"], 4)
        for k, v in m.items():
            agg_acc.setdefault(k, []).append(v)
    agg = {k: round(sum(v) / len(v), 3) for k, v in agg_acc.items()}
    print("RERANK-QWEN3-INSTRUCT:", agg, flush=True)
    with open("/kaggle/working/rerank-qwen3-instruct_qlevel.json", "w") as f:
        json.dump(per_q, f, indent=1)
    with open("/kaggle/working/run_manifest.json", "w") as f:
        json.dump({"model": MODEL_ID, "revision": REVISION, "instruction": INSTRUCTION,
                   "topn": TOPN, "recipe": "causal-lm-yes-no-logit",
                   "transformers": "4.57.6", "aggregate": agg}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
