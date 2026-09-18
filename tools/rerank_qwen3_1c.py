"""Phase 1C: rerank hybrid-instruct top-50 with real Qwen3-Reranker-0.6B.

Uses the model's DOCUMENTED Transformers recipe (causal-LM yes/no logit
scoring), because ST 5.1.2's CrossEncoder can only load a sequence-class
head (random weights here -- the Phase-1B 0.174 number is RETRACTED, it scored
with an uninitialized head).

Instruction = the ONE predeclared domain text (configs/v3/qwen_instruction.txt).
Batched forward with left padding. Score = P(yes | yes/no).
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.experiment_v3proto import _load

MODEL_ID = 'Qwen/Qwen3-Reranker-0.6B'
REVISION = 'e61197ed45024b0ed8a2d74b80b4d909f1255473'
INSTR = (ROOT / 'configs/v3/qwen_instruction.txt').read_text().strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                          padding_side='left', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=REVISION, trust_remote_code=True,
    torch_dtype=torch.float32).eval()
token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")
MAXLEN = 1024  # docs ~150 toks + template ~100; nothing of value beyond this
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
          "\"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PREFIX_TOKS = tokenizer.encode(PREFIX, add_special_tokens=False)
SUFFIX_TOKS = tokenizer.encode(SUFFIX, add_special_tokens=False)


def fmt(query, doc):
    return f"<Instruct>: {INSTR}\n<Query>: {query}\n<Document>: {doc}"


@torch.no_grad()
def score_batch(pairs):
    texts = [fmt(q, d) for q, d in pairs]
    enc = tokenizer(texts, padding=False, truncation='longest_first',
                    return_attention_mask=False,
                    max_length=MAXLEN - len(PREFIX_TOKS) - len(SUFFIX_TOKS))
    ids = [PREFIX_TOKS + e + SUFFIX_TOKS for e in enc['input_ids']]
    batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    logits = model(**batch).logits[:, -1, :]
    t = logits[:, token_true_id]
    f = logits[:, token_false_id]
    return torch.stack([f, t], dim=1).log_softmax(dim=1)[:, 1].exp().tolist()


def main():
    docs, queries, qrels = _load()
    txt = {d.doc_id: d.text for d in docs}
    hyb = {}
    with open(ROOT / 'runs/v3proto/hybrid-real-instruct.trec') as f:
        for line in f:
            qid, _, did, _, _, _ = line.split()
            hyb.setdefault(qid, []).append(did)
    out = {}
    ckpt = ROOT / 'runs/v3proto/rerank-qwen3-instruct_partial.json'
    if ckpt.exists():
        out = json.loads(ckpt.read_text())
        print(f'resumed {len(out)} queries', flush=True)
    for qi, q in enumerate(queries):
        qid = q['_id']
        if qid in out:
            continue
        cand = hyb[qid][:50]
        scores = []
        for i in range(0, len(cand), 16):
            chunk = [(q["text"], txt[c]) for c in cand[i:i + 16]]
            scores.extend(score_batch(chunk))
        top = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
        rest = [d for d in hyb[qid] if d not in top]
        out[qid] = top + rest
        ckpt.write_text(json.dumps(out))
        print(f'{len(out)}/20 {qid}', flush=True)
    with open(ROOT / 'runs/v3proto/rerank-qwen3-instruct_rankings.json', 'w') as f:
        json.dump(out, f)
    with open(ROOT / 'runs/v3proto/rerank-qwen3-instruct.trec', 'w') as f:
        for qid, r in out.items():
            for i, did in enumerate(r, start=1):
                f.write(f'{qid} Q0 {did} {i} {1000-i} rerank-qwen3-instruct\n')
    import ir_measures
    from ir_measures import nDCG, Recall, RR, AP, Qrel, ScoredDoc
    qr = [Qrel(query_id=qid, doc_id=did, relevance=g)
          for q in queries for did, g in qrels[q['_id']].items()]
    run = [ScoredDoc(query_id=qid, doc_id=did, score=1000 - i)
           for qid, r in out.items() for i, did in enumerate(r)]
    res = ir_measures.calc_aggregate([nDCG@10, Recall@10, Recall@20, RR, AP], qr, run)
    print('RERANK-QWEN3-INSTRUCT:', {str(k): round(float(v), 3) for k, v in res.items()})
    ql = {}
    for row in ir_measures.iter_calc([nDCG@10], qr, run):
        ql[row.query_id] = round(float(row.value), 4)
    json.dump(ql, open(ROOT / 'runs/v3proto/rerank-qwen3-instruct_qlevel.json', 'w'), indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
