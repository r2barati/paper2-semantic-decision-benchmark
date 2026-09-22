"""R0 static-RAG and A3 bounded-agent orchestration.

Budgets: R0 = exactly 1 retrieval (verbatim warning query) + 1 synthesis call.
A3 = hypotheses call + retrieval(query_1) + optional retrieval(query_2) + synth call.
A3 issues at most 2 retrieval calls, then stops. No simulator access here.
`llm` is an injected callable (vLLM client on Kaggle, deterministic stub in tests).
"""

from __future__ import annotations

import hashlib
import time
from typing import Callable

from . import prompts
from .arms import CallRecord, extract_belief

LlmFn = Callable[[str, str], tuple[str, dict]]  # (system, user) -> (raw, usage)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def run_R0(warning_id: str, warning: str, corpus, llm: LlmFn):
    t0 = time.time()
    docs = corpus.retrieve(warning, k=3)
    user = ("Warning:\n" + warning + "\n\nRetrieved evidence:\n"
            + "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in docs))
    raw, usage = llm(prompts.R0_SYNTH_SYS, user)
    from .arms import parse_json_strict

    belief = extract_belief(parse_json_strict(raw), "R0")
    rec = CallRecord(arm="R0", warning_id=warning_id,
                     prompt_sha=_sha(prompts.R0_SYNTH_SYS),
                     prompt_tokens=int(usage.get("prompt_tokens", 0)),
                     completion_tokens=int(usage.get("completion_tokens", 0)),
                     retrieval_calls=1, reasoning_stages=1,
                     latency_s=time.time() - t0)
    struct = {"query": warning, "evidence_ids": [d["doc_id"] for d in docs]}
    return belief, struct, docs, [rec]


def run_A3(warning_id: str, warning: str, corpus, llm: LlmFn):
    from .arms import parse_json_strict

    t0 = time.time()
    recs: list[CallRecord] = []
    # Call 1: hypotheses + first query.
    raw1, u1 = llm(prompts.A3_HYPO_SYS, "Warning:\n" + warning)
    h1 = parse_json_strict(raw1)
    recs.append(CallRecord(arm="A3", warning_id=warning_id,
                           prompt_sha=_sha(prompts.A3_HYPO_SYS),
                           prompt_tokens=int(u1.get("prompt_tokens", 0)),
                           completion_tokens=int(u1.get("completion_tokens", 0)),
                           retrieval_calls=0, reasoning_stages=1,
                           latency_s=time.time() - t0))
    q1 = h1.get("query_1") or warning
    docs1 = corpus.retrieve(q1, k=3)
    # Call 2: synthesize; model may supply one reformulated query_2, executed once.
    user2 = ("Warning:\n" + warning + "\n\nHypotheses:\n" + str(h1.get("hypotheses"))
             + "\n\nEvidence round 1 (query_1):\n"
             + "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in docs1))
    t1 = time.time()
    raw2, u2 = llm(prompts.A3_SYNTH_SYS, user2)
    s2 = parse_json_strict(raw2)
    docs2: list[dict] = []
    n_ret = 1
    if s2.get("query_2_used") and s2.get("query_2"):
        docs2 = corpus.retrieve(s2["query_2"], k=3)
        n_ret = 2
    belief = extract_belief(s2, "A3")
    recs.append(CallRecord(arm="A3", warning_id=warning_id,
                           prompt_sha=_sha(prompts.A3_SYNTH_SYS),
                           prompt_tokens=int(u2.get("prompt_tokens", 0)),
                           completion_tokens=int(u2.get("completion_tokens", 0)),
                           retrieval_calls=n_ret, reasoning_stages=2,
                           latency_s=time.time() - t1))
    assert n_ret <= 2, "A3 retrieval budget exceeded"
    struct = {"hypotheses": h1.get("hypotheses"), "query_1": q1,
              "query_2_used": bool(s2.get("query_2_used")),
              "query_2": s2.get("query_2", ""),
              "evidence_ids": [d["doc_id"] for d in docs1 + docs2]}
    return belief, struct, docs1 + docs2, recs
