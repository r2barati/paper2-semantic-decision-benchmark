"""Per-warning inference (Correction 4): one belief per (warning x arm), cached, replayed across seeds."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import prompts
from .agent import run_A3, run_R0
from .arms import PRIOR_BELIEF, CallRecord, extract_belief, parse_json_strict


def cache_key(arm: str, prompt_sha: str, warning: str) -> str:
    return hashlib.sha256(f"{arm}||{prompt_sha}||{warning}".encode()).hexdigest()[:16]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def run_arm_warning(arm: str, warning_id: str, warning: str, corpus, llm,
                    tolerance_note: str = "") -> tuple[dict, dict, list, list[CallRecord]]:
    """Returns (belief_dict, struct, docs, records). Zero-tolerance parsing at smoke;
    caller applies fallback policy for full run."""
    t0 = time.time()
    if arm == "R0":
        b, struct, docs, recs = run_R0(warning_id, warning, corpus, llm)
        return b.as_dict(), struct, docs, recs
    if arm == "A3":
        b, struct, docs, recs = run_A3(warning_id, warning, corpus, llm)
        return b.as_dict(), struct, docs, recs
    if arm == "A2":
        raw1, u1 = llm(prompts.A1_DELIB_SYS, "Warning:\n" + warning)
        s1 = parse_json_strict(raw1)
        rec1 = CallRecord(arm="A2", warning_id=warning_id, prompt_sha=_sha(prompts.A1_DELIB_SYS),
                          prompt_tokens=int(u1.get("prompt_tokens", 0)),
                          completion_tokens=int(u1.get("completion_tokens", 0)),
                          retrieval_calls=0, reasoning_stages=1, latency_s=time.time() - t0)
        t1 = time.time()
        user2 = ("Warning:\n" + warning + "\n\nInitial structured belief:\n" + json.dumps(s1))
        raw2, u2 = llm(prompts.A2_VERIFY_SYS, user2)
        belief = extract_belief(parse_json_strict(raw2), "A2")
        rec2 = CallRecord(arm="A2", warning_id=warning_id, prompt_sha=_sha(prompts.A2_VERIFY_SYS),
                          prompt_tokens=int(u2.get("prompt_tokens", 0)),
                          completion_tokens=int(u2.get("completion_tokens", 0)),
                          retrieval_calls=0, reasoning_stages=2, latency_s=time.time() - t1)
        struct = {"initial": s1, "verification": parse_json_strict(raw2)}
        return belief.as_dict(), struct, [], [rec1, rec2]
    sys_prompt = {"A0": prompts.A0_DIRECT_SYS, "A1": prompts.A1_DELIB_SYS}[arm]
    raw, usage = llm(sys_prompt, "Warning:\n" + warning)
    belief = extract_belief(parse_json_strict(raw), arm)
    stages = 1 if arm == "A0" else 1
    rec = CallRecord(arm=arm, warning_id=warning_id, prompt_sha=_sha(sys_prompt),
                     prompt_tokens=int(usage.get("prompt_tokens", 0)),
                     completion_tokens=int(usage.get("completion_tokens", 0)),
                     retrieval_calls=0, reasoning_stages=stages, latency_s=time.time() - t0)
    struct = parse_json_strict(raw) if arm == "A1" else {"raw_len": len(raw)}
    docs: list = []
    return belief.as_dict(), struct, docs, [rec]
