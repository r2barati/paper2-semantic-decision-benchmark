"""V3 evidence consumers C0/C1/C3 (Phase-1 prototype).

Every consumer returns the SAME BeliefSchema, convertible to the frozen
RegimeInterpretation contract. LLMs produce INFORMATION only: no simulator
state, no orders, no controller access.

C0 deterministic reference: RuleBased on concatenated top-k (reuse, no LLM).
C1 naive RAG: concat top-k -> 1 LLM call (REGIME_EXTRACTION_PROMPT shape).
C3 provenance-aware: 1 LLM call per doc (entity/event/freshness/stance) ->
  frozen source-weighted aggregation + abstain-to-prior (tau=0.5, PREDECLARED,
  not tuned).

Cache key: sha256(model || prompt_sha || docset_hash). Qwen3-8B runs through
the same OpenAI-compatible client via LLM_BASE_URL (pending compute).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.interpreter import (REGIME_EXTRACTION_PROMPT,
                             no_info_regime_belief, rule_based_regime_extract)

CACHE_DIR = Path(os.environ.get("PAPER2_V3_CACHE_DIR",
                                 str(ROOT / "results" / "v3proto" / "llm_cache")))

FROZEN_C1_SHA = "66e6890ba9c5464c"
FROZEN_C3_SHA = "5966cadde90e8236"


def assert_frozen_prompts():
    assert prompt_sha(PROMPT_C1) == FROZEN_C1_SHA, "C1 prompt drift!"
    assert prompt_sha(PROMPT_C3_DOC) == FROZEN_C3_SHA, "C3 prompt drift!"
    assert ABSTAIN_TAU == 0.5, "tau drift!"

PROMPT_C1 = REGIME_EXTRACTION_PROMPT  # reuse frozen V1/V2 prompt shape

PROMPT_C3_DOC = (
    "You are an operational risk analyst. Given ONE evidence document and the "
    "operator's own node described below, extract structured evidence. Respond ONLY "
    "with valid JSON matching this exact schema:\n"
    '{"entity_match": <true|false>, "event": <"supplier_delay"|"demand_surge"|"normal"|"none">, '
    '"fresh": <true|false>, "stance": <"support"|"refute"|"na">, '
    '"confidence": <float 0-1>}\n'
    "entity_match: whether the document concerns the operator's own node. "
    "fresh: whether it describes the current window (not resolved/closed). "
    "stance: support/refute relative to its own event claim. "
    "Do not include any other text."
)

ABSTAIN_TAU = 0.5  # predeclared; NOT tuned on any evaluation data


def prompt_sha(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class BeliefSchema(BaseModel):
    p_normal: float
    p_supplier_delay: float
    p_demand_surge: float
    abstain: bool = False
    confidence: float = 0.0
    evidence_ids: list[str] = []

    @field_validator("p_normal", "p_supplier_delay", "p_demand_surge")
    @classmethod
    def _range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"probability out of range: {v}")
        return float(v)

    @field_validator("confidence")
    @classmethod
    def _conf(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence out of range: {v}")
        return float(v)

    @model_validator(mode="after")
    def _sums_to_one(self):
        s = self.p_normal + self.p_supplier_delay + self.p_demand_surge
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"probabilities sum to {s}, not 1")
        return self

    def to_regime_interpretation(self):
        from src.interpreter import RegimeInterpretation
        if self.abstain:
            return no_info_regime_belief()
        return RegimeInterpretation(
            regime_probabilities={"normal": self.p_normal,
                                  "supplier_delay": self.p_supplier_delay,
                                  "demand_surge": self.p_demand_surge})


def _parse_json(raw):
    """Transport-level unwrap: strip thinking traces + code fences. No content change."""
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _client():
    key = []
    env_path = ROOT / ".env"
    if env_path.exists():
        key = [l.strip().split("=", 1)[1] for l in open(env_path)
               if l.strip().startswith("OPENAI_API_KEY")]
    if not key:
        # Kaggle/vLLM path: endpoint carries no key; any non-empty placeholder works
        key = [os.environ.get("LLM_API_KEY", "local")]
    if not key or not key[0]:
        raise RuntimeError("no LLM API key (.env OPENAI_API_KEY or $LLM_API_KEY)")
    import openai
    return openai.OpenAI(api_key=key[0],
                         base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))


def _llm_json(model, prompt, user, max_tokens=512):
    key_src = f"{model}||{prompt_sha(prompt)}||{hashlib.sha256(user.encode()).hexdigest()[:16]}"
    fname = hashlib.sha256(key_src.encode()).hexdigest()[:16] + ".json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hit = CACHE_DIR / fname
    if hit.exists():
        return json.loads(hit.read_text()), True
    client = _client()
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": user}])
    raw = r.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = {"model": model, "prompt_sha": prompt_sha(prompt),
               "raw": raw,
               "usage": (r.usage.model_dump() if r.usage else {})}
    hit.write_text(json.dumps(payload))
    return payload, False


def consume_c0(top_docs):
    """Deterministic reference: RuleBased over concatenated evidence."""
    text = "\n\n".join(d["text"] for d in top_docs)
    if not text.strip():
        b = no_info_regime_belief()
    else:
        b = rule_based_regime_extract(text)
    p = b.normalized().regime_probabilities
    return BeliefSchema(p_normal=p["normal"], p_supplier_delay=p["supplier_delay"],
                        p_demand_surge=p["demand_surge"], abstain=False,
                        confidence=1.0, evidence_ids=[d["doc_id"] for d in top_docs])


def consume_c1(top_docs, query, model):
    """Naive concat RAG: one LLM call over concatenated top-k."""
    if not top_docs:
        return BeliefSchema(p_normal=1/3, p_supplier_delay=1/3, p_demand_surge=1/3,
                            abstain=True, confidence=0.0, evidence_ids=[])
    user = (f"Operator information need: {query}\n\nEvidence:\n" +
            "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in top_docs))
    payload, _ = _llm_json(model, PROMPT_C1, user)
    probs = _parse_json(payload["raw"])
    s = probs["normal"] + probs["supplier_delay"] + probs["demand_surge"]
    return BeliefSchema(p_normal=probs["normal"]/s,
                        p_supplier_delay=probs["supplier_delay"]/s,
                        p_demand_surge=probs["demand_surge"]/s, abstain=False,
                        confidence=min(1.0, s / 1.0) if s <= 1.5 else 0.5,
                        evidence_ids=[d["doc_id"] for d in top_docs])


def consume_c3(top_docs, query, model, own_node):
    """Provenance-aware: per-doc extraction + frozen weighted aggregation."""
    if not top_docs:
        return BeliefSchema(p_normal=1/3, p_supplier_delay=1/3, p_demand_surge=1/3,
                            abstain=True, confidence=0.0, evidence_ids=[])
    per_doc = []
    for d in top_docs:
        user = (f"Operator's own node: {own_node}\n\nEvidence document:\n{d['text']}")
        payload, _ = _llm_json(model, PROMPT_C3_DOC, user)
        per_doc.append((d["doc_id"], _parse_json(payload["raw"])))
    # Frozen aggregation: weight = confidence * entity_match * fresh; refute negates.
    agg = {"normal": 0.0, "supplier_delay": 0.0, "demand_surge": 0.0}
    support = 0.0
    for _, e in per_doc:
        w = float(e["confidence"]) * (1.0 if e["entity_match"] else 0.2) * \
            (1.0 if e["fresh"] else 0.2)
        ev = e["event"]
        if ev in agg and e["stance"] == "support":
            agg[ev] += w
            support = max(support, w)
        elif ev in agg and e["stance"] == "refute":
            agg[ev] -= 0.5 * w
    if support < ABSTAIN_TAU:
        return BeliefSchema(p_normal=1/3, p_supplier_delay=1/3, p_demand_surge=1/3,
                            abstain=True, confidence=support,
                            evidence_ids=[d["doc_id"] for d in top_docs])
    # softmax over aggregated support + prior smoothing
    import math
    prior = {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30}
    scores = {k: agg[k] + 0.5 * prior[k] for k in agg}
    m = max(scores.values())
    ex = {k: math.exp(v - m) for k, v in scores.items()}
    tot = sum(ex.values())
    return BeliefSchema(p_normal=ex["normal"]/tot,
                        p_supplier_delay=ex["supplier_delay"]/tot,
                        p_demand_surge=ex["demand_surge"]/tot, abstain=False,
                        confidence=min(1.0, support),
                        evidence_ids=[d["doc_id"] for d in top_docs])


def freeze_prompts(outpath):
    out = {"C1_prompt_sha": prompt_sha(PROMPT_C1),
           "C3_doc_prompt_sha": prompt_sha(PROMPT_C3_DOC),
           "ABSTAIN_TAU": ABSTAIN_TAU,
           "C1_prompt": PROMPT_C1, "C3_doc_prompt": PROMPT_C3_DOC}
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    Path(outpath).write_text(json.dumps(out, indent=2))
    return out
