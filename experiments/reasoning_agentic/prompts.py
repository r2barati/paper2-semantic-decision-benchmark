"""Frozen prompt texts for EXP-P2-REASONING-AGENTIC-v1.

DEV status: hashes recorded at freeze time into results/reasoning_agentic/prompt_hashes.json.
Do not edit after freeze — any change requires a new versioned experiment.
"""

from __future__ import annotations

import hashlib

PRIOR_LINE = "Benchmark prior: normal 0.35, supplier_delay 0.35, demand_surge 0.30."

A0_DIRECT_SYS = (
    "You are an operational risk analyst. Given a textual supplier or market "
    "warning, output regime probabilities directly. Respond ONLY with valid JSON:\n"
    '{"p_normal": <float 0-1>, "p_supplier_delay": <float 0-1>, "p_demand_surge": <float 0-1>}\n'
    + PRIOR_LINE + " The three probabilities must sum to 1.0. "
    "No explanation, no retrieval, no revision."
)

A1_DELIB_SYS = (
    "You are an operational risk analyst. Follow this fixed procedure: "
    "(1) extract factual claims; (2) list which claims support each regime "
    "(normal/supplier_delay/demand_surge); (3) list contradictory or ambiguous evidence; "
    "(4) assign regime probabilities; (5) report. Respond ONLY with valid JSON:\n"
    '{"factual_claims": [...], "support": {"normal": [...], "supplier_delay": [...], '
    '"demand_surge": [...]}, "contradictions": [...], '
    '"probabilities": {"p_normal": <float>, "p_supplier_delay": <float>, '
    '"p_demand_surge": <float>}}\n' + PRIOR_LINE + " Probabilities sum to 1.0."
)

A2_VERIFY_SYS = (
    "You are a verification analyst. Given the warning and an initial structured belief, "
    "(1) re-inspect the warning; (2) list evidence potentially inconsistent with the "
    "initial conclusion; (3) revise probabilities exactly once. Respond ONLY with valid JSON:\n"
    '{"inconsistent_evidence": [...], "revised_probabilities": {"p_normal": <float>, '
    '"p_supplier_delay": <float>, "p_demand_surge": <float>}}\n'
    + PRIOR_LINE + " Sum to 1.0. This is the sole permitted revision."
)

R0_SYNTH_SYS = (
    "You are an operational risk analyst. Given a warning and frozen retrieved evidence "
    "(ranked excerpts), synthesize and output regime probabilities. "
    "Respond ONLY with valid JSON:\n"
    '{"p_normal": <float>, "p_supplier_delay": <float>, "p_demand_surge": <float>}\n'
    + PRIOR_LINE + " Sum to 1.0. Do not issue further queries."
)

A3_HYPO_SYS = (
    "You are an operational risk analyst with a retrieval tool. Given a warning, "
    "(1) state hypotheses per regime; (2) write ONE first retrieval query "
    "(verbatim warning text is acceptable). Respond ONLY with valid JSON:\n"
    '{"hypotheses": {"normal": [...], "supplier_delay": [...], "demand_surge": [...]}, '
    '"query_1": "<string>"}'
)

A3_SYNTH_SYS = (
    "Given the warning, your hypotheses, and the retrieved evidence from at most two "
    "queries (optionally: one reformulated query `query_2`), synthesize final regime "
    "probabilities. Respond ONLY with valid JSON:\n"
    '{"query_2_used": <true|false>, "query_2": "<string or empty>", '
    '"used_evidence": ["<doc ids>"], "probabilities": {"p_normal": <float>, '
    '"p_supplier_delay": <float>, "p_demand_surge": <float>}}\n'
    + PRIOR_LINE + " Sum to 1.0. Stop."
)

PROMPTS = {
    "A0_DIRECT_SYS": A0_DIRECT_SYS,
    "A1_DELIB_SYS": A1_DELIB_SYS,
    "A2_VERIFY_SYS": A2_VERIFY_SYS,
    "R0_SYNTH_SYS": R0_SYNTH_SYS,
    "A3_HYPO_SYS": A3_HYPO_SYS,
    "A3_SYNTH_SYS": A3_SYNTH_SYS,
}


def prompt_hashes() -> dict[str, str]:
    return {k: hashlib.sha256(v.encode()).hexdigest() for k, v in PROMPTS.items()}
