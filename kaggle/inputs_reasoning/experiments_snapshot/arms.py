"""Arm schemas + LLM-call accounting for EXP-P2-REASONING-AGENTIC-v1.

One belief per (warning x arm); retrieval budgets enforced here:
A0/A1/A2 = 0, R0 = 1, A3 <= 2.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, field_validator, model_validator

ARMS = ("A0", "A1", "A2", "R0", "A3")
RETRIEVAL_BUDGET = {"A0": 0, "A1": 0, "A2": 0, "R0": 1, "A3": 2}
LLM_CALLS = {"A0": 1, "A1": 1, "A2": 2, "R0": 1, "A3": 2}


class Belief(BaseModel):
    p_normal: float
    p_supplier_delay: float
    p_demand_surge: float

    @field_validator("p_normal", "p_supplier_delay", "p_demand_surge")
    @classmethod
    def _range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"probability out of range: {v}")
        return float(v)

    @model_validator(mode="after")
    def _sums_to_one(self):
        s = self.p_normal + self.p_supplier_delay + self.p_demand_surge
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"probabilities sum to {s}, not 1")
        return self

    def as_dict(self) -> dict:
        return {"normal": self.p_normal, "supplier_delay": self.p_supplier_delay,
                "demand_surge": self.p_demand_surge}


@dataclass
class CallRecord:
    arm: str
    warning_id: str
    prompt_sha: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_calls: int = 0
    reasoning_stages: int = 0
    latency_s: float = 0.0
    fallback: bool = False


def parse_json_strict(raw: str) -> dict:
    """Transport-level unwrap only (thinking-trace strip + fences). No content edit."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def extract_belief(payload: dict, arm: str) -> Belief:
    if arm in ("A0", "R0"):
        return Belief(p_normal=payload["p_normal"],
                      p_supplier_delay=payload["p_supplier_delay"],
                      p_demand_surge=payload["p_demand_surge"])
    if arm == "A1":
        p = payload["probabilities"]
        return Belief(p_normal=p["p_normal"], p_supplier_delay=p["p_supplier_delay"],
                      p_demand_surge=p["p_demand_surge"])
    if arm == "A2":
        p = payload["revised_probabilities"]
        return Belief(p_normal=p["p_normal"], p_supplier_delay=p["p_supplier_delay"],
                      p_demand_surge=p["p_demand_surge"])
    if arm == "A3":
        p = payload["probabilities"]
        return Belief(p_normal=p["p_normal"], p_supplier_delay=p["p_supplier_delay"],
                      p_demand_surge=p["p_demand_surge"])
    raise ValueError(arm)


PRIOR_BELIEF = Belief(p_normal=0.35, p_supplier_delay=0.35, p_demand_surge=0.30)
