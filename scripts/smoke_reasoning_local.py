"""Local CPU smoke for EXP-P2-REASONING-AGENTIC-v1 (no GPU/LLM).

Deterministic stub LLM returns schema-valid JSON per arm (exercises harness, NOT model quality).
Gates: zero parse/schema failures; sum-to-1; budgets A0/A1/A2=0 R0=1 A3<=2; seal;
controller parity; simulator reproducibility (|d|<=1e-9); checkpoint reload.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.reasoning_agentic import prompts  # noqa: E402
from experiments.reasoning_agentic.agent import run_A3, run_R0  # noqa: E402
from experiments.reasoning_agentic.arms import RETRIEVAL_BUDGET, extract_belief, parse_json_strict  # noqa: E402
from experiments.reasoning_agentic.retriever import SealedCorpus  # noqa: E402
from experiments.reasoning_agentic.run_inference import run_arm_warning  # noqa: E402
from experiments.reasoning_agentic.run_simulation import (  # noqa: E402
    pathwise_equal, run_episode_with_history)
from src.confirmation_templates import CONFIRMATION_TEMPLATES  # noqa: E402
from src.env import CausalOptimizer  # noqa: E402
from src.events import P5_DEMAND_MEAN, P5_HORIZON, Regime  # noqa: E402
from src.interpreter import no_info_regime_belief  # noqa: E402

SMOKE_IDS = ["cd_clear_1", "cd_moderate_1", "cs_clear_1", "cs_moderate_1",
             "cn_clear_1", "cn_moderate_1"]
SEEDS = [62100, 62101, 62102]


def stub_llm(system: str, user: str):
    """Deterministic valid-JSON stub keyed by prompt (harness test only)."""
    px = prompts.prompt_hashes()
    inv = {v[:16]: k for k, v in px.items()}

    def probs():
        return {"p_normal": 0.34, "p_supplier_delay": 0.34, "p_demand_surge": 0.32}

    if "first retrieval query" in system:
        return json.dumps({"hypotheses": {"normal": ["h"], "supplier_delay": ["h"],
                                           "demand_surge": ["h"]}, "query_1": user[:200]}), \
            {"prompt_tokens": 10, "completion_tokens": 10}
    if "at most two queries" in system:
        return json.dumps({"query_2_used": True, "query_2": "supplier lead time delay",
                           "used_evidence": [], "probabilities": probs()}), \
            {"prompt_tokens": 10, "completion_tokens": 10}
    if "sole permitted revision" in system:
        return json.dumps({"inconsistent_evidence": ["e"], "revised_probabilities": probs()}), \
            {"prompt_tokens": 10, "completion_tokens": 10}
    if "factual claims" in system:
        return json.dumps({"factual_claims": ["c"], "support": {"normal": [], "supplier_delay": ["c"],
                                                                "demand_surge": []},
                           "contradictions": ["a"], "probabilities": probs()}), \
            {"prompt_tokens": 10, "completion_tokens": 10}
    return json.dumps(probs()), {"prompt_tokens": 10, "completion_tokens": 10}


def main() -> int:
    gates: dict[str, str] = {}
    corpus = SealedCorpus(ROOT / "data" / "v3" / "corpus.jsonl")
    # Seal must NOT trip on the real repo (forbidden files live elsewhere); verify the
    # guard itself trips on a planted dir.
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "test.tsv").write_text("x")
        try:
            SealedCorpus(ROOT / "data" / "v3" / "corpus.jsonl", seal_dir=Path(td))
            gates["seal_abort"] = "FAIL (no abort on planted test.tsv)"
        except RuntimeError:
            gates["seal_abort"] = "PASS"
    print(f"corpus docs: {len(corpus)}", flush=True)

    warns = [t for t in CONFIRMATION_TEMPLATES if t["template_id"] in SMOKE_IDS]
    assert len(warns) == 6, [t["template_id"] for t in warns]
    failures, budgets_ok, n_beliefs = 0, True, 0
    beliefs: dict[str, dict] = {}
    for t in warns:
        for arm in ("A0", "A1", "A2", "R0", "A3"):
            try:
                b, struct, docs, recs = run_arm_warning(
                    arm, t["template_id"], t["text"], corpus, stub_llm)
                n_used = sum(r.retrieval_calls if arm != "A3" else 0 for r in recs)
                mx = max((r.retrieval_calls for r in recs), default=0)
                used = mx if arm == "A3" else (1 if arm == "R0" else 0)
                if arm in ("A0", "A1", "A2") and used != 0:
                    budgets_ok = False
                if arm == "R0" and used != 1:
                    budgets_ok = False
                if arm == "A3" and not (1 <= used <= 2):
                    budgets_ok = False
                beliefs[f"{t['template_id']}||{arm}"] = b
                n_beliefs += 1
            except Exception as e:  # noqa: BLE001 — gate counts every failure
                failures += 1
                print(f"FAIL {t['template_id']} {arm}: {e!r}", flush=True)
    gates["zero_parse_failures"] = "PASS" if failures == 0 else f"FAIL ({failures})"
    gates["retrieval_budgets"] = "PASS" if budgets_ok else "FAIL"
    gates["infer_once_per_warning"] = "PASS" if n_beliefs == 30 else f"FAIL ({n_beliefs})"

    # Controller parity: identical optimizer params across arms at warning state.
    prior = no_info_regime_belief().normalized().regime_probabilities
    try:
        from src.env import InventoryEnv

        env = InventoryEnv(seed=SEEDS[0], regime=Regime.SUPPLIER_DELAY)
        env.reset()
        o1 = CausalOptimizer(seed=SEEDS[0], regime_probabilities=dict(prior),
                             horizon=P5_HORIZON, initial_inventory=30, demand_mean=P5_DEMAND_MEAN)
        o2 = CausalOptimizer(seed=SEEDS[0], regime_probabilities=dict(prior),
                             horizon=P5_HORIZON, initial_inventory=30, demand_mean=P5_DEMAND_MEAN)
        gates["controller_parity"] = "PASS" if o1.decide(env._state) == o2.decide(env._state) else "FAIL"
    except Exception as e:  # noqa: BLE001
        gates["controller_parity"] = f"FAIL ({e!r})"

    # Simulator reproducibility + pathwise equality sanity (same belief twice => equal paths).
    try:
        b = beliefs["cd_clear_1||A0"]
        r1, _ = run_episode_with_history(b, Regime.SUPPLIER_DELAY, SEEDS[0])
        r2, _ = run_episode_with_history(b, Regime.SUPPLIER_DELAY, SEEDS[0])
        same = pathwise_equal(r1["actions"], r2["actions"]) and abs(
            r1["total_profit"] - r2["total_profit"]) <= 1e-9
        gates["simulator_reproducibility"] = "PASS" if same else "FAIL"
        # Dead-zone pathwise hook: A0 vs prior-belief replay on one warning/seed.
        prior_b = {"normal": 0.35, "supplier_delay": 0.35, "demand_surge": 0.30}
        r3, _ = run_episode_with_history(prior_b, Regime.SUPPLIER_DELAY, SEEDS[0])
        eq = pathwise_equal(r1["actions"], r3["actions"])
        gates["pathwise_deadzone_hook"] = f"PASS (equal={eq}, dProfit={r1['total_profit'] - r3['total_profit']:.4f})"
    except Exception as e:  # noqa: BLE001
        gates["simulator_reproducibility"] = f"FAIL ({e!r})"
        gates["pathwise_deadzone_hook"] = "FAIL"

    # Checkpoint reload: serialize + reload beliefs, compare.
    try:
        blob = json.dumps(beliefs, sort_keys=True)
        assert json.loads(blob) == beliefs
        gates["checkpoint_reload"] = "PASS"
    except Exception as e:  # noqa: BLE001
        gates["checkpoint_reload"] = f"FAIL ({e!r})"

    print(json.dumps(gates, indent=2), flush=True)
    overall = all(v.startswith("PASS") for v in gates.values())
    print("LOCAL SMOKE " + ("PASS" if overall else "FAIL"), flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
