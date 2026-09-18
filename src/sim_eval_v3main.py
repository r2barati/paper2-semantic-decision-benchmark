"""Sequential-decision evaluation path for the frozen V3 main experiment.

PROTOCOL (from results/v3main/NOTES.md): NO economic simulation before the
belief freeze (tools/freeze_beliefs_v3main.py gate) is complete. Full LLM
beliefs are NOT ready.

Consequences enforced by this module:

* No ``gym_invmgmt`` import anywhere in this file (controlled or gym path).
* No ``InventoryEnv`` construction, no ``env.step`` / ``env.reset``, no
  episode returns computed anywhere in the pre-freeze task.
* The ONLY executable path pre-freeze is the controller-only mapping
  ``belief row -> RegimeInterpretation -> frozen controller ACTION`` via
  :func:`belief_row_to_interpretation` + :func:`interpretation_to_controller_action`,
  which instantiates the frozen ``CausalOptimizer`` planning computation on a
  synthetic initial state. Planning (receding-horizon LP) is not simulation:
  no environment is constructed or stepped, no profit/return is computed,
  no files are written.
* Every evaluation-path function that would step the simulator or consume
  episode returns takes an explicit ``beliefs_dir`` argument and calls
  :func:`require_frozen_beliefs` FIRST, which raises :class:`FreezeGateError`
  unless the directory carries an assembled, error-free
  ``belief_manifest.json`` (written by ``tools/freeze_beliefs_v3main.py``)
  whose recorded SHA-256 matches the belief parquet(s) on disk. Post-freeze
  these stubs still raise ``NotImplementedError`` until wired; they are
  scaffolding, not an execution request.

Frozen code reused BY REFERENCE (never reimplemented here):

* Controller: ``src.env:CausalOptimizer`` (frozen receding-horizon LP),
  driven per episode by ``src.experiment_phase5:_run_p5_episode`` and used
  with ``CONTROLLER = "CausalOptimizer"`` in ``src/experiment_phase7.py``
  and ``src/experiment_retrieval.py``; replayed via ``benchmark/run.py``
  (``--experiment controlled`` -> ``src.experiment_phase7``).
* Simulator entry points (guarded stubs only, never imported at module
  scope): ``src.env:InventoryEnv``, ``src.env:HindsightOracle``,
  ``src.experiment_phase5:_run_p5_episode``.
* Value metrics: ``src.metrics:signed_sivr`` with the prior-ledger
  negative-OIV guard (``NEGATIVE_ORACLE_REFERENCE_VALUE`` /
  ``ZERO_OR_NEAR_ZERO_REFERENCE_VALUE`` statuses; never ``abs``-normalized).
* Beliefs: ``src.beliefs_v3main`` row schema
  (query_id/system/consumer/model/k/doc_ids/p_normal/p_supplier_delay/
  p_demand_surge/abstain/confidence) converted via
  ``src.consumers_v3:BeliefSchema.to_regime_interpretation``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.consumers_v3 import BeliefSchema

# ---------------------------------------------------------------------------
# Freeze gate + frozen references (no simulator imports at module scope)
# ---------------------------------------------------------------------------

#: Frozen downstream controller shared by every treatment (belief is the only
#: thing that varies). Reference: src/env.py::CausalOptimizer.
FROZEN_CONTROLLER = "CausalOptimizer"

#: Controlled-simulator / runner entry points used by prior phases. Strings
#: only, so importing this module can never pull in simulator code. Guarded
#: stubs resolve these lazily AFTER require_frozen_beliefs passes.
SIM_ENTRY_POINTS = {
    "controller": "src.env:CausalOptimizer",
    "controlled_env": "src.env:InventoryEnv",
    "hindsight": "src.env:HindsightOracle",
    "episode_runner": "src.experiment_phase5:_run_p5_episode",
    "prior_controlled_runners": (
        "src/experiment_phase7.py (CONTROLLER=CausalOptimizer), "
        "src/experiment_retrieval.py (CONTROLLER=CausalOptimizer), "
        "benchmark/run.py --experiment controlled"
    ),
    "gym_entry": "src/experiment_phase8b.py + src/gym_adapter.py (gym-invmgmt==0.2.1)",
    "value_metric": "src.metrics:signed_sivr",
}

MANIFEST_NAME = "belief_manifest.json"


class FreezeGateError(RuntimeError):
    """Raised when any simulation-adjacent path runs without freeze proof."""


def require_frozen_beliefs(beliefs_dir: str | Path | None) -> dict[str, Any]:
    """Verify an assembled + freeze-manifest-verified beliefs directory.

    Checks: directory exists; ``belief_manifest.json`` exists and parses;
    every file entry in ``manifest["errors"]`` is an empty list; every
    parquet listed in ``manifest["files"]`` exists on disk with matching
    SHA-256. Returns the manifest dict. Raises :class:`FreezeGateError`
    on ANY violation (never silently patch).
    """
    if beliefs_dir is None:
        raise FreezeGateError(
            "no beliefs_dir supplied: simulation-adjacent paths require an "
            "explicit assembled + freeze-manifest-verified beliefs directory "
            "(see tools/freeze_beliefs_v3main.py). No simulation before the "
            "belief freeze."
        )
    d = Path(beliefs_dir)
    manifest_path = d / MANIFEST_NAME
    if not d.is_dir():
        raise FreezeGateError(f"beliefs_dir is not a directory: {d}")
    if not manifest_path.exists():
        raise FreezeGateError(
            f"{MANIFEST_NAME} missing in {d}: beliefs not frozen "
            "(run tools/freeze_beliefs_v3main.py after full assembly)."
        )
    manifest = json.loads(manifest_path.read_text())
    errors = manifest.get("errors", {})
    if not errors:
        raise FreezeGateError(f"{manifest_path} records no frozen files.")
    bad = {k: v for k, v in errors.items() if v}
    if bad:
        raise FreezeGateError(
            f"freeze manifest reports errors for {sorted(bad)}: "
            "simulation refused."
        )
    for fname, info in manifest.get("files", {}).items():
        p = d / fname
        if not p.exists():
            raise FreezeGateError(f"manifest file missing on disk: {p}")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha != info.get("sha256"):
            raise FreezeGateError(f"SHA-256 mismatch (not frozen): {p}")
    return manifest


# ---------------------------------------------------------------------------
# 1. Controller-only mapping (the ONLY pre-freeze executable path)
# ---------------------------------------------------------------------------

def _field(row: Any, key: str, default: Any = None) -> Any:
    """Read one field from a beliefs row (dict, Mapping, or pandas Series)."""
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def belief_row_to_interpretation(row: Mapping[str, Any]):
    """Map one beliefs-parquet row to a frozen ``RegimeInterpretation``.

    Pure information mapping via ``BeliefSchema.to_regime_interpretation``
    (abstain rows map to the benchmark prior). No simulator contact.
    """
    doc_ids = _field(row, "doc_ids", None)
    if doc_ids is None:
        doc_ids = _field(row, "evidence_ids", [])
    belief = BeliefSchema(
        p_normal=float(_field(row, "p_normal")),
        p_supplier_delay=float(_field(row, "p_supplier_delay")),
        p_demand_surge=float(_field(row, "p_demand_surge")),
        abstain=bool(_field(row, "abstain", False)),
        confidence=float(_field(row, "confidence", 0.0)),
        evidence_ids=[str(d) for d in (doc_ids if doc_ids is not None else [])],
    )
    return belief.to_regime_interpretation()


def interpretation_to_controller_action(
    interpretation,
    *,
    seed: int = 0,
    t: int = 0,
    on_hand: float | None = None,
) -> dict[str, Any]:
    """Map a ``RegimeInterpretation`` to a frozen-controller ACTION.

    Controller-only: builds the frozen ``src.env:CausalOptimizer`` with the
    normalized regime belief and solves its receding-horizon LP once
    (``decide``) on a synthetic initial ``InventoryState``. No
    ``InventoryEnv`` is constructed, no ``step``/``reset`` is called, no
    ``gym_invmgmt`` is imported, no profit/return is computed, nothing is
    written to disk. (``decide`` records the planned order in the
    controller's own planning pipeline; that is controller state, not
    simulator state.)
    """
    # Lazy imports keep simulator-adjacent code out of module scope; these
    # resolve the frozen controller/typing only -- never the environment.
    from src.env import CausalOptimizer, InventoryState, INITIAL_INVENTORY
    from src.events import P5_DEMAND_MEAN, P5_HORIZON

    belief = interpretation.normalized()
    probs = dict(belief.regime_probabilities)
    controller = CausalOptimizer(
        seed=int(seed),
        regime_probabilities=probs,
        horizon=P5_HORIZON,
        initial_inventory=INITIAL_INVENTORY,
        demand_mean=P5_DEMAND_MEAN,
    )
    state = InventoryState(
        time=int(t),
        on_hand=float(INITIAL_INVENTORY if on_hand is None else on_hand),
        pipeline=0.0,
    )
    action = float(controller.decide(state))
    return {
        "action": action,
        "most_likely_regime": belief.most_likely_regime,
        "regime_probabilities": probs,
        "controller": FROZEN_CONTROLLER,
        "seed": int(seed),
        "t": int(t),
        "on_hand": float(state.on_hand),
    }


def map_belief_rows_to_actions(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """In-memory beliefs-row -> interpretation -> controller-action mapping.

    No simulator stepping, no returns, no file writes. Each output carries
    its query/system/k/consumer keys plus the mapped action.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        mapped = interpretation_to_controller_action(
            belief_row_to_interpretation(row), seed=seed
        )
        mapped.update({
            "query_id": _field(row, "query_id", None),
            "system": _field(row, "system", None),
            "consumer": _field(row, "consumer", None),
            "k": _field(row, "k", None),
        })
        out.append(mapped)
    return out


def load_c0_reference_rows(
    parquet_path: str | Path,
    *,
    query_ids: Sequence[str],
    system: str = "bm25",
    k: int = 3,
):
    """Read-only loader for the deterministic C0 reference smoke subset.

    Reads the beliefs parquet and returns the C0 rows for the requested
    queries/system/k, sorted by query_id. No simulator contact, no writes.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    sub = df[
        (df["query_id"].isin(list(query_ids)))
        & (df["consumer"] == "C0")
        & (df["system"] == system)
        & (df["k"] == k)
    ].sort_values("query_id").reset_index(drop=True)
    if len(sub) != len(query_ids):
        raise ValueError(
            f"expected {len(query_ids)} C0 reference rows, found {len(sub)}"
        )
    return sub


# ---------------------------------------------------------------------------
# 2. Full evaluation-path STUBS (NOT executed pre-freeze)
# ---------------------------------------------------------------------------

#: Oracle ladder: each rung varies the belief source only; the frozen
#: controller is identical everywhere. HindsightOracle is a clairvoyant
#: reference for decomposition, never the SIVR denominator.
ORACLE_LADDER: tuple[str, ...] = (
    "NoInfo",
    "ActualRetrieval",
    "OracleRelevant",
    "OracleFactual",
    "PerfectBelief",
    "HindsightOracle",
)

#: Four-gap decomposition rungs (differences of the query-averaged,
#: seed-averaged benchmark return J along the ladder / controller axis).
FOUR_GAPS: tuple[str, ...] = (
    "retrieval_gap",            # J(ActualRetrieval) - J(NoInfo)
    "relevance_to_factual_gap",  # J(OracleFactual) - J(OracleRelevant)
    "interpretation_gap",       # J(PerfectBelief) - J(ActualRetrieval|factual)
    "downstream_control_gap",   # J(HindsightOracle) - J(PerfectBelief)
)


def oracle_ladder_spec() -> list[dict[str, str]]:
    """Pure spec: ordered ladder rungs and their frozen belief sources."""
    return [
        {"rung": "NoInfo", "belief_source": "benchmark prior (no evidence)"},
        {"rung": "ActualRetrieval",
         "belief_source": "frozen beliefs parquet row for the retrieving system"},
        {"rung": "OracleRelevant",
         "belief_source": "frozen beliefs parquet oracle-relevant rows"},
        {"rung": "OracleFactual",
         "belief_source": "frozen beliefs parquet oracle-factual rows"},
        {"rung": "PerfectBelief",
         "belief_source": "degenerate belief on the query true_regime"},
        {"rung": "HindsightOracle",
         "belief_source": "clairvoyant reference (src.env:HindsightOracle); "
                          "decomposition only, never the SIVR denominator"},
    ]


def four_gap_spec() -> list[dict[str, str]]:
    """Pure spec: the four decomposition gaps and their J-differences."""
    return [
        {"gap": "retrieval_gap",
         "definition": "J(ActualRetrieval) - J(NoInfo)"},
        {"gap": "relevance_to_factual_gap",
         "definition": "J(OracleFactual) - J(OracleRelevant)"},
        {"gap": "interpretation_gap",
         "definition": "J(PerfectBelief) - J(best evidence-grounded belief); "
                       "consumer/interpretation quality with evidence fixed"},
        {"gap": "downstream_control_gap",
         "definition": "J(HindsightOracle) - J(PerfectBelief); control "
                       "ceiling with semantics fixed"},
    ]


def paired_seed_plan(n_seeds: int, seed_start: int) -> dict[str, Any]:
    """Pure spec: paired fresh seeds shared across treatments.

    Query is the primary unit of analysis; seeds capture environment
    uncertainty only. Every treatment x ladder rung sees the SAME seed set
    (paired design), disjoint from all prior-phase seed ranges.
    """
    return {
        "seeds": [int(seed_start) + i for i in range(int(n_seeds))],
        "seed_start": int(seed_start),
        "n_seeds": int(n_seeds),
        "shared_across_treatments": True,
        "primary_unit": "query",
        "seed_role": "environment uncertainty only",
    }


def run_treatment_episodes(
    *,
    beliefs_dir: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """STUB (not executed): run paired episodes for one treatment.

    Post-freeze wiring: :func:`require_frozen_beliefs` then, per
    (query, seed), resolve the frozen belief row and call
    ``src.experiment_phase5:_run_p5_episode`` with
    ``controller="CausalOptimizer"`` (cf. ``src/experiment_retrieval.py``).
    """
    require_frozen_beliefs(beliefs_dir)
    raise NotImplementedError(
        "stub: episode execution wires up only after the belief freeze "
        "(tools/freeze_beliefs_v3main.py gate)."
    )


def evaluate_oracle_ladder(
    *,
    beliefs_dir: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """STUB (not executed): evaluate the full oracle ladder.

    Post-freeze wiring: :func:`require_frozen_beliefs`, then
    :func:`paired_seed_plan` seeds shared across all rungs in
    :func:`oracle_ladder_spec`, one ``_run_p5_episode`` per
    (query, rung, seed) with the frozen controller, query-averaged J via
    ``src.metrics:weighted_benchmark_return``.
    """
    require_frozen_beliefs(beliefs_dir)
    raise NotImplementedError(
        "stub: ladder evaluation wires up only after the belief freeze "
        "(tools/freeze_beliefs_v3main.py gate)."
    )


def decompose_four_gaps(
    *,
    beliefs_dir: str | Path | None = None,
    J: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """STUB (not executed): four-gap decomposition over ladder returns.

    Gap structure (see :func:`four_gap_spec`); computed from the
    query-averaged, seed-averaged benchmark returns J only after the
    freeze gate passes.
    """
    require_frozen_beliefs(beliefs_dir)
    raise NotImplementedError(
        "stub: gap decomposition wires up only after the belief freeze "
        "(tools/freeze_beliefs_v3main.py gate)."
    )


def compute_sivr_oiv_guarded(
    *,
    beliefs_dir: str | Path | None = None,
    j_condition: float,
    j_no_info: float,
    j_oracle_semantic: float,
) -> dict[str, Any]:
    """SIVR/OIV via the prior-ledger guard (freeze-gated).

    Uses ``src.metrics:signed_sivr`` BY REFERENCE: SIVR is reported only
    with status ``VALID``; a non-positive oracle reference returns NaN /
    the ``NEGATIVE_ORACLE_REFERENCE_VALUE`` /
    ``ZERO_OR_NEAR_ZERO_REFERENCE_VALUE`` status instead of an
    information-value interpretation (never ``abs``-normalized).
    Requires the freeze gate because its inputs are episode returns.
    """
    require_frozen_beliefs(beliefs_dir)
    from src.metrics import signed_sivr  # lazy: value metric only, no simulator

    return signed_sivr(
        float(j_condition), float(j_no_info), float(j_oracle_semantic)
    ).as_dict()


def smoke_controller_mapping(
    parquet_path: str | Path,
    query_ids: Sequence[str] = ("v3-q001", "v3-q002", "v3-q003"),
    *,
    system: str = "bm25",
    k: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Allowed pre-freeze smoke: C0 rows -> interpretation -> ACTION.

    In-memory only: reads the reference parquet, maps rows, returns the
    actions plus wall-clock runtime. Writes no files, computes no returns,
    steps no simulator.
    """
    t0 = time.perf_counter()
    sub = load_c0_reference_rows(
        parquet_path, query_ids=list(query_ids), system=system, k=k
    )
    actions = map_belief_rows_to_actions(
        (r for _, r in sub.iterrows()), seed=seed
    )
    ok = len(actions) == len(query_ids) and all(
        a["action"] == a["action"] and a["action"] >= 0.0 for a in actions
    )
    return {
        "ok": bool(ok),
        "n": len(actions),
        "query_ids": list(query_ids),
        "system": system,
        "k": int(k),
        "actions": actions,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "files_written": 0,
        "returns_computed": False,
        "simulator_stepped": False,
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(
        description="Pre-freeze controller-mapping smoke (no simulation)."
    )
    ap.add_argument("--parquet", default="results/v3main/beliefs_gpt-4o-2024-11-20.parquet")
    ap.add_argument("--queries", nargs="*", default=["v3-q001", "v3-q002", "v3-q003"])
    ap.add_argument("--system", default="bm25")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = smoke_controller_mapping(
        a.parquet, a.queries, system=a.system, k=a.k, seed=a.seed
    )
    print(json.dumps(
        {k: v for k, v in res.items() if k != "actions"}, indent=2
    ))
    for row in res["actions"]:
        print(f"  {row['query_id']} {row['system']}@k{row['k']} "
              f"{row['most_likely_regime']} action={row['action']:.4f}")
