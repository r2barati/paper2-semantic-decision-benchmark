"""One-pass deterministic v5 analysis.

This script computes only the preregistered controller, selective-consumption,
cross-family, and persistence estimands from the frozen episode table. It uses
the signed definitions literally and refuses to analyze an unfrozen input.
"""

from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
SIM_MANIFEST = ROOT / "versions/sem2act-v5/manifests/simulation_manifest.json"
ASSEMBLY_MANIFEST = ROOT / "versions/sem2act-v5/manifests/belief_assembly.json"
PREFLIGHT = ROOT / "versions/sem2act-v5/manifests/model_preflight.json"
EPISODES = ROOT / "versions/sem2act-v5/results/episodes.parquet"
OUT = ROOT / "versions/sem2act-v5/results/analysis"
CANONICAL_MANIFEST = ROOT / "versions/sem2act-v5/manifests/RESULT_MANIFEST.json"
B = 5000
SEED = 42
REGIMES = ("normal", "supplier_delay", "demand_surge")
WEIGHTS = {regime: 1 / 3 for regime in REGIMES}
QWEN = "Qwen/Qwen3-8B-AWQ"
FAMILIES = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions() -> dict[str, str]:
    names = ("numpy", "pandas", "pyarrow", "scipy", "pydantic")
    out = {"python": platform.python_version()}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "unavailable"
    return out


def subset(
    frame: pd.DataFrame,
    *,
    model: str,
    system: str,
    consumer: str,
    controller: str,
    method: str = "always_consume",
    condition: str = "persistent",
    horizon: int = 40,
) -> pd.DataFrame:
    mask = (
        (frame["model"] == model)
        & (frame["system"] == system)
        & (frame["consumer"] == consumer)
        & (frame["controller"] == controller)
        & (frame["method"] == method)
        & (frame["condition"] == condition)
        & (frame["horizon"] == horizon)
    )
    return frame[mask].copy()


def ref_subset(
    frame: pd.DataFrame,
    *,
    model: str,
    comparison_system: str,
    consumer: str,
    controller: str,
    condition: str = "persistent",
    horizon: int = 40,
) -> pd.DataFrame:
    mask = (
        (frame["model"] == model)
        & (frame["system"] == "noinfo")
        & (frame["comparison_system"] == comparison_system)
        & (frame["consumer"] == consumer)
        & (frame["controller"] == controller)
        & (frame["method"] == "noinfo")
        & (frame["condition"] == condition)
        & (frame["horizon"] == horizon)
    )
    return frame[mask].copy()


def paired_difference(arm: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    keys = ["query_id", "seed"]
    if arm.empty or ref.empty:
        raise SystemExit("empty arm/reference in preregistered contrast")
    left = arm[keys + ["true_regime", "profit"]].rename(
        columns={"profit": "arm_profit"}
    )
    right = ref[keys + ["profit"]].rename(columns={"profit": "ref_profit"})
    out = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(out) != len(left) or len(out) != len(right):
        raise SystemExit("paired key coverage mismatch")
    out["diff"] = out["arm_profit"] - out["ref_profit"]
    return out[keys + ["true_regime", "diff"]]


def matrices(diff: pd.DataFrame):
    mats = {}
    for regime in REGIMES:
        part = diff[diff["true_regime"] == regime]
        if part.empty:
            raise SystemExit(f"missing regime {regime}")
        table = part.pivot(index="query_id", columns="seed", values="diff").sort_index()
        if table.isna().any().any():
            raise SystemExit("missing crossed-bootstrap cell")
        mats[regime] = table.to_numpy(dtype=float)
    return mats


def balanced_stat(mats: dict[str, np.ndarray]) -> float:
    return float(sum(WEIGHTS[regime] * np.mean(mats[regime]) for regime in REGIMES))


def bootstrap_effect(diff: pd.DataFrame) -> dict:
    mats = matrices(diff)
    point = balanced_stat(mats)
    rng = np.random.default_rng(SEED)
    draws = np.empty(B)
    for b in range(B):
        total = 0.0
        for regime in REGIMES:
            matrix = mats[regime]
            qi = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            si = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
            total += WEIGHTS[regime] * float(matrix[np.ix_(qi, si)].mean())
        draws[b] = total
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = (1 + np.count_nonzero(np.abs(draws) >= abs(point))) / (B + 1)
    return {
        "estimate": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(p),
        "n_queries": int(sum(m.shape[0] for m in mats.values())),
        "n_seeds": int(next(iter(mats.values())).shape[1]),
    }


def bootstrap_interaction(cells: dict[tuple[str, int, str], pd.DataFrame]) -> dict:
    mats = {}
    for key, diff in cells.items():
        mats[key] = matrices(diff)
    required = {
        (condition, horizon, "rerank")
        for condition in ("persistent", "reset")
        for horizon in (1, 40)
    }
    if set(mats) != required:
        raise SystemExit("persistence interaction cell set is incomplete")
    point = 0.0
    for regime in REGIMES:
        point += WEIGHTS[regime] * float(
            np.mean(mats[("persistent", 40, "rerank")][regime])
            - np.mean(mats[("persistent", 1, "rerank")][regime])
            - np.mean(mats[("reset", 40, "rerank")][regime])
            + np.mean(mats[("reset", 1, "rerank")][regime])
        )
    rng = np.random.default_rng(SEED)
    draws = np.empty(B)
    for b in range(B):
        value = 0.0
        for regime in REGIMES:
            reference = mats[("persistent", 40, "rerank")][regime]
            qi = rng.integers(0, reference.shape[0], size=reference.shape[0])
            si = rng.integers(0, reference.shape[1], size=reference.shape[1])
            value += WEIGHTS[regime] * float(
                mats[("persistent", 40, "rerank")][regime][np.ix_(qi, si)].mean()
                - mats[("persistent", 1, "rerank")][regime][np.ix_(qi, si)].mean()
                - mats[("reset", 40, "rerank")][regime][np.ix_(qi, si)].mean()
                + mats[("reset", 1, "rerank")][regime][np.ix_(qi, si)].mean()
            )
        draws[b] = value
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = (1 + np.count_nonzero(np.abs(draws) >= abs(point))) / (B + 1)
    return {
        "estimate_I_40_1": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(p),
        "null": "H0: I_40_1 = 0",
        "bootstrap": {"replicates": B, "seed": SEED, "paired_crossed": True},
    }


def holm(rows: list[dict], p_key: str = "p_value") -> list[dict]:
    ordered = sorted(range(len(rows)), key=lambda i: rows[i][p_key])
    adjusted = [0.0] * len(rows)
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (len(rows) - rank) * rows[index][p_key])
        adjusted[index] = min(1.0, running)
    for index, row in enumerate(rows):
        row["p_holm"] = float(adjusted[index])
        row["reject_holm_05"] = bool(adjusted[index] < 0.05)
    return rows

def effect_row(frame, *, model, system, consumer, controller) -> dict:
    arm = subset(
        frame, model=model, system=system, consumer=consumer,
        controller=controller,
    )
    ref = ref_subset(
        frame, model=model, comparison_system=system, consumer=consumer,
        controller=controller,
    )
    result = bootstrap_effect(paired_difference(arm, ref))
    return {
        "model": model, "system": system, "consumer": consumer,
        "controller": controller, "comparison": "delta_J_vs_NoInfo",
        **result,
    }


def cross_family_row(frame, family: str, system: str, consumer: str) -> dict:
    qwen_arm = subset(
        frame, model=QWEN, system=system, consumer=consumer,
        controller="CausalOptimizer",
    )
    qwen_ref = ref_subset(
        frame, model=QWEN, comparison_system=system, consumer=consumer,
        controller="CausalOptimizer",
    )
    family_arm = subset(
        frame, model=family, system=system, consumer=consumer,
        controller="CausalOptimizer",
    )
    family_ref = ref_subset(
        frame, model=family, comparison_system=system, consumer=consumer,
        controller="CausalOptimizer",
    )
    qwen_diff = paired_difference(qwen_arm, qwen_ref)
    family_diff = paired_difference(family_arm, family_ref)
    left = qwen_diff.rename(columns={"diff": "qwen_diff"})
    right = family_diff.rename(columns={"diff": "family_diff"})
    merged = left.merge(right, on=["query_id", "seed", "true_regime"],
                         validate="one_to_one")
    merged["diff"] = merged["family_diff"] - merged["qwen_diff"]
    result = bootstrap_effect(merged[["query_id", "seed", "true_regime", "diff"]])
    return {
        "family": family, "system": system, "consumer": consumer,
        "comparison": "family_delta_minus_Qwen_delta",
        **result,
    }


def selective_row(frame, consumer: str) -> dict:
    raw = subset(
        frame, model=QWEN, system="rerank", consumer=consumer,
        controller="CausalOptimizer", method="always_consume",
    )
    selective = subset(
        frame, model=QWEN, system="rerank", consumer=consumer,
        controller="CausalOptimizer", method="selective_entropy",
    )
    left = selective[["query_id", "seed", "true_regime", "profit"]].rename(
        columns={"profit": "selective_profit"}
    )
    right = raw[["query_id", "seed", "profit"]].rename(
        columns={"profit": "always_profit"}
    )
    merged = left.merge(right, on=["query_id", "seed", "true_regime"],
                         validate="one_to_one")
    merged["diff"] = merged["selective_profit"] - merged["always_profit"]
    result = bootstrap_effect(merged[["query_id", "seed", "true_regime", "diff"]])
    return {
        "model": QWEN, "system": "rerank", "consumer": consumer,
        "controller": "CausalOptimizer",
        "comparison": "selective_entropy_minus_always_consume",
        **result,
    }


def persistence_results(frame: pd.DataFrame) -> dict:
    cells = {}
    deltas = {}
    for condition in ("persistent", "reset"):
        for horizon in (1, 5, 10, 20, 40):
            arm = subset(
                frame, model=QWEN, system="rerank", consumer="C1",
                controller="CausalOptimizer", condition=condition,
                horizon=horizon,
            )
            ref = ref_subset(
                frame, model=QWEN, comparison_system="rerank", consumer="C1",
                controller="CausalOptimizer", condition=condition,
                horizon=horizon,
            )
            diff = paired_difference(arm, ref)
            deltas[f"{condition}/T{horizon}"] = bootstrap_effect(diff)
            if horizon in (1, 40):
                cells[(condition, horizon, "rerank")] = diff
    interaction = bootstrap_interaction(cells)
    per_period = {}
    for key, value in deltas.items():
        horizon = int(key.rsplit("T", 1)[1])
        per_period[key] = value["estimate"] / horizon
    return {
        "primary_arm": {
            "model": QWEN, "system": "rerank", "consumer": "C1",
            "controller": "CausalOptimizer", "release_t": 12,
        },
        "delta_definition": "J_Rerank - J_NoInfo",
        "deltas": deltas,
        "per_period_delta": per_period,
        "interaction": interaction,
    }

def main() -> int:
    for path in (PROTOCOL, SIM_MANIFEST, ASSEMBLY_MANIFEST, PREFLIGHT, EPISODES):
        if not path.exists():
            raise SystemExit(f"required frozen input missing: {path}")
    protocol = json.loads(PROTOCOL.read_text())
    simulation_manifest = json.loads(SIM_MANIFEST.read_text())
    if protocol["protocol_hash"] != "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022":
        raise SystemExit("protocol hash drift")
    if simulation_manifest.get("status") != "frozen" or simulation_manifest.get("qrels_read") is not False:
        raise SystemExit("simulation is not a frozen qrel-free input")
    frame = pd.read_parquet(EPISODES)
    required = {
        "model", "system", "comparison_system", "consumer", "controller",
        "method", "condition", "horizon", "query_id", "true_regime", "seed",
        "profit",
    }
    if not required <= set(frame.columns):
        raise SystemExit("episode schema incomplete")
    if frame.empty:
        raise SystemExit("episode table empty")

    causal = [
        effect_row(frame, model=QWEN, system=system, consumer=consumer,
                   controller="CausalOptimizer")
        for system in ("bm25", "rerank", "oracle")
        for consumer in ("C1", "C3")
    ]
    base = [
        effect_row(frame, model=QWEN, system=system, consumer=consumer,
                   controller="BeliefBaseStock")
        for system in ("bm25", "rerank", "oracle")
        for consumer in ("C1", "C3")
    ]
    causal = holm(causal)
    base = holm(base)
    # Keep explicit six-test families so multiplicity cannot be misread as one
    # 12-test family.
    for row in causal:
        row["holm_family"] = "CausalOptimizer six effects"
    for row in base:
        row["holm_family"] = "BeliefBaseStock six effects"

    safety_rows = []
    for system in ("bm25", "rerank", "oracle"):
        for consumer in ("C1", "C3"):
            arm = subset(
                frame, model=QWEN, system=system, consumer=consumer,
                controller="WorstCaseBaseStock",
            )
            ref = ref_subset(
                frame, model=QWEN, comparison_system=system, consumer=consumer,
                controller="WorstCaseBaseStock",
            )
            diff = paired_difference(arm, ref)
            maximum = float(np.max(np.abs(diff["diff"])))
            if maximum > 1e-9:
                raise SystemExit("evidence-insensitive safety control changed action")
            safety_rows.append({
                "system": system, "consumer": consumer,
                "controller": "WorstCaseBaseStock",
                "max_abs_paired_delta": maximum,
                "excluded_from_confirmatory_family": True,
            })

    cross_family = []
    for family in FAMILIES:
        for system in ("rerank", "oracle"):
            for consumer in ("C1", "C3"):
                cross_family.append(cross_family_row(frame, family, system, consumer))
    cross_family = holm(cross_family)

    selective = [
        selective_row(frame, consumer) for consumer in ("C1", "C3")
    ]
    selective = holm(selective)

    outputs = {
        "controller_effects": causal + base,
        "safety_control": safety_rows,
        "cross_family": cross_family,
        "selective_consumption": selective,
        "persistence": persistence_results(frame),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "v5_analysis.json"
    result_path.write_text(json.dumps(outputs, indent=2) + "\n")
    model_runs = {}
    for family in ("qwen", "llama", "mistral"):
        run_path = ROOT / f"versions/sem2act-v5/runtime/model_outputs/{family}/run_manifest.json"
        if not run_path.exists():
            raise SystemExit(f"accepted {family} run manifest missing")
        run = json.loads(run_path.read_text())
        model_runs[family] = {
            "manifest_sha256": sha(run_path),
            "model": run.get("model"),
            "revision": run.get("revision"),
            "expected_calls": run.get("expected_calls"),
            "actual_calls": run.get("actual_calls"),
            "cache_count": run.get("n_cache_files", run.get("actual_calls")),
            "n_fail": run.get("n_fail"),
        }
    dataset_versions = {}
    version_dir = ROOT / "versions/sem2act-v5/manifests/dataset_versions"
    if version_dir.exists():
        dataset_versions = {
            path.name: sha(path) for path in sorted(version_dir.glob("*.json"))
        }
    source_paths = {
        "assembly_script": ROOT / "versions/sem2act-v5/experiments/assemble_beliefs.py",
        "simulation_script": ROOT / "versions/sem2act-v5/experiments/simulate_confirmation.py",
        "analysis_script": Path(__file__).resolve(),
        "signed_estimands": ROOT / "versions/sem2act-v5/experiments/signed_estimands.py",
        "selective_entropy": ROOT / "versions/sem2act-v5/experiments/selective_entropy.py",
        "controllers": ROOT / "versions/sem2act-v5/experiments/controllers.py",
        "environment": ROOT / "src/env.py",
        "events": ROOT / "src/events.py",
        "base_stock": ROOT / "src/controller_basestock.py",
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-confirmation-analysis",
        "status": "frozen",
        "protocol_hash": protocol["protocol_hash"],
        "bootstrap": {
            "method": "crossed query-seed bootstrap",
            "replicates": B,
            "seed": SEED,
            "weights": "balanced",
        },
        "holm_families": {
            "controller": 6,
            "cross_family": 8,
            "selective_consumption": 2,
        },
        "primary_null": "H0: I_40_1 = 0",
        "positive_interaction_interpretation": (
            "Positive I_40_1 amplifies the Rerank-versus-NoInfo effect under "
            "persistence relative to reset; the sign of Delta J determines "
            "benefit versus harm."
        ),
        "qrels_read": False,
        "excluded_runs": [],
        "model_runs": model_runs,
        "dataset_versions": dataset_versions,
        "rng_provenance": {"bootstrap_seed": SEED},
        "package_versions": versions(),
        "script_hashes": {name: sha(path) for name, path in source_paths.items()},
        "input_hashes": {
            "episodes": sha(EPISODES),
            "simulation_manifest": sha(SIM_MANIFEST),
            "belief_assembly": sha(ASSEMBLY_MANIFEST),
            "model_preflight": sha(PREFLIGHT),
            "protocol": sha(PROTOCOL),
        },
        "outputs": {
            "analysis": str(result_path.relative_to(ROOT)),
            "analysis_sha256": sha(result_path),
            "episodes": str(EPISODES.relative_to(ROOT)),
        },
    }
    CANONICAL_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "RESULT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
