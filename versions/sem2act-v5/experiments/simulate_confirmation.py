"""Run the frozen v5 CPU replay after model beliefs are accepted.

The simulator is deterministic conditional on frozen beliefs and paired seeds.
It implements the persistent rollout and the explicitly diagnostic state-reset
counterfactual. No retrieval or model call occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.controller_basestock import BeliefBaseStock
from src.env import CausalOptimizer, INITIAL_INVENTORY, InventoryEnv
from src.events import (
    P5_DEMAND_MEAN,
    P5_EVENT_DURATION,
    P5_EVENT_START,
    REGIME_PARAMS,
    REGIME_PRIOR,
    Regime,
)

SEEDS = list(range(61000, 61020))
HORIZONS = (1, 5, 10, 20, 40)
RELEASE_T = 12
PRIOR = {
    (key.value if hasattr(key, "value") else str(key)): float(value)
    for key, value in REGIME_PRIOR.items()
}
PRIOR = {key: PRIOR[key] for key in ("normal", "supplier_delay", "demand_surge")}
CONTROLLERS_QWEN = ("CausalOptimizer", "BeliefBaseStock", "WorstCaseBaseStock")
CONTROLLERS_REPLICATION = ("CausalOptimizer",)
REGIMES = tuple(PRIOR)
PROTOCOL = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
BELIEFS = ROOT / "versions/sem2act-v5/runtime/assembled_beliefs.jsonl"
OUT = ROOT / "versions/sem2act-v5/results/episodes.parquet"


def load_worst_case():
    path = ROOT / "versions/sem2act-v5/experiments/controllers.py"
    spec = importlib.util.spec_from_file_location("v5_controllers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WorstCaseBaseStock


WorstCaseBaseStock = load_worst_case()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_probs(row: dict) -> dict[str, float]:
    values = {
        "normal": float(row["p_normal"]),
        "supplier_delay": float(row["p_supplier_delay"]),
        "demand_surge": float(row["p_demand_surge"]),
    }
    if any(value < 0 or value > 1 for value in values.values()):
        raise ValueError("belief probability out of range")
    if abs(sum(values.values()) - 1.0) > 1e-6:
        raise ValueError("belief probabilities do not sum to one")
    return values


def effective_belief(probs: dict[str, float], selective: bool) -> dict[str, float]:
    if not selective:
        return dict(probs)
    from importlib.util import spec_from_file_location, module_from_spec
    path = ROOT / "versions/sem2act-v5/experiments/selective_entropy.py"
    spec = spec_from_file_location("v5_selective_entropy", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.shrink_to_prior(probs, PRIOR)


def make_controller(name: str, seed: int, horizon: int, probs: dict[str, float]):
    if name == "CausalOptimizer":
        return CausalOptimizer(
            seed=seed,
            regime_probabilities=dict(probs),
            horizon=horizon,
            initial_inventory=INITIAL_INVENTORY,
            demand_mean=P5_DEMAND_MEAN,
        )
    if name == "BeliefBaseStock":
        return BeliefBaseStock(probs)
    if name == "WorstCaseBaseStock":
        return WorstCaseBaseStock()
    raise ValueError(f"unknown controller: {name}")


def update_controller(controller, name: str, probs: dict[str, float]):
    if name == "CausalOptimizer":
        controller.set_regime_probabilities(dict(probs))
        return controller
    if name == "BeliefBaseStock":
        return BeliefBaseStock(probs)
    return controller


def demand_mean(regime: Regime, t: int) -> float:
    params = REGIME_PARAMS[regime]
    if P5_EVENT_START <= t < P5_EVENT_START + P5_EVENT_DURATION:
        return P5_DEMAND_MEAN * params["demand_multiplier"]
    return P5_DEMAND_MEAN

def run_persistent(
    seed: int,
    true_regime: str,
    arm_probs: dict[str, float],
    controller_name: str,
    horizon: int,
    release_t: int,
    selective: bool,
) -> float:
    regime = Regime(true_regime)
    env = InventoryEnv(seed=seed, regime=regime, horizon=horizon)
    env.reset()
    controller = make_controller(controller_name, seed, horizon, PRIOR)
    released = release_t <= 0
    if released:
        controller = update_controller(
            controller, controller_name, effective_belief(arm_probs, selective)
        )
    total = 0.0
    for t in range(horizon):
        if not released and t >= release_t:
            released = True
            controller = update_controller(
                controller, controller_name,
                effective_belief(arm_probs, selective)
            )
        state = env._state
        order = controller.decide(state)
        next_state = env.step(order)
        total += float(next_state.period_profit)
    return total


def run_reset(
    seed: int,
    true_regime: str,
    arm_probs: dict[str, float],
    controller_name: str,
    horizon: int,
    release_t: int,
    selective: bool,
) -> float:
    regime = Regime(true_regime)
    # Draw the aligned demand path once. Each one-step rollout below consumes
    # the draw at its original time after resetting inventory and pipeline.
    rng = np.random.default_rng(seed)
    demand_path = [int(rng.poisson(demand_mean(regime, t))) for t in range(horizon)]
    total = 0.0
    for t in range(horizon):
        env = InventoryEnv(seed=seed, regime=regime, horizon=horizon)
        env.reset()
        replay_rng = np.random.default_rng(seed)
        for prior_t in range(t):
            replay_rng.poisson(demand_mean(regime, prior_t))
        env.rng = replay_rng
        env._state.time = t
        env._state.lead_time = env._get_regime_lead_time(t)
        controller = make_controller(controller_name, seed, horizon, PRIOR)
        if t >= release_t:
            controller = update_controller(
                controller, controller_name,
                effective_belief(arm_probs, selective)
            )
        order = controller.decide(env._state)
        next_state = env.step(order)
        if int(next_state.demand) != demand_path[t]:
            raise RuntimeError("reset counterfactual demand alignment drift")
        total += float(next_state.period_profit)
    return total


def run_task(task: dict) -> dict:
    if task["condition"] == "persistent":
        profit = run_persistent(
            task["seed"], task["true_regime"], task["probs"],
            task["controller"], task["horizon"], task["release_t"],
            task["selective"],
        )
    elif task["condition"] == "reset":
        profit = run_reset(
            task["seed"], task["true_regime"], task["probs"],
            task["controller"], task["horizon"], task["release_t"],
            task["selective"],
        )
    else:
        raise ValueError("unknown environment condition")
    return {
        "model": task["model"],
        "system": task["system"],
        "comparison_system": task["comparison_system"],
        "consumer": task["consumer"],
        "controller": task["controller"],
        "method": task["method"],
        "condition": task["condition"],
        "horizon": task["horizon"],
        "release_t": task["release_t"],
        "query_id": task["query_id"],
        "true_regime": task["true_regime"],
        "seed": task["seed"],
        "profit": profit,
    }


def task_key(task: dict) -> tuple:
    return (
        task["model"], task["system"], task["comparison_system"],
        task["consumer"], task["controller"], task["method"],
        task["condition"], task["horizon"], task["query_id"], task["seed"],
    )


def add_task(tasks: dict[tuple, dict], *, model: str, system: str,
             comparison_system: str, consumer: str, controller: str,
             method: str, condition: str, horizon: int, query_id: str,
             true_regime: str, seed: int, probs: dict[str, float],
             release_t: int = RELEASE_T, selective: bool = False):
    task = {
        "model": model, "system": system,
        "comparison_system": comparison_system, "consumer": consumer,
        "controller": controller, "method": method,
        "condition": condition, "horizon": horizon,
        "release_t": release_t, "query_id": query_id,
        "true_regime": true_regime, "seed": seed, "probs": probs,
        "selective": selective,
    }
    tasks[task_key(task)] = task


def load_belief_rows() -> list[dict]:
    if not BELIEFS.exists():
        raise SystemExit("assembled beliefs are required before simulation")
    rows = [json.loads(line) for line in BELIEFS.read_text().splitlines() if line]
    if not rows:
        raise SystemExit("assembled beliefs are empty")
    return rows

def build_tasks(rows: list[dict]) -> dict[tuple, dict]:
    tasks = {}
    qwen = "Qwen/Qwen3-8B-AWQ"
    for row in rows:
        model = row["model"]
        systems = CONTROLLERS_QWEN if model == qwen else CONTROLLERS_REPLICATION
        for controller in systems:
            add_task(
                tasks, model=model, system=row["system"],
                comparison_system=row["system"], consumer=row["consumer"],
                controller=controller, method="always_consume",
                condition="persistent", horizon=40,
                query_id=row["query_id"], true_regime=row["true_regime"],
                seed=0, probs=as_probs(row),
            )
            if model == qwen and row["system"] == "rerank":
                add_task(
                    tasks, model=model, system=row["system"],
                    comparison_system=row["system"], consumer=row["consumer"],
                    controller=controller, method="selective_entropy",
                    condition="persistent", horizon=40,
                    query_id=row["query_id"], true_regime=row["true_regime"],
                    seed=0, probs=as_probs(row), selective=True,
                )
    # Expand the unique arm specifications to the 20 preregistered paired seeds.
    base = list(tasks.values())
    tasks = {}
    for task in base:
        for seed in SEEDS:
            task_copy = dict(task)
            task_copy["seed"] = seed
            tasks[task_key(task_copy)] = task_copy

    # Add the primary persistence arm and its matched NoInfo references.
    persistence_rows = [
        row for row in rows
        if row["model"] == qwen and row["system"] == "rerank"
        and row["consumer"] == "C1"
    ]
    for row in persistence_rows:
        probs = as_probs(row)
        for horizon in HORIZONS:
            for condition in ("persistent", "reset"):
                for system, system_probs, method in (
                    ("rerank", probs, "always_consume"),
                    ("noinfo", PRIOR, "noinfo"),
                ):
                    for seed in SEEDS:
                        add_task(
                            tasks, model=qwen, system=system,
                            comparison_system="rerank", consumer="C1",
                            controller="CausalOptimizer", method=method,
                            condition=condition, horizon=horizon,
                            query_id=row["query_id"],
                            true_regime=row["true_regime"], seed=seed,
                            probs=system_probs,
                            selective=False,
                            release_t=RELEASE_T if system == "rerank" else horizon + 1,
                        )
    # Add NoInfo references for every ordinary arm. Their key is shared across
    # retrieval systems, so repeated references are deduplicated by task_key.
    for row in rows:
        model = row["model"]
        controllers = CONTROLLERS_QWEN if model == qwen else CONTROLLERS_REPLICATION
        for controller in controllers:
            for seed in SEEDS:
                add_task(
                    tasks, model=model, system="noinfo",
                    comparison_system=row["system"], consumer=row["consumer"],
                    controller=controller, method="noinfo",
                    condition="persistent", horizon=40,
                    query_id=row["query_id"], true_regime=row["true_regime"],
                    seed=seed, probs=PRIOR, release_t=41,
                )
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["protocol_hash"] != "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022":
        raise SystemExit("protocol hash drift")
    rows = load_belief_rows()
    tasks = build_tasks(rows)
    print(f"tasks: {len(tasks)}", flush=True)
    with mp.Pool(args.workers) as pool:
        results = list(pool.imap(run_task, list(tasks.values()), chunksize=8))
    frame = pd.DataFrame(results).sort_values(
        ["model", "system", "comparison_system", "consumer", "controller",
         "method", "condition", "horizon", "query_id", "seed"]
    ).reset_index(drop=True)
    if frame.duplicated([
        "model", "system", "comparison_system", "consumer", "controller",
        "method", "condition", "horizon", "query_id", "seed"
    ]).any():
        raise SystemExit("duplicate episode key")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-confirmation-simulation",
        "status": "frozen",
        "protocol_hash": protocol["protocol_hash"],
        "qrels_read": False,
        "seeds": SEEDS,
        "horizons": HORIZONS,
        "release_t": RELEASE_T,
        "reset_definition": (
            "one-step rollouts reset inventory/pipeline/controller while replaying "
            "the aligned hidden demand draw at each original time"
        ),
        "n_tasks": len(tasks),
        "n_episode_rows": len(frame),
        "input_sha256": {"beliefs": sha(BELIEFS)},
        "script_sha256": sha(Path(__file__).resolve()),
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha(OUT),
    }
    manifest_path = ROOT / "versions/sem2act-v5/manifests/simulation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
