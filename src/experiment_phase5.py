"""Phase 5: Multi-Regime Information-Value Experiment.

Tests whether advance semantic information has decision value in a
multi-regime operational environment where the true regime is hidden.

Design:
- 3 latent regimes: Normal, SupplierDelay, DemandSurge
- Warning at t=12; event starts at t=20; horizon=40
- NoInfo uses prior; PerfectSemantic knows true regime
- CausalOptimizer plans under beliefs
- OIV = J_PerfectSemantic,O - J_NoInfo,O (information value)
- SIVR_O = (J_sensor,O - J_NoInfo,O) / (J_PerfectSemantic,O - J_NoInfo,O)

Gate A: OIV must be consistently positive and economically meaningful.
Gate B: trajectory analysis must show anticipatory action before event onset.
Only after both gates pass: evaluate RuleBased + LLM sensors.
"""

from __future__ import annotations

import csv
import json
import os
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.events import (
    Regime, REGIME_PRIOR, REGIME_PARAMS, REGIME_WARNING_TEMPLATES,
    P5_HORIZON, P5_NORMAL_LEAD_TIME, P5_DEMAND_MEAN,
    P5_EVENT_START, P5_EVENT_DURATION, P5_WARNING_TIME,
)
from src.env import (
    InventoryEnv, InventoryState, CausalOptimizer, HindsightOracle,
    REVENUE_PER_UNIT, ORDERING_FIXED_COST, ORDERING_VARIABLE_COST,
    HOLDING_COST_PER_UNIT, STOCKOUT_COST_PER_UNIT,
    INITIAL_INVENTORY, HORIZON, DEMAND_MEAN,
)
from src.interpreter import (
    RegimeInterpretation, no_info_regime_belief, perfect_semantic_regime_belief,
    rule_based_regime_extract, llm_regime_interpret_with_result, load_dotenv,
)
from src.metrics import compute_episode_metrics, information_value, signed_sivr

# Results directory
default_results_dir = Path(__file__).resolve().parent.parent / "results" / "phase5"
RESULTS_DIR = Path(os.environ.get("PAPER2_PHASE5_RESULTS_DIR", str(default_results_dir)))

# --- Sensor/Controller labels ---
SENSOR_NOINFO = "NoInfo"
SENSOR_PERFECT = "PerfectSemantic"
SENSOR_RULEBASED = "RuleBased"
SENSOR_LLM = "LLM"
CONTROLLER_OPTIMIZER = "CausalOptimizer"
CONTROLLER_HEURISTIC = "Heuristic"
CONDITION_HINDSIGHT = "HindsightOracle"


@dataclass
class P5EpisodeResult:
    """Results from a single Phase 5 episode."""
    seed: int
    regime: str
    template_id: str
    ambiguity_level: str
    sensor: str
    controller: str
    total_profit: float
    fill_rate: float
    total_lost_sales: float
    avg_inventory: float
    periods_stockout: float
    # Regime belief
    belief_normal: float
    belief_delay: float
    belief_surge: float
    most_likely_regime: str
    belief_correct: bool
    # Parameters
    est_lt_increase: int
    est_duration: int
    est_demand_multiplier: float


def _run_p5_episode(
    seed: int,
    regime: Regime,
    sensor: str,
    controller: str,
    template_id: str = "",
    ambiguity_level: str = "",
    regime_interp: Optional[RegimeInterpretation] = None,
    warning_time: int = P5_WARNING_TIME,
    event_start: int = P5_EVENT_START,
    event_duration: int = P5_EVENT_DURATION,
) -> tuple[P5EpisodeResult, list[InventoryState]]:
    """Run a single Phase 5 episode.

    Returns (result, history) where history is the full list of period states.
    """
    # Create environment with the true regime
    env = InventoryEnv(seed=seed, regime=regime)
    env.reset()

    # Create CausalOptimizer with belief-based planning
    if regime_interp is None:
        regime_interp = no_info_regime_belief()

    belief = regime_interp.normalized()
    regime_probs = belief.regime_probabilities

    # Warning-release protocol.  The text is published at `warning_time`, so
    # before that period NO sensor has read anything and every controller must
    # plan under the benchmark prior.  From `warning_time` onward the
    # interpreted belief replaces the prior while the operational state and
    # in-transit pipeline carry over unchanged.
    #
    # Until September 2026 this parameter was accepted and then ignored: the
    # interpreted belief was applied from t=0, which granted every semantic
    # sensor eight periods of advance notice the protocol did not give it.
    prior_probs = no_info_regime_belief().normalized().regime_probabilities
    warning_released = warning_time <= 0

    if controller == CONTROLLER_OPTIMIZER:
        optimizer = CausalOptimizer(
            seed=seed,
            regime_probabilities=regime_probs if warning_released else prior_probs,
            horizon=P5_HORIZON,
            initial_inventory=INITIAL_INVENTORY,
            demand_mean=P5_DEMAND_MEAN,
        )
    else:
        optimizer = None

    history: list[InventoryState] = []

    for t in range(P5_HORIZON):
        state = env._state

        if not warning_released and t >= warning_time:
            warning_released = True
            if optimizer is not None:
                optimizer.set_regime_probabilities(regime_probs)

        if controller == CONTROLLER_OPTIMIZER and optimizer is not None:
            order_qty = optimizer.decide(state)
        else:
            # Simple base-stock heuristic
            target = P5_DEMAND_MEAN * P5_NORMAL_LEAD_TIME + 1.65 * np.sqrt(
                P5_NORMAL_LEAD_TIME * P5_DEMAND_MEAN
            )
            inv_pos = state.on_hand + state.pipeline
            order_qty = max(0.0, target - inv_pos)

        next_state = env.step(order_qty)
        history.append(next_state)

    # Compute metrics
    metrics = compute_episode_metrics(history, seed=seed, condition=f"{sensor}+{controller}")
    belief_correct = belief.most_likely_regime == regime.value

    result = P5EpisodeResult(
        seed=seed,
        regime=regime.value,
        template_id=template_id,
        ambiguity_level=ambiguity_level,
        sensor=sensor,
        controller=controller,
        total_profit=metrics.total_profit,
        fill_rate=metrics.fill_rate,
        total_lost_sales=metrics.total_lost_sales,
        avg_inventory=metrics.avg_inventory,
        periods_stockout=metrics.periods_stockout,
        belief_normal=regime_probs.get(Regime.NORMAL.value, 0.0),
        belief_delay=regime_probs.get(Regime.SUPPLIER_DELAY.value, 0.0),
        belief_surge=regime_probs.get(Regime.DEMAND_SURGE.value, 0.0),
        most_likely_regime=belief.most_likely_regime,
        belief_correct=belief_correct,
        est_lt_increase=belief.estimated_lt_increase,
        est_duration=belief.estimated_duration,
        est_demand_multiplier=belief.estimated_demand_multiplier,
    )

    return result, history


def run_information_value_grid(
    num_seeds: int = 15,
    seed_base: int = 1000,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the predefined information-value grid.

    Evaluates NoInfo+CausalOptimizer vs PerfectSemantic+CausalOptimizer
    across all regimes. This is Gate A.

    Returns dict with grid results and OIV metrics.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    regimes = [Regime.NORMAL, Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]
    seeds = [seed_base + i for i in range(num_seeds)]

    # Results storage
    grid_results = []
    episode_results = []
    all_histories = {}

    print("Phase 5 Information-Value Grid")
    print(f"Regimes: {[r.value for r in regimes]}")
    print(f"Seeds: {num_seeds} | Warning time: t={P5_WARNING_TIME}")
    print(f"Event start: t={P5_EVENT_START} | Duration: {P5_EVENT_DURATION}")
    print(f"Normal LT: {P5_NORMAL_LEAD_TIME} | Base demand: {P5_DEMAND_MEAN}")
    print()

    for regime in regimes:
        no_info_profits = []
        perfect_profits = []
        hindsight_profits = []

        for seed in seeds:
            # NoInfo + CausalOptimizer (uses prior)
            ni_result, ni_hist = _run_p5_episode(
                seed=seed, regime=regime,
                sensor=SENSOR_NOINFO, controller=CONTROLLER_OPTIMIZER,
                regime_interp=no_info_regime_belief(),
            )
            no_info_profits.append(ni_result.total_profit)
            episode_results.append(ni_result)
            all_histories[f"no_info_{regime.value}_{seed}"] = ni_hist

            # PerfectSemantic + CausalOptimizer (knows true regime)
            ps_interp = perfect_semantic_regime_belief(regime)
            ps_result, ps_hist = _run_p5_episode(
                seed=seed, regime=regime,
                sensor=SENSOR_PERFECT, controller=CONTROLLER_OPTIMIZER,
                regime_interp=ps_interp,
            )
            perfect_profits.append(ps_result.total_profit)
            episode_results.append(ps_result)
            all_histories[f"perfect_{regime.value}_{seed}"] = ps_hist

            # HindsightOracle
            ho = HindsightOracle(seed=seed, regime=regime)
            ho_history, ho_orders = ho.compute_optimal_trajectory()
            ho_profit = sum(s.period_profit for s in ho_history)
            hindsight_profits.append(ho_profit)

        ni_mean = np.mean(no_info_profits)
        ps_mean = np.mean(perfect_profits)
        ho_mean = np.mean(hindsight_profits)
        oiv = ps_mean - ni_mean

        grid_results.append({
            "regime": regime.value,
            "noinfo_profit": float(ni_mean),
            "perfect_profit": float(ps_mean),
            "hindsight_profit": float(ho_mean),
            "oiv": float(oiv),
            "oiv_pct": float(oiv / ho_mean * 100) if ho_mean > 0 else 0.0,
        })

        print(f"  {regime.value:20s}: NoInfo={ni_mean:8.1f}  Perfect={ps_mean:8.1f}  "
              f"Hindsight={ho_mean:8.1f}  OIV={oiv:+8.1f} ({oiv/ho_mean*100:+.1f}%)")

    # Save grid results
    with open(output_dir / "information_value_grid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "regime", "noinfo_profit", "perfect_profit", "hindsight_profit",
            "oiv", "oiv_pct",
        ])
        writer.writeheader()
        writer.writerows(grid_results)

    # Save episode results
    _save_p5_episodes(episode_results, output_dir)

    # Gate A check
    positive_oiv = sum(1 for g in grid_results if g["oiv"] > 10.0)
    total_regimes = len(grid_results)
    gate_a_passed = positive_oiv >= 2  # at least 2 of 3 regimes show positive OIV

    print(f"\nGate A: {positive_oiv}/{total_regimes} regimes show positive OIV")
    print(f"Gate A {'PASSED' if gate_a_passed else 'FAILED'}")

    return {
        "grid_results": grid_results,
        "episode_results": episode_results,
        "gate_a_passed": gate_a_passed,
    }


def run_trajectory_diagnostics(
    num_seeds: int = 5,
    seed_base: int = 1000,
    output_dir: Optional[Path] = None,
) -> dict:
    """Diagnose WHY perfect information helps by comparing trajectories.

    Compares NoInfo vs PerfectSemantic order trajectories around
    warning/event periods for representative paired episodes.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    regimes = [Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]
    seeds = [seed_base + i for i in range(num_seeds)]

    diagnostics = []

    print("\nTrajectory Diagnostics")
    print(f"Comparing NoInfo vs PerfectSemantic order trajectories")

    for regime in regimes:
        for seed in seeds[:3]:  # 3 representative seeds
            # NoInfo trajectory
            ni_result, ni_hist = _run_p5_episode(
                seed=seed, regime=regime,
                sensor=SENSOR_NOINFO, controller=CONTROLLER_OPTIMIZER,
                regime_interp=no_info_regime_belief(),
            )

            # PerfectSemantic trajectory
            ps_interp = perfect_semantic_regime_belief(regime)
            ps_result, ps_hist = _run_p5_episode(
                seed=seed, regime=regime,
                sensor=SENSOR_PERFECT, controller=CONTROLLER_OPTIMIZER,
                regime_interp=ps_interp,
            )

            # Compare around warning/event periods
            for t in range(max(0, P5_WARNING_TIME - 2), min(P5_HORIZON, P5_EVENT_START + 5)):
                ni_state = ni_hist[t]
                ps_state = ps_hist[t]

                diagnostics.append({
                    "regime": regime.value,
                    "seed": seed,
                    "period": t,
                    "phase": (
                        "pre_warning" if t < P5_WARNING_TIME
                        else "post_warning_pre_event" if t < P5_EVENT_START
                        else "during_event"
                    ),
                    "ni_order": ni_state.order_quantity,
                    "ps_order": ps_state.order_quantity,
                    "ni_on_hand": ni_state.on_hand,
                    "ps_on_hand": ps_state.on_hand,
                    "ni_pipeline": ni_state.pipeline,
                    "ps_pipeline": ps_state.pipeline,
                    "ni_lost_sales": ni_state.lost_sales,
                    "ps_lost_sales": ps_state.lost_sales,
                    "order_diff": ps_state.order_quantity - ni_state.order_quantity,
                    "inventory_diff": ps_state.on_hand - ni_state.on_hand,
                })

            # Summary
            pre_event_ni_orders = [h.order_quantity for h in ni_hist[P5_WARNING_TIME:P5_EVENT_START]]
            pre_event_ps_orders = [h.order_quantity for h in ps_hist[P5_WARNING_TIME:P5_EVENT_START]]
            during_event_ni_lost = sum(h.lost_sales for h in ni_hist[P5_EVENT_START:P5_EVENT_START+5])
            during_event_ps_lost = sum(h.lost_sales for h in ps_hist[P5_EVENT_START:P5_EVENT_START+5])

            print(f"  {regime.value} seed={seed}: "
                  f"Pre-event orders NI={np.mean(pre_event_ni_orders):.1f} PS={np.mean(pre_event_ps_orders):.1f} "
                  f"| Event lost-sales NI={during_event_ni_lost:.0f} PS={during_event_ps_lost:.0f}")

    # Save diagnostics
    if diagnostics:
        with open(output_dir / "trajectory_diagnostics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=diagnostics[0].keys())
            writer.writeheader()
            writer.writerows(diagnostics)

    # Gate B check: PerfectSemantic must order differently before event
    pre_event_diffs = [d["order_diff"] for d in diagnostics if d["phase"] == "post_warning_pre_event"]
    mean_diff = np.mean(pre_event_diffs) if pre_event_diffs else 0.0
    gate_b_passed = abs(mean_diff) > 1.0  # at least 1 unit average difference

    print(f"\nGate B: Mean pre-event order difference = {mean_diff:+.1f}")
    print(f"Gate B {'PASSED' if gate_b_passed else 'FAILED'}")

    return {
        "diagnostics": diagnostics,
        "gate_b_passed": gate_b_passed,
        "mean_pre_event_order_diff": float(mean_diff),
    }


def run_semantic_sensor_evaluation(
    num_seeds: int = 15,
    seed_base: int = 1000,
    output_dir: Optional[Path] = None,
    llm_models: Optional[list[str]] = None,
) -> dict:
    """Evaluate RuleBased and LLM sensors under the frozen benchmark.

    Only called after Gate A and Gate B pass.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
        os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"

    regimes = [Regime.NORMAL, Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]
    seeds = [seed_base + i for i in range(num_seeds)]

    sensors = [SENSOR_NOINFO, SENSOR_RULEBASED, SENSOR_PERFECT]
    if llm_models:
        for model in llm_models:
            sensors.append(f"{SENSOR_LLM}:{model}")

    episode_results = []
    sensor_results = defaultdict(lambda: defaultdict(list))
    regime_beliefs = []

    print("\nPhase 5 Semantic Sensor Evaluation")
    print(f"Sensors: {sensors}")

    for regime in regimes:
        # Get templates for this regime
        regime_templates = [
            t for t in REGIME_WARNING_TEMPLATES if t["regime"] == regime
        ]

        for seed in seeds:
            for tmpl in regime_templates:
                text = tmpl["text"]
                tid = tmpl["template_id"]
                alevel = tmpl["ambiguity_level"]

                for sensor in sensors:
                    # Get regime interpretation from sensor
                    if sensor == SENSOR_NOINFO:
                        ri = no_info_regime_belief()
                    elif sensor == SENSOR_PERFECT:
                        ri = perfect_semantic_regime_belief(regime)
                    elif sensor == SENSOR_RULEBASED:
                        ri = rule_based_regime_extract(text)
                    elif sensor.startswith(f"{SENSOR_LLM}:"):
                        model = sensor.split(":", 1)[1]
                        ri, _ = llm_regime_interpret_with_result(text, model=model)
                    else:
                        ri = no_info_regime_belief()

                    # Run episode
                    result, history = _run_p5_episode(
                        seed=seed, regime=regime,
                        sensor=sensor, controller=CONTROLLER_OPTIMIZER,
                        template_id=tid, ambiguity_level=alevel,
                        regime_interp=ri,
                    )
                    episode_results.append(result)
                    sensor_results[sensor][regime.value].append(result.total_profit)

                    # Record belief
                    belief = ri.normalized()
                    regime_beliefs.append({
                        "sensor": sensor,
                        "regime": regime.value,
                        "template_id": tid,
                        "seed": seed,
                        "belief_normal": belief.regime_probabilities.get("normal", 0.0),
                        "belief_delay": belief.regime_probabilities.get("supplier_delay", 0.0),
                        "belief_surge": belief.regime_probabilities.get("demand_surge", 0.0),
                        "most_likely": belief.most_likely_regime,
                        "correct": belief.most_likely_regime == regime.value,
                    })

    # Compute SIVR_O per sensor
    no_info_by_regime = {
        r.value: np.mean(sensor_results[SENSOR_NOINFO][r.value])
        for r in regimes if sensor_results[SENSOR_NOINFO][r.value]
    }
    perfect_by_regime = {
        r.value: np.mean(sensor_results[SENSOR_PERFECT][r.value])
        for r in regimes if sensor_results[SENSOR_PERFECT][r.value]
    }

    sivr_results = []
    for sensor in sensors:
        for regime in regimes:
            ni_profit = no_info_by_regime.get(regime.value, 0.0)
            ps_profit = perfect_by_regime.get(regime.value, 0.0)
            sensor_profit = np.mean(sensor_results[sensor][regime.value]) if sensor_results[sensor][regime.value] else 0.0
            sivr = signed_sivr(sensor_profit, ni_profit, ps_profit).value

            sivr_results.append({
                "sensor": sensor,
                "regime": regime.value,
                "noinfo_profit": float(ni_profit),
                "perfect_profit": float(ps_profit),
                "sensor_profit": float(sensor_profit),
                "sivr": float(sivr),
            })

    # Save results
    _save_p5_episodes(episode_results, output_dir / "semantic_sensor_results.csv")

    with open(output_dir / "sivr_optimizer.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sensor", "regime", "noinfo_profit", "perfect_profit",
            "sensor_profit", "sivr",
        ])
        writer.writeheader()
        writer.writerows(sivr_results)

    with open(output_dir / "belief_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=regime_beliefs[0].keys())
        writer.writeheader()
        writer.writerows(regime_beliefs)

    # Print summary
    print("\nSIVR_O by Sensor and Regime:")
    for row in sivr_results:
        print(f"  {row['sensor']:20s} {row['regime']:15s} SIVR={row['sivr']:.3f} "
              f"(sensor={row['sensor_profit']:.1f} ni={row['noinfo_profit']:.1f} "
              f"ps={row['perfect_profit']:.1f})")

    # Belief accuracy
    for sensor in sensors:
        sensor_beliefs = [b for b in regime_beliefs if b["sensor"] == sensor]
        if sensor_beliefs:
            acc = np.mean([b["correct"] for b in sensor_beliefs])
            print(f"  {sensor:20s} belief accuracy: {acc:.3f}")

    return {
        "episode_results": episode_results,
        "sivr_results": sivr_results,
        "regime_beliefs": regime_beliefs,
    }


def _save_p5_episodes(results: list[P5EpisodeResult], path: Path) -> None:
    """Save Phase 5 episode results to CSV."""
    if isinstance(path, Path) and path.suffix == ".csv":
        filepath = path
    else:
        filepath = Path(path) / "episode_results.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "regime", "template_id", "ambiguity_level",
            "sensor", "controller", "total_profit", "fill_rate",
            "total_lost_sales", "avg_inventory", "periods_stockout",
            "belief_normal", "belief_delay", "belief_surge",
            "most_likely_regime", "belief_correct",
            "est_lt_increase", "est_duration", "est_demand_multiplier",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def run_phase5(
    num_seeds: int = 15,
    seed_base: int = 1000,
    llm_models: Optional[list[str]] = None,
) -> dict:
    """Run the complete Phase 5 experiment pipeline.

    Steps:
    1. Gate A: information value grid
    2. Gate B: trajectory diagnostics
    3. If both pass: semantic sensor evaluation
    """
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 5: MULTI-REGIME INFORMATION-VALUE EXPERIMENT")
    print("=" * 70)
    print(f"Regimes: {[r.value for r in Regime]}")
    print(f"Prior: {REGIME_PRIOR}")
    print(f"Horizon: {P5_HORIZON} | Normal LT: {P5_NORMAL_LEAD_TIME}")
    print(f"Warning: t={P5_WARNING_TIME} | Event: t={P5_EVENT_START} dur={P5_EVENT_DURATION}")
    print()

    # Gate A: Information value grid
    gate_a = run_information_value_grid(num_seeds=num_seeds, seed_base=seed_base)

    # Gate B: Trajectory diagnostics
    gate_b = run_trajectory_diagnostics(num_seeds=min(num_seeds, 5), seed_base=seed_base)

    # Gate check
    if not gate_a["gate_a_passed"]:
        print("\n" + "=" * 70)
        print("GATE A FAILED: Insufficient information value for advance sensing.")
        print("Diagnose why advance information does not change optimal decisions.")
        print("=" * 70)
        return {"gate_a": gate_a, "gate_b": gate_b, "semantic_eval": None}

    if not gate_b["gate_b_passed"]:
        print("\n" + "=" * 70)
        print("GATE B FAILED: Perfect information does not change actions before event.")
        print("Cannot claim advance-information value.")
        print("=" * 70)
        return {"gate_a": gate_a, "gate_b": gate_b, "semantic_eval": None}

    print("\n" + "=" * 70)
    print("GATES A AND B PASSED — Proceeding to semantic sensor evaluation")
    print("=" * 70)

    # Semantic sensor evaluation
    semantic_eval = run_semantic_sensor_evaluation(
        num_seeds=num_seeds, seed_base=seed_base, llm_models=llm_models,
    )

    return {
        "gate_a": gate_a,
        "gate_b": gate_b,
        "semantic_eval": semantic_eval,
    }


if __name__ == "__main__":
    import sys
    use_llm = bool(os.environ.get("LLM_API_KEY"))
    models = None
    if use_llm and len(sys.argv) > 1:
        models = sys.argv[1:]
    elif use_llm:
        models = ["gpt-4o-mini"]

    run_phase5(llm_models=models)
