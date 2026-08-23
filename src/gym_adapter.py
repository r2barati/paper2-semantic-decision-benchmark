"""Phase 8: Adapter for Paper-1 Gymnasium inventory environment.

Creates a bridge between the semantic interpreter output (regime belief)
and the Paper-1 CoreEnv controller, without modifying Paper-1 code.

Architecture:
  Warning text
    -> Semantic interpreter
    -> Probabilistic regime belief (Normal / DemandSurge)
    -> BeliefAdapter (maps belief to controller params)
    -> Paper-1 CoreEnv step()
    -> Operational return
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PAPER1_PATH = Path("/Users/sanamimani/Paper 1/final-github-clean/gym-invmgmt-paper")
if str(PAPER1_PATH) not in sys.path:
    sys.path.insert(0, str(PAPER1_PATH))

from gym_invmgmt.core_env import CoreEnv

from src.events import Regime, REGIME_PRIOR
from src.interpreter import RegimeInterpretation, no_info_regime_belief


REGIME_LIST_2 = [Regime.NORMAL, Regime.DEMAND_SURGE]
REGIME_LABELS_2 = [r.value for r in REGIME_LIST_2]

DEFAULT_BASE_MU = 10.0
SURGE_MULTIPLIER = 2.0
DEFAULT_NUM_PERIODS = 30
DEFAULT_EVENT_START = 15
DEFAULT_EVENT_DURATION = 15
DEFAULT_WARNING_TIME = 10
DEFAULT_SCENARIO = "serial"


@dataclass
class GymEpisodeResult:
    seed: int
    regime: str
    sensor: str
    total_reward: float
    total_profit: float
    total_demand: float
    total_sold: float
    total_backlog: float
    fill_rate: float
    avg_inventory: float
    inventory_position_final: float
    belief_normal: float
    belief_surge: float
    most_likely_regime: str
    belief_correct: bool
    periods: int


def make_env(
    seed: int,
    scenario: str = DEFAULT_SCENARIO,
    num_periods: int = DEFAULT_NUM_PERIODS,
    base_mu: float = DEFAULT_BASE_MU,
    backlog: bool = True,
) -> CoreEnv:
    demand_config = {
        "type": "stationary",
        "base_mu": base_mu,
        "use_goodwill": False,
    }
    env = CoreEnv(
        scenario=scenario,
        demand_config=demand_config,
        num_periods=num_periods,
        backlog=backlog,
    )
    env.reset(seed=seed)
    return env


def make_surge_env(
    seed: int,
    scenario: str = DEFAULT_SCENARIO,
    num_periods: int = DEFAULT_NUM_PERIODS,
    base_mu: float = DEFAULT_BASE_MU,
    shock_mag: float = SURGE_MULTIPLIER,
    shock_time: int = DEFAULT_EVENT_START,
    backlog: bool = True,
) -> CoreEnv:
    demand_config = {
        "type": "stationary",
        "base_mu": base_mu,
        "use_goodwill": False,
        "effects": ["shock"],
        "shock_time": shock_time,
        "shock_mag": shock_mag,
    }
    env = CoreEnv(
        scenario=scenario,
        demand_config=demand_config,
        num_periods=num_periods,
        backlog=backlog,
    )
    env.reset(seed=seed)
    return env


class BeliefAdaptiveController:
    """Maps a 2-class regime belief to a base-stock order-up-to policy.

    The effective demand rate is computed as:
        effective_mu = P(Normal) * base_mu + P(DemandSurge) * base_mu * surge_multiplier

    For each managed node, the order-up-to target is:
        target = effective_mu * avg_incoming_lead_time + safety_factor * sqrt(avg_lt * effective_mu)

    At each step, the order quantity for each reorder link is:
        order = max(0, target - inventory_position)
    where inventory_position = on_hand + in_transit (pipeline).
    """

    def __init__(
        self,
        env: CoreEnv,
        regime_probabilities: dict[str, float],
        base_mu: float = DEFAULT_BASE_MU,
        surge_multiplier: float = SURGE_MULTIPLIER,
        safety_factor: float = 1.65,
    ):
        self.env = env
        self.regime_probs = dict(regime_probabilities)
        self.base_mu = base_mu
        self.surge_multiplier = surge_multiplier
        self.safety_factor = safety_factor
        self.network = env.network
        self._precompute_params()

    def _precompute_params(self):
        p_normal = self.regime_probs.get("normal", 0.5)
        p_surge = self.regime_probs.get("demand_surge", 0.5)
        self.effective_mu = p_normal * self.base_mu + p_surge * self.base_mu * self.surge_multiplier

        self.node_targets = {}
        for node in self.network.main_nodes:
            successors = list(self.network.graph.successors(node))
            if not successors:
                self.node_targets[node] = 0.0
                continue

            total_downstream_demand = 0.0
            for succ in successors:
                edge = self.network.graph.edges.get((node, succ))
                if edge and ("demand_dist" in edge or "p" in edge):
                    total_downstream_demand += self.effective_mu

            lead_times = []
            for pred_idx, reorder_idx, L in self.network.pred_reorder_indices.get(node, []):
                lead_times.append(L)
            avg_lt = np.mean(lead_times) if lead_times else 2.0

            node_C = self.network.graph.nodes[node].get("C", float("inf"))
            safety_stock = self.safety_factor * np.sqrt(max(avg_lt, 1) * self.effective_mu)
            pipeline = total_downstream_demand * avg_lt
            target = pipeline + safety_stock
            if node_C < float("inf"):
                target = min(target, node_C * 3)
            self.node_targets[node] = target

    def get_action(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        n_actions = len(self.network.reorder_links)
        actions = np.zeros(n_actions)

        for i, (supplier, buyer) in enumerate(self.network.reorder_links):
            target = self.node_targets.get(buyer, 0.0)

            buyer_idx = self.network.node_map.get(buyer, None)
            if buyer_idx is None:
                actions[i] = self.effective_mu
                continue

            current_inv = float(self.env.X[self.env.period, buyer_idx])

            pipeline = 0.0
            for pred_idx, reorder_idx, L in self.network.pred_reorder_indices.get(buyer, []):
                if L > 0:
                    pipeline += float(self.env.Y[self.env.period, reorder_idx])

            inv_pos = current_inv + pipeline
            order = max(0.0, target - inv_pos)
            actions[i] = order

        return actions

    def __call__(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        return self.get_action(obs, current_period)


def _build_controller(
    env: CoreEnv,
    regime_probs: dict[str, float],
    base_mu: float = DEFAULT_BASE_MU,
) -> BeliefAdaptiveController:
    """Build a controller bound to a specific env instance."""
    return BeliefAdaptiveController(env, regime_probs, base_mu=base_mu)


def run_gym_episode(
    seed: int,
    regime: Regime,
    sensor: str,
    regime_probabilities: dict[str, float],
    scenario: str = DEFAULT_SCENARIO,
    num_periods: int = DEFAULT_NUM_PERIODS,
    base_mu: float = DEFAULT_BASE_MU,
    event_start: int = DEFAULT_EVENT_START,
    shock_mag: float = SURGE_MULTIPLIER,
) -> tuple[GymEpisodeResult, list]:
    """Run one episode: create env + controller from regime belief, execute.

    The controller is built AFTER the env is created, ensuring they share
    the same network topology and internal state.
    """
    if regime == Regime.DEMAND_SURGE:
        env = make_surge_env(
            seed=seed, scenario=scenario, num_periods=num_periods,
            base_mu=base_mu, shock_mag=shock_mag, shock_time=event_start,
        )
    else:
        env = make_env(
            seed=seed, scenario=scenario, num_periods=num_periods,
            base_mu=base_mu,
        )

    controller = _build_controller(env, regime_probabilities, base_mu=base_mu)

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    trajectory = []

    for t in range(num_periods):
        action = controller.get_action(obs, t)
        action = np.clip(action, 0, float(env.action_space.high[0]))
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward

        trajectory.append({
            "period": t,
            "obs": obs.copy(),
            "action": action.copy(),
            "reward": reward,
            "inventory": env.X[t + 1].copy() if t + 1 < env.X.shape[0] else env.X[-1].copy(),
        })

        obs = next_obs
        if truncated:
            break

    total_demand = float(np.sum(env.D))
    total_sold = float(np.sum(env.R))
    total_backlog = float(np.sum(env.U))
    fill_rate = total_sold / total_demand if total_demand > 0 else 1.0
    avg_inventory = float(np.mean(env.X[1:env.period + 1])) if env.period > 0 else 0.0

    node_inv_pos = 0.0
    for node in env.network.main_nodes:
        node_i = env.network.node_map[node]
        node_inv_pos += env.X[env.period, node_i]
        for pred_idx, reorder_idx, L in env.network.pred_reorder_indices.get(node, []):
            node_inv_pos += env.Y[env.period, reorder_idx]

    belief_normal = regime_probabilities.get("normal", 0.0)
    belief_surge = regime_probabilities.get("demand_surge", 0.0)
    most_likely = "normal" if belief_normal >= belief_surge else "demand_surge"

    result = GymEpisodeResult(
        seed=seed,
        regime=regime.value,
        sensor=sensor,
        total_reward=float(total_reward),
        total_profit=float(total_reward),
        total_demand=total_demand,
        total_sold=total_sold,
        total_backlog=total_backlog,
        fill_rate=fill_rate,
        avg_inventory=avg_inventory,
        inventory_position_final=node_inv_pos,
        belief_normal=belief_normal,
        belief_surge=belief_surge,
        most_likely_regime=most_likely,
        belief_correct=(most_likely == regime.value),
        periods=env.period,
    )

    return result, trajectory


def get_no_info_probs() -> dict[str, float]:
    belief = no_info_regime_belief()
    p_surge = belief.regime_probabilities.get("demand_surge", 0.0)
    return {"normal": 1.0 - p_surge, "demand_surge": p_surge}


def get_perfect_semantic_probs(true_regime: Regime) -> dict[str, float]:
    probs = {r.value: 0.0 for r in REGIME_LIST_2}
    if true_regime.value in probs:
        probs[true_regime.value] = 1.0
    else:
        probs["normal"] = 1.0
    return probs


def get_rulebased_probs(text: str) -> dict[str, float]:
    from src.interpreter import rule_based_regime_extract
    ri = rule_based_regime_extract(text)
    belief = ri.normalized()
    p_normal = belief.regime_probabilities.get("normal", 0.0)
    p_surge = belief.regime_probabilities.get("demand_surge", 0.0)
    total = p_normal + p_surge
    if total > 0:
        return {"normal": p_normal / total, "demand_surge": p_surge / total}
    return {"normal": 0.5, "demand_surge": 0.5}


def get_tfidf_probs(text: str, model) -> dict[str, float]:
    result = model.predict_regime(text)
    ri = result.to_regime_interpretation()
    belief = ri.normalized()
    p_normal = belief.regime_probabilities.get("normal", 0.0)
    p_surge = belief.regime_probabilities.get("demand_surge", 0.0)
    total = p_normal + p_surge
    if total > 0:
        return {"normal": p_normal / total, "demand_surge": p_surge / total}
    return {"normal": 0.5, "demand_surge": 0.5}


def get_llm_probs(text: str, model: str = "gpt-4o") -> dict[str, float]:
    from src.interpreter import llm_regime_interpret_with_result
    ri, _ = llm_regime_interpret_with_result(text, model=model)
    belief = ri.normalized()
    p_normal = belief.regime_probabilities.get("normal", 0.0)
    p_surge = belief.regime_probabilities.get("demand_surge", 0.0)
    total = p_normal + p_surge
    if total > 0:
        return {"normal": p_normal / total, "demand_surge": p_surge / total}
    return {"normal": 0.5, "demand_surge": 0.5}
