"""Phase 9A: Adapter for SupplierCapacityDrop event on divergent topology.

Adds supply-side disruption without modifying Paper-1 code by wrapping
the environment step to temporarily reduce factory capacity during shock periods.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER1_PATH = Path(os.environ.get(
    "GYM_INVMGMT_PATH",
    REPO_ROOT / "third_party" / "gym-invmgmt-paper",
))
if str(PAPER1_PATH) not in sys.path:
    sys.path.insert(0, str(PAPER1_PATH))

from gym_invmgmt.core_env import CoreEnv

from src.events import Regime

DEFAULT_SCENARIO = "network"
DEFAULT_NUM_PERIODS = 30
DEFAULT_BASE_MU = 50.0
DEFAULT_NORMAL_CAPACITY = 90
DEFAULT_DISRUPTED_CAPACITY = 30
DEFAULT_EVENT_START = 10
DEFAULT_EVENT_END = 25
DEFAULT_AFFECTED_FACTORY = 4


@dataclass
class GymEpisodeResult9:
    seed: int
    regime: str
    sensor: str
    total_reward: float
    total_demand: float
    total_retail_sales: float
    fill_rate: float
    avg_inventory: float
    inventory_position_final: float
    belief_normal: float
    belief_capacity_drop: float
    most_likely_regime: str
    belief_correct: bool
    periods: int


class CapacityDropWrapper:
    """Wraps CoreEnv to temporarily reduce factory capacity during shock."""

    def __init__(
        self,
        env: CoreEnv,
        affected_factory: int = DEFAULT_AFFECTED_FACTORY,
        normal_capacity: float = DEFAULT_NORMAL_CAPACITY,
        disrupted_capacity: float = DEFAULT_DISRUPTED_CAPACITY,
        event_start: int = DEFAULT_EVENT_START,
        event_end: int = DEFAULT_EVENT_END,
    ):
        self.env = env
        self.affected_factory = affected_factory
        self.normal_capacity = normal_capacity
        self.disrupted_capacity = disrupted_capacity
        self.event_start = event_start
        self.event_end = event_end
        self._original_capacity = env.network.graph.nodes[affected_factory].get("C", normal_capacity)

    def step(self, action: np.ndarray):
        t = self.env.period
        if self.event_start <= t < self.event_end:
            self.env.network.graph.nodes[self.affected_factory]["C"] = self.disrupted_capacity
        else:
            self.env.network.graph.nodes[self.affected_factory]["C"] = self.normal_capacity
        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.network.graph.nodes[self.affected_factory]["C"] = self._original_capacity
        return self.env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)


class SupplyBeliefAdaptiveController:
    """Maps a 2-class belief {P(Normal), P(CapacityDrop)} to a base-stock policy.

    The belief modulates the safety stock level:
    - When P(CapacityDrop) is high, the controller builds MORE safety stock
      before the disruption, anticipating reduced supply throughput.
    - During the disruption, the factory's live capacity caps actual throughput.

    effective_safety = safety_factor * (1 + p_cd * BUFFER_AMPLIFICATION)
    """

    BUFFER_AMPLIFICATION = 2.0

    def __init__(
        self,
        env_or_wrapper,
        regime_probabilities: dict[str, float],
        base_mu: float = DEFAULT_BASE_MU,
        normal_capacity: float = DEFAULT_NORMAL_CAPACITY,
        disrupted_capacity: float = DEFAULT_DISRUPTED_CAPACITY,
        affected_factory: int = DEFAULT_AFFECTED_FACTORY,
        safety_factor: float = 1.65,
    ):
        if isinstance(env_or_wrapper, CapacityDropWrapper):
            self.env = env_or_wrapper.env
            self._wrapper = env_or_wrapper
        else:
            self.env = env_or_wrapper
            self._wrapper = None

        self.regime_probs = dict(regime_probabilities)
        self.base_mu = base_mu
        self.normal_capacity = normal_capacity
        self.disrupted_capacity = disrupted_capacity
        self.affected_factory = affected_factory
        self.safety_factor = safety_factor
        self.network = self.env.network

        self.p_normal = self.regime_probs.get("normal", 0.5)
        self.p_cd = self.regime_probs.get("supplier_capacity_drop", 0.5)
        self.effective_safety = safety_factor * (1.0 + self.p_cd * self.BUFFER_AMPLIFICATION)

        self._precompute()

    def _precompute(self):
        self.node_lt = {}
        self.node_downstream = {}
        for node in self.network.main_nodes:
            lead_times = []
            for pred_idx, reorder_idx, L in self.network.pred_reorder_indices.get(node, []):
                lead_times.append(L)
            self.node_lt[node] = np.mean(lead_times) if lead_times else 2.0

            successors = list(self.network.graph.successors(node))
            total = 0.0
            for succ in successors:
                edge = self.network.graph.edges.get((node, succ))
                if edge and ("demand_dist" in edge or "p" in edge):
                    total += self.base_mu
            self.node_downstream[node] = total

        self._factory_outbound_links = {}
        for i, (supplier, buyer) in enumerate(self.network.reorder_links):
            if supplier == self.affected_factory:
                if supplier not in self._factory_outbound_links:
                    self._factory_outbound_links[supplier] = []
                self._factory_outbound_links[supplier].append(i)

    def get_action(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        n_actions = len(self.network.reorder_links)
        actions = np.zeros(n_actions)

        for i, (supplier, buyer) in enumerate(self.network.reorder_links):
            buyer_idx = self.network.node_map.get(buyer, None)
            if buyer_idx is None:
                actions[i] = self.base_mu
                continue

            avg_lt = self.node_lt.get(buyer, 2.0)
            ds_demand = self.node_downstream.get(buyer, self.base_mu)

            safety_stock = self.effective_safety * np.sqrt(max(avg_lt, 1) * self.base_mu)
            pipeline = ds_demand * avg_lt
            target = pipeline + safety_stock

            buyer_C = self.network.graph.nodes[buyer].get("C", float("inf"))
            if buyer_C < float("inf"):
                target = min(target, buyer_C * 3)

            current_inv = float(self.env.X[self.env.period, buyer_idx])
            pipeline_inv = 0.0
            for pred_idx, reorder_idx, L in self.network.pred_reorder_indices.get(buyer, []):
                if L > 0:
                    pipeline_inv += float(self.env.Y[self.env.period, reorder_idx])

            inv_pos = current_inv + pipeline_inv
            order = max(0.0, target - inv_pos)

            actions[i] = order

        return actions

    def __call__(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        return self.get_action(obs, current_period)


def make_capacity_drop_env(
    seed: int,
    scenario: str = DEFAULT_SCENARIO,
    num_periods: int = DEFAULT_NUM_PERIODS,
    base_mu: float = DEFAULT_BASE_MU,
    affected_factory: int = DEFAULT_AFFECTED_FACTORY,
    normal_capacity: float = DEFAULT_NORMAL_CAPACITY,
    disrupted_capacity: float = DEFAULT_DISRUPTED_CAPACITY,
    event_start: int = DEFAULT_EVENT_START,
    event_end: int = DEFAULT_EVENT_END,
    backlog: bool = True,
    config_path: str | None = None,
    noise_scale: float | None = None,
) -> CapacityDropWrapper:
    demand_config = {
        "type": "stationary",
        "base_mu": base_mu,
        "use_goodwill": False,
    }
    if noise_scale is not None:
        demand_config["noise_scale"] = noise_scale
    env = CoreEnv(
        scenario=scenario,
        demand_config=demand_config,
        num_periods=num_periods,
        backlog=backlog,
        config_path=config_path,
    )
    env.reset(seed=seed)
    return CapacityDropWrapper(
        env,
        affected_factory=affected_factory,
        normal_capacity=normal_capacity,
        disrupted_capacity=disrupted_capacity,
        event_start=event_start,
        event_end=event_end,
    )


def make_normal_env(
    seed: int,
    scenario: str = DEFAULT_SCENARIO,
    num_periods: int = DEFAULT_NUM_PERIODS,
    base_mu: float = DEFAULT_BASE_MU,
    backlog: bool = True,
    config_path: str | None = None,
    noise_scale: float | None = None,
) -> CoreEnv:
    demand_config = {
        "type": "stationary",
        "base_mu": base_mu,
        "use_goodwill": False,
    }
    if noise_scale is not None:
        demand_config["noise_scale"] = noise_scale
    env = CoreEnv(
        scenario=scenario,
        demand_config=demand_config,
        num_periods=num_periods,
        backlog=backlog,
        config_path=config_path,
    )
    env.reset(seed=seed)
    return env


def run_phase9_episode(
    seed: int,
    regime: Regime,
    sensor: str,
    regime_probabilities: dict[str, float],
    scenario: str = DEFAULT_SCENARIO,
    num_periods: int = DEFAULT_NUM_PERIODS,
    base_mu: float = DEFAULT_BASE_MU,
    event_start: int = DEFAULT_EVENT_START,
    event_end: int = DEFAULT_EVENT_END,
    affected_factory: int = DEFAULT_AFFECTED_FACTORY,
    normal_capacity: float = DEFAULT_NORMAL_CAPACITY,
    disrupted_capacity: float = DEFAULT_DISRUPTED_CAPACITY,
    config_path: str | None = None,
    noise_scale: float | None = None,
    backlog: bool = True,
) -> tuple[GymEpisodeResult9, list]:
    if regime == Regime.SUPPLIER_CAPACITY_DROP:
        wrapper = make_capacity_drop_env(
            seed=seed, scenario=scenario, num_periods=num_periods,
            base_mu=base_mu, affected_factory=affected_factory,
            normal_capacity=normal_capacity, disrupted_capacity=disrupted_capacity,
            event_start=event_start, event_end=event_end,
            config_path=config_path, noise_scale=noise_scale, backlog=backlog,
        )
        controller = SupplyBeliefAdaptiveController(
            wrapper, regime_probabilities, base_mu=base_mu,
            normal_capacity=normal_capacity, disrupted_capacity=disrupted_capacity,
            affected_factory=affected_factory,
        )
    else:
        wrapper = make_normal_env(
            seed=seed, scenario=scenario, num_periods=num_periods,
            base_mu=base_mu, config_path=config_path, noise_scale=noise_scale,
            backlog=backlog,
        )
        controller = SupplyBeliefAdaptiveController(
            wrapper, regime_probabilities, base_mu=base_mu,
            normal_capacity=normal_capacity, disrupted_capacity=disrupted_capacity,
            affected_factory=affected_factory,
        )

    obs, info = wrapper.reset(seed=seed)
    total_reward = 0.0
    trajectory = []

    for t in range(num_periods):
        action = controller.get_action(obs, t)
        action = np.clip(action, 0, float(wrapper.action_space.high[0]))
        next_obs, reward, terminated, truncated, step_info = wrapper.step(action)
        total_reward += reward

        trajectory.append({
            "period": t,
            "obs": obs.copy(),
            "action": action.copy(),
            "reward": reward,
            "inventory": wrapper.X[t + 1].copy() if t + 1 < wrapper.X.shape[0] else wrapper.X[-1].copy(),
        })

        obs = next_obs
        if truncated:
            break

    total_demand = float(np.sum(wrapper.D))
    retail_link_idx = None
    for idx, (supplier, buyer) in enumerate(wrapper.network.reorder_links):
        if buyer == 1:
            retail_link_idx = idx
            break
    if retail_link_idx is not None:
        total_retail_sales = float(np.sum(wrapper.S[:, retail_link_idx]))
    else:
        total_retail_sales = float(np.sum(wrapper.R))
    fill_rate = total_retail_sales / total_demand if total_demand > 0 else 1.0
    avg_inventory = float(np.mean(wrapper.X[1:wrapper.period + 1])) if wrapper.period > 0 else 0.0

    node_inv_pos = 0.0
    for node in wrapper.network.main_nodes:
        node_i = wrapper.network.node_map[node]
        node_inv_pos += wrapper.X[wrapper.period, node_i]
        for pred_idx, reorder_idx, L in wrapper.network.pred_reorder_indices.get(node, []):
            node_inv_pos += wrapper.Y[wrapper.period, reorder_idx]

    belief_normal = regime_probabilities.get("normal", 0.0)
    belief_cd = regime_probabilities.get("supplier_capacity_drop", 0.0)
    most_likely = "normal" if belief_normal >= belief_cd else "supplier_capacity_drop"

    result = GymEpisodeResult9(
        seed=seed,
        regime=regime.value,
        sensor=sensor,
        total_reward=float(total_reward),
        total_demand=total_demand,
        total_retail_sales=total_retail_sales,
        fill_rate=fill_rate,
        avg_inventory=avg_inventory,
        inventory_position_final=node_inv_pos,
        belief_normal=belief_normal,
        belief_capacity_drop=belief_cd,
        most_likely_regime=most_likely,
        belief_correct=(most_likely == regime.value),
        periods=wrapper.period,
    )

    return result, trajectory


def get_no_info_probs_9() -> dict[str, float]:
    return {"normal": 0.5, "supplier_capacity_drop": 0.5}


def get_perfect_semantic_probs_9(true_regime: Regime) -> dict[str, float]:
    probs = {"normal": 0.0, "supplier_capacity_drop": 0.0}
    if true_regime.value in probs:
        probs[true_regime.value] = 1.0
    else:
        probs["normal"] = 1.0
    return probs
