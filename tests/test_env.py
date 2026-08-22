"""Tests for the inventory environment."""

import numpy as np
from src.env import InventoryEnv, InventoryState, HORIZON
from src.events import SupplierDisruption, DEFAULT_DISRUPTION


def test_same_seed_same_demand():
    """Same seed produces identical demand trajectories."""
    env1 = InventoryEnv(seed=42)
    env2 = InventoryEnv(seed=42)
    s1 = env1.reset()
    s2 = env2.reset()
    demands1, demands2 = [], []
    for _ in range(HORIZON):
        s1 = env1.step(10)
        s2 = env2.step(10)
        demands1.append(s1.demand)
        demands2.append(s2.demand)
    assert demands1 == demands2, "Same seed must produce identical demand trajectories"


def test_different_seeds_differ():
    """Different seeds produce different demand trajectories."""
    env1 = InventoryEnv(seed=42)
    env2 = InventoryEnv(seed=99)
    s1 = env1.reset()
    s2 = env2.reset()
    demands1, demands2 = [], []
    for _ in range(HORIZON):
        s1 = env1.step(10)
        s2 = env2.step(10)
        demands1.append(s1.demand)
        demands2.append(s2.demand)
    assert demands1 != demands2, "Different seeds should produce different trajectories"


def test_disruption_changes_lead_time():
    """Hidden disruption changes lead time at the correct periods."""
    disc = SupplierDisruption(
        start_time=18, duration=8, normal_lead_time=2, disrupted_lead_time=5,
    )
    env = InventoryEnv(seed=1000, disruption=disc)
    state = env.reset()
    lt_before = []
    lt_during = []
    lt_after = []
    for _ in range(HORIZON):
        state = env.step(5)
        t = state.time
        if t < disc.start_time:
            lt_before.append(state.lead_time)
        elif disc.start_time <= t < disc.end_time:
            lt_during.append(state.lead_time)
        else:
            lt_after.append(state.lead_time)
    assert all(lt == 2 for lt in lt_before), "Lead time should be 2 before disruption"
    assert all(lt == 5 for lt in lt_during), "Lead time should be 5 during disruption"
    assert all(lt == 2 for lt in lt_after), "Lead time should be 2 after disruption"


def test_text_generation_does_not_modify_dynamics():
    """Text generation / interpretation does not affect environment dynamics."""
    disc = DEFAULT_DISRUPTION
    env_a = InventoryEnv(seed=1000, disruption=disc)
    env_b = InventoryEnv(seed=1000, disruption=disc)
    s_a = env_a.reset()
    s_b = env_b.reset()
    for _ in range(HORIZON):
        s_a = env_a.step(8)
        s_b = env_b.step(8)
    assert s_a.demand == s_b.demand
    assert s_a.cumulative_profit == s_b.cumulative_profit


def test_initial_inventory():
    """Environment starts with correct initial inventory."""
    env = InventoryEnv(seed=1000, initial_inventory=50)
    state = env.reset()
    assert state.on_hand == 50.0


def test_lost_sales():
    """Unmet demand is lost and counted correctly."""
    env = InventoryEnv(seed=1000, demand_mean=20.0, initial_inventory=5)
    state = env.reset()
    state = env.step(0)  # order 0, demand drawn from Poisson(20)
    assert state.lost_sales >= 0, "Lost sales cannot be negative"
    assert state.sales + state.lost_sales == state.demand, "Sales + lost must equal demand"
    assert state.on_hand >= 0, "On-hand cannot go negative with lost-sales"


def test_profit_calculation():
    """Period profit equals revenue minus costs."""
    from src.env import REVENUE_PER_UNIT, ORDERING_FIXED_COST, ORDERING_VARIABLE_COST, HOLDING_COST_PER_UNIT, STOCKOUT_COST_PER_UNIT
    env = InventoryEnv(seed=42)
    state = env.reset()
    state = env.step(10)
    expected_revenue = state.sales * REVENUE_PER_UNIT
    expected_ordering = ORDERING_FIXED_COST + ORDERING_VARIABLE_COST * 10
    expected_holding = HOLDING_COST_PER_UNIT * state.on_hand
    expected_stockout = STOCKOUT_COST_PER_UNIT * state.lost_sales
    expected_profit = expected_revenue - expected_ordering - expected_holding - expected_stockout
    assert abs(state.period_profit - expected_profit) < 1e-9, (
        f"Profit mismatch: {state.period_profit} vs {expected_profit}"
    )
