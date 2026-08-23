"""Tests for Phase 9 components: CapacityDrop, templates, controller, YAML scaling."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.events import Regime
from src.capacity_drop_templates import (
    ALL_CAPACITY_DROP_TEMPLATES,
    CAPACITY_DROP_TEMPLATES,
    NORMAL_TEMPLATES,
    TRAIN_TEMPLATE_IDS,
    TEST_TEMPLATE_IDS,
    TRAIN_TEMPLATES,
    TEST_TEMPLATES,
    Regime9,
)
from src.gym_adapter9 import (
    CapacityDropWrapper,
    SupplyBeliefAdaptiveController,
    make_capacity_drop_env,
    make_normal_env,
    run_phase9_episode,
    get_no_info_probs_9,
    get_perfect_semantic_probs_9,
    DEFAULT_BASE_MU,
    DEFAULT_NORMAL_CAPACITY,
    DEFAULT_DISRUPTED_CAPACITY,
)


# ── Events ──────────────────────────────────────────────────────────────

def test_regime_capacity_drop_exists():
    assert hasattr(Regime, "SUPPLIER_CAPACITY_DROP")
    assert Regime.SUPPLIER_CAPACITY_DROP.value == "supplier_capacity_drop"


# ── Templates ───────────────────────────────────────────────────────────

def test_template_counts():
    assert len(NORMAL_TEMPLATES) == 6
    assert len(CAPACITY_DROP_TEMPLATES) == 10
    assert len(ALL_CAPACITY_DROP_TEMPLATES) == 16


def test_template_keys():
    for t in ALL_CAPACITY_DROP_TEMPLATES:
        assert "template_id" in t
        assert "ambiguity_level" in t
        assert "regime" in t
        assert "text" in t
        assert t["ambiguity_level"] in ("clear", "moderate", "vague")


def test_train_test_split_disjoint():
    assert len(set(TRAIN_TEMPLATE_IDS) & set(TEST_TEMPLATE_IDS)) == 0


def test_train_test_split_covers_all():
    all_ids = {t["template_id"] for t in ALL_CAPACITY_DROP_TEMPLATES}
    covered = set(TRAIN_TEMPLATE_IDS) | set(TEST_TEMPLATE_IDS)
    assert covered == all_ids


def test_train_templates_match_ids():
    train_ids = {t["template_id"] for t in TRAIN_TEMPLATES}
    assert train_ids == set(TRAIN_TEMPLATE_IDS)


def test_test_templates_match_ids():
    test_ids = {t["template_id"] for t in TEST_TEMPLATES}
    assert test_ids == set(TEST_TEMPLATE_IDS)


def test_train_has_both_regimes():
    train_regimes = {t["regime"] for t in TRAIN_TEMPLATES}
    assert Regime9.NORMAL in train_regimes
    assert Regime9.SUPPLIER_CAPACITY_DROP in train_regimes


def test_vague_cd_templates_all_held_out():
    vague_cd_ids = {t["template_id"] for t in ALL_CAPACITY_DROP_TEMPLATES
                    if t["ambiguity_level"] == "vague" and t["regime"] == Regime9.SUPPLIER_CAPACITY_DROP}
    for tid in vague_cd_ids:
        assert tid in TEST_TEMPLATE_IDS, f"vague CD template {tid} should be held out"


# ── CapacityDropWrapper ────────────────────────────────────────────────

def test_capacity_drop_wrapper_basic():
    w = make_capacity_drop_env(seed=4000)
    obs, info = w.reset(seed=4000)
    assert obs.shape[0] > 0
    assert w.action_space is not None


def test_capacity_drop_wrapper_capacity_reduction():
    w = make_capacity_drop_env(
        seed=4000,
        event_start=10, event_end=15,
        normal_capacity=90, disrupted_capacity=30,
    )
    w.reset(seed=4000)
    for t in range(10):
        action = np.array([0.0] * w.action_space.shape[0])
        w.step(action)
    node_c = w.env.network.graph.nodes[4].get("C", 90)
    assert node_c == 90
    action = np.array([0.0] * w.action_space.shape[0])
    w.step(action)
    node_c = w.env.network.graph.nodes[4].get("C", 90)
    assert node_c == 30


def test_capacity_drop_wrapper_restores_capacity():
    w = make_capacity_drop_env(
        seed=4000,
        event_start=10, event_end=15,
        normal_capacity=90, disrupted_capacity=30,
    )
    w.reset(seed=4000)
    for t in range(16):
        action = np.array([0.0] * w.action_space.shape[0])
        w.step(action)
    node_c = w.env.network.graph.nodes[4].get("C", 90)
    assert node_c == 90


# ── Controller ──────────────────────────────────────────────────────────

def test_controller_buffer_amplification():
    w = make_capacity_drop_env(seed=4000)
    probs_cd = {"normal": 0.0, "supplier_capacity_drop": 1.0}
    probs_normal = {"normal": 1.0, "supplier_capacity_drop": 0.0}

    ctrl_cd = SupplyBeliefAdaptiveController(
        w, probs_cd, base_mu=DEFAULT_BASE_MU,
        normal_capacity=DEFAULT_NORMAL_CAPACITY,
        disrupted_capacity=DEFAULT_DISRUPTED_CAPACITY,
    )
    ctrl_norm = SupplyBeliefAdaptiveController(
        w, probs_normal, base_mu=DEFAULT_BASE_MU,
        normal_capacity=DEFAULT_NORMAL_CAPACITY,
        disrupted_capacity=DEFAULT_DISRUPTED_CAPACITY,
    )
    assert ctrl_cd.effective_safety > ctrl_norm.effective_safety


def test_controller_no_info_belief():
    p = get_no_info_probs_9()
    assert p["normal"] == 0.5
    assert p["supplier_capacity_drop"] == 0.5


def test_controller_perfect_belief_normal():
    p = get_perfect_semantic_probs_9(Regime.NORMAL)
    assert p["normal"] == 1.0
    assert p["supplier_capacity_drop"] == 0.0


def test_controller_perfect_belief_cd():
    p = get_perfect_semantic_probs_9(Regime.SUPPLIER_CAPACITY_DROP)
    assert p["normal"] == 0.0
    assert p["supplier_capacity_drop"] == 1.0


def test_controller_action_non_negative():
    w = make_capacity_drop_env(seed=4000)
    obs, _ = w.reset(seed=4000)
    ctrl = SupplyBeliefAdaptiveController(
        w, get_no_info_probs_9(), base_mu=DEFAULT_BASE_MU,
        normal_capacity=DEFAULT_NORMAL_CAPACITY,
        disrupted_capacity=DEFAULT_DISRUPTED_CAPACITY,
    )
    action = ctrl.get_action(obs, 0)
    assert np.all(action >= 0)


# ── Episode runner ──────────────────────────────────────────────────────

def test_run_episode_capacity_drop():
    result, traj = run_phase9_episode(
        seed=4000, regime=Regime.SUPPLIER_CAPACITY_DROP,
        sensor="NoInfo",
        regime_probabilities=get_no_info_probs_9(),
    )
    assert result.regime == "supplier_capacity_drop"
    assert result.periods > 0
    assert result.total_demand > 0
    assert 0.0 <= result.fill_rate <= 1.0


def test_run_episode_normal():
    result, traj = run_phase9_episode(
        seed=4000, regime=Regime.NORMAL,
        sensor="NoInfo",
        regime_probabilities=get_no_info_probs_9(),
    )
    assert result.regime == "normal"
    assert result.periods > 0


def test_oracle_beats_no_info_capacity_drop():
    ni_results = []
    oracle_results = []
    for seed in range(4000, 4005):
        ni, _ = run_phase9_episode(seed=seed, regime=Regime.SUPPLIER_CAPACITY_DROP,
                                    sensor="NoInfo", regime_probabilities=get_no_info_probs_9())
        oc, _ = run_phase9_episode(seed=seed, regime=Regime.SUPPLIER_CAPACITY_DROP,
                                    sensor="OracleSemantic",
                                    regime_probabilities=get_perfect_semantic_probs_9(Regime.SUPPLIER_CAPACITY_DROP))
        ni_results.append(ni.total_reward)
        oracle_results.append(oc.total_reward)
    assert np.mean(oracle_results) > np.mean(ni_results)


# ── Custom YAML (Phase 9B) ─────────────────────────────────────────────

def test_custom_yaml_short_lead():
    yaml_path = str(ROOT / "results" / "phase9b_robustness" / "divergent_short_lead.yaml")
    if not Path(yaml_path).exists():
        pytest.skip("short_lead YAML not generated")
    w = make_capacity_drop_env(seed=4000, scenario="custom", config_path=yaml_path)
    obs, info = w.reset(seed=4000)
    assert obs.shape[0] > 0
    for e, lt in w.network.lead_times.items():
        if e[0] in (4, 5, 6) and e[1] in (2, 3):
            assert lt <= 6, f"Short-lead L should be <=6, got {lt} for {e}"


def test_custom_yaml_long_lead():
    yaml_path = str(ROOT / "results" / "phase9b_robustness" / "divergent_long_lead.yaml")
    if not Path(yaml_path).exists():
        pytest.skip("long_lead YAML not generated")
    w = make_capacity_drop_env(seed=4000, scenario="custom", config_path=yaml_path)
    obs, info = w.reset(seed=4000)
    assert obs.shape[0] > 0
    for e, lt in w.network.lead_times.items():
        if e[0] in (4, 5, 6) and e[1] in (2, 3):
            assert lt >= 15, f"Long-lead L should be >=15, got {lt} for {e}"


def test_noise_scale_zero():
    w = make_capacity_drop_env(seed=4000, noise_scale=0.0)
    obs, info = w.reset(seed=4000)
    assert obs.shape[0] > 0


def test_backlog_false():
    w = make_capacity_drop_env(seed=4000, backlog=False)
    obs, info = w.reset(seed=4000)
    assert obs.shape[0] > 0
    assert w.backlog is False
