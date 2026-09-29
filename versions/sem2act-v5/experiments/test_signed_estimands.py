from __future__ import annotations

import pytest

from signed_estimands import delta_j, persistence_interaction


def test_delta_is_rerank_minus_noinfo():
    assert delta_j(120.0, 100.0) == pytest.approx(20.0)
    assert delta_j(80.0, 100.0) == pytest.approx(-20.0)


def test_interaction_is_persistent_minus_reset():
    persistent = {1: -2.0, 40: -20.0}
    reset = {1: -2.0, 40: -5.0}
    assert persistence_interaction(persistent, reset) == pytest.approx(-15.0)


def test_positive_interaction_for_persistent_benefit():
    persistent = {1: 2.0, 40: 20.0}
    reset = {1: 2.0, 40: 5.0}
    assert persistence_interaction(persistent, reset) == pytest.approx(15.0)


def test_missing_horizon_is_rejected():
    with pytest.raises(ValueError):
        persistence_interaction({1: 0.0, 20: 1.0}, {1: 0.0, 40: 1.0})
