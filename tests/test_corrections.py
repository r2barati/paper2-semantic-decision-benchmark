"""Tests for the post-audit estimands, metrics, and offline evidence path."""

import math

from src.metrics import (
    SIVR_NEGATIVE_ORACLE_REFERENCE_VALUE,
    SIVR_VALID,
    SIVR_ZERO_OR_NEAR_ZERO_REFERENCE_VALUE,
    family_seed_bootstrap_ci,
    signed_sivr,
    standard_brier_score,
)


def test_signed_sivr_positive_reference():
    result = signed_sivr(1040, 1000, 1080)
    assert result.status == SIVR_VALID
    assert result.oiv == 80
    assert result.value == 0.5


def test_signed_sivr_negative_reference_is_diagnostic_only():
    result = signed_sivr(-90, -100, -120)
    assert result.status == SIVR_NEGATIVE_ORACLE_REFERENCE_VALUE
    assert result.oiv == -20
    assert result.value == -0.5


def test_signed_sivr_near_zero_is_nan():
    result = signed_sivr(1.0, 1.0, 1.0 + 1e-12)
    assert result.status == SIVR_ZERO_OR_NEAR_ZERO_REFERENCE_VALUE
    assert math.isnan(result.value)


def test_standard_multiclass_brier_score():
    assert math.isclose(standard_brier_score(
        {"normal": 0.8, "surge": 0.2}, "normal", ["normal", "surge"]
    ), 0.08)


def test_family_bootstrap_is_deterministic_and_paired():
    values = {
        "normal": {
            "family_a": {"v1": [1.0, 2.0], "v2": [3.0, 4.0]},
            "family_b": {"v1": [5.0, 7.0]},
        },
        "surge": {"family_c": {"v1": [9.0, 11.0]}},
    }
    left = family_seed_bootstrap_ci(values,
                                    regime_weights={"normal": 0.5, "surge": 0.5},
                                    n_boot=100, seed=17)
    right = family_seed_bootstrap_ci(values,
                                     regime_weights={"normal": 0.5, "surge": 0.5},
                                     n_boot=100, seed=17)
    assert left["mean"] == right["mean"]
    assert (left["bootstrap_samples"] == right["bootstrap_samples"]).all()
    assert left["bootstrap_hierarchy"] == "regime -> template_family -> variant -> paired_seed"
