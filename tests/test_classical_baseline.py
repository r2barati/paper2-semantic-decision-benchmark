"""Phase 7 tests for the classical NLP baseline.

Covers model behavior, leakage prevention, integration with the
downstream pipeline, and metric consistency.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.events import Regime, REGIME_WARNING_TEMPLATES, REGIME_PRIOR
from src.interpreter import (
    RegimeInterpretation, no_info_regime_belief, perfect_semantic_regime_belief,
)
from src.confirmation_templates import CONFIRMATION_TEMPLATES
from src.classical_baseline import (
    TFIDFLogReg, build_dataset, grouped_train_test_split,
    evaluate_classification, leakage_audit, compute_dataset_manifest,
    cross_validate_on_original, _template_family, REGIME_LABELS,
)
from src.experiment_phase5 import _run_p5_episode
from src.experiment_phase6 import brier_score, log_loss_safe, aggregate_sivr


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def original_templates():
    return REGIME_WARNING_TEMPLATES


@pytest.fixture
def confirmation_templates():
    return CONFIRMATION_TEMPLATES


@pytest.fixture
def trained_model(original_templates):
    texts = [t["text"] for t in original_templates]
    labels = [t["regime"].value if isinstance(t["regime"], Regime) else t["regime"]
              for t in original_templates]
    model = TFIDFLogReg(seed=42)
    model.fit(texts, labels)
    return model


@pytest.fixture
def trained_model_calibrated(original_templates):
    texts = [t["text"] for t in original_templates]
    labels = [t["regime"].value if isinstance(t["regime"], Regime) else t["regime"]
              for t in original_templates]
    model = TFIDFLogReg(seed=42, calibrate=True, cal_method="isotonic")
    model.fit(texts, labels)
    return model


# =============================================================================
# Model behavior tests
# =============================================================================

class TestModelBehavior:
    def test_output_probabilities_sum_to_one(self, trained_model, confirmation_templates):
        texts = [t["text"] for t in confirmation_templates]
        proba = trained_model.predict_proba(texts)
        sums = np.sum(proba, axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_output_probabilities_within_range(self, trained_model, confirmation_templates):
        texts = [t["text"] for t in confirmation_templates]
        proba = trained_model.predict_proba(texts)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_regime_labels_correct(self, trained_model):
        classes = set(trained_model.classes_)
        expected = {"normal", "supplier_delay", "demand_surge"}
        assert classes == expected

    def test_deterministic_training(self, original_templates):
        texts = [t["text"] for t in original_templates]
        labels = [t["regime"].value if isinstance(t["regime"], Regime) else t["regime"]
                  for t in original_templates]
        m1 = TFIDFLogReg(seed=42)
        m1.fit(texts, labels)
        m2 = TFIDFLogReg(seed=42)
        m2.fit(texts, labels)
        p1 = m1.predict_proba(texts)
        p2 = m2.predict_proba(texts)
        np.testing.assert_array_equal(p1, p2)

    def test_serialization_preserves_outputs(self, trained_model, confirmation_templates):
        texts = [t["text"] for t in confirmation_templates]
        orig_proba = trained_model.predict_proba(texts)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            trained_model.save(path)
            loaded = TFIDFLogReg.load(path)
            loaded_proba = loaded.predict_proba(texts)
            np.testing.assert_array_almost_equal(orig_proba, loaded_proba)

    def test_serialization_preserves_metadata(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            trained_model.save(path)
            loaded = TFIDFLogReg.load(path)
            assert loaded.is_fitted_ == trained_model.is_fitted_
            assert loaded.train_size_ == trained_model.train_size_
            assert loaded.vocabulary_size_ == trained_model.vocabulary_size_

    def test_predict_regime_returns_valid_result(self, trained_model):
        result = trained_model.predict_regime("Normal operations expected")
        assert isinstance(result.regime_probabilities, dict)
        assert set(result.regime_probabilities.keys()) == {"normal", "supplier_delay", "demand_surge"}
        assert abs(sum(result.regime_probabilities.values()) - 1.0) < 1e-6
        assert all(0.0 <= v <= 1.0 for v in result.regime_probabilities.values())

    def test_predict_regime_estimated_params(self, trained_model):
        result = trained_model.predict_regime("Lead times will increase from 4 to 8 periods")
        assert isinstance(result.estimated_lt_increase, int)
        assert isinstance(result.estimated_duration, int)
        assert isinstance(result.estimated_demand_multiplier, float)

    def test_calibrated_probabilities_sum_to_one(self, trained_model_calibrated, confirmation_templates):
        texts = [t["text"] for t in confirmation_templates]
        proba = trained_model_calibrated.predict_proba(texts)
        sums = np.sum(proba, axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_calibrated_probabilities_within_range(self, trained_model_calibrated, confirmation_templates):
        texts = [t["text"] for t in confirmation_templates]
        proba = trained_model_calibrated.predict_proba(texts)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


# =============================================================================
# Leakage prevention tests
# =============================================================================

class TestLeakagePrevention:
    def test_test_templates_absent_from_training(self, original_templates, confirmation_templates):
        orig_ids = {t["template_id"] for t in original_templates}
        conf_ids = {t["template_id"] for t in confirmation_templates}
        assert len(orig_ids & conf_ids) == 0

    def test_no_template_id_overlap_in_split(self, original_templates, confirmation_templates):
        all_texts, all_labels, all_ids, all_families = build_dataset(
            original_templates + confirmation_templates
        )
        test_ids = set(t["template_id"] for t in confirmation_templates)
        train_texts, train_labels, train_ids, train_families, \
            eval_texts, eval_labels, eval_ids, eval_families = \
            grouped_train_test_split(all_texts, all_labels, all_ids, all_families, test_ids)

        audit = leakage_audit(train_ids, eval_ids, train_families, eval_families)
        assert audit["no_template_leak"]

    def test_vectorizer_fitted_on_train_only(self, original_templates):
        model = TFIDFLogReg(seed=42)
        texts = [t["text"] for t in original_templates]
        labels = [t["regime"].value if isinstance(t["regime"], Regime) else t["regime"]
                  for t in original_templates]
        model.fit(texts, labels)
        assert model.is_fitted_
        assert model.train_size_ == len(texts)

    def test_calibration_does_not_use_test_data(self, original_templates, confirmation_templates):
        model = TFIDFLogReg(seed=42, calibrate=True, cal_method="isotonic")
        train_texts = [t["text"] for t in original_templates]
        train_labels = [t["regime"].value if isinstance(t["regime"], Regime) else t["regime"]
                        for t in original_templates]
        model.fit(train_texts, train_labels)
        assert model.calibrated_model_ is not None

        test_texts = [t["text"] for t in confirmation_templates]
        test_proba = model.predict_proba(test_texts)
        assert test_proba.shape == (len(confirmation_templates), 3)

    def test_leakage_audit_covers_all_requirements(self, original_templates, confirmation_templates):
        all_texts, all_labels, all_ids, all_families = build_dataset(
            original_templates + confirmation_templates
        )
        test_ids = set(t["template_id"] for t in confirmation_templates)
        train_texts, train_labels, train_ids, train_families, \
            eval_texts, eval_labels, eval_ids, eval_families = \
            grouped_train_test_split(all_texts, all_labels, all_ids, all_families, test_ids)

        audit = leakage_audit(train_ids, eval_ids, train_families, eval_families)
        assert audit["no_template_leak"]
        assert len(audit["train_families"]) > 0
        assert len(audit["test_families"]) > 0
        assert "train_templates" in audit
        assert "test_templates" in audit


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    def test_classical_interpreter_implements_interface(self, trained_model):
        result = trained_model.predict_regime("Test warning text")
        ri = result.to_regime_interpretation()
        assert isinstance(ri, RegimeInterpretation)
        assert hasattr(ri, "regime_probabilities")
        assert hasattr(ri, "most_likely_regime")
        assert hasattr(ri, "normalized")

    def test_probabilities_reach_optimizer(self, trained_model):
        result = trained_model.predict_regime("Test text")
        ri = result.to_regime_interpretation()
        belief = ri.normalized()

        from src.env import CausalOptimizer
        optimizer = CausalOptimizer(
            seed=1000,
            regime_probabilities=belief.regime_probabilities,
            horizon=40,
            initial_inventory=30,
            demand_mean=8.0,
        )
        from src.env import InventoryState
        state = InventoryState(time=0, on_hand=30, pipeline=0, order_quantity=0,
                               lost_sales=0, cumulative_profit=0, demand=8)
        order = optimizer.decide(state)
        assert isinstance(order, float)
        assert order >= 0.0

    def test_downstream_episode_completes(self, trained_model, confirmation_templates):
        tmpl = confirmation_templates[0]
        result_baseline = trained_model.predict_regime(tmpl["text"])
        ri = result_baseline.to_regime_interpretation()

        episode_result, history = _run_p5_episode(
            seed=2000, regime=Regime.SUPPLIER_DELAY,
            sensor="TFIDF_LogReg", controller="CausalOptimizer",
            template_id=tmpl["template_id"],
            ambiguity_level=tmpl["ambiguity_level"],
            regime_interp=ri,
        )
        assert len(history) == 40
        assert episode_result.total_profit > 0

    def test_no_future_leakage(self, trained_model, confirmation_templates):
        tmpl = confirmation_templates[0]
        result_baseline = trained_model.predict_regime(tmpl["text"])
        ri = result_baseline.to_regime_interpretation()
        belief = ri.normalized()

        assert isinstance(belief.estimated_lt_increase, int)
        assert isinstance(belief.estimated_duration, int)
        assert isinstance(belief.estimated_demand_multiplier, float)
        assert belief.estimated_demand_multiplier > 0

    def test_normalized_belief_sums_to_one(self, trained_model, confirmation_templates):
        for tmpl in confirmation_templates[:5]:
            result = trained_model.predict_regime(tmpl["text"])
            ri = result.to_regime_interpretation()
            belief = ri.normalized()
            total = sum(belief.regime_probabilities.values())
            np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_same_seed_same_world_across_sensors(self, trained_model, confirmation_templates):
        tmpl = confirmation_templates[0]
        regime = Regime.SUPPLIER_DELAY

        result_baseline = trained_model.predict_regime(tmpl["text"])
        ri = result_baseline.to_regime_interpretation()

        from src.experiment_phase6 import rule_based_regime_extract
        ri_rb = rule_based_regime_extract(tmpl["text"])

        _, hist_tf = _run_p5_episode(
            seed=2000, regime=regime, sensor="TFIDF_LogReg",
            controller="CausalOptimizer", regime_interp=ri,
        )
        _, hist_rb = _run_p5_episode(
            seed=2000, regime=regime, sensor="RuleBased",
            controller="CausalOptimizer", regime_interp=ri_rb,
        )
        for t in range(40):
            assert hist_tf[t].demand == hist_rb[t].demand


# =============================================================================
# Metrics tests
# =============================================================================

class TestMetrics:
    def test_brier_score_computed(self, trained_model, confirmation_templates):
        texts = [t["text"] for t in confirmation_templates]
        proba = trained_model.predict_proba(texts)
        classes = trained_model.classes_

        for i, tmpl in enumerate(confirmation_templates):
            true_regime = tmpl["regime"].value if isinstance(tmpl["regime"], Regime) else tmpl["regime"]
            pred = {cls: float(proba[i, j]) for j, cls in enumerate(classes)}
            brier = brier_score(pred, true_regime)
            assert 0.0 <= brier <= 2.0

    def test_log_loss_computed(self, trained_model, confirmation_templates):
        texts = [t["text"] for t in confirmation_templates]
        proba = trained_model.predict_proba(texts)
        classes = trained_model.classes_

        for i, tmpl in enumerate(confirmation_templates):
            true_regime = tmpl["regime"].value if isinstance(tmpl["regime"], Regime) else tmpl["regime"]
            pred = {cls: float(proba[i, j]) for j, cls in enumerate(classes)}
            ll = log_loss_safe(pred, true_regime)
            assert ll >= 0.0

    def test_aggregate_sivr_computed(self):
        sensor_profits = np.array([100.0, 110.0, 105.0])
        ni_profits = np.array([90.0, 95.0, 92.0])
        ps_profits = np.array([120.0, 125.0, 118.0])
        sivr = aggregate_sivr(sensor_profits, ni_profits, ps_profits)
        assert isinstance(sivr, float)
        assert not math.isnan(sivr)

    def test_evaluate_classification(self):
        y_true = ["normal", "supplier_delay", "demand_surge", "normal"]
        y_pred = ["normal", "supplier_delay", "normal", "normal"]
        proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.5, 0.2, 0.3],
            [0.9, 0.05, 0.05],
        ])
        classes = np.array(["normal", "supplier_delay", "demand_surge"])
        result = evaluate_classification(y_true, y_pred, proba, classes)
        assert "accuracy" in result
        assert "brier_macro" in result
        assert "log_loss" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_classification_metrics_on_held_out(self, trained_model, confirmation_templates):
        texts = [t["text"] for t in confirmation_templates]
        labels = [t["regime"].value if isinstance(t["regime"], Regime) else t["regime"]
                  for t in confirmation_templates]
        proba = trained_model.predict_proba(texts)
        pred = trained_model.predict(texts)
        classes = trained_model.classes_
        result = evaluate_classification(labels, pred, proba, classes)
        assert result["accuracy"] > 0.5
        assert result["brier_macro"] < 1.0
        assert result["log_loss"] < 5.0


# =============================================================================
# Cross-validation tests
# =============================================================================

class TestCrossValidation:
    def test_cv_returns_valid_results(self, original_templates):
        result = cross_validate_on_original(original_templates, n_splits=3, seed=42)
        assert result["n_folds"] == 3
        assert 0.0 <= result["avg_accuracy"] <= 1.0
        assert result["avg_brier_macro"] >= 0.0
        assert result["avg_log_loss"] >= 0.0
        assert len(result["folds"]) == 3

    def test_cv_folds_have_consistent_structure(self, original_templates):
        result = cross_validate_on_original(original_templates, n_splits=3, seed=42)
        for fold in result["folds"]:
            assert "fold" in fold
            assert "train_size" in fold
            assert "val_size" in fold
            assert "accuracy" in fold
            assert fold["train_size"] + fold["val_size"] == len(original_templates)


# =============================================================================
# Template family tests
# =============================================================================

class TestTemplateFamily:
    def test_template_family_mapping(self):
        assert _template_family("delay_clear_1") == "delay"
        assert _template_family("normal_moderate_2") == "normal"
        assert _template_family("surge_vague_3") == "surge"
        assert _template_family("cd_clear_1") == "delay"
        assert _template_family("cn_moderate_2") == "normal"
        assert _template_family("cs_vague_3") == "surge"

    def test_build_dataset(self, original_templates):
        texts, labels, ids, families = build_dataset(original_templates)
        assert len(texts) == len(original_templates)
        assert len(labels) == len(original_templates)
        assert len(ids) == len(original_templates)
        assert len(families) == len(original_templates)
        assert all(isinstance(t, str) for t in texts)
        assert all(l in REGIME_LABELS for l in labels)
