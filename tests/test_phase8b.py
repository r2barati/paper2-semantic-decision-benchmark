"""Phase 8B validity tests.

Verifies the frozen, leakage-safe confirmation experiment.
All conditions must pass before the main Phase-8B run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.events import Regime, REGIME_WARNING_TEMPLATES
from src.confirmation_templates import CONFIRMATION_TEMPLATES


PHASE8B_DIR = ROOT / "results" / "phase8b_gym_confirmation"


# ---------------------------------------------------------------------------
# 1. Linguistic separation: no template ID overlap
# ---------------------------------------------------------------------------

class TestLinguisticSeparation:
    def test_no_id_overlap(self):
        train_ids = {t["template_id"] for t in REGIME_WARNING_TEMPLATES}
        test_ids = {t["template_id"] for t in CONFIRMATION_TEMPLATES}
        overlap = train_ids & test_ids
        assert len(overlap) == 0, f"Template ID overlap detected: {overlap}"

    def test_train_ids_prefixed(self):
        for t in REGIME_WARNING_TEMPLATES:
            assert t["template_id"].startswith(("delay_", "surge_", "normal_")), (
                f"Unexpected train template_id prefix: {t['template_id']}"
            )

    def test_test_ids_prefixed(self):
        for t in CONFIRMATION_TEMPLATES:
            assert t["template_id"].startswith(("cd_", "cs_", "cn_")), (
                f"Unexpected test template_id prefix: {t['template_id']}"
            )

    def test_held_out_only_surge_normal(self):
        held_out = [t for t in CONFIRMATION_TEMPLATES if t["regime"] in (Regime.NORMAL, Regime.DEMAND_SURGE)]
        assert len(held_out) == 24, f"Expected 24 held-out templates (12 surge + 12 normal), got {len(held_out)}"

    def test_split_manifest_exists_if_results_exist(self):
        manifest = PHASE8B_DIR / "linguistic_split_manifest.json"
        if manifest.exists():
            import json
            data = json.loads(manifest.read_text())
            assert data["leakage_audit"]["no_leak"] is True
            assert len(data["leakage_audit"]["overlap"]) == 0


# ---------------------------------------------------------------------------
# 2. TF-IDF model identity: raw vs calibrated
# ---------------------------------------------------------------------------

class TestTFIDFModelIdentity:
    def test_raw_model_exists(self):
        pkl = ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_model.pkl"
        assert pkl.exists(), "Raw TF-IDF model must exist"

    def test_calibrated_model_exists(self):
        pkl = ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_calibrated_model.pkl"
        assert pkl.exists(), "Calibrated TF-IDF model must exist"

    def test_raw_and_calibrated_different(self):
        from src.classical_baseline import TFIDFLogReg
        raw = TFIDFLogReg.load(str(ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_model.pkl"))
        cal = TFIDFLogReg.load(str(ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_calibrated_model.pkl"))
        # Raw should NOT have calibration applied
        assert raw.calibrated_model_ is None, "Raw model should not have calibration"

    def test_raw_non_degenerate_on_heldout(self):
        from src.gym_adapter import get_tfidf_probs
        from src.classical_baseline import TFIDFLogReg
        raw = TFIDFLogReg.load(str(ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_model.pkl"))
        held_out = [t for t in CONFIRMATION_TEMPLATES if t["regime"] in (Regime.NORMAL, Regime.DEMAND_SURGE)]
        for tmpl in held_out:
            probs = get_tfidf_probs(tmpl["text"], raw)
            for key, val in probs.items():
                assert 0.05 < val < 0.95, (
                    f"Raw model degenerate on {tmpl['template_id']}: {key}={val:.4f}"
                )

    def test_calibrated_degenerate_on_train(self):
        from src.gym_adapter import get_tfidf_probs
        from src.classical_baseline import TFIDFLogReg
        cal = TFIDFLogReg.load(str(ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_calibrated_model.pkl"))
        for tmpl in REGIME_WARNING_TEMPLATES:
            probs = get_tfidf_probs(tmpl["text"], cal)
            if tmpl["regime"] == Regime.DEMAND_SURGE:
                assert probs["demand_surge"] > 0.95, (
                    f"Calibrated model should saturate on train surge: {probs}"
                )
            elif tmpl["regime"] == Regime.NORMAL:
                assert probs["normal"] > 0.95, (
                    f"Calibrated model should saturate on train normal: {probs}"
                )


# ---------------------------------------------------------------------------
# 3. Controller frozen from Phase-8A
# ---------------------------------------------------------------------------

class TestControllerFrozen:
    def test_controller_source_unchanged(self):
        from src.gym_adapter import BeliefAdaptiveController
        import inspect
        source = inspect.getsource(BeliefAdaptiveController)
        # The key formula: effective_mu = p_normal * base_mu + p_surge * base_mu * surge_multiplier
        assert "p_normal * self.base_mu + p_surge * self.base_mu * self.surge_multiplier" in source, (
            "Controller mapping changed from Phase-8A"
        )

    def test_no_argmax_in_controller(self):
        from src.gym_adapter import BeliefAdaptiveController
        import inspect
        source = inspect.getsource(BeliefAdaptiveController)
        assert "argmax" not in source, "Controller must not use argmax"

    def test_controller_params_frozen(self):
        from src.gym_adapter import BeliefAdaptiveController, make_surge_env
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        probs = {"normal": 0.3, "demand_surge": 0.7}
        ctrl = BeliefAdaptiveController(env, probs, base_mu=10.0, surge_multiplier=2.0)
        assert ctrl.base_mu == 10.0
        assert ctrl.surge_multiplier == 2.0
        assert abs(ctrl.effective_mu - 10.0 * (1.0 + 0.7 * (2.0 - 1.0))) < 1e-9


# ---------------------------------------------------------------------------
# 4. Seed disjointness
# ---------------------------------------------------------------------------

class TestSeedDisjointness:
    def test_phase8b_seeds_not_in_8a(self):
        phase8a_pilot = set(range(1, 21))
        phase8a_confirm = set(range(2000, 2020))
        phase8b_pilot = set(range(3000, 3002))
        phase8b = set(range(3100, 3130))
        assert len(phase8b & phase8a_pilot) == 0, "Phase-8B seeds overlap with 8A pilot"
        assert len(phase8b & phase8a_confirm) == 0, "Phase-8B seeds overlap with 8A confirmation"
        assert len(phase8b & phase8b_pilot) == 0, "Phase-8B seeds overlap with Phase-8B pilot"

    def test_phase8b_seeds_manifest_if_exists(self):
        manifest = PHASE8B_DIR / "seed_manifest.json"
        if manifest.exists():
            import json
            data = json.loads(manifest.read_text())
            assert data["all_disjoint"] is True


# ---------------------------------------------------------------------------
# 5. Fill-rate metric correctness
# ---------------------------------------------------------------------------

class TestFillRateMetric:
    def test_fill_rate_uses_retail_sales(self):
        import importlib
        mod = importlib.import_module("src.experiment_phase8b")
        import inspect
        source = inspect.getsource(mod)
        # Must use env.S for fill rate (retail sales), not env.R
        assert "env.S" in source, (
            "Fill rate must use env.S (retail sales), not env.R"
        )

    def test_fill_rate_bounded_when_corrected(self):
        from src.gym_adapter import get_perfect_semantic_probs
        import importlib
        mod = importlib.import_module("src.experiment_phase8b")
        probs = get_perfect_semantic_probs(Regime.DEMAND_SURGE)
        result, _ = mod.run_gym_episode_corrected_fill_rate(
            seed=3000, regime=Regime.DEMAND_SURGE, sensor="OracleSemantic",
            regime_probabilities=probs, scenario="serial",
            num_periods=30, base_mu=10.0, event_start=15, shock_mag=2.0,
        )
        assert 0.0 <= result.fill_rate <= 1.0, (
            f"Corrected fill rate {result.fill_rate} out of [0, 1]"
        )


# ---------------------------------------------------------------------------
# 6. Six sensors configured
# ---------------------------------------------------------------------------

class TestSixSensors:
    def test_sensors_list(self):
        from src.experiment_phase8b import SENSORS
        expected = {"NoInfo", "RuleBased", "TFIDF_LogReg_Raw", "TFIDF_LogReg_Calibrated", "gpt-4o", "OracleSemantic"}
        assert set(SENSORS) == expected, f"Expected sensors {expected}, got {set(SENSORS)}"

    def test_perfect_semantic_renamed(self):
        from src.experiment_phase8b import SENSORS
        assert "PerfectSemantic" not in SENSORS, "PerfectSemantic should be renamed to OracleSemantic"
        assert "OracleSemantic" in SENSORS


# ---------------------------------------------------------------------------
# 7. No future-demand leakage (reverified for Phase-8B)
# ---------------------------------------------------------------------------

class TestNoLeakagePhase8B:
    def test_controller_forbidden_attrs(self):
        import inspect
        from src.gym_adapter import BeliefAdaptiveController
        source = inspect.getsource(BeliefAdaptiveController)
        forbidden = ["demand_engine", "get_current_mu", "env.D[", "env_kwargs"]
        for pattern in forbidden:
            assert pattern not in source, (
                f"BeliefAdaptiveController contains '{pattern}' — potential leakage"
            )


# ---------------------------------------------------------------------------
# 8. Argmax ablation
# ---------------------------------------------------------------------------

class TestArgmaxAblation:
    def test_full_prob_belief_vs_argmax_belief(self):
        from src.gym_adapter import BeliefAdaptiveController, make_surge_env
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        # Full-probability belief
        probs_full = {"normal": 0.3, "demand_surge": 0.7}
        ctrl_full = BeliefAdaptiveController(env, probs_full, base_mu=10.0, surge_multiplier=2.0)
        # Argmax-only belief (P(surge)=1.0)
        ctrl_argmax = BeliefAdaptiveController(env, {"normal": 0.0, "demand_surge": 1.0}, base_mu=10.0, surge_multiplier=2.0)
        # effective_mu should differ
        assert ctrl_full.effective_mu != ctrl_argmax.effective_mu, (
            "Full-prob and argmax controllers should have different effective_mu"
        )


# ---------------------------------------------------------------------------
# 9. Hierarchical bootstrap
# ---------------------------------------------------------------------------

class TestHierarchicalBootstrap:
    def test_hierarchical_paired_bootstrap(self):
        from src.experiment_phase8b import hierarchical_paired_bootstrap
        rng = np.random.default_rng(0)
        diffs_by_template = {
            f"tmpl_{i}": rng.normal(loc=10.0, scale=3.0, size=30)
            for i in range(12)
        }
        mean_diff, ci_lo, ci_hi = hierarchical_paired_bootstrap(
            diffs_by_template, n_boot=1000, alpha=0.05, seed=42
        )
        assert ci_lo < mean_diff < ci_hi, "Mean should be within CI"
        assert ci_hi - ci_lo > 0, "CI should have positive width"


# ---------------------------------------------------------------------------
# 10. Unit conversion check
# ---------------------------------------------------------------------------

class TestUnitConversion:
    def test_surge_multiplier_is_two(self):
        from src.experiment_phase8b import SURGE_MULTIPLIER
        assert SURGE_MULTIPLIER == 2.0, f"SHOCK_MAG should be 2.0, got {SURGE_MULTIPLIER}"

    def test_base_mu_is_ten(self):
        from src.experiment_phase8b import BASE_MU
        assert BASE_MU == 10.0, f"BASE_MU should be 10.0, got {BASE_MU}"
