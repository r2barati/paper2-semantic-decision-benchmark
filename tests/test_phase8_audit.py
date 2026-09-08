"""Phase 8 validity audit tests.

These tests verify the scientific validity of the Phase 8 experiment
without modifying any existing results or code paths.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Paper 1" / "final-github-clean" / "gym-invmgmt-paper"))

from src.events import Regime, REGIME_WARNING_TEMPLATES


# ---------------------------------------------------------------------------
# Audit 1: Paper-1 provenance
# ---------------------------------------------------------------------------

class TestAudit1Provenance:
    def test_gym_invmgmt_importable(self):
        import gym_invmgmt
        assert gym_invmgmt.__file__ is not None

    def test_gym_invmgmt_version(self):
        import gym_invmgmt
        assert hasattr(gym_invmgmt, "__version__")
        assert gym_invmgmt.__version__ == "0.1.0"

    def test_core_env_source_path(self):
        import inspect
        from gym_invmgmt.core_env import CoreEnv
        f = inspect.getfile(CoreEnv)
        assert "Paper 1" in f or "gym-invmgmt-paper" in f

    def test_core_env_matches_installed(self):
        import gym_invmgmt
        init_file = Path(gym_invmgmt.__file__).resolve()
        assert init_file.parent.name == "gym_invmgmt"
        assert (init_file.parent / "core_env.py").exists()


# ---------------------------------------------------------------------------
# Audit 2: No future-demand leakage
# ---------------------------------------------------------------------------

class TestAudit2NoLeakage:
    def test_d_empty_at_reset(self):
        from src.gym_adapter import make_surge_env
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        env.reset(seed=42)
        assert np.all(env.D == 0), "env.D must be all zeros immediately after reset"

    def test_d_filled_incrementally(self):
        from src.gym_adapter import make_surge_env
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        env.reset(seed=42)
        action = np.array([10.0, 10.0, 10.0])
        for t in range(5):
            env.step(action)
            d_nz = np.count_nonzero(env.D)
            assert d_nz == t + 1, f"After {t+1} steps, D should have {t+1} nonzero entries, got {d_nz}"

    def test_x_y_future_rows_zero_at_t0(self):
        from src.gym_adapter import make_surge_env
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        env.reset(seed=42)
        assert np.all(env.X[1:] == 0), "X rows after period 0 must be zero at reset"
        assert np.all(env.Y[1:] == 0), "Y rows after period 0 must be zero at reset"

    def test_controller_does_not_read_demand_engine(self):
        import inspect
        from src.gym_adapter import BeliefAdaptiveController
        source = inspect.getsource(BeliefAdaptiveController)
        forbidden = ["demand_engine", "get_current_mu", "env.D[", "env_kwargs"]
        for pattern in forbidden:
            assert pattern not in source, (
                f"BeliefAdaptiveController contains '{pattern}' — potential leakage channel"
            )

    def test_controller_does_not_use_obs(self):
        import inspect
        from src.gym_adapter import BeliefAdaptiveController
        source = inspect.getsource(BeliefAdaptiveController.get_action)
        assert "obs[" not in source, "get_action must not parse obs (would bypass sanitization)"

    def test_no_info_no_leakage_across_seeds(self):
        from src.gym_adapter import run_gym_episode, get_no_info_probs
        probs = get_no_info_probs()
        rewards = []
        for seed in [2000, 2001, 2002]:
            result, _ = run_gym_episode(
                seed=seed, regime=Regime.DEMAND_SURGE, sensor="NoInfo",
                regime_probabilities=probs, scenario="serial",
                num_periods=30, base_mu=10.0, event_start=15, shock_mag=2.0,
            )
            rewards.append(result.total_reward)
        assert len(set(rewards)) == len(rewards), "Different seeds must produce different rewards"

    def test_noinfo_identical_across_templates(self):
        from src.gym_adapter import run_gym_episode, get_no_info_probs
        probs = get_no_info_probs()
        rewards = {}
        for tmpl in REGIME_WARNING_TEMPLATES:
            result, _ = run_gym_episode(
                seed=2000, regime=tmpl["regime"], sensor="NoInfo",
                regime_probabilities=probs, scenario="serial",
                num_periods=30, base_mu=10.0, event_start=15, shock_mag=2.0,
            )
            rewards[tmpl["template_id"]] = result.total_reward
        unique_rewards = set(rewards.values())
        assert len(unique_rewards) == 2, (
            f"NoInfo should produce only 2 unique rewards (one per regime), got {len(unique_rewards)}"
        )


# ---------------------------------------------------------------------------
# Audit 4: Surge timing
# ---------------------------------------------------------------------------

class TestAudit4SurgeTiming:
    def test_mu_before_shock(self):
        from src.gym_adapter import make_surge_env
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        for t in range(15):
            mu = env.demand_engine.get_current_mu(t)
            assert mu == 10.0, f"mu({t}) should be 10.0, got {mu}"

    def test_mu_during_shock(self):
        from src.gym_adapter import make_surge_env
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        for t in range(15, 30):
            mu = env.demand_engine.get_current_mu(t)
            assert mu == 20.0, f"mu({t}) should be 20.0, got {mu}"

    def test_shock_increases_demand(self):
        from src.gym_adapter import make_surge_env
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        env.reset(seed=42)
        action = np.array([15.0, 15.0, 15.0])
        pre_shock_demands = []
        post_shock_demands = []
        for t in range(30):
            env.step(action)
            if t < 10:
                pre_shock_demands.append(env.D[t, 0])
            elif t >= 15:
                post_shock_demands.append(env.D[t, 0])
        pre_mean = np.mean(pre_shock_demands)
        post_mean = np.mean(post_shock_demands)
        assert post_mean > pre_mean, (
            f"Post-shock mean ({post_mean:.1f}) should exceed pre-shock ({pre_mean:.1f})"
        )


# ---------------------------------------------------------------------------
# Audit 7: TF-IDF model identity
# ---------------------------------------------------------------------------

class TestAudit7TFIDFIdentity:
    def test_calibrated_model_loaded(self):
        pkl = ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_calibrated_model.pkl"
        assert pkl.exists(), "Calibrated TF-IDF model must exist"
        from src.classical_baseline import TFIDFLogReg
        model = TFIDFLogReg.load(str(pkl))
        assert model.calibrated_model_ is not None, "Model must have calibration applied"

    def test_calibrated_is_confident_but_not_degenerate_in_sample(self):
        """In-sample confidence, measured rather than assumed.

        The pre-September-2026 pipeline fitted the TF-IDF vocabulary on all 18
        training texts *before* the internal calibration split, and isotonic
        regression on six calibration points then drove in-sample
        probabilities to exactly 0/1.  The old assertions (`> 0.95`, `> 0.99`)
        encoded that leak as an expectation.

        With the vocabulary fitted inside each fold the model is still
        confident on its own training data -- which is precisely why the
        confirmation experiments use held-out templates -- but no longer
        numerically degenerate.  This test pins the corrected behaviour and
        reports the observed range instead of asserting a saturation artifact.
        """
        from src.classical_baseline import TFIDFLogReg
        pkl = ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_calibrated_model.pkl"
        model = TFIDFLogReg.load(str(pkl))

        p_true = []
        for tmpl in REGIME_WARNING_TEMPLATES:
            result = model.predict_regime(tmpl["text"])
            p_true.append(result.regime_probabilities.get(tmpl["regime"].value, 0.0))

        assert min(p_true) > 0.5, (
            f"calibrated model should still rank the true regime first in sample; "
            f"observed min P(true)={min(p_true):.4f}"
        )
        # Fixing the vocabulary leak raised the observed floor from a fully
        # degenerate 0/1 map to roughly 0.83, but ISOTONIC regression on six
        # calibration points still saturates some in-sample texts at exactly
        # 1.0.  That is scikit-learn's documented small-sample behaviour, not
        # a leak, and the sigmoid arm below is the alternative that avoids it.
        n_degenerate = sum(1 for p in p_true if p >= 1.0 - 1e-12)
        assert n_degenerate < len(p_true), (
            "every in-sample probability is degenerate, which would indicate "
            "the fold vocabulary leak has returned"
        )

    def test_sigmoid_calibration_avoids_small_sample_degeneracy(self):
        """The small-sample-appropriate calibration arm.

        With three folds over 18 texts each calibration set holds six points.
        Isotonic regression saturates there; Platt/sigmoid scaling fits two
        parameters and does not.  Both arms are shipped so the paper can
        report the comparison instead of picking one silently.
        """
        from src.classical_baseline import TFIDFLogReg
        base = ROOT / "results" / "phase7_classical_baseline"
        sig = TFIDFLogReg.load(str(base / "tfidf_logreg_calibrated_sigmoid.pkl"))

        p_true = [
            sig.predict_regime(t["text"]).regime_probabilities.get(t["regime"].value, 0.0)
            for t in REGIME_WARNING_TEMPLATES
        ]
        assert min(p_true) > 0.5, f"sigmoid should still rank truth first: {min(p_true):.4f}"
        assert max(p_true) < 1.0 - 1e-9, (
            f"sigmoid calibration should not saturate; max P(true)={max(p_true):.6f}"
        )

    def test_fold_ensemble_control_exists_and_differs_from_calibrated(self):
        """The control that separates calibration from ensembling.

        `CalibratedClassifierCV` refits one classifier per fold, so the
        calibrated pipeline differs from the single-model pipeline in *two*
        ways.  The fold-ensemble checkpoint is the arm that isolates the
        ensembling half; without it the raw-vs-calibrated contrast is a
        comparison of training pipelines, not of calibration.
        """
        from src.classical_baseline import TFIDFLogReg, VARIANT_FOLD_ENSEMBLE
        base = ROOT / "results" / "phase7_classical_baseline"
        ens = TFIDFLogReg.load(str(base / "tfidf_logreg_fold_ensemble.pkl"))
        cal = TFIDFLogReg.load(str(base / "tfidf_logreg_calibrated_model.pkl"))
        assert ens.variant == VARIANT_FOLD_ENSEMBLE
        assert ens.calibrated_model_ is None, "control must carry no calibration map"
        assert ens.fold_models_ and len(ens.fold_models_) == ens.cv

        text = REGIME_WARNING_TEMPLATES[0]["text"]
        pe = ens.predict_regime(text).regime_probabilities
        pc = cal.predict_regime(text).regime_probabilities
        assert max(abs(pe[k] - pc.get(k, 0.0)) for k in pe) > 1e-6, (
            "fold ensemble and calibrated model must be distinguishable, "
            "otherwise the control arm is vacuous"
        )

    def test_fold_sizes_are_recorded(self):
        """Training/calibration counts must be reportable, not guessed."""
        from src.classical_baseline import TFIDFLogReg
        pkl = ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_calibrated_model.pkl"
        model = TFIDFLogReg.load(str(pkl))
        assert model.fold_sizes_, "calibration fold sizes must be persisted"
        assert all(f["calibration"] > 0 for f in model.fold_sizes_)
        assert sum(f["calibration"] for f in model.fold_sizes_) == model.train_size_


# ---------------------------------------------------------------------------
# Audit 8: Controller probability sensitivity
# ---------------------------------------------------------------------------

class TestAudit8ControllerSensitivity:
    def test_effective_mu_linear_in_p_surge(self):
        from src.gym_adapter import make_surge_env, BeliefAdaptiveController
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        for p_surge in [0.0, 0.25, 0.5, 0.75, 1.0]:
            probs = {"normal": 1.0 - p_surge, "demand_surge": p_surge}
            ctrl = BeliefAdaptiveController(env, probs, base_mu=10.0, surge_multiplier=2.0)
            expected = 10.0 * (1.0 + p_surge)
            assert abs(ctrl.effective_mu - expected) < 1e-9, (
                f"P(surge)={p_surge}: effective_mu={ctrl.effective_mu} != {expected}"
            )

    def test_node_targets_vary_with_probability(self):
        from src.gym_adapter import make_surge_env, BeliefAdaptiveController
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        targets_low = {}
        targets_high = {}
        for p_surge, targets in [(0.0, targets_low), (1.0, targets_high)]:
            probs = {"normal": 1.0 - p_surge, "demand_surge": p_surge}
            ctrl = BeliefAdaptiveController(env, probs, base_mu=10.0, surge_multiplier=2.0)
            targets.update(ctrl.node_targets)
        for node in targets_low:
            assert targets_high[node] > targets_low[node], (
                f"Node {node}: target at P=1.0 ({targets_high[node]:.2f}) "
                f"should exceed target at P=0.0 ({targets_low[node]:.2f})"
            )

    def test_actions_diverge_after_initial_buffer(self):
        from src.gym_adapter import run_gym_episode
        from src.gym_adapter import make_surge_env, BeliefAdaptiveController
        env0 = make_surge_env(seed=42, scenario="serial", num_periods=30,
                              base_mu=10.0, shock_mag=2.0, shock_time=15)
        env1 = make_surge_env(seed=42, scenario="serial", num_periods=30,
                              base_mu=10.0, shock_mag=2.0, shock_time=15)
        probs0 = {"normal": 1.0, "demand_surge": 0.0}
        probs1 = {"normal": 0.0, "demand_surge": 1.0}
        ctrl0 = BeliefAdaptiveController(env0, probs0, base_mu=10.0)
        ctrl1 = BeliefAdaptiveController(env1, probs1, base_mu=10.0)

        obs0, _ = env0.reset(seed=42)
        obs1, _ = env1.reset(seed=42)
        actions0 = []
        actions1 = []
        for t in range(30):
            a0 = ctrl0.get_action(obs0, t)
            a1 = ctrl1.get_action(obs1, t)
            actions0.append(a0.copy())
            actions1.append(a1.copy())
            a0c = np.clip(a0, 0, float(env0.action_space.high[0]))
            a1c = np.clip(a1, 0, float(env1.action_space.high[0]))
            obs0, _, _, _, _ = env0.step(a0c)
            obs1, _, _, _, _ = env1.step(a1c)

        diffs = [np.sum(np.abs(a0 - a1)) for a0, a1 in zip(actions0, actions1)]
        first_diff_t = next((t for t, d in enumerate(diffs) if d > 1e-6), None)
        assert first_diff_t is not None, "Actions must diverge at some point"
        assert first_diff_t <= 3, f"Actions should diverge by t=3, diverged at t={first_diff_t}"


# ---------------------------------------------------------------------------
# Audit 9: TF-IDF equals OracleSemantic
# ---------------------------------------------------------------------------

class TestAudit9TFIDFVersusOracle:
    """Is the classical interpreter distinguishable from the oracle?

    The original tests asserted that calibrated TF-IDF *equalled* the oracle
    belief on training templates.  That was true, and it was a defect: it made
    the interpreter an oracle proxy on its own training data.  The tests are
    inverted here.  What the confirmation design actually requires is that on
    HELD-OUT templates the interpreter is clearly not an oracle proxy; the
    in-sample case is documented, not asserted as a target.
    """

    @staticmethod
    def _model():
        from src.classical_baseline import TFIDFLogReg
        pkl = ROOT / "results" / "phase7_classical_baseline" / "tfidf_logreg_calibrated_model.pkl"
        return TFIDFLogReg.load(str(pkl))

    def test_not_oracle_equivalent_on_heldout_templates(self):
        from src.gym_adapter import get_perfect_semantic_probs, get_tfidf_probs
        from src.confirmation_templates import CONFIRMATION_TEMPLATES

        model = self._model()
        deviations = []
        for tmpl in CONFIRMATION_TEMPLATES:
            regime = tmpl["regime"]
            if regime not in (Regime.NORMAL, Regime.DEMAND_SURGE):
                continue
            oracle = get_perfect_semantic_probs(regime)
            got = get_tfidf_probs(tmpl["text"], model)
            deviations.append(max(abs(oracle[k] - got[k]) for k in ("normal", "demand_surge")))

        assert deviations, "no held-out normal/surge templates found"
        assert max(deviations) > 1e-3, (
            "calibrated TF-IDF is numerically indistinguishable from the oracle "
            "on held-out text, so the confirmation experiment would be comparing "
            "the oracle against itself"
        )

    def test_in_sample_two_class_belief_is_still_oracle_equivalent(self):
        """Records a real limitation rather than asserting it away.

        Two projections must not be confused:

        * The 3-class classifier output is confident but not degenerate
          in-sample once the fold vocabulary leak is fixed (min P(true) ~ 0.83).
        * The gym adapter renormalises onto {normal, demand_surge}, discarding
          the supplier_delay mass.  Under that projection the residual
          probability on the *other* relevant class is zero for every training
          template, so the belief the controller actually consumes collapses
          to exactly the oracle belief.

        This is why the confirmation experiments must use held-out templates,
        and it is the fact the paper states.  The complementary held-out test
        above is the one that licenses the design.
        """
        from src.gym_adapter import get_tfidf_probs

        model = self._model()
        considered, degenerate = 0, 0
        for tmpl in REGIME_WARNING_TEMPLATES:
            if tmpl["regime"] not in (Regime.NORMAL, Regime.DEMAND_SURGE):
                continue
            considered += 1
            p_true = get_tfidf_probs(tmpl["text"], model)[tmpl["regime"].value]
            assert p_true > 0.5, f"{tmpl['template_id']}: P(true)={p_true:.4f}"
            if p_true >= 1.0 - 1e-12:
                degenerate += 1

        assert considered == 12, f"expected 12 in-sample normal/surge templates, got {considered}"
        assert degenerate == considered, (
            "in-sample two-class beliefs are no longer oracle-equivalent; the "
            "documented limitation has changed and the manuscript text that "
            f"describes it must be updated ({degenerate}/{considered} degenerate)"
        )

    def test_three_class_in_sample_belief_is_not_degenerate(self):
        """The complementary projection, so the two are never conflated."""
        model = self._model()
        p_true = [
            model.predict_regime(t["text"]).regime_probabilities.get(t["regime"].value, 0.0)
            for t in REGIME_WARNING_TEMPLATES
        ]
        assert min(p_true) > 0.5
        assert min(p_true) < 1.0 - 1e-9, (
            "the 3-class in-sample distribution should retain some mass off "
            f"the true class; observed min P(true)={min(p_true):.6f}"
        )


# ---------------------------------------------------------------------------
# Audit 11: Statistical pairing
# ---------------------------------------------------------------------------

class TestAudit11Statistical:
    def test_operational_results_exist(self):
        csv_path = ROOT / "results" / "phase8_gym_replication" / "operational_results.csv"
        assert csv_path.exists(), "operational_results.csv must exist"

    def test_20_seeds_present(self):
        import pandas as pd
        df = pd.read_csv(ROOT / "results" / "phase8_gym_replication" / "operational_results.csv")
        seeds = sorted(df["seed"].unique())
        assert len(seeds) == 20, f"Expected 20 seeds, got {len(seeds)}"
        assert seeds[0] == 2000, f"First seed should be 2000, got {seeds[0]}"
        assert seeds[-1] == 2019, f"Last seed should be 2019, got {seeds[-1]}"

    def test_all_five_sensors_present(self):
        import pandas as pd
        df = pd.read_csv(ROOT / "results" / "phase8_gym_replication" / "operational_results.csv")
        expected = {"NoInfo", "RuleBased", "TFIDF_LogReg", "gpt-4o", "PerfectSemantic"}
        actual = set(df["sensor"].unique())
        assert expected == actual, f"Expected sensors {expected}, got {actual}"

    def test_rows_per_sensor(self):
        import pandas as pd
        df = pd.read_csv(ROOT / "results" / "phase8_gym_replication" / "operational_results.csv")
        counts = df.groupby("sensor").size()
        for sensor, count in counts.items():
            assert count == 240, f"{sensor}: expected 240 rows (20 seeds × 12 templates), got {count}"


# ---------------------------------------------------------------------------
# Audit 4 (supplementary): Episode horizon
# ---------------------------------------------------------------------------

class TestAudit4Horizon:
    def test_episode_length(self):
        from src.gym_adapter import make_surge_env, BeliefAdaptiveController, get_no_info_probs
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        probs = get_no_info_probs()
        ctrl = BeliefAdaptiveController(env, probs, base_mu=10.0)
        obs, _ = env.reset(seed=42)
        steps = 0
        for t in range(30):
            action = ctrl.get_action(obs, t)
            action = np.clip(action, 0, float(env.action_space.high[0]))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert steps == 30, f"Episode should run 30 periods, ran {steps}"

    def test_shock_observed_within_episode(self):
        from src.gym_adapter import make_surge_env, BeliefAdaptiveController, get_perfect_semantic_probs
        env = make_surge_env(seed=42, scenario="serial", num_periods=30,
                             base_mu=10.0, shock_mag=2.0, shock_time=15)
        probs = get_perfect_semantic_probs(Regime.DEMAND_SURGE)
        ctrl = BeliefAdaptiveController(env, probs, base_mu=10.0)
        obs, _ = env.reset(seed=42)
        pre_shock = []
        post_shock = []
        for t in range(30):
            action = ctrl.get_action(obs, t)
            action = np.clip(action, 0, float(env.action_space.high[0]))
            obs, reward, terminated, truncated, info = env.step(action)
            if t < 10:
                pre_shock.append(reward)
            elif t >= 15:
                post_shock.append(reward)
        assert len(pre_shock) == 10
        assert len(post_shock) == 15
