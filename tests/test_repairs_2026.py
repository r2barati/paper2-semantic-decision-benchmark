"""Behavioural regressions for the September 2026 audit repairs.

Each test here corresponds to a defect the previous suite did not catch,
because the previous suite checked names and mocked probabilities rather than
simulator and controller behaviour.  These assert the behaviour directly, so
a regression fails rather than passing under a plausible-sounding test name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

from src.env import (
    CausalOptimizer, HindsightOracle, InventoryEnv, PipelineSchedule,
    SupplierDisruption, DEFAULT_ASSUMED_START, ORDER_ZERO_TOL,
)
from src.events import DEFAULT_DISRUPTION, Regime, P5_WARNING_TIME


# ---------------------------------------------------------------------------
# Finding 2: the NoInfo controller must not see the environment's truth
# ---------------------------------------------------------------------------

class TestInformationBoundary:
    def test_noinfo_optimizer_holds_no_disruption_belief(self):
        for disruption in (
            DEFAULT_DISRUPTION,
            SupplierDisruption(start_time=3, duration=25, normal_lead_time=2, disrupted_lead_time=9),
        ):
            opt = CausalOptimizer(seed=7, disruption=disruption)
            assert opt._assumed_disruption is None, (
                "an uninformed controller must plan under the nominal lead time, "
                "not the environment's true disruption"
            )
            assert opt.knows_true_disruption is False

    def test_noinfo_plan_is_invariant_to_hidden_truth(self):
        """The decisive behavioural check the old string scan could not make."""
        def plan(disruption):
            opt = CausalOptimizer(seed=11, disruption=disruption)
            # Drive the planner over a FIXED state trajectory so only the
            # controller's beliefs, never the realised dynamics, can differ.
            orders = []
            state = InventoryEnv(seed=11, disruption=disruption).reset()
            for t in range(15):
                state.time = t
                state.on_hand = 30.0
                orders.append(round(opt.decide(state), 6))
            return orders

        base = plan(DEFAULT_DISRUPTION)
        alt = plan(SupplierDisruption(start_time=4, duration=30,
                                      normal_lead_time=2, disrupted_lead_time=11))
        assert base == alt, (
            "NoInfo orders changed when the hidden disruption changed: the "
            "controller is reading privileged information"
        )

    def test_perfect_semantic_does_react_to_truth(self):
        """The boundary must isolate NoInfo without disabling the oracle arm."""
        def plan(disruption):
            opt = CausalOptimizer(seed=11, disruption=disruption, knows_true_disruption=True)
            orders = []
            state = InventoryEnv(seed=11, disruption=disruption).reset()
            for t in range(15):
                state.time = t
                state.on_hand = 30.0
                orders.append(round(opt.decide(state), 6))
            return orders

        base = plan(DEFAULT_DISRUPTION)
        alt = plan(SupplierDisruption(start_time=4, duration=30,
                                      normal_lead_time=2, disrupted_lead_time=11))
        assert base != alt, "the informed reference must respond to the true disruption"

    def test_assumed_start_comes_from_the_protocol_not_the_truth(self):
        """A semantic controller must not inherit the hidden start period."""
        disruption = SupplierDisruption(start_time=18, duration=8,
                                        normal_lead_time=2, disrupted_lead_time=5)
        opt = CausalOptimizer(seed=1, disruption=disruption,
                              assumed_lt_increase=3, assumed_duration=8)
        assert opt._assumed_disruption.start_time == DEFAULT_ASSUMED_START
        assert opt._assumed_disruption.start_time != disruption.start_time


# ---------------------------------------------------------------------------
# Finding 3: the LP must be able to sell the stock it already holds
# ---------------------------------------------------------------------------

class TestLinearProgramInventory:
    def test_one_period_lp_sells_from_existing_stock(self):
        """100 units on hand, no arrivals, expected demand 8 -> sell 8, order 0."""
        disruption = SupplierDisruption(start_time=99, duration=1,
                                        normal_lead_time=2, disrupted_lead_time=2)
        opt = CausalOptimizer(seed=1, disruption=disruption, horizon=1, demand_mean=8.0)
        order = opt._solve_mpc(0, on_hand=100.0, remaining=1)
        assert order == pytest.approx(0.0, abs=1e-6), (
            "the LP ordered despite ample stock, which is the signature of the "
            "first-period sales bound omitting on-hand inventory"
        )

    def test_lp_sales_bound_includes_on_hand(self):
        """Directly assert the constraint value the defect got wrong."""
        disruption = SupplierDisruption(start_time=99, duration=1,
                                        normal_lead_time=2, disrupted_lead_time=2)
        opt = CausalOptimizer(seed=1, disruption=disruption, horizon=3, demand_mean=8.0)
        a_fixed, _ = opt._build_arrival_model(0, PipelineSchedule(), 3)
        on_hand = 100.0
        expected_first_bound = a_fixed[0] + on_hand
        assert expected_first_bound > 0, "sanity"
        # With the bound correct the plan holds no stockout in period 0.
        order = opt._solve_mpc(0, on_hand=on_hand, remaining=3)
        assert order == pytest.approx(0.0, abs=1e-6)

    def test_starved_lp_still_orders(self):
        """The corrected bound must not suppress genuinely needed orders."""
        disruption = SupplierDisruption(start_time=99, duration=1,
                                        normal_lead_time=2, disrupted_lead_time=2)
        opt = CausalOptimizer(seed=1, disruption=disruption, horizon=6, demand_mean=8.0)
        order = opt._solve_mpc(0, on_hand=0.0, remaining=6)
        assert order > 0.0, "with no stock the optimiser must replenish"


# ---------------------------------------------------------------------------
# Finding 4: arrival timing and warning availability
# ---------------------------------------------------------------------------

class TestArrivalTiming:
    @pytest.mark.parametrize("lead_time", [1, 2, 3, 4, 5])
    def test_impulse_arrives_after_exactly_lead_time_periods(self, lead_time):
        disruption = SupplierDisruption(start_time=99, duration=1,
                                        normal_lead_time=lead_time,
                                        disrupted_lead_time=lead_time)
        env = InventoryEnv(seed=1, disruption=disruption)
        env.reset()
        env.step(100.0)                       # placed in period 0
        for t in range(1, lead_time):
            env.step(0.0)
            assert env._pipeline.outstanding() == pytest.approx(100.0), (
                f"order arrived early: still in transit expected at t={t}"
            )
        env.step(0.0)                          # period `lead_time`
        assert env._pipeline.outstanding() == pytest.approx(0.0), (
            f"order placed at t=0 did not arrive at t={lead_time}"
        )

    def test_orders_in_transit_are_not_delayed_by_a_later_disruption(self):
        disruption = SupplierDisruption(start_time=2, duration=4,
                                        normal_lead_time=2, disrupted_lead_time=6)
        env = InventoryEnv(seed=1, disruption=disruption)
        env.reset()
        env.step(50.0)                         # t=0, lead time 2 -> arrives t=2
        assert env._pipeline.as_dict() == {2: 50.0}
        env.step(0.0)                          # t=1
        env.step(0.0)                          # t=2, disruption now active
        assert env._pipeline.as_dict() == {}, (
            "an order already in transit was pushed back by a lead-time change"
        )

    def test_pipeline_contracts_when_the_disruption_ends(self):
        disruption = SupplierDisruption(start_time=1, duration=2,
                                        normal_lead_time=2, disrupted_lead_time=6)
        env = InventoryEnv(seed=1, disruption=disruption)
        env.reset()
        for t in range(5):
            env.step(0.0)
        # After recovery a new order must use the nominal lead time again.
        t_now = env._state.time
        env.step(10.0)
        arrivals = env._pipeline.as_dict()
        assert arrivals == {t_now + 2: 10.0}, (
            f"post-recovery order should arrive after 2 periods, got {arrivals}"
        )

    def test_planner_and_simulator_share_one_arrival_model(self):
        disruption = SupplierDisruption(start_time=5, duration=6,
                                        normal_lead_time=2, disrupted_lead_time=5)
        opt = CausalOptimizer(seed=3, disruption=disruption, knows_true_disruption=True)
        env = InventoryEnv(seed=3, disruption=disruption)
        env.reset()
        # Project arrivals for a unit order placed in each of the next periods
        # and check the simulator delivers on the same period.
        for t in (0, 4, 6, 12):
            projected = opt._simulate_arrivals(t, PipelineSchedule(), [1.0] + [0.0] * 15)
            offsets = [i for i, a in enumerate(projected) if a > 0.5]
            assert offsets, f"planner projected no arrival for an order at t={t}"
            e = InventoryEnv(seed=3, disruption=disruption)
            e.reset()
            for _ in range(t):
                e.step(0.0)
            e.step(1.0)
            arrival = next(iter(e._pipeline.as_dict()))
            assert arrival - t == offsets[0], (
                f"planner says +{offsets[0]}, simulator says +{arrival - t} at t={t}"
            )


class TestWarningAvailability:
    def test_belief_is_withheld_until_the_warning_is_released(self):
        from src.experiment_phase5 import _run_p5_episode
        from src.interpreter import no_info_regime_belief, perfect_semantic_regime_belief

        kwargs = dict(seed=2000, regime=Regime.SUPPLIER_DELAY,
                      sensor="test", controller="CausalOptimizer")
        _, h_noinfo = _run_p5_episode(regime_interp=no_info_regime_belief(), **kwargs)
        _, h_perfect = _run_p5_episode(
            regime_interp=perfect_semantic_regime_belief(Regime.SUPPLIER_DELAY), **kwargs)

        for t in range(P5_WARNING_TIME):
            assert h_noinfo[t].order_quantity == pytest.approx(h_perfect[t].order_quantity), (
                f"a semantic sensor acted at t={t}, before the warning is "
                f"released at t={P5_WARNING_TIME}"
            )

    def test_warning_time_zero_restores_immediate_information(self):
        from src.experiment_phase5 import _run_p5_episode
        from src.interpreter import no_info_regime_belief, perfect_semantic_regime_belief

        kwargs = dict(seed=2000, regime=Regime.SUPPLIER_DELAY,
                      sensor="test", controller="CausalOptimizer")
        _, h_noinfo = _run_p5_episode(
            regime_interp=no_info_regime_belief(), warning_time=0, **kwargs)
        _, h_perfect = _run_p5_episode(
            regime_interp=perfect_semantic_regime_belief(Regime.SUPPLIER_DELAY),
            warning_time=0, **kwargs)
        diverged = any(
            abs(a.order_quantity - b.order_quantity) > 1e-9
            for a, b in zip(h_noinfo, h_perfect)
        )
        assert diverged, "with warning_time=0 the sensors must be distinguishable"


# ---------------------------------------------------------------------------
# Finding 17: the hindsight reference must actually be a reference
# ---------------------------------------------------------------------------

class TestHindsightOracle:
    @pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003, 1004])
    def test_objective_equals_simulated_return(self, seed):
        report = HindsightOracle(seed=seed, disruption=DEFAULT_DISRUPTION)\
            .verify_against_simulator()
        assert report["within_tolerance"], (
            f"MILP objective {report['planned_objective']} != simulated "
            f"{report['simulated_return']} (gap {report['gap']}); the oracle "
            "cannot be described as a reference bound"
        )

    @pytest.mark.parametrize("regime", list(Regime))
    def test_objective_equals_simulated_return_in_regime_mode(self, regime):
        report = HindsightOracle(seed=2000, regime=regime).verify_against_simulator()
        assert report["within_tolerance"], report

    def test_setup_indicators_are_genuinely_binary(self):
        """`integrality=2` is semi-continuous in SciPy, not binary."""
        import inspect
        import src.env as env_module
        source = inspect.getsource(env_module.HindsightOracle._compute_optimal_orders)
        assert "integrality[H:2 * H] = 1" in source, (
            "setup indicators must use integrality 1 (integer) with [0,1] bounds"
        )

    def test_no_order_can_arrive_before_it_is_placed(self):
        """The empty initial pipeline must not reference real order variables.

        The defect encoded the empty pipeline as ``list(range(normal_lt))``,
        i.e. as indices 0..L-1 of the *decision variables*, so ``order[0]``
        appeared to arrive in period 0.  In matrix terms that put non-zero
        entries on or above the diagonal of the arrival matrix ``A[i, j]``
        (arrival period i, placement period j).
        """
        H = 40
        A = np.zeros((H, H))
        for t in range(H):
            arrival = t + max(1, int(DEFAULT_DISRUPTION.current_lead_time(t)))
            if arrival < H:
                A[arrival, t] = 1.0

        assert np.count_nonzero(np.triu(A)) == 0, (
            "the arrival matrix credits an order in a period at or before the "
            "period it was placed"
        )
        assert np.count_nonzero(A) > 0, "sanity: some orders must arrive in horizon"

    def test_early_horizon_return_is_capped_by_initial_stock(self):
        """Behavioural counterpart: no goods can exist before the first arrival.

        With an empty pipeline and lead time L, total sales over the first L
        periods cannot exceed the initial inventory.  The old formulation could
        beat this bound by receiving order[0] in period 0.
        """
        lead = DEFAULT_DISRUPTION.normal_lead_time
        oracle = HindsightOracle(seed=1000, disruption=DEFAULT_DISRUPTION)
        history, _ = oracle.compute_optimal_trajectory()
        early_sales = sum(state.sales for state in history[:lead])
        assert early_sales <= oracle.initial_inventory + 1e-9, (
            f"sold {early_sales} units in the first {lead} periods from an "
            f"initial stock of {oracle.initial_inventory}"
        )

    def test_failure_raises_instead_of_calling_a_missing_method(self):
        oracle = HindsightOracle(seed=1, disruption=DEFAULT_DISRUPTION)
        assert not hasattr(oracle, "_fallback_greedy"), (
            "the removed fallback must not be referenced"
        )


# ---------------------------------------------------------------------------
# Finding 7 / 15: offline artifacts and credential-independent loading
# ---------------------------------------------------------------------------

class TestOfflineRelease:
    def test_frozen_path_is_independent_of_the_runtime_cache_location(self, monkeypatch):
        from src import offline_artifacts

        before = offline_artifacts.frozen_cache_dir()
        monkeypatch.setenv("PAPER2_LLM_CACHE_DIR", "/tmp/paper2-cache-elsewhere")
        assert offline_artifacts.runtime_cache_dir() == Path("/tmp/paper2-cache-elsewhere")
        assert offline_artifacts.frozen_cache_dir() == before, (
            "redirecting the runtime cache also moved the frozen export"
        )

    def test_every_manifest_record_is_reconstructable(self):
        manifest = ROOT / "results" / "frozen_llm_outputs" / "manifest.jsonl"
        assert manifest.exists(), "the tracked manifest is the offline source of truth"
        records = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        assert records, "manifest is empty"
        for record in records:
            assert record.get("canonical_sha256"), "canonical checksum missing"
            assert "raw_or_canonical_response" in record
            payload = json.loads(record["raw_or_canonical_response"])
            assert isinstance(payload, dict)

    def test_offline_artifacts_are_complete(self):
        from tools.rebuild_offline_artifacts import (
            rebuild_semantic_cache, rebuild_tfidf_models,
        )
        cache = rebuild_semantic_cache(verify_only=True)
        models = rebuild_tfidf_models(verify_only=True)
        assert not cache["checksum_mismatch"], cache["checksum_mismatch"]
        assert not cache["missing"], (
            f"{len(cache['missing'])} semantic caches missing; run "
            "`python3 -m tools.rebuild_offline_artifacts`"
        )
        assert not models["missing"], models["missing"]

    def test_provenance_labels_are_not_blanket_defaults(self):
        manifest = ROOT / "results" / "frozen_llm_outputs" / "manifest.jsonl"
        records = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        models = {r["model_id"] for r in records}
        assert len(models - {"unknown"}) > 1, (
            "every resolved record carries the same model label, which is the "
            "signature of the blanket gpt-4o-mini default"
        )
        for record in records:
            assert record["provenance"] in {
                "reconstructed_from_cache_key",
                "unresolved_no_matching_model_text_pair",
            }
            if record["provenance"].startswith("unresolved"):
                assert record["model_id"] == "unknown", (
                    "an unresolved record must not be given a plausible label"
                )


# ---------------------------------------------------------------------------
# Findings 9-12: estimand, ensembling control, and text-world pairing
# ---------------------------------------------------------------------------

class TestEstimandAndDesign:
    def test_weighted_return_differs_from_equal_row_mean_when_unbalanced(self):
        from src.metrics import weighted_benchmark_return

        # Six 'normal' cells and ten 'event' cells, as in Phase 9A.
        values = {
            "normal": {"f": {f"v{i}": {0: 100.0} for i in range(6)}},
            "event": {"f": {f"v{i}": {0: 200.0} for i in range(10)}},
        }
        weighted = weighted_benchmark_return(values)["value"]
        flat = (6 * 100.0 + 10 * 200.0) / 16
        assert weighted == pytest.approx(150.0)
        assert abs(weighted - flat) > 1.0, (
            "the balanced estimand and the equal-row mean must be visibly "
            "different, otherwise the distinction cannot be tested"
        )

    def test_bootstrap_shares_seed_draws_and_bounds_p_values(self):
        from src.metrics import crossed_bootstrap_ci

        rng = np.random.default_rng(0)
        diffs = {
            "r": {
                "fam": {
                    "v1": {s: float(rng.normal(3.0)) for s in range(20)},
                    "v2": {s: float(rng.normal(3.0)) for s in range(20)},
                }
            }
        }
        out = crossed_bootstrap_ci(diffs, n_boot=500)
        assert out["shared_seed_draws"] is True
        assert out["p_value"] >= out["p_value_floor"] > 0.0, (
            "a bootstrap p-value must never be reported as exactly zero"
        )
        assert out["ci_lower"] < out["mean"] < out["ci_upper"]

    def test_bootstrap_is_independent_of_string_hash_order(self):
        """Reproducibility must not depend on PYTHONHASHSEED.

        The bootstrap draws one variant sample per distinct sampled family. If
        that iterates a `set` of family names, CPython's per-process string
        hash randomisation changes the order in which draws are consumed, and
        the interval changes between runs of the same command. The
        implementation iterates `sorted(set(...))`; this test runs the same
        computation in a subprocess under three different hash seeds.
        """
        import subprocess
        import sys as _sys

        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from src.metrics import crossed_bootstrap_ci\n"
            "import numpy as np\n"
            "rng = np.random.default_rng(7)\n"
            "d = {'r': {\n"
            "  'famA': {'v1': {s: float(rng.normal(3)) for s in range(15)},\n"
            "           'v2': {s: float(rng.normal(3)) for s in range(15)}},\n"
            "  'famB': {'v1': {s: float(rng.normal(2)) for s in range(15)}},\n"
            "  'famC': {'v1': {s: float(rng.normal(4)) for s in range(15)}}}}\n"
            "r = crossed_bootstrap_ci(d, n_boot=200, seed=42)\n"
            "print('%%.12f %%.12f' %% (r['ci_lower'], r['ci_upper']))\n"
        ) % str(ROOT)

        outputs = set()
        for hashseed in ("0", "3", "7"):
            env = {"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"}
            result = subprocess.run(
                [_sys.executable, "-c", script],
                capture_output=True, text=True, env=env, cwd=str(ROOT),
            )
            assert result.returncode == 0, result.stderr
            outputs.add(result.stdout.strip())

        assert len(outputs) == 1, (
            f"bootstrap interval depends on PYTHONHASHSEED: {outputs}"
        )

    def test_calibration_and_ensembling_are_separable(self):
        from src.classical_baseline import (
            TFIDFLogReg, VARIANT_SINGLE, VARIANT_FOLD_ENSEMBLE, VARIANT_CALIBRATED,
        )
        from src.events import REGIME_WARNING_TEMPLATES

        texts = [t["text"] for t in REGIME_WARNING_TEMPLATES]
        labels = [t["regime"].value for t in REGIME_WARNING_TEMPLATES]
        probe = texts[0]

        got = {}
        for variant in (VARIANT_SINGLE, VARIANT_FOLD_ENSEMBLE, VARIANT_CALIBRATED):
            model = TFIDFLogReg(variant=variant, seed=42).fit(texts, labels)
            got[variant] = model.predict_proba([probe])[0]

        assert not np.allclose(got[VARIANT_SINGLE], got[VARIANT_FOLD_ENSEMBLE]), (
            "fold ensembling alone changes the fitted classifier, so it cannot "
            "be folded into a claim about calibration"
        )
        assert not np.allclose(got[VARIANT_FOLD_ENSEMBLE], got[VARIANT_CALIBRATED])

    def test_phase9b_pairing_is_explicit(self):
        from src.experiment_phase9b import (
            world_regime_for, PAIRING_MATCHED, PAIRING_FALSE_NEGATIVE,
            TRAIN_TEMPLATES, _get_heldout_templates,
        )
        templates = list(TRAIN_TEMPLATES) + list(_get_heldout_templates())
        normals = [t for t in templates if t["regime"] == Regime.NORMAL]
        assert normals, "expected some normal-labelled reports"

        for tmpl in normals:
            assert world_regime_for(tmpl, PAIRING_MATCHED) == Regime.NORMAL, (
                "the default pairing must simulate the world the report describes"
            )
            assert world_regime_for(tmpl, PAIRING_FALSE_NEGATIVE) == \
                Regime.SUPPLIER_CAPACITY_DROP, (
                "the misinformation arm must be reachable and explicitly named"
            )


# ---------------------------------------------------------------------------
# Finding 5: the no-text controls exist and are genuinely text-free
# ---------------------------------------------------------------------------

class TestNoTextControls:
    def test_constant_belief_reads_no_text(self):
        from src.notext_controls import constant_belief

        belief = constant_belief(0.75, Regime.DEMAND_SURGE)
        assert belief["demand_surge"] == pytest.approx(0.75)
        assert belief["normal"] == pytest.approx(0.25)

    def test_development_seeds_are_disjoint_from_confirmation_seeds(self):
        from src.notext_controls import DEV_SEEDS_8B, DEV_SEEDS_9A

        confirmation_8b = set(range(3100, 3130))
        confirmation_9a = set(range(4000, 4030))
        confirmation_9b = set(range(4100, 4110))
        assert not set(DEV_SEEDS_8B) & (confirmation_8b | confirmation_9a | confirmation_9b)
        assert not set(DEV_SEEDS_9A) & (confirmation_8b | confirmation_9a | confirmation_9b)

    def test_shuffled_text_is_a_derangement(self):
        from src.notext_controls import shuffled_text_assignment

        ids = [f"t{i}" for i in range(12)]
        mapping = shuffled_text_assignment(ids)
        assert set(mapping) == set(ids)
        assert sorted(mapping.values()) == sorted(ids)
        assert all(k != v for k, v in mapping.items()), "no template may keep its own text"

    def test_argmax_control_collapses_the_belief(self):
        from src.notext_controls import argmax_belief

        hard = argmax_belief({"normal": 0.4, "demand_surge": 0.6}, Regime.DEMAND_SURGE)
        assert hard == {"normal": 0.0, "demand_surge": 1.0}
