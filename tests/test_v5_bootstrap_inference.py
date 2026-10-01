"""Synthetic unit tests for the frozen-then-amended V5 bootstrap inference.

No episodes, no models, no network. Validates the null-centered bootstrap
p-value formula and Holm behavior on constructed diff frames, plus the
selective_row merge repair against the equivalent paired-difference path.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v5_analyze_confirmation_test",
    ROOT / "versions/sem2act-v5/experiments/analyze_confirmation.py",
)
an = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(an)


def make_diff(offset, nq=12, ns=5, spread=0.2):
    rows = []
    reg = ["normal", "supplier_delay", "demand_surge"]
    for i in range(nq):
        for j in range(ns):
            noise = (((i * 7 + j * 13) % 5) - 2) * spread
            rows.append({"query_id": f"q{i:04d}", "seed": 61000 + j,
                         "true_regime": reg[i % 3], "diff": offset + noise})
    return pd.DataFrame(rows)


class BootstrapPTests(unittest.TestCase):
    def test_strong_positive_gives_small_p(self):
        out = an.bootstrap_effect(make_diff(10.0))
        self.assertAlmostEqual(out["estimate"], 10.0, places=1)
        self.assertLess(out["p_value"], 0.01)
        self.assertGreater(out["ci_lo"], 0)

    def test_strong_negative_gives_small_p(self):
        out = an.bootstrap_effect(make_diff(-10.0))
        self.assertAlmostEqual(out["estimate"], -10.0, places=1)
        self.assertLess(out["p_value"], 0.01)
        self.assertLess(out["ci_hi"], 0)

    def test_true_zero_gives_nonsmall_p(self):
        rows = []
        reg = ["normal", "supplier_delay", "demand_surge"]
        for i in range(12):
            for j in range(5):
                rows.append({"query_id": f"q{i:04d}", "seed": 61000 + j,
                             "true_regime": reg[i % 3],
                             "diff": 1.0 if (i + j) % 2 else -1.0})
        out = an.bootstrap_effect(pd.DataFrame(rows))
        self.assertAlmostEqual(out["estimate"], 0.0, places=9)
        self.assertEqual(out["p_value"], 1.0)
        self.assertLess(out["ci_lo"], 0)
        self.assertGreater(out["ci_hi"], 0)

    def test_sign_symmetry(self):
        pos = an.bootstrap_effect(make_diff(4.0))
        neg = an.bootstrap_effect(make_diff(-4.0))
        self.assertEqual(pos["p_value"], neg["p_value"])
        self.assertAlmostEqual(neg["estimate"], -pos["estimate"], places=9)
        self.assertAlmostEqual(neg["ci_lo"], -pos["ci_hi"], places=9)
        self.assertAlmostEqual(neg["ci_hi"], -pos["ci_lo"], places=9)


class HolmTests(unittest.TestCase):
    def test_step_down_matches_hand_computation(self):
        rows = [{"p_value": 0.01}, {"p_value": 0.02}, {"p_value": 0.03}]
        out = an.holm(rows)
        # step-down with running-maximum monotonicity: (3)(.01)=.03,
        # max(.03,(2)(.02)=.04), max(.04,(1)(.03)=.03)
        self.assertEqual([r["p_holm"] for r in out], [0.03, 0.04, 0.04])
        self.assertTrue(all(r["reject_holm_05"] for r in out))

    def test_cap_and_monotonicity(self):
        rows = [{"p_value": 0.03}, {"p_value": 0.01}]
        out = an.holm(rows)
        self.assertEqual([r["p_holm"] for r in out], [0.03, 0.02])
        rows = [{"p_value": 0.4}, {"p_value": 0.5}]
        out = an.holm(rows)
        self.assertEqual([r["p_holm"] for r in out], [0.8, 0.8])
        self.assertTrue(all(not r["reject_holm_05"] for r in out))


class SelectiveRepairTests(unittest.TestCase):
    def _frame(self):
        rows = []
        for i in range(6):
            for j in range(3):
                for method, delta in (("always_consume", 0.0), ("selective_entropy", 2.0)):
                    rows.append({
                        "model": an.QWEN, "system": "rerank",
                        "comparison_system": "rerank", "consumer": "C1",
                        "controller": "CausalOptimizer", "method": method,
                        "condition": "persistent", "horizon": 40,
                        "query_id": f"q{i:04d}", "seed": 61000 + j,
                        "true_regime": ["normal", "supplier_delay", "demand_surge"][i % 3],
                        "profit": 100.0 + i + 0.1 * j + delta,
                    })
        return pd.DataFrame(rows)

    def test_repaired_helper_matches_equivalent_estimand(self):
        frame = self._frame()
        got = an.selective_row(frame, "C1")
        sel = an.subset(frame, model=an.QWEN, system="rerank", consumer="C1",
                        controller="CausalOptimizer", method="selective_entropy")
        alw = an.subset(frame, model=an.QWEN, system="rerank", consumer="C1",
                        controller="CausalOptimizer", method="always_consume")
        want = an.bootstrap_effect(an.paired_difference(sel, alw))
        want = {"model": an.QWEN, "system": "rerank", "consumer": "C1",
                "controller": "CausalOptimizer",
                "comparison": "selective_entropy_minus_always_consume", **want}
        self.assertEqual(got, want)
        self.assertAlmostEqual(got["estimate"], 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
