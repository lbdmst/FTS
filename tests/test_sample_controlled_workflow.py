from __future__ import annotations

import unittest

import numpy as np

from evals.sample_controlled_workflow import _run_expected_signal_checks


class SampleAtomicWorkflowTests(unittest.TestCase):
    def test_level_shift_signal_checks_accept_relative_shift_on_nonzero_baseline(self) -> None:
        case = {
            "case_id": "generated_atomic_shift_07",
            "length": 100,
            "expected_plan_outline": {
                "steps": [
                    {
                        "primitive": "add_level_shift",
                        "effect_type": "additive",
                        "parameters": {"anchor_timestep": 55, "shift": -2.5},
                    }
                ]
            },
            "expected_signal_checks": {
                "length": 100,
                "point_values": [
                    {"timestep": 54, "value": 0.0, "tolerance": 1e-6},
                    {"timestep": 55, "value": -2.5, "tolerance": 1e-6},
                    {"timestep": 99, "value": -2.5, "tolerance": 1e-6},
                ]
            },
        }
        series = np.full(100, 0.625, dtype=float)
        series[55:] -= 2.5

        checks = _run_expected_signal_checks(case, series)

        self.assertTrue(all(item["passed"] for item in checks), checks)


if __name__ == "__main__":
    unittest.main()
