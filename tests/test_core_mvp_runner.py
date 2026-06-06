from __future__ import annotations

import unittest

from evals.run_core_mvp import run_cases
from schema_validator import validate_payload


class CoreMVPRunnerTests(unittest.TestCase):
    def test_sample_core_mvp_cases_pass_and_emit_schema_valid_reports(self) -> None:
        payload = run_cases()

        self.assertEqual(payload["aggregate"]["cases"], 2.0)
        self.assertEqual(payload["aggregate"]["pass_rate"], 1.0)
        self.assertIsNone(payload["aggregate"]["crr"])
        self.assertIsNone(payload["aggregate"]["hus"])
        for case in payload["cases"]:
            self.assertTrue(case["passed"], case["failure_reasons"])
            self.assertEqual(validate_payload("eval_report", case["eval_report"])["status"], "valid")


if __name__ == "__main__":
    unittest.main()
