from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.run_baseline_benchmark import run_baseline


def _category_case_count(category: str) -> int:
    root = Path(__file__).resolve().parents[1] / "data" / "controlled_gold_cases"
    total = 0
    for path in (root / "single_claim_cases.json", root / "multi_claim_cases.json", root / "robustness_cases.json"):
        cases = json.loads(path.read_text(encoding="utf-8"))
        total += sum(1 for case in cases if case.get("category") == category)
    return total


class BaselineBenchmarkTests(unittest.TestCase):
    def test_oracle_baseline_is_upper_bound_on_atomic_cases(self) -> None:
        payload = run_baseline("oracle", category="atomic")

        self.assertEqual(payload["aggregate"]["cases"], float(_category_case_count("atomic")))
        self.assertEqual(payload["aggregate"]["pass_rate"], 1.0)
        self.assertEqual(payload["aggregate"]["csr"], 1.0)
        self.assertEqual(payload["aggregate"]["primitive_f1"], 1.0)
        self.assertEqual(payload["failure_taxonomy"]["failed_cases"], 0)

    def test_no_rejection_ablation_hallucinates_unsupported_adversarial_cases(self) -> None:
        payload = run_baseline("oracle_no_rejection", category="adversarial")

        self.assertEqual(payload["aggregate"]["hus"], 1.0)
        self.assertEqual(payload["aggregate"]["crr"], 0.0)
        self.assertLess(payload["aggregate"]["pass_rate"], 1.0)
        self.assertGreater(payload["failure_taxonomy"]["counts"]["unsupported_hallucination"], 0)

    def test_rule_parser_uses_shared_evaluator_and_serializes_as_strict_json(self) -> None:
        payload = run_baseline("rule_parser", category="atomic")

        self.assertEqual(payload["aggregate"]["cases"], float(_category_case_count("atomic")))
        self.assertIn("csr", payload["aggregate"])
        self.assertIn("primitive_f1", payload["aggregate"])
        self.assertGreater(payload["failure_taxonomy"]["counts"]["wrong_primitive"], 0)
        json.dumps(payload["aggregate"], allow_nan=False)

    def test_repair_flag_records_plan_repair_summary(self) -> None:
        payload = run_baseline("rule_parser", category="atomic", repair=True)

        self.assertTrue(payload["repair_enabled"])
        self.assertEqual(payload["repair_summary"]["cases"], _category_case_count("atomic"))
        self.assertIn("total_iterations", payload["repair_summary"])

    def test_llm_direct_array_output_adapter_scores_series_without_valid_plan(self) -> None:
        case = {
            "case_id": "flat_array_case",
            "category": "atomic",
            "description": "The series remains flat at 2.0 across all 4 timesteps.",
            "length": 4,
            "expected_candidate_primitives": ["add_flat"],
            "expected_plan_outline": {
                "needs_library_extension": False,
                "unresolved_semantics": [],
                "steps": [
                    {
                        "primitive": "add_flat",
                        "effect_type": "overwrite",
                        "parameters": {"start_timestep": 0, "end_timestep": 3, "target_value": 2.0},
                        "semantic_role": "constant signal",
                    }
                ],
            },
            "expected_signal_checks": {
                "length": 4,
                "constant_segments": [{"start": 0, "end": 3, "value": 2.0, "tolerance": 1e-6}],
            },
            "numeric_range": {"min": 0.0, "max": 4.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            out_dir = root / "outputs"
            out_dir.mkdir()
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (out_dir / "flat_array_case.txt").write_text('{"series": [2, 2, 2, 2]}', encoding="utf-8")

            payload = run_baseline("llm_direct_array", paths=[cases_path], output_dir=out_dir)

        self.assertEqual(payload["aggregate"]["csr"], 1.0)
        self.assertEqual(payload["aggregate"]["primitive_f1"], 0.0)
        self.assertTrue(payload["cases"][0]["passed"], payload["cases"][0]["failure_reasons"])
        self.assertEqual(payload["cases"][0]["eval_report"]["plan_validation_status"]["status"], "not_run")
        self.assertEqual(payload["failure_taxonomy"]["failed_cases"], 0)

    def test_llm_direct_code_output_adapter_scores_executable_signal_without_valid_plan(self) -> None:
        case = {
            "case_id": "flat_code_case",
            "category": "atomic",
            "description": "The series remains flat at 2.0 across all 4 timesteps.",
            "length": 4,
            "expected_candidate_primitives": ["add_flat"],
            "expected_plan_outline": {
                "needs_library_extension": False,
                "unresolved_semantics": [],
                "steps": [
                    {
                        "primitive": "add_flat",
                        "effect_type": "overwrite",
                        "parameters": {"start_timestep": 0, "end_timestep": 3, "target_value": 2.0},
                        "semantic_role": "constant signal",
                    }
                ],
            },
            "expected_signal_checks": {
                "length": 4,
                "constant_segments": [{"start": 0, "end": 3, "value": 2.0, "tolerance": 1e-6}],
            },
            "numeric_range": {"min": 0.0, "max": 4.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            out_dir = root / "outputs"
            out_dir.mkdir()
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (out_dir / "flat_code_case.txt").write_text(
                "import numpy as np\nseries = np.full(4, 2.0)",
                encoding="utf-8",
            )

            payload = run_baseline("llm_direct_code", paths=[cases_path], output_dir=out_dir)

        self.assertEqual(payload["aggregate"]["csr"], 1.0)
        self.assertTrue(payload["cases"][0]["passed"], payload["cases"][0]["failure_reasons"])
        self.assertEqual(payload["cases"][0]["eval_report"]["plan_validation_status"]["status"], "not_run")
        self.assertEqual(payload["failure_taxonomy"]["failed_cases"], 0)

    def test_llm_direct_code_output_adapter_records_execution_failures(self) -> None:
        case = {
            "case_id": "bad_code_case",
            "category": "atomic",
            "description": "The series remains flat at 2.0 across all 4 timesteps.",
            "length": 4,
            "expected_candidate_primitives": ["add_flat"],
            "expected_plan_outline": {
                "needs_library_extension": False,
                "unresolved_semantics": [],
                "steps": [
                    {
                        "primitive": "add_flat",
                        "effect_type": "overwrite",
                        "parameters": {"start_timestep": 0, "end_timestep": 3, "target_value": 2.0},
                        "semantic_role": "constant signal",
                    }
                ],
            },
            "expected_signal_checks": {
                "length": 4,
                "constant_segments": [{"start": 0, "end": 3, "value": 2.0, "tolerance": 1e-6}],
            },
            "numeric_range": {"min": 0.0, "max": 4.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            out_dir = root / "outputs"
            out_dir.mkdir()
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (out_dir / "bad_code_case.txt").write_text(
                "import numpy as np\nseries = np.array([2, 2, 2])",
                encoding="utf-8",
            )

            payload = run_baseline("llm_direct_code", paths=[cases_path], output_dir=out_dir)

        self.assertEqual(payload["aggregate"]["csr"], 0.0)
        self.assertIn("Direct code output length 3 does not match expected length 4", payload["cases"][0]["eval_report"]["execution_status"]["errors"][0])
        self.assertIn("length_mismatch", payload["failure_taxonomy"]["case_labels"]["bad_code_case"])


if __name__ == "__main__":
    unittest.main()
