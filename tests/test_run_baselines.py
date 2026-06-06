from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.run_baselines import render_markdown, run_methods


class MultiBaselineRunnerTests(unittest.TestCase):
    def test_runner_marks_missing_llm_outputs_as_skipped(self) -> None:
        payload = run_methods(methods=["oracle_program", "llm_direct_array"], category="atomic")

        statuses = {item["method"]: item["status"] for item in payload["results"]}
        self.assertEqual(statuses["oracle_program"], "completed")
        self.assertEqual(statuses["llm_direct_array"], "skipped")
        self.assertEqual(payload["results"][1]["skip_reason"], "missing_output_dir_or_api_key")

    def test_runner_writes_markdown_ready_summary(self) -> None:
        payload = run_methods(methods=["oracle_program"], category="atomic")
        markdown = render_markdown(payload)

        self.assertIn("| Method | Status | Cases |", markdown)
        self.assertIn("oracle_program", markdown)
        json.dumps(payload, allow_nan=False)

    def test_mixed_runner_requires_method_specific_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_methods(methods=["oracle_program", "llm_direct_array"], category="atomic", output_root=Path(tmp))

        result = {item["method"]: item for item in payload["results"]}
        self.assertEqual(result["oracle_program"]["status"], "completed")
        self.assertEqual(result["llm_direct_array"]["status"], "skipped")
        self.assertEqual(result["llm_direct_array"]["skip_reason"], "missing_method_output_dir")

    def test_prompt_only_method_dir_is_not_treated_as_llm_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_dir = root / "llm_direct_array" / "prompts"
            prompt_dir.mkdir(parents=True)
            (prompt_dir / "atomic_flat_constant.txt").write_text("prompt only", encoding="utf-8")

            payload = run_methods(methods=["llm_direct_array"], category="atomic", output_root=root)

        result = payload["results"][0]
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["skip_reason"], "missing_method_output_files")

    def test_ours_full_saved_plan_output_uses_shared_evaluator(self) -> None:
        case = {
            "case_id": "saved_flat_case",
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
            "numeric_range": {"min": 0.0, "max": 4.0},
        }
        plan = {
            "version": "1.0",
            "input_description": case["description"],
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": {
                "length": 4,
                "numeric_range": {"min": 0.0, "max": 4.0},
                "dt": None,
                "seed": None,
            },
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 3, "target_value": 2.0},
                    "semantic_role": "constant signal",
                    "effect_type": "overwrite",
                    "confidence": 1.0,
                    "source_text": case["description"],
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
            "assumptions": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            out_root = root / "outputs"
            out_root.mkdir()
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (out_root / "saved_flat_case.json").write_text(json.dumps({"plan": plan}), encoding="utf-8")

            payload = run_methods(methods=["ours_full"], paths=[cases_path], output_root=out_root)

        result = payload["results"][0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["aggregate"]["csr"], 1.0)
        self.assertTrue(result["cases"][0]["passed"], result["cases"][0]["failure_reasons"])

    def test_verbalts_external_series_output_uses_shared_evaluator(self) -> None:
        case = {
            "case_id": "verbalts_flat_case",
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
            "numeric_range": {"min": 0.0, "max": 4.0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            out_root = root / "outputs"
            out_root.mkdir()
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (out_root / "verbalts_flat_case.json").write_text(
                json.dumps({"generated_series": [2.0, 2.0, 2.0, 2.0], "model": "VerbalTS"}),
                encoding="utf-8",
            )

            payload = run_methods(methods=["verbalts"], paths=[cases_path], output_root=out_root)

        result = payload["results"][0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["aggregate"]["csr"], 1.0)
        self.assertTrue(result["cases"][0]["passed"], result["cases"][0]["failure_reasons"])
        self.assertEqual(result["cases"][0]["eval_report"]["plan_validation_status"]["status"], "not_run")

    def test_verbalts_missing_outputs_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_methods(methods=["verbalts"], category="atomic", output_root=Path(tmp))

        result = payload["results"][0]
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["skip_reason"], "missing_method_output_files")


if __name__ == "__main__":
    unittest.main()
