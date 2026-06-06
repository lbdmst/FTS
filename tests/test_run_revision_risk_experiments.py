from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.run_revision_risk_experiments import (
    build_public_baseline_matrix,
    run_edit_locality_benchmark,
    run_public_generator_comparison,
    run_revision_risk_experiments,
)


class RevisionRiskExperimentTests(unittest.TestCase):
    def test_public_baseline_matrix_names_public_generators(self) -> None:
        matrix = build_public_baseline_matrix()
        names = {item["baseline_id"] for item in matrix["baselines"]}

        self.assertIn("t2s_text_to_series", names)
        self.assertIn("verbalts", names)
        self.assertIn(
            matrix["status"],
            {"comparison_protocol_ready_external_runs_pending", "t2s_and_verbalts_case_level_2500_completed"},
        )

    def test_edit_locality_benchmark_has_local_oracle_advantage(self) -> None:
        payload = run_edit_locality_benchmark()
        aggregate = {item["method"]: item for item in payload["aggregate"]}

        self.assertLess(aggregate["local_program_edit_oracle"]["mean_outside_drift"], 0.01)
        self.assertGreater(
            aggregate["local_program_edit_oracle"]["mean_locality_score"],
            aggregate["global_rerun_control"]["mean_locality_score"],
        )

    def test_runner_writes_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = run_revision_risk_experiments(
                report_dir=root,
                latency_methods=["oracle_program"],
                latency_repeats=1,
            )

            json_path = root / "revision_risk_report.json"
            md_path = root / "revision_risk_report.md"
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("edit_locality", payload)
            self.assertIn("Public Text-Conditioned Baselines", md_path.read_text(encoding="utf-8"))
            self.assertTrue((root / "public_generator_inputs" / "public_generator_input_manifest.json").exists())
            self.assertEqual(payload["public_generator_inputs"]["status"], "completed")
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["version"], "1.0")

    def test_public_generator_outputs_can_be_scored(self) -> None:
        case = {
            "case_id": "public_flat",
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
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            output_dir = root / "public_outputs" / "t2s_text_to_series"
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            output_dir.mkdir(parents=True)
            (output_dir / "public_flat.json").write_text(
                json.dumps({"generated_series": [2.0, 2.0, 2.0, 2.0], "seconds": 0.2, "seed": 7}),
                encoding="utf-8",
            )

            payload = run_public_generator_comparison(output_root=root / "public_outputs", case_paths=[cases_path])
            results = {item["baseline_id"]: item for item in payload["results"]}

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(results["t2s_text_to_series"]["status"], "completed")
        self.assertEqual(results["t2s_text_to_series"]["aggregate"]["csr"], 1.0)


if __name__ == "__main__":
    unittest.main()
