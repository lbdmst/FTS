from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.run_research_experiments import run_research_experiments


class ResearchExperimentRunnerTests(unittest.TestCase):
    def test_runs_end_to_end_on_tiny_case_file(self) -> None:
        case = {
            "case_id": "research_flat",
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
            "global_context": {"length": 4, "numeric_range": {"min": 0.0, "max": 4.0}, "dt": None, "seed": None},
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
            workflow_root = root / "workflow"
            case_dir = workflow_root / "research_flat"
            report_dir = root / "reports"
            case_dir.mkdir(parents=True)
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (case_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (case_dir / "generated_series.json").write_text(json.dumps([2, 2, 2, 2]), encoding="utf-8")

            manifest = run_research_experiments(
                report_dir=report_dir,
                case_paths=[cases_path],
                workflow_dirs=[workflow_root],
                methods=["oracle_program", "ours_full", "llm_direct_array"],
                ablations=["full_oracle"],
            )

            tables = Path(manifest["paths"]["research_tables_md"]).read_text(encoding="utf-8")
            baseline_exists = Path(manifest["paths"]["baseline_results_json"]).exists()
            claim_evidence_path = manifest["paths"]["claim_evidence_json"]
            claim_evidence_exists = Path(claim_evidence_path).exists() if claim_evidence_path is not None else False

        self.assertNotIn("oracle_program", tables)
        self.assertIn("ours_full", tables)
        self.assertIn("llm_direct_array", tables)
        self.assertTrue(baseline_exists)
        self.assertIsNotNone(claim_evidence_path)
        self.assertTrue(claim_evidence_exists)

    def test_t2s_outputs_are_written_under_report_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            t2s_output_root = report_dir / "custom_t2s_outputs"

            with patch(
                "evals.run_research_experiments.generate_many_outputs",
                return_value={"status": "completed", "output_root": str(t2s_output_root), "summaries": []},
            ) as generate_many, patch(
                "evals.run_research_experiments.evaluate_t2s_llm_outputs",
                return_value={
                    "version": "1.0",
                    "benchmark": "T2S TSFragment-600K direct LLM baseline outputs",
                    "case_file": "fake_cases.json",
                    "output_root": str(t2s_output_root),
                    "methods": ["llm_direct_array"],
                    "total_cases": 1,
                    "results": [
                        {
                            "method": "llm_direct_array",
                            "status": "completed",
                            "case_file": "fake_cases.json",
                            "output_dir": str(t2s_output_root / "llm_direct_array"),
                            "cases": 1,
                            "successful_cases": 1,
                            "success_rate": 1.0,
                            "mean_wape": 0.0,
                            "median_wape": 0.0,
                            "mean_corr": 1.0,
                            "median_corr": 1.0,
                            "mean_mrr": 0.0,
                            "median_mrr": 0.0,
                            "mean_raw_mse": 0.0,
                            "median_raw_mse": 0.0,
                            "failure_counts": {},
                            "records": [],
                        }
                    ],
                },
            ), patch(
                "evals.run_research_experiments.run_methods",
                return_value={
                    "version": "1.0",
                    "methods": ["oracle_program", "llm_direct_array"],
                    "category": None,
                    "results": [],
                },
            ), patch(
                "evals.run_research_experiments.run_ablations",
                return_value={"version": "1.0", "ablations": [], "results": []},
            ):
                manifest = run_research_experiments(
                    report_dir=report_dir,
                    methods=["oracle_program", "llm_direct_array"],
                    ablations=["full_oracle"],
                    run_t2s_llm_baselines=True,
                    t2s_llm_output_root=t2s_output_root,
                    skip_revision_risks=True,
                )

                eval_exists = Path(manifest["paths"]["t2s_llm_eval_json"]).exists()
                comparison_json_exists = Path(manifest["paths"]["t2s_main_benchmark_json"]).exists()
                comparison_md_exists = Path(manifest["paths"]["t2s_main_benchmark_md"]).exists()

        generate_many.assert_called_once()
        self.assertTrue(eval_exists)
        self.assertTrue(comparison_json_exists)
        self.assertTrue(comparison_md_exists)
        self.assertEqual(manifest["paths"]["t2s_llm_output_root"], str(t2s_output_root))
        self.assertTrue(str(report_dir) in manifest["paths"]["t2s_main_benchmark_json"])


if __name__ == "__main__":
    unittest.main()
