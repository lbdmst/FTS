from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.collect_faithts_outputs import collect_outputs
from evals.run_baseline_benchmark import run_baseline


class CollectFaithTSOutputsTests(unittest.TestCase):
    def test_collects_saved_plan_payload_for_faithts_saved_plan_runner(self) -> None:
        case = {
            "case_id": "collect_flat",
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
            case_dir = workflow_root / "collect_flat"
            output_dir = root / "saved"
            case_dir.mkdir(parents=True)
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (case_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (case_dir / "generated_series.json").write_text(json.dumps([2, 2, 2, 2]), encoding="utf-8")

            summary = collect_outputs(case_paths=[cases_path], workflow_dirs=[workflow_root], output_dir=output_dir)
            payload = run_baseline("ours_full", paths=[cases_path], output_dir=output_dir)

        self.assertEqual(summary["collected"], 1)
        self.assertEqual(payload["aggregate"]["csr"], 1.0)
        self.assertTrue(payload["cases"][0]["passed"], payload["cases"][0]["failure_reasons"])

    def test_collect_can_apply_static_rejector(self) -> None:
        case = {
            "case_id": "collect_mean_reversion",
            "category": "adversarial",
            "description": "After deviating upward, the series is pulled back toward its long-run mean.",
            "length": 4,
            "expected_candidate_primitives": [],
            "expected_plan_outline": {
                "needs_library_extension": True,
                "unresolved_semantics": [{"semantic": "mean_reversion", "reason": "unsupported"}],
                "steps": [],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            workflow_root = root / "workflow"
            case_dir = workflow_root / "collect_mean_reversion"
            output_dir = root / "saved"
            case_dir.mkdir(parents=True)
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (case_dir / "plan.json").write_text(json.dumps({}), encoding="utf-8")
            (case_dir / "generated_series.json").write_text(json.dumps([0, 1, 0.5, 0]), encoding="utf-8")

            summary = collect_outputs(
                case_paths=[cases_path],
                workflow_dirs=[workflow_root],
                output_dir=output_dir,
                apply_rejector=True,
            )
            payload = run_baseline("ours_full", paths=[cases_path], output_dir=output_dir)

        self.assertEqual(summary["collected"], 1)
        self.assertEqual(payload["aggregate"]["crr"], 1.0)
        self.assertTrue(payload["cases"][0]["passed"], payload["cases"][0]["failure_reasons"])

    def test_rejected_saved_plan_ignores_stale_generated_series(self) -> None:
        case = {
            "case_id": "collect_rejected_empty_series",
            "category": "adversarial",
            "description": "After deviating upward, the series is pulled back toward its long-run mean.",
            "length": 4,
            "expected_candidate_primitives": [],
            "expected_plan_outline": {
                "needs_library_extension": True,
                "unresolved_semantics": [{"semantic": "mean_reversion", "reason": "unsupported"}],
                "steps": [],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            workflow_root = root / "workflow"
            case_dir = workflow_root / "collect_rejected_empty_series"
            output_dir = root / "saved"
            case_dir.mkdir(parents=True)
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            (case_dir / "plan.json").write_text(json.dumps({}), encoding="utf-8")
            (case_dir / "generated_series.json").write_text(json.dumps([]), encoding="utf-8")

            collect_outputs(case_paths=[cases_path], workflow_dirs=[workflow_root], output_dir=output_dir, apply_rejector=True)
            payload = run_baseline("ours_full", paths=[cases_path], output_dir=output_dir)

        self.assertEqual(payload["aggregate"]["crr"], 1.0)
        self.assertEqual(payload["aggregate"]["hus"], 0.0)
        self.assertTrue(payload["cases"][0]["passed"], payload["cases"][0]["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
