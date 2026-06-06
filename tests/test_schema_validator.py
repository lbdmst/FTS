import json
import tempfile
import unittest
from pathlib import Path

from schema_validator import validate_instance


ROOT = Path(__file__).resolve().parent.parent


class SchemaValidatorTests(unittest.TestCase):
    def test_registry_validates_against_primitive_schema(self) -> None:
        report = validate_instance("primitive", ROOT / "atomic_timeseries" / "registry.json")
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["schema_path"], "schemas/primitive.schema.json")

    def test_plan_validation_reports_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_plan = Path(tmpdir) / "bad_plan.json"
            bad_plan.write_text(json.dumps({"version": "1.0"}))
            report = validate_instance("plan", bad_plan)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(report["errors"])

    def test_eval_validation_reports_valid_payload(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "Flat series around 3.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "plan_validation_status": {
                "status": "valid",
                "schema_path": "schemas/plan.schema.json",
                "target_path": "artifacts/plan.json",
                "errors": [],
            },
            "code_generation_status": {
                "status": "success",
                "draft_generated": True,
                "final_generated": True,
                "artifacts": ["artifacts/draft.py", "artifacts/final.py"],
                "errors": [],
            },
            "execution_status": {
                "status": "success",
                "signal_summary": {
                    "length": 128,
                    "min": 3.0,
                    "max": 3.0,
                    "mean": 3.0,
                    "std": 0.0,
                },
                "errors": [],
            },
            "semantic_alignment_score": {
                "overall": 1.0,
                "description_to_plan": 1.0,
                "plan_to_code": 1.0,
                "code_to_signal": 1.0,
            },
            "unresolved_semantics": [],
            "failure_reasons": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "eval_report.json"
            report_path.write_text(json.dumps(payload))
            report = validate_instance("eval_report", report_path)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["schema_path"], "schemas/eval_report.schema.json")


if __name__ == "__main__":
    unittest.main()
