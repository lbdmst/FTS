from __future__ import annotations

import unittest

from schema_validator import validate_payload
from faithts.rejector import apply_static_rejector, detect_unsupported_semantics


class StaticRejectorTests(unittest.TestCase):
    def test_detects_mean_reversion_as_unsupported(self) -> None:
        decision = detect_unsupported_semantics("The series is pulled back toward its long-run mean.")

        self.assertTrue(decision.should_reject)
        self.assertEqual(decision.semantic, "mean_reversion")

    def test_apply_static_rejector_builds_schema_valid_unresolved_plan(self) -> None:
        plan = {
            "version": "1.0",
            "input_description": "placeholder",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": {"length": 8, "numeric_range": {"min": 0.0, "max": 1.0}},
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 0.0},
                    "semantic_role": "fallback",
                    "effect_type": "overwrite",
                    "confidence": 0.1,
                    "source_text": "fallback",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }

        repaired = apply_static_rejector(
            plan,
            description="After deviating upward, the series is pulled back toward its long-run mean.",
            length=8,
            numeric_range={"min": 0.0, "max": 1.0},
        )

        self.assertTrue(repaired["needs_library_extension"])
        self.assertEqual(repaired["steps"][0]["kind"], "unresolved")
        self.assertEqual(validate_payload("plan", repaired)["status"], "valid")


if __name__ == "__main__":
    unittest.main()
