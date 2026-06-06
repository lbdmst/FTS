from __future__ import annotations

import unittest

import numpy as np

from faithts_pipeline.hybrid_pipeline import execute_plan, parse_plan
from schema_validator import validate_payload
from faithts.repair import repair_plan


class RepairLoopTests(unittest.TestCase):
    def test_repair_fills_schema_fields_and_effect_type(self) -> None:
        plan = {
            "input_description": "Noise is added late.",
            "global_context": {"length": 10, "numeric_range": {"min": -1, "max": 1}},
            "steps": [
                {
                    "primitive": "add_noise",
                    "effect_type": "overwrite",
                    "parameters": {"start_timestep": 5, "end_timestep": 9, "noise_scale": 0.1},
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }

        result = repair_plan(plan, series_length=10)

        self.assertEqual(result.schema_status, "valid", result.changes)
        self.assertEqual(validate_payload("plan", result.plan)["status"], "valid")
        step = result.plan["steps"][0]
        self.assertEqual(step["kind"], "primitive")
        self.assertEqual(step["effect_type"], "additive")
        self.assertEqual(step["semantic_layer"], "texture")
        self.assertEqual(step["parameters"]["random_seed"], 7)
        self.assertIn("fixed effect_type for add_noise", result.changes)
        self.assertIn("fixed semantic_layer for add_noise", result.changes)

    def test_repair_handles_list_metadata_fields(self) -> None:
        plan = {
            "version": "1.0",
            "input_description": "A level shift starts at timestep 5.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": {"length": 10, "numeric_range": {"min": -1, "max": 1}},
            "steps": [
                {
                    "id": "s1",
                    "kind": "primitive",
                    "primitive": "add_level_shift",
                    "parameters": {"anchor_timestep": 5, "shift": 1.0},
                    "semantic_role": "level shift",
                    "semantic_layer": "segment_structure",
                    "priority": 0,
                    "protected_constraints": [],
                    "semantic_cues": [],
                    "effect_type": "additive",
                    "confidence": 0.8,
                    "source_text": "level shift starts at timestep 5",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }

        result = repair_plan(plan, series_length=10)

        self.assertEqual(result.schema_status, "valid", result.changes)
        self.assertEqual(result.plan["steps"][0]["semantic_layer"], "global_scaffold")
        self.assertIn("fixed semantic_layer for add_level_shift", result.changes)

    def test_repair_converts_unknown_primitive_to_unresolved(self) -> None:
        plan = {
            "version": "1.0",
            "input_description": "Use an unsupported oscillator.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": {"length": 10, "numeric_range": {"min": -1, "max": 1}},
            "steps": [
                {
                    "id": "s1",
                    "kind": "primitive",
                    "primitive": "add_mean_reversion",
                    "parameters": {},
                    "semantic_role": "mean reversion",
                    "effect_type": "additive",
                    "confidence": 0.8,
                    "source_text": "mean reversion",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }

        result = repair_plan(plan, series_length=10)

        self.assertEqual(result.schema_status, "valid", result.changes)
        self.assertTrue(result.plan["needs_library_extension"])
        self.assertEqual(result.plan["steps"][0]["kind"], "unresolved")
        self.assertEqual(result.plan["unresolved_semantics"][0]["id"], "s1")

    def test_repair_removes_noise_when_description_forbids_fresh_random_noise(self) -> None:
        plan = {
            "version": "1.0",
            "input_description": (
                "The fluctuations become more volatile after timestep 40, meaning the existing swings "
                "widen around the same baseline rather than introducing fresh random noise."
            ),
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": {"length": 100, "numeric_range": {"min": -1, "max": 1}},
            "steps": [
                {
                    "id": "s1",
                    "kind": "primitive",
                    "primitive": "add_volatility",
                    "parameters": {"start_timestep": 41, "end_timestep": 99, "volatility_scale": 0.5},
                    "semantic_role": "more volatile",
                    "effect_type": "overwrite",
                    "confidence": 0.8,
                    "source_text": "more volatile after timestep 40",
                    "needs_library_extension": False,
                },
                {
                    "id": "s2",
                    "kind": "primitive",
                    "primitive": "add_noise",
                    "parameters": {"start_timestep": 41, "end_timestep": 99, "noise_scale": 0.1, "random_seed": 7},
                    "semantic_role": "fresh random noise",
                    "effect_type": "additive",
                    "confidence": 0.5,
                    "source_text": "fresh random noise",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }

        result = repair_plan(plan, series_length=100)

        self.assertEqual(result.schema_status, "valid", result.changes)
        self.assertEqual([step["primitive"] for step in result.plan["steps"]], ["add_volatility"])
        self.assertEqual(result.plan["steps"][0]["parameters"]["start_timestep"], 40)
        self.assertIn("removed add_noise contradicted by fresh-random-noise negation", result.changes)
        self.assertIn("aligned start_timestep with explicit after-timestep boundary", result.changes)

    def test_repair_recovers_missing_gap_workflow_failure(self) -> None:
        plan = {
            "version": "1.0",
            "input_description": "From timestep 2 to 4, the data is missing entirely rather than merely dipping.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": {"length": 8, "numeric_range": {"min": -1, "max": 1}},
            "steps": [
                {
                    "id": "workflow_failure_step",
                    "kind": "unresolved",
                    "description": "Workflow failed before a schema-valid executable plan was available.",
                    "semantic_role": "workflow failure",
                    "confidence": 1.0,
                    "source_text": "From timestep 2 to 4, the data is missing entirely rather than merely dipping.",
                    "reason": "Series contains non-finite values during plan execution.",
                    "needs_library_extension": True,
                }
            ],
            "unresolved_semantics": [
                {
                    "id": "workflow_failure",
                    "description": "Workflow failed before a schema-valid executable plan was available.",
                    "source_text": "From timestep 2 to 4, the data is missing entirely rather than merely dipping.",
                    "reason": "Series contains non-finite values during plan execution.",
                }
            ],
            "needs_library_extension": True,
        }

        result = repair_plan(plan, series_length=8)

        self.assertEqual(result.schema_status, "valid", result.changes)
        self.assertFalse(result.plan["needs_library_extension"])
        self.assertEqual(result.plan["unresolved_semantics"], [])
        self.assertEqual(result.plan["steps"][0]["primitive"], "add_gap")
        parsed = parse_plan(result.plan, candidate_primitives=["add_gap"], series_length=8)
        series = execute_plan(parsed, result.plan["input_description"], 8)
        self.assertTrue(np.isnan(series[2:5]).all())

    def test_repair_clamps_vague_jitter_scale_without_touching_explicit_scale(self) -> None:
        base = {
            "version": "1.0",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": {"length": 20, "numeric_range": {"min": -1, "max": 1}},
            "steps": [
                {
                    "id": "s1",
                    "kind": "primitive",
                    "primitive": "add_noise",
                    "parameters": {"start_timestep": 10, "end_timestep": 19, "noise_scale": 1.0, "random_seed": 7},
                    "semantic_role": "random jitter",
                    "effect_type": "additive",
                    "confidence": 0.8,
                    "source_text": "random jitter",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        vague = dict(base, input_description="The last timesteps become noisy with random jitter.")
        explicit = dict(base, input_description="The last timesteps become noisy with random jitter of scale 1.0.")

        vague_result = repair_plan(vague, series_length=20)
        explicit_result = repair_plan(explicit, series_length=20)

        self.assertEqual(vague_result.plan["steps"][0]["parameters"]["noise_scale"], 0.2)
        self.assertEqual(explicit_result.plan["steps"][0]["parameters"]["noise_scale"], 1.0)

    def test_repair_extends_flat_baseline_before_persistent_level_shift(self) -> None:
        plan = {
            "version": "1.0",
            "input_description": (
                "The first half is flat at 1.5. At timestep 50 the series jumps up by 2.0 "
                "into a new level, and from timestep 70 to 90 it levels off into a plateau at 4.0."
            ),
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": {"length": 100, "numeric_range": {"min": -1, "max": 6}, "dt": None, "seed": None},
            "steps": [
                {
                    "id": "s1",
                    "kind": "primitive",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 49, "target_value": 1.5},
                    "semantic_role": "initial flat regime",
                    "semantic_layer": "local_event",
                    "effect_type": "overwrite",
                    "confidence": 0.9,
                    "source_text": "first half is flat at 1.5",
                    "needs_library_extension": False,
                },
                {
                    "id": "s2",
                    "kind": "primitive",
                    "primitive": "add_level_shift",
                    "parameters": {"anchor_timestep": 50, "shift": 2.0},
                    "semantic_role": "regime jump",
                    "semantic_layer": "global_scaffold",
                    "effect_type": "additive",
                    "confidence": 0.9,
                    "source_text": "jumps up by 2.0 into a new level",
                    "needs_library_extension": False,
                },
                {
                    "id": "s3",
                    "kind": "primitive",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 70, "end_timestep": 90, "target_value": 4.0},
                    "semantic_role": "late plateau",
                    "semantic_layer": "local_event",
                    "effect_type": "overwrite",
                    "confidence": 0.9,
                    "source_text": "from timestep 70 to 90 it levels off into a plateau at 4.0",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
            "assumptions": [],
        }

        result = repair_plan(plan, series_length=100)

        self.assertEqual(result.schema_status, "valid", result.changes)
        self.assertEqual(result.plan["steps"][0]["parameters"]["end_timestep"], 99)
        self.assertIn(
            "extended flat baseline before add_level_shift from end_timestep 49 to 99",
            result.changes,
        )
        parsed = parse_plan(result.plan, candidate_primitives=["add_flat", "add_level_shift"], series_length=100)
        series = execute_plan(parsed, result.plan["input_description"], 100)
        self.assertTrue(np.allclose(series[:50], 1.5))
        self.assertTrue(np.allclose(series[50:70], 3.5))
        self.assertTrue(np.allclose(series[70:91], 4.0))
        self.assertTrue(np.allclose(series[91:], 3.5))


if __name__ == "__main__":
    unittest.main()
