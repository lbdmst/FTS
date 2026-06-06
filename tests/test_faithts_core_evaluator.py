from __future__ import annotations

import unittest

import numpy as np

from atomic_timeseries import add_flat, add_peak, add_plateau, add_ramp
from schema_validator import validate_payload
from faithts import aggregate_results, build_eval_report_from_result, evaluate_case, evaluate_direct_signal_case


def _registry_validation() -> dict:
    return {
        "status": "valid",
        "schema_path": "schemas/primitive.schema.json",
        "target_path": "atomic_timeseries/registry.json",
        "errors": [],
    }


def _global_context(length: int = 32) -> dict:
    return {
        "length": length,
        "numeric_range": {"min": -10.0, "max": 10.0},
        "dt": None,
        "seed": 7,
    }


def _primitive_step(step_id: str, primitive: str, params: dict, *, effect_type: str, source_text: str) -> dict:
    return {
        "id": step_id,
        "kind": "primitive",
        "primitive": primitive,
        "parameters": params,
        "semantic_role": primitive.replace("add_", ""),
        "effect_type": effect_type,
        "confidence": 1.0,
        "source_text": source_text,
        "needs_library_extension": False,
    }


def _base_plan(steps: list[dict], *, length: int = 32) -> dict:
    return {
        "version": "1.0",
        "input_description": "Flat first, then ramp.",
        "registry_validation": _registry_validation(),
        "global_context": _global_context(length),
        "steps": steps,
        "unresolved_semantics": [],
        "needs_library_extension": False,
        "assumptions": [],
    }


class FaithTSCoreEvaluatorTests(unittest.TestCase):
    def test_supported_case_scores_semantic_constraints(self) -> None:
        spec = {
            "case_id": "core_supported",
            "description": "Flat at 2 for ten steps, then ramp from 2 to 4.",
            "length": 32,
            "features": [
                {
                    "id": "f_flat",
                    "semantic": "flat",
                    "expected_primitives": ["add_flat"],
                    "window": [0, 9],
                    "parameters": {"target_value": 2.0},
                    "effect_type": "overwrite",
                    "checker": "check_flat",
                    "required": True,
                },
                {
                    "id": "f_ramp",
                    "semantic": "linear_ramp",
                    "expected_primitives": ["add_ramp"],
                    "window": [10, 19],
                    "parameters": {"start_value": 2.0, "end_value": 4.0},
                    "effect_type": "overwrite",
                    "checker": "check_ramp",
                    "required": True,
                },
            ],
        }
        plan = _base_plan(
            [
                _primitive_step(
                    "s1",
                    "add_flat",
                    {"start_timestep": 0, "end_timestep": 9, "target_value": 2.0},
                    effect_type="overwrite",
                    source_text="flat at 2",
                ),
                _primitive_step(
                    "s2",
                    "add_ramp",
                    {"start_timestep": 10, "end_timestep": 19, "start_value": 2.0, "end_value": 4.0},
                    effect_type="overwrite",
                    source_text="ramp from 2 to 4",
                ),
            ]
        )
        series = np.zeros(32, dtype=float)
        series = add_flat(series, start_timestep=0, end_timestep=9, target_value=2.0)
        series = add_ramp(series, start_timestep=10, end_timestep=19, start_value=2.0, end_value=4.0)

        result = evaluate_case(semantic_spec=spec, plan=plan, series=series)

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.constraint_satisfaction_rate, 1.0)
        self.assertEqual(result.primitive_f1, 1.0)
        self.assertEqual(result.window_iou, 1.0)
        self.assertEqual(result.parameter_error, 0.0)
        self.assertFalse(result.false_rejection)

    def test_semantic_failure_is_taxonomized(self) -> None:
        spec = {
            "case_id": "core_semantic_failure",
            "description": "Ramp from 2 to 4.",
            "length": 32,
            "features": [
                {
                    "id": "f_ramp",
                    "semantic": "linear_ramp",
                    "expected_primitives": ["add_ramp"],
                    "window": [10, 19],
                    "parameters": {"start_value": 2.0, "end_value": 4.0},
                    "effect_type": "overwrite",
                    "checker": "check_ramp",
                    "required": True,
                }
            ],
        }
        plan = _base_plan(
            [
                _primitive_step(
                    "s1",
                    "add_ramp",
                    {"start_timestep": 10, "end_timestep": 19, "start_value": 2.0, "end_value": 3.0},
                    effect_type="overwrite",
                    source_text="ramp from 2 to 3",
                )
            ]
        )
        series = add_ramp(np.zeros(32), start_timestep=10, end_timestep=19, start_value=2.0, end_value=3.0)

        result = evaluate_case(semantic_spec=spec, plan=plan, series=series)

        self.assertFalse(result.passed)
        self.assertLess(result.constraint_satisfaction_rate, 1.0)
        self.assertIn("F_semantic", {item["code"] for item in result.failure_reasons})

    def test_primitive_f1_accepts_equivalent_primitives(self) -> None:
        spec = {
            "case_id": "core_equivalent_primitive",
            "description": "A plateau from timestep 4 through 12.",
            "length": 32,
            "features": [
                {
                    "id": "f_plateau",
                    "semantic": "flat",
                    "expected_primitives": ["add_flat", "add_plateau"],
                    "window": [4, 12],
                    "parameters": {"target_value": 3.0},
                    "effect_type": "overwrite",
                    "checker": "check_flat",
                    "required": True,
                }
            ],
        }
        plan = _base_plan(
            [
                _primitive_step(
                    "s1",
                    "add_plateau",
                    {"start_timestep": 4, "end_timestep": 12, "target_value": 3.0},
                    effect_type="overwrite",
                    source_text="plateau from timestep 4 through 12",
                )
            ]
        )
        series = add_plateau(np.zeros(32), start_timestep=4, end_timestep=12, target_value=3.0)

        result = evaluate_case(semantic_spec=spec, plan=plan, series=series)

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.primitive_f1, 1.0)

    def test_explicit_local_event_window_overrides_width(self) -> None:
        spec = {
            "case_id": "core_explicit_local_window",
            "description": "A rounded peak at timestep 10 within the local window from 6 to 14.",
            "length": 32,
            "features": [
                {
                    "id": "f_peak",
                    "semantic": "peak",
                    "expected_primitives": ["add_peak"],
                    "window": [6, 14],
                    "parameters": {"timestep": 10, "target_value": 5.0},
                    "effect_type": "additive",
                    "checker": "check_peak",
                    "required": True,
                }
            ],
        }
        plan = _base_plan(
            [
                _primitive_step(
                    "s1",
                    "add_peak",
                    {"timestep": 10, "target_value": 5.0, "width": 1.5, "start_timestep": 6, "end_timestep": 14},
                    effect_type="additive",
                    source_text="peak at timestep 10 within the local window from 6 to 14",
                )
            ]
        )
        series = add_peak(
            np.zeros(32),
            timestep=10,
            target_value=5.0,
            width=1.5,
            start_timestep=6,
            end_timestep=14,
        )

        result = evaluate_case(semantic_spec=spec, plan=plan, series=series)

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.window_iou, 1.0)

    def test_compositional_scaffold_is_checked_at_step_trace(self) -> None:
        description = "Flat at 2 throughout, with a spike to 5 at timestep 10."
        spec = {
            "case_id": "core_compositional_trace",
            "description": description,
            "length": 32,
            "features": [
                {
                    "id": "f_flat",
                    "semantic": "flat",
                    "expected_primitives": ["add_flat"],
                    "window": [0, 31],
                    "parameters": {"target_value": 2.0},
                    "effect_type": "overwrite",
                    "checker": "check_flat",
                    "required": True,
                },
                {
                    "id": "f_spike",
                    "semantic": "spike",
                    "expected_primitives": ["add_spike"],
                    "window": [10, 10],
                    "parameters": {"timestep": 10, "target_value": 5.0},
                    "effect_type": "additive",
                    "checker": "check_spike",
                    "required": True,
                },
            ],
        }
        plan = _base_plan(
            [
                _primitive_step(
                    "s1",
                    "add_flat",
                    {"start_timestep": 0, "end_timestep": 31, "target_value": 2.0},
                    effect_type="overwrite",
                    source_text="Flat at 2 throughout",
                ),
                _primitive_step(
                    "s2",
                    "add_spike",
                    {"timestep": 10, "target_value": 5.0},
                    effect_type="additive",
                    source_text="spike to 5 at timestep 10",
                ),
            ],
            length=32,
        )
        series = np.full(32, 2.0)
        series[10] = 5.0

        result = evaluate_case(semantic_spec=spec, plan=plan, series=series, execution_success=True)

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.constraint_satisfaction_rate, 1.0)

    def test_direct_signal_case_does_not_require_plan_schema_validation(self) -> None:
        spec = {
            "case_id": "core_direct_signal",
            "description": "Flat at 2 throughout, with a spike to 5 at timestep 10.",
            "length": 32,
            "features": [
                {
                    "id": "f_flat",
                    "semantic": "flat",
                    "expected_primitives": ["add_flat"],
                    "window": [0, 31],
                    "parameters": {"target_value": 2.0},
                    "effect_type": "overwrite",
                    "checker": "check_flat",
                    "required": True,
                },
                {
                    "id": "f_spike",
                    "semantic": "spike",
                    "expected_primitives": ["add_spike"],
                    "window": [10, 10],
                    "parameters": {"timestep": 10, "target_value": 5.0},
                    "effect_type": "additive",
                    "checker": "check_spike",
                    "required": True,
                },
            ],
        }
        series = np.full(32, 2.0)
        series[10] = 5.0
        signal_checks = {
            "length": 32,
            "point_values": [{"timestep": 10, "value": 5.0, "tolerance": 1e-6}],
            "constant_segments": [
                {"start": 0, "end": 9, "value": 2.0, "tolerance": 1e-6},
                {"start": 11, "end": 31, "value": 2.0, "tolerance": 1e-6},
            ],
        }

        result = evaluate_direct_signal_case(
            semantic_spec=spec,
            series=series,
            execution_success=True,
            signal_checks=signal_checks,
        )
        report = build_eval_report_from_result(result, spec["description"])

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertFalse(result.plan_validation_applicable)
        self.assertEqual(report["plan_validation_status"]["status"], "not_run")
        self.assertEqual(validate_payload("eval_report", report)["status"], "valid")

    def test_unsupported_semantics_correct_rejection_builds_schema_valid_report(self) -> None:
        spec = {
            "case_id": "core_unsupported",
            "description": "The signal should show long-memory mean reversion.",
            "length": 32,
            "features": [],
            "unsupported_semantics": [
                {
                    "id": "u1",
                    "description": "long-memory mean reversion",
                    "reason": "No registered primitive models this dynamic.",
                }
            ],
        }
        plan = {
            "version": "1.0",
            "input_description": spec["description"],
            "registry_validation": _registry_validation(),
            "global_context": _global_context(32),
            "steps": [
                {
                    "id": "u1",
                    "kind": "unresolved",
                    "description": "long-memory mean reversion",
                    "semantic_role": "unsupported dynamics",
                    "confidence": 0.95,
                    "source_text": "long-memory mean reversion",
                    "reason": "No registered primitive models this dynamic.",
                    "needs_library_extension": True,
                }
            ],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "long-memory mean reversion",
                    "source_text": "long-memory mean reversion",
                    "reason": "No registered primitive models this dynamic.",
                }
            ],
            "needs_library_extension": True,
            "assumptions": [],
        }

        result = evaluate_case(semantic_spec=spec, plan=plan, series=None, execution_success=False)
        report = build_eval_report_from_result(result, spec["description"])

        self.assertTrue(result.passed, result.failure_reasons)
        self.assertTrue(result.correct_rejection)
        self.assertFalse(result.hallucinated_unsupported)
        self.assertEqual(validate_payload("eval_report", report)["status"], "valid")

    def test_aggregate_reports_core_metrics(self) -> None:
        rejected_spec = {
            "case_id": "unsupported_hallucinated",
            "description": "Unsupported dynamics.",
            "length": 16,
            "features": [],
            "unsupported_semantics": [{"id": "u1", "description": "unsupported", "reason": "unsupported"}],
        }
        hallucinating_plan = _base_plan(
            [
                _primitive_step(
                    "s1",
                    "add_flat",
                    {"start_timestep": 0, "end_timestep": 15, "target_value": 1.0},
                    effect_type="overwrite",
                    source_text="fake unsupported dynamics",
                )
            ],
            length=16,
        )
        result = evaluate_case(
            semantic_spec=rejected_spec,
            plan=hallucinating_plan,
            series=np.ones(16),
            execution_success=True,
        )

        metrics = aggregate_results([result])

        self.assertEqual(metrics["cases"], 1.0)
        self.assertEqual(metrics["crr"], 0.0)
        self.assertEqual(metrics["hus"], 1.0)


if __name__ == "__main__":
    unittest.main()
