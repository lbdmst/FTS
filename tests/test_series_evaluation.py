import tempfile
import unittest
from pathlib import Path

import numpy as np

from evals.series_evaluation import compute_error_metrics, render_comparison_plot, write_case_artifacts
from evals.tsfragment_eval_workflow import (
    extract_real_series_plan_payload,
    normalize_plan_for_real_series,
    remove_generic_real_seasonality,
    resolve_representable_real_unresolved,
)


class TSFragmentSeriesEvaluationTests(unittest.TestCase):
    def test_compute_error_metrics_matches_expected_values(self) -> None:
        truth = np.array([1.0, 2.0, 3.0], dtype=float)
        generated = np.array([1.0, 1.0, 5.0], dtype=float)
        metrics = compute_error_metrics(generated, truth)
        self.assertAlmostEqual(metrics["mse"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], (0.0 + 0.5 + (2.0 / 3.0)) / 3.0)
        self.assertAlmostEqual(metrics["wape"], 3.0 / 6.0)
        self.assertAlmostEqual(metrics["range_normalized_mse"], (5.0 / 3.0) / 4.0)
        self.assertAlmostEqual(metrics["range_normalized_rmse"], np.sqrt(5.0 / 3.0) / 2.0)
        self.assertAlmostEqual(metrics["max_abs_error"], 2.0)

    def test_render_comparison_plot_writes_png(self) -> None:
        truth = np.array([0.0, 1.0, 0.5, 1.5], dtype=float)
        generated = np.array([0.0, 0.8, 0.7, 1.3], dtype=float)
        metrics = compute_error_metrics(generated, truth)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.png"
            render_comparison_plot(
                case_id="demo_case",
                description="A short demo description.",
                reference_series=truth,
                generated_series=generated,
                metrics=metrics,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_write_case_artifacts_writes_expected_files(self) -> None:
        record = {
            "case_id": "demo_case",
            "description": "Demo description",
            "true_series": [0.0, 1.0],
            "expected_true_series_summary": {"length": 2},
            "candidate_primitives": ["add_flat"],
            "retrieved_primitives": ["add_flat"],
            "prompt": "prompt text",
            "raw_model_text": "{\"steps\": []}",
            "structured_plan": {"steps": []},
            "generated_code": "import numpy as np\n",
            "generated_series": [0.0, 1.0],
            "generated_series_summary": {"length": 2},
            "step_outputs": [{"id": "step_1"}],
            "comparison_metrics": {"mse": 0.0, "mrr": 0.0},
            "status": "success",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "demo_case"
            paths = write_case_artifacts(case_dir, record)
            self.assertTrue((case_dir / "description.txt").exists())
            self.assertTrue((case_dir / "reference_series.json").exists())
            self.assertTrue((case_dir / "structured_plan.json").exists())
            self.assertTrue((case_dir / "generated_code.py").exists())
            self.assertTrue((case_dir / "generated_series.json").exists())
            self.assertTrue((case_dir / "evaluation_metrics.json").exists())
            self.assertIn("record", paths)

    def test_extract_real_series_plan_payload_allows_null_numeric_range_for_later_repair(self) -> None:
        raw_text = """
```json
{
  "version": "1.0",
  "input_description": "demo",
  "global_context": {
    "length": 24,
    "numeric_range": {"min": null, "max": null},
    "dt": null,
    "seed": null
  },
  "steps": [
    {
      "id": "step_1",
      "kind": "primitive",
      "primitive": "add_trend",
      "parameters": {"start_timestep": 0, "end_timestep": 5, "slope": null},
      "semantic_role": "initial decline",
      "semantic_layer": "segment_structure",
      "priority": 1,
      "protected_constraints": [],
      "semantic_cues": [],
      "effect_type": "additive",
      "confidence": 0.9,
      "source_text": "initial decline",
      "needs_library_extension": false
    }
  ],
  "unresolved_semantics": [],
  "needs_library_extension": false
}
```
"""

        payload = extract_real_series_plan_payload(raw_text)

        self.assertEqual(payload["global_context"]["numeric_range"], {"min": None, "max": None})

    def test_normalize_plan_for_real_series_fills_executable_parameters(self) -> None:
        reference = np.array([10.0, 8.0, 6.0, 7.0, 9.0], dtype=float)
        plan = {
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "parameters": {"start_timestep": 0, "end_timestep": 2, "end_value": None},
                },
                {
                    "id": "step_2",
                    "kind": "primitive",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 3, "end_timestep": 5, "target_value": None},
                },
                {
                    "id": "step_3",
                    "kind": "primitive",
                    "primitive": "add_peak",
                    "parameters": {"timestep": None},
                },
            ]
        }
        changes = normalize_plan_for_real_series(plan, reference=reference)
        self.assertTrue(changes)
        self.assertEqual(plan["steps"][0]["parameters"]["start_value"], 10.0)
        self.assertEqual(plan["steps"][0]["parameters"]["end_value"], 6.0)
        self.assertAlmostEqual(plan["steps"][1]["parameters"]["target_value"], 8.0)
        self.assertEqual(plan["steps"][1]["parameters"]["end_timestep"], 4)
        self.assertEqual(plan["steps"][2]["parameters"]["timestep"], 0)
        self.assertEqual(plan["steps"][2]["parameters"]["target_value"], 10.0)

    def test_normalize_plan_for_real_series_honors_one_based_data_point_values(self) -> None:
        reference = np.array(
            [
                16.672,
                17.938,
                18.501,
                19.416,
                19.064,
                17.798,
                17.094,
                16.813,
                16.602,
                16.391,
                16.25,
                16.109,
                16.321,
                16.039,
                16.18,
                15.687,
                15.547,
                15.476,
                16.602,
                17.305,
                17.235,
                18.009,
                18.431,
                18.642,
            ],
            dtype=float,
        )
        description = (
            "The time series starts at a peak of 19.416 at the 4.0 data point, "
            "then exhibits a decline to 15.476 by the 18.0 data point. "
            "A slight recovery occurs afterward, with values fluctuating between 16.180 and 18.642 "
            "from the 19.0 to 24.0 data points."
        )
        plan = {
            "input_description": description,
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_peak",
                    "effect_type": "additive",
                    "source_text": "The time series starts at a peak of 19.416 at the 4.0 data point",
                    "parameters": {"timestep": 4},
                },
                {
                    "id": "step_2",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "effect_type": "overwrite",
                    "source_text": "then exhibits a decline to 15.476 by the 18.0 data point",
                    "parameters": {"start_timestep": 4, "end_timestep": 18},
                },
                {
                    "id": "step_3",
                    "kind": "primitive",
                    "primitive": "add_volatility",
                    "effect_type": "overwrite",
                    "source_text": (
                        "with values fluctuating between 16.180 and 18.642 "
                        "from the 19.0 to 24.0 data points"
                    ),
                    "parameters": {"start_timestep": 19, "end_timestep": 24},
                },
            ],
        }

        changes = normalize_plan_for_real_series(plan, reference=reference)

        by_id = {step["id"]: step for step in plan["steps"] if "id" in step}
        self.assertEqual(by_id["step_1"]["parameters"]["timestep"], 3)
        self.assertAlmostEqual(by_id["step_1"]["parameters"]["target_value"], 19.416)
        self.assertEqual(by_id["step_2"]["parameters"]["start_timestep"], 3)
        self.assertEqual(by_id["step_2"]["parameters"]["end_timestep"], 17)
        self.assertAlmostEqual(by_id["step_2"]["parameters"]["start_value"], 19.416)
        self.assertAlmostEqual(by_id["step_2"]["parameters"]["end_value"], 15.476)
        self.assertEqual(by_id["step_3"]["parameters"]["start_timestep"], 18)
        self.assertEqual(by_id["step_3"]["parameters"]["end_timestep"], 23)
        self.assertTrue(any("explicit caption endpoint value" in change for change in changes))

    def test_normalize_plan_for_real_series_adds_exact_guards_for_fallback_hard_anchors(self) -> None:
        reference = np.linspace(16.672, 18.431, 48)
        description = (
            "The time series begins at 16.672 and shows an initial upward trend, "
            "peaking at 22.230 around the 28.0 data point. Following the peak, "
            "the values experience some fluctuations, with a slight decrease to 17.094 by the 7.0 point."
        )
        plan = {
            "input_description": description,
            "steps": [
                {
                    "id": "fallback_step_1",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "effect_type": "overwrite",
                    "source_text": description,
                    "parameters": {"start_timestep": 0, "end_timestep": 16, "start_value": 16.672, "end_value": 18.0},
                },
                {
                    "id": "fallback_step_2",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "effect_type": "overwrite",
                    "source_text": description,
                    "parameters": {"start_timestep": 17, "end_timestep": 32, "start_value": 18.0, "end_value": 18.2},
                },
                {
                    "id": "fallback_step_3",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "effect_type": "overwrite",
                    "source_text": description,
                    "parameters": {"start_timestep": 33, "end_timestep": 47, "start_value": 18.2, "end_value": 18.431},
                },
            ],
        }

        changes = normalize_plan_for_real_series(plan, reference=reference)

        guards = [
            step for step in plan["steps"]
            if step.get("kind") == "primitive" and step.get("primitive") == "add_outlier"
        ]
        guard_params = {(step["parameters"]["timestep"], step["parameters"]["target_value"]) for step in guards}
        self.assertIn((27, 22.230), guard_params)
        self.assertIn((6, 17.094), guard_params)
        self.assertTrue(any("explicit caption value" in change for change in changes))
        self.assertEqual(plan["steps"][0]["parameters"]["end_timestep"], 16)

    def test_normalize_plan_for_real_series_converts_absolute_trend_to_ramp(self) -> None:
        reference = np.array([100.0, 90.0, 80.0, 85.0], dtype=float)
        plan = {
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_trend",
                    "effect_type": "additive",
                    "parameters": {"start_timestep": 0, "end_timestep": 2},
                }
            ]
        }

        changes = normalize_plan_for_real_series(plan, reference=reference, available_primitives={"add_ramp"})

        self.assertIn("converted step_1 add_trend to absolute add_ramp for real-series endpoints", changes)
        converted = next(step for step in plan["steps"] if step.get("id") == "step_1")
        self.assertEqual(converted["primitive"], "add_ramp")
        self.assertEqual(converted["effect_type"], "overwrite")
        self.assertEqual(converted["parameters"]["start_value"], 100.0)
        self.assertEqual(converted["parameters"]["end_value"], 80.0)

    def test_normalize_plan_for_real_series_repairs_invalid_real_case_parameters(self) -> None:
        reference = np.array([1.0, 2.0, 3.0, 2.5], dtype=float)
        plan = {
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_plateau",
                    "parameters": {"start_timestep": 1, "end_timestep": 2},
                },
                {
                    "id": "step_2",
                    "kind": "primitive",
                    "primitive": "add_noise",
                    "parameters": {"noise_scale": "noise_scale"},
                },
                {
                    "id": "step_3",
                    "kind": "primitive",
                    "primitive": "add_seasonality",
                    "parameters": {"amplitude": 0.0, "period": 0.0},
                },
                {
                    "id": "step_4",
                    "kind": "primitive",
                    "primitive": "add_change_point",
                    "parameters": {"anchor_timestep": 2},
                },
            ]
        }

        changes = normalize_plan_for_real_series(plan, reference=reference)

        by_id = {step["id"]: step for step in plan["steps"] if "id" in step}
        self.assertAlmostEqual(by_id["step_1"]["parameters"]["target_value"], 2.5)
        self.assertGreater(by_id["step_2"]["parameters"]["noise_scale"], 0.0)
        self.assertGreater(by_id["step_3"]["parameters"]["amplitude"], 0.0)
        self.assertGreater(by_id["step_3"]["parameters"]["period"], 0.0)
        self.assertNotIn("step_4", [step["id"] for step in plan["steps"]])
        self.assertIn("removed step_4 no-op add_change_point", changes)

    def test_normalize_plan_for_real_series_adds_baseline_for_additive_gap(self) -> None:
        reference = np.array([10.0, 9.0, 8.0, 8.5, 9.5, 9.0, 8.0], dtype=float)
        plan = {
            "input_description": "Decline followed by fluctuations and a later peak.",
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "effect_type": "overwrite",
                    "source_text": "initial decline",
                    "parameters": {"start_timestep": 0, "end_timestep": 2},
                },
                {
                    "id": "step_2",
                    "kind": "primitive",
                    "primitive": "add_noise",
                    "effect_type": "additive",
                    "source_text": "fluctuations",
                    "parameters": {"start_timestep": 3, "end_timestep": 5, "noise_scale": 0.1},
                },
                {
                    "id": "step_3",
                    "kind": "primitive",
                    "primitive": "add_peak",
                    "effect_type": "additive",
                    "source_text": "later peak",
                    "parameters": {"timestep": 4},
                },
            ],
        }

        changes = normalize_plan_for_real_series(plan, reference=reference)

        scaffold_steps = [step for step in plan["steps"] if str(step["id"]).startswith("real_scaffold_")]
        self.assertTrue(scaffold_steps, changes)
        self.assertEqual(scaffold_steps[0]["primitive"], "add_ramp")
        self.assertEqual(scaffold_steps[0]["parameters"]["start_timestep"], 3)
        self.assertEqual(scaffold_steps[0]["parameters"]["end_timestep"], 5)
        self.assertIn("start_timestep", plan["steps"][-1]["parameters"])
        self.assertIn("end_timestep", plan["steps"][-1]["parameters"])
        self.assertTrue(any("inserted reference baseline" in change for change in changes))
        self.assertTrue(any("bounded step_3 local peak window" in change for change in changes))

    def test_normalize_plan_for_real_series_expands_several_peaks_into_multiple_events(self) -> None:
        reference = np.array(
            [10.0, 9.0, 8.0, 9.5, 8.5, 7.5, 8.0, 9.2, 8.6, 8.1, 8.8, 9.7, 8.9, 8.2, 7.8],
            dtype=float,
        )
        plan = {
            "input_description": "The series declines and then has several peaks around the middle and late portions.",
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "effect_type": "overwrite",
                    "parameters": {"start_timestep": 0, "end_timestep": 4},
                },
                {
                    "id": "step_2",
                    "kind": "primitive",
                    "primitive": "add_peak",
                    "effect_type": "additive",
                    "parameters": {"timestep": 7},
                },
            ],
        }

        changes = normalize_plan_for_real_series(plan, reference=reference)

        peak_steps = [step for step in plan["steps"] if step.get("primitive") == "add_peak"]
        self.assertGreaterEqual(len(peak_steps), 3)
        self.assertTrue(any("expanded `several peaks`" in change for change in changes))

    def test_normalize_plan_for_real_series_inserts_background_for_uncovered_windows(self) -> None:
        reference = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0], dtype=float)
        plan = {
            "input_description": "Decline, then a later decline with an uncovered middle interval.",
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "effect_type": "overwrite",
                    "parameters": {"start_timestep": 0, "end_timestep": 1},
                },
                {
                    "id": "step_2",
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "effect_type": "overwrite",
                    "parameters": {"start_timestep": 5, "end_timestep": 6},
                },
            ],
        }

        changes = normalize_plan_for_real_series(plan, reference=reference)

        scaffold_steps = [step for step in plan["steps"] if str(step.get("id", "")).startswith("real_background_")]
        self.assertEqual(len(scaffold_steps), 1)
        self.assertEqual(scaffold_steps[0]["parameters"]["start_timestep"], 2)
        self.assertEqual(scaffold_steps[0]["parameters"]["end_timestep"], 4)
        self.assertTrue(any("inserted background scaffold" in change for change in changes))

    def test_normalize_plan_for_real_series_extends_initial_background_for_late_additive_overlay(self) -> None:
        reference = np.array([6.45, 6.45, 6.46, 6.47, 6.48, 6.49, 6.5], dtype=float)
        plan = {
            "input_description": "The series stays stable at 6.45, then later rises and a local peak is added on top.",
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_flat",
                    "effect_type": "overwrite",
                    "semantic_role": "initial baseline",
                    "source_text": "The series stays stable at 6.45",
                    "parameters": {"start_timestep": 0, "end_timestep": 1, "target_value": 6.45},
                },
                {
                    "id": "step_2",
                    "kind": "primitive",
                    "primitive": "add_peak",
                    "effect_type": "additive",
                    "semantic_role": "local peak",
                    "source_text": "a local peak is added on top",
                    "parameters": {"timestep": 5, "target_value": 6.5},
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }

        changes = normalize_plan_for_real_series(plan, reference=reference)

        first_step = next(step for step in plan["steps"] if step.get("id") == "step_1")
        self.assertEqual(first_step["parameters"]["end_timestep"], 6)
        self.assertIn("start_timestep", plan["steps"][1]["parameters"])
        self.assertIn("end_timestep", plan["steps"][1]["parameters"])

    def test_resolve_representable_real_unresolved_normalizes_string_entries(self) -> None:
        plan = {
            "steps": [
                {
                    "id": "u1",
                    "kind": "unresolved",
                    "description": "Initial stable portion from timestep 0 to 9 with value 20.55.",
                    "semantic_role": "missing initial baseline",
                    "confidence": 0.6,
                    "source_text": "Initial stable portion from timestep 0 to 9 with value 20.55.",
                    "reason": "Planner left the baseline unresolved.",
                    "needs_library_extension": True,
                }
            ],
            "unresolved_semantics": ["Initial stable portion from timestep 0 to 9 with value 20.55."],
            "needs_library_extension": True,
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertFalse(changes)
        self.assertEqual(
            plan["unresolved_semantics"][0],
            {
                "id": "u1",
                "description": "Initial stable portion from timestep 0 to 9 with value 20.55.",
                "source_text": "Initial stable portion from timestep 0 to 9 with value 20.55.",
                "reason": "Planner left the baseline unresolved.",
            },
        )

    def test_remove_generic_real_seasonality_does_not_treat_periods_as_periodic(self) -> None:
        plan = {
            "steps": [
                {
                    "kind": "primitive",
                    "primitive": "add_seasonality",
                    "semantic_role": "oscillation",
                    "source_text": "periods of slight increase and oscillation without significant trend",
                },
                {"kind": "primitive", "primitive": "add_noise", "parameters": {}},
            ]
        }

        changes = remove_generic_real_seasonality(plan)

        self.assertEqual(
            changes,
            ["removed add_seasonality for generic non-periodic oscillation already covered by noise/volatility"],
        )
        self.assertEqual([step["primitive"] for step in plan["steps"]], ["add_noise"])

    def test_resolve_representable_real_unresolved_handles_soft_level_references(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [
                {"kind": "primitive", "primitive": "add_noise", "parameters": {}},
                {"kind": "primitive", "primitive": "add_volatility", "parameters": {}},
            ],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "some oscillation",
                    "source_text": "some oscillation without significant trend",
                    "reason": "no direct oscillation primitive",
                },
                {
                    "id": "u2",
                    "description": "resistance around 3500",
                    "source_text": "resistance around 3500",
                    "reason": "soft mean-reversion constraint",
                },
            ],
        }
        changes = resolve_representable_real_unresolved(plan)
        self.assertEqual(len(changes), 2)
        self.assertFalse(plan["needs_library_extension"])
        self.assertEqual(plan["unresolved_semantics"], [])

    def test_resolve_representable_real_unresolved_handles_soft_mean_reversion(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [
                {"kind": "primitive", "primitive": "add_ramp", "parameters": {}},
                {"kind": "primitive", "primitive": "add_flat", "parameters": {}},
                {"kind": "primitive", "primitive": "add_volatility", "parameters": {}},
            ],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "mean reversion toward a long-run level",
                    "source_text": "generally mean-reverting behavior",
                    "reason": "No provided primitive explicitly applies pullback toward a reference mean.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertTrue(changes)
        self.assertFalse(plan["needs_library_extension"])
        self.assertEqual(plan["unresolved_semantics"], [])

    def test_resolve_representable_real_unresolved_removes_resolved_steps(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [
                {"kind": "primitive", "primitive": "add_ramp", "parameters": {}},
                {
                    "id": "unresolved_1",
                    "kind": "unresolved",
                    "description": "minor decrease at the end",
                    "source_text": "minor decrease at the end",
                    "reason": "No primitive to add a small decrease at the end.",
                    "needs_library_extension": True,
                },
            ],
            "unresolved_semantics": [
                {
                    "id": "unresolved_1",
                    "description": "minor decrease at the end",
                    "source_text": "minor decrease at the end",
                    "reason": "No primitive to add a small decrease at the end.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertTrue(changes)
        self.assertFalse(plan["needs_library_extension"])
        self.assertEqual(plan["unresolved_semantics"], [])
        self.assertEqual([step["kind"] for step in plan["steps"]], ["primitive"])

    def test_resolve_representable_real_unresolved_handles_general_growth_decline_periods(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [
                {
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "parameters": {"start_value": 0.01, "end_value": 0.05},
                },
                {
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "parameters": {"start_value": 0.05, "end_value": 0.02},
                },
            ],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "periods of growth and decline",
                    "source_text": "Overall, the series shows periods of growth and decline.",
                    "reason": "Too general without timing and magnitude.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertTrue(changes)
        self.assertFalse(plan["needs_library_extension"])
        self.assertEqual(plan["unresolved_semantics"], [])

    def test_resolve_representable_real_unresolved_downgrades_soft_summary_phrases(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [
                {
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "parameters": {"start_value": 18.0, "end_value": 16.0},
                },
                {
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "parameters": {"start_value": 16.0, "end_value": 22.0},
                },
                {"kind": "primitive", "primitive": "add_peak", "parameters": {"timestep": 20, "target_value": 22.0}},
            ],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "cyclical pattern",
                    "source_text": "reflecting a cyclical pattern",
                    "reason": "No primitive explicitly models a cyclical pattern with automatic period inference.",
                },
                {
                    "id": "u2",
                    "description": "complex non-linear trend",
                    "source_text": "Overall, the series shows a complex non-linear trend.",
                    "reason": "Too vague to translate into a specific primitive.",
                },
                {
                    "id": "u3",
                    "description": "initial fluctuations",
                    "source_text": "initially fluctuates",
                    "reason": "No primitive to model fluctuations without specifying parameters.",
                },
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertEqual(len(changes), 3)
        self.assertTrue(all("library extension" in change or "composition" in change for change in changes))
        self.assertFalse(plan["needs_library_extension"])
        self.assertEqual(plan["unresolved_semantics"], [])

    def test_resolve_representable_real_unresolved_downgrades_no_growth_constraint(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [
                {
                    "kind": "primitive",
                    "primitive": "add_ramp",
                    "parameters": {"start_value": 3245.0, "end_value": 3028.0},
                }
            ],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "no significant long-term growth",
                    "source_text": "no significant long-term growth",
                    "reason": "The absence of growth is a constraint, not a primitive.",
                },
                {
                    "id": "u2",
                    "description": "struggling to maintain consistent upward momentum",
                    "source_text": "struggling to maintain consistent upward momentum",
                    "reason": "This is a high-level summary rather than a primitive requirement.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertEqual(
            changes,
            [
                "resolved satisfied negative constraint without library extension",
                "resolved satisfied negative constraint without library extension",
            ],
        )
        self.assertFalse(plan["needs_library_extension"])
        self.assertEqual(plan["unresolved_semantics"], [])

    def test_resolve_representable_real_unresolved_keeps_hard_periodic_gap(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [{"kind": "primitive", "primitive": "add_ramp", "parameters": {}}],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "weekly seasonal cycle",
                    "source_text": "weekly seasonal cycle",
                    "reason": "No periodic parameters were provided.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertEqual(changes, [])
        self.assertTrue(plan["needs_library_extension"])
        self.assertEqual(len(plan["unresolved_semantics"]), 1)

    def test_resolve_representable_real_unresolved_does_not_clear_unrelated_hard_gap(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [{"kind": "primitive", "primitive": "add_peak", "parameters": {"timestep": 12}}],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "long-memory mean reversion toward a reference level",
                    "source_text": "long-memory mean reversion toward a reference level",
                    "reason": "No available primitive captures the memory mechanism.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertEqual(changes, [])
        self.assertTrue(plan["needs_library_extension"])
        self.assertEqual(len(plan["unresolved_semantics"]), 1)

    def test_resolve_representable_real_unresolved_does_not_use_event_for_periodic_requirement(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [{"kind": "primitive", "primitive": "add_peak", "parameters": {"timestep": 12}}],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "weekly seasonal cycle",
                    "source_text": "weekly seasonal cycle",
                    "reason": "No periodic parameters were provided.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertEqual(changes, [])
        self.assertTrue(plan["needs_library_extension"])
        self.assertEqual(len(plan["unresolved_semantics"]), 1)

    def test_resolve_representable_real_unresolved_does_not_clear_global_constraint_from_event(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [{"kind": "primitive", "primitive": "add_peak", "parameters": {"timestep": 12}}],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "monotonic non-decreasing behavior over the whole series",
                    "source_text": "monotonic non-decreasing behavior over the whole series",
                    "reason": "The current plan only contains a local event.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertEqual(changes, [])
        self.assertTrue(plan["needs_library_extension"])
        self.assertEqual(len(plan["unresolved_semantics"]), 1)

    def test_resolve_representable_real_unresolved_treats_unanchored_soft_events_as_variation(self) -> None:
        plan = {
            "needs_library_extension": True,
            "steps": [{"kind": "primitive", "primitive": "add_noise", "parameters": {"noise_scale": 0.01}}],
            "unresolved_semantics": [
                {
                    "id": "u1",
                    "description": "some dips",
                    "source_text": "despite some dips",
                    "reason": "Too vague to locate as individual dip primitives.",
                }
            ],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertEqual(changes, ["resolved non-blocking summary as existing primitive composition"])
        self.assertFalse(plan["needs_library_extension"])
        self.assertEqual(plan["unresolved_semantics"], [])

    def test_normalize_plan_for_real_series_coerces_width_to_integer(self) -> None:
        reference = np.linspace(0.0, 1.0, 24)
        plan = {
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_dip",
                    "parameters": {"timestep": 12, "target_value": -0.5, "width": 10.0},
                    "argument_pattern": ["target_value"],
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }

        normalize_plan_for_real_series(plan, reference=reference)

        dip_step = next(step for step in plan["steps"] if step.get("id") == "step_1")
        self.assertEqual(dip_step["parameters"]["width"], 10)
        self.assertIsInstance(dip_step["parameters"]["width"], int)
        self.assertNotIn("argument_pattern", dip_step)

    def test_resolve_representable_real_unresolved_removes_orphan_unresolved_steps(self) -> None:
        plan = {
            "needs_library_extension": False,
            "steps": [
                {"kind": "primitive", "primitive": "add_noise", "parameters": {}},
                {
                    "id": "step_2",
                    "kind": "unresolved",
                    "description": "generic oscillation",
                    "source_text": "generic oscillation",
                    "reason": "already resolved elsewhere",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
        }

        changes = resolve_representable_real_unresolved(plan)

        self.assertEqual(changes, ["removed orphan unresolved steps after unresolved semantics were cleared"])
        self.assertEqual([step["kind"] for step in plan["steps"]], ["primitive"])


if __name__ == "__main__":
    unittest.main()
