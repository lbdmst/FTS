from __future__ import annotations

import os
import json
import unittest
from unittest import mock

import numpy as np

from faithts_pipeline.hybrid_pipeline import _ordered_steps, compile_plan_to_code, execute_plan, extract_json_payload, parse_plan
from faithts_pipeline.workflow import (
    DEFAULT_MAX_RETRIEVED_PRIMITIVES,
    ExternalLLMConfigurationError,
    LLMGeneration,
    OpenAICompatibleLLMClient,
    OpenAICompatibleConfig,
    RetrievalExample,
    build_eval_report,
    build_llm_client,
    build_parameter_inference_prompt,
    build_prompt,
    extend_background_steps_for_additive_overlays,
    extract_parameter_payload,
    load_registry_primitives,
    merge_inferred_parameters,
    retrieve_primitives_for_caption,
    evaluate_generated_code,
    execute_generated_code,
    extract_text_from_openai_compatible_response,
    load_primitive_specs,
    load_retrieval_examples,
    run_workflow_for_example,
    select_examples,
    summarize_series,
)
from schema_validator import validate_payload


def _global_context(length: int, *, min_value: float = -1.0, max_value: float = 100.0) -> dict[str, object]:
    return {
        "length": length,
        "numeric_range": {"min": min_value, "max": max_value},
        "dt": None,
        "seed": None,
    }


class StubLLMClient:
    def generate_code(self, prompt: str, example) -> LLMGeneration:
        del example
        if "inferring primitive call parameters" in prompt:
            return LLMGeneration(
                provider="stub",
                model="stub-model",
                raw_text=json.dumps(
                    {
                        "steps": [
                            {
                                "id": "step_1",
                                "primitive": "add_flat",
                                "parameters": {
                                    "start_timestep": 0,
                                    "end_timestep": 7,
                                    "target_value": 7.5,
                                },
                            }
                        ]
                    }
                ),
                response_id="stub-2",
                finish_reason="stop",
            )
        return LLMGeneration(
            provider="stub",
            model="stub-model",
            raw_text=json.dumps(
                {
                    "version": "1.0",
                    "input_description": "One flat overwrite segment at 7.5.",
                    "registry_validation": {
                        "status": "valid",
                        "schema_path": "schemas/primitive.schema.json",
                        "target_path": "atomic_timeseries/registry.json",
                        "errors": [],
                    },
                    "global_context": _global_context(8, min_value=0.0, max_value=10.0),
                    "steps": [
                        {
                            "kind": "primitive",
                            "id": "step_1",
                            "effect_type": "overwrite",
                            "primitive": "add_flat",
                            "parameters": {},
                            "semantic_role": "flat segment",
                            "confidence": 1.0,
                            "source_text": "flat at 7.5",
                            "needs_library_extension": False,
                        }
                    ],
                    "unresolved_semantics": [],
                    "needs_library_extension": False,
                    "assumptions": [],
                }
            ),
            response_id="stub-1",
            finish_reason="stop",
        )


class StubNoiseLLMClient:
    def generate_code(self, prompt: str, example) -> LLMGeneration:
        del example
        if "inferring primitive call parameters" in prompt:
            return LLMGeneration(
                provider="stub",
                model="stub-model",
                raw_text=json.dumps(
                    {
                        "steps": [
                            {
                                "id": "step_1",
                                "primitive": "add_noise",
                                "parameters": {
                                    "start_timestep": 0,
                                    "end_timestep": 7,
                                    "noise_scale": 0.1,
                                    "random_seed": 7,
                                },
                            }
                        ]
                    }
                ),
                response_id="stub-noise-2",
                finish_reason="stop",
            )
        return LLMGeneration(
            provider="stub",
            model="stub-model",
            raw_text=json.dumps(
                {
                    "version": "1.0",
                    "input_description": "A noisy segment with Gaussian jitter.",
                    "registry_validation": {
                        "status": "valid",
                        "schema_path": "schemas/primitive.schema.json",
                        "target_path": "atomic_timeseries/registry.json",
                        "errors": [],
                    },
                    "global_context": _global_context(8, min_value=-1000.0, max_value=1000.0),
                    "steps": [
                        {
                            "kind": "primitive",
                            "id": "step_1",
                            "effect_type": "additive",
                            "primitive": "add_noise",
                            "parameters": {},
                            "semantic_role": "noise",
                            "confidence": 1.0,
                            "source_text": "Gaussian jitter",
                            "needs_library_extension": False,
                        }
                    ],
                    "unresolved_semantics": [],
                    "needs_library_extension": False,
                    "assumptions": [],
                }
            ),
            response_id="stub-noise-1",
            finish_reason="stop",
        )


class PrototypeCodegenTests(unittest.TestCase):
    def test_prompt_includes_caption_candidates_and_specs(self) -> None:
        examples = load_retrieval_examples()
        specs = load_primitive_specs()
        example = select_examples(examples, [3])[0]
        retrieved = retrieve_primitives_for_caption(example.caption, specs, example.candidate_primitives)
        prompt = build_prompt(example, retrieved, numeric_range=(0.0, 10.0))
        self.assertIn(example.caption, prompt)
        self.assertIn("add_flat", prompt)
        self.assertIn("signature:", prompt)
        self.assertIn("Primitive specs:", prompt)
        self.assertIn("effect_type:", prompt)
        self.assertIn("Do not emit `unresolved` for a negated semantic", prompt)
        self.assertIn("use `parameters` to encode temporal scope", prompt)
        self.assertIn("several peaks", prompt)

    def test_build_parameter_inference_prompt_includes_selection_plan_and_arg_constraints(self) -> None:
        examples = load_retrieval_examples()
        example = select_examples(examples, [3])[0]
        registry = load_registry_primitives()
        selection_plan = {
            "version": "1.0",
            "input_description": example.caption,
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=10.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {},
                    "semantic_role": "flat segment",
                    "confidence": 1.0,
                    "source_text": "flat at 7.5",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        prompt = build_parameter_inference_prompt(
            example=example,
            selection_plan=selection_plan,
            registry_primitives=registry,
            series_length=8,
            numeric_range=(0.0, 10.0),
        )
        self.assertIn('"primitive": "add_flat"', prompt)
        self.assertIn('"required_args": [', prompt)
        self.assertIn('"constraints": {', prompt)

    def test_merge_inferred_parameters_preserves_selected_primitives(self) -> None:
        selection_plan = {
            "version": "1.0",
            "input_description": "One flat overwrite segment at 7.5.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=10.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {},
                    "semantic_role": "flat segment",
                    "confidence": 1.0,
                    "source_text": "flat at 7.5",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        inferred = extract_parameter_payload(
            json.dumps(
                {
                    "steps": [
                        {
                            "id": "step_1",
                            "primitive": "add_flat",
                            "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 7.5},
                        }
                    ]
                }
            )
        )
        merged = merge_inferred_parameters(selection_plan, inferred)
        self.assertEqual(merged["steps"][0]["parameters"]["target_value"], 7.5)

    def test_merge_inferred_parameters_preserves_selected_temporal_scope(self) -> None:
        selection_plan = {
            "version": "1.0",
            "input_description": "One flat overwrite segment at 7.5.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=10.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7},
                    "semantic_role": "flat segment",
                    "confidence": 1.0,
                    "source_text": "flat at 7.5",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        inferred = extract_parameter_payload(
            json.dumps(
                {
                    "steps": [
                        {
                            "id": "step_1",
                            "primitive": "add_flat",
                            "parameters": {"target_value": 7.5},
                        }
                    ]
                }
            )
        )
        merged = merge_inferred_parameters(selection_plan, inferred)
        self.assertEqual(merged["steps"][0]["parameters"]["start_timestep"], 0)
        self.assertEqual(merged["steps"][0]["parameters"]["end_timestep"], 7)
        self.assertEqual(merged["steps"][0]["parameters"]["target_value"], 7.5)

    def test_merge_inferred_parameters_drops_none_values(self) -> None:
        selection_plan = {
            "version": "1.0",
            "input_description": "A downward trend after the plateau.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(100, min_value=0.0, max_value=10.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "additive",
                    "primitive": "add_trend",
                    "parameters": {"start_timestep": 20, "end_timestep": 80},
                    "semantic_role": "downward trend",
                    "confidence": 1.0,
                    "source_text": "a downward trend begins",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        inferred = extract_parameter_payload(
            json.dumps(
                {
                    "steps": [
                        {
                            "id": "step_1",
                            "primitive": "add_trend",
                            "parameters": {"slope": None, "start_offset": 0.0, "end_offset": -1.0},
                        }
                    ]
                }
            )
        )
        merged = merge_inferred_parameters(selection_plan, inferred)
        self.assertNotIn("slope", merged["steps"][0]["parameters"])
        self.assertEqual(merged["steps"][0]["parameters"]["start_offset"], 0.0)
        self.assertEqual(merged["steps"][0]["parameters"]["end_offset"], -1.0)

    def test_extend_background_steps_for_additive_overlays_extends_flat_baseline(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series starts flat at 4.0, then from timestep 20 to 79 a periodic sine oscillation is added on top.",
            "registry_validation": {"status": "valid", "schema_path": "", "target_path": "", "errors": []},
            "global_context": _global_context(100, min_value=0.0, max_value=10.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 19, "target_value": 4.0},
                    "semantic_role": "initial flat segment",
                    "confidence": 1.0,
                    "source_text": "The series starts flat at 4.0",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_seasonality",
                    "parameters": {"start_timestep": 20, "end_timestep": 79, "amplitude": 1.5, "period": 12.0},
                    "semantic_role": "periodic sine oscillation",
                    "confidence": 1.0,
                    "source_text": "then from timestep 20 to 79 a periodic sine oscillation is added on top.",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        normalized = extend_background_steps_for_additive_overlays(payload, series_length=100)
        self.assertEqual(normalized["steps"][0]["parameters"]["end_timestep"], 99)

    def test_extend_background_steps_for_additive_overlays_extends_trend_baseline(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series trends upward, then oscillation is added on top.",
            "registry_validation": {"status": "valid", "schema_path": "", "target_path": "", "errors": []},
            "global_context": _global_context(100, min_value=0.0, max_value=10.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "additive",
                    "primitive": "add_trend",
                    "parameters": {"start_timestep": 0, "end_timestep": 19, "slope": 0.1},
                    "semantic_role": "background trend",
                    "confidence": 1.0,
                    "source_text": "The series trends upward",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_seasonality",
                    "parameters": {"start_timestep": 20, "end_timestep": 79, "amplitude": 1.5, "period": 12.0},
                    "semantic_role": "periodic sine oscillation",
                    "confidence": 1.0,
                    "source_text": "oscillation is added on top.",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        normalized = extend_background_steps_for_additive_overlays(payload, series_length=100)
        self.assertEqual(normalized["steps"][0]["parameters"]["end_timestep"], 99)

    def test_extend_background_steps_for_additive_overlays_extends_initial_flat_baseline_for_growth(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series begins near 30.6 and remains fairly stable before entering a curved growth phase.",
            "registry_validation": {"status": "valid", "schema_path": "", "target_path": "", "errors": []},
            "global_context": _global_context(100, min_value=30.0, max_value=32.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 9, "target_value": 30.6},
                    "semantic_role": "initial baseline",
                    "confidence": 1.0,
                    "source_text": "The series begins near 30.6 and remains fairly stable",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "overwrite",
                    "primitive": "add_growth",
                    "parameters": {"start_timestep": 10, "end_timestep": 43, "end_value": 31.4},
                    "semantic_role": "curved growth phase",
                    "confidence": 1.0,
                    "source_text": "entering a curved growth phase from roughly timestep 10 to 43",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_3",
                    "effect_type": "additive",
                    "primitive": "add_level_shift",
                    "parameters": {"start_timestep": 53, "shift": -0.2},
                    "semantic_role": "late negative shift",
                    "confidence": 1.0,
                    "source_text": "starting around timestep 53, the level shifts downward",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        normalized = extend_background_steps_for_additive_overlays(payload, series_length=100)
        self.assertEqual(normalized["steps"][0]["parameters"]["end_timestep"], 99)

    def test_extend_background_steps_for_additive_overlays_extends_initial_flat_baseline_for_seasonality(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The time series starts from a stable baseline near 70.39 and then develops a repeating oscillatory pattern.",
            "registry_validation": {"status": "valid", "schema_path": "", "target_path": "", "errors": []},
            "global_context": _global_context(100, min_value=69.0, max_value=71.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 9, "target_value": 70.39},
                    "semantic_role": "stable baseline",
                    "confidence": 1.0,
                    "source_text": "starts from a stable baseline near 70.39",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_seasonality",
                    "parameters": {"start_timestep": 10, "end_timestep": 67, "amplitude": 0.1, "period": 10.0},
                    "semantic_role": "oscillatory pattern",
                    "confidence": 1.0,
                    "source_text": "develops a repeating oscillatory pattern from about timestep 10 to 67",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        normalized = extend_background_steps_for_additive_overlays(payload, series_length=100)
        self.assertEqual(normalized["steps"][0]["parameters"]["end_timestep"], 99)

    def test_retrieve_primitives_for_caption_uses_bounded_subset(self) -> None:
        specs = load_primitive_specs()
        caption = "A mostly flat series with a sharp peak and a small dip later."
        retrieved = retrieve_primitives_for_caption(caption, specs, retrieval_hints=["add_flat", "add_peak", "add_dip"])
        names = [spec.function_name for spec in retrieved]
        self.assertIn("add_flat", names)
        self.assertIn("add_peak", names)
        self.assertLessEqual(len(names), DEFAULT_MAX_RETRIEVED_PRIMITIVES)

    def test_retrieve_primitives_for_caption_adds_texture_and_plateau_for_compositional_caption(self) -> None:
        specs = load_primitive_specs()
        caption = (
            "The series stays stable at first, then declines with small fluctuations, "
            "shows an early peak, and finally settles into a late plateau."
        )
        retrieved = retrieve_primitives_for_caption(
            caption,
            specs,
            retrieval_hints=["add_flat", "add_peak", "add_trend", "add_ramp"],
            max_candidates=10,
        )
        names = [spec.function_name for spec in retrieved]
        self.assertIn("add_noise", names)
        self.assertIn("add_plateau", names)
        self.assertIn("add_peak", names)

    def test_execute_generated_code_returns_series(self) -> None:
        code = """
import numpy as np
from atomic_timeseries import add_flat

series = np.zeros(8, dtype=float)
series = add_flat(series, start_timestep=0, end_timestep=7, target_value=2.5)
"""
        result = execute_generated_code(code, ["add_flat"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (8,))
        self.assertTrue(np.allclose(result, 2.5))

    def test_execute_generated_code_rejects_unsafe_imports(self) -> None:
        code = """
import os
series = [1, 2, 3]
"""
        with self.assertRaisesRegex(RuntimeError, "Only `import numpy as np` is allowed"):
            execute_generated_code(code, ["add_flat"])

    def test_execute_generated_code_rejects_numpy_import_from(self) -> None:
        code = """
from numpy import linspace
from atomic_timeseries import add_flat

series = np.zeros(8, dtype=float)
series = add_flat(series, start_timestep=0, end_timestep=7, target_value=2.5)
"""
        with self.assertRaisesRegex(RuntimeError, "from atomic_timeseries import"):
            execute_generated_code(code, ["add_flat"])

    def test_execute_generated_code_rejects_disallowed_numpy_imported_symbols(self) -> None:
        code = """
import numpy as np
from atomic_timeseries import add_flat

series = np.zeros(8, dtype=float)
series = np.mean(series)
"""
        with self.assertRaisesRegex(RuntimeError, "Disallowed numpy call: mean"):
            execute_generated_code(code, ["add_flat"])

    def test_execute_generated_code_rejects_function_definitions(self) -> None:
        code = """
import numpy as np
from atomic_timeseries import add_flat

def helper():
    return 1

series = np.zeros(8, dtype=float)
series = add_flat(series, start_timestep=0, end_timestep=7, target_value=2.5)
"""
        with self.assertRaisesRegex(RuntimeError, "Unsupported Python construct in generated code: FunctionDef"):
            execute_generated_code(code, ["add_flat"])

    def test_execute_generated_code_rejects_unassigned_primitive_call(self) -> None:
        code = """
import numpy as np
from atomic_timeseries import add_flat

series = np.zeros(8, dtype=float)
add_flat(series, start_timestep=0, end_timestep=7, target_value=2.5)
"""
        with self.assertRaisesRegex(RuntimeError, "must be assigned back to `series`"):
            execute_generated_code(code, ["add_flat"])

    def test_execute_generated_code_rejects_additive_as_setter(self) -> None:
        code = """
import numpy as np
from atomic_timeseries import add_trend

series = np.zeros(8, dtype=float)
series = add_trend(series, start_timestep=0, end_timestep=7, start_value=series[0], end_value=2.5)
"""
        with self.assertRaisesRegex(RuntimeError, "setter-style value"):
            execute_generated_code(code, ["add_trend"])

    def test_extract_text_from_responses_payload(self) -> None:
        payload = {
            "id": "resp_123",
            "model": "gpt-x",
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "```python\nprint('x')\n```"},
                    ],
                    "finish_reason": "stop",
                }
            ],
        }
        text, finish_reason = extract_text_from_openai_compatible_response(payload)
        self.assertIn("print('x')", text)
        self.assertEqual(finish_reason, "stop")

    def test_openai_config_from_env_requires_api_key_and_model(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ExternalLLMConfigurationError):
                OpenAICompatibleConfig.from_env()

    def test_build_llm_client_openai_compatible(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "PROTOTYPE_CODEGEN_API_KEY": "test-key",
                "PROTOTYPE_CODEGEN_MODEL": "test-model",
            },
            clear=True,
        ):
            client = build_llm_client("openai-compatible")
        self.assertIsInstance(client, OpenAICompatibleLLMClient)

    def test_run_workflow_for_example_captures_prompt_and_metadata(self) -> None:
        examples = load_retrieval_examples()
        specs = load_primitive_specs()
        example = select_examples(examples, [1])[0]
        result = run_workflow_for_example(example, specs, StubLLMClient(), series_length=8, numeric_range=(0.0, 10.0))
        self.assertEqual(result.example_index, 1)
        self.assertEqual(result.llm_provider, "stub")
        self.assertEqual(result.llm_model, "stub-model")
        self.assertIn(example.caption, result.prompt)
        self.assertTrue(result.retrieved_primitives)
        self.assertIn("steps", result.structured_plan)
        self.assertTrue(result.contract_valid)
        self.assertTrue(result.executed)
        self.assertTrue(result.usable_success)
        self.assertEqual(result.series_summary["length"], 8)
        self.assertEqual(result.eval_report["execution_status"]["status"], "success")

    def test_run_workflow_for_example_returns_result_even_when_evaluation_has_failure_categories(self) -> None:
        specs = load_primitive_specs()
        example = RetrievalExample(
            index=1,
            caption="A noisy segment with Gaussian jitter that ends at 3.0.",
            candidate_primitives=["add_noise"],
            retrieval_status="gold-case",
            orchestration_level_descriptors=[],
            raw_record={"numeric_range": {"min": -1000.0, "max": 1000.0}},
        )
        result = run_workflow_for_example(
            example,
            specs,
            StubNoiseLLMClient(),
            series_length=8,
            numeric_range=(-1000.0, 1000.0),
        )
        self.assertTrue(result.executed)
        self.assertIn("endpoint_mismatch", result.failure_categories)
        self.assertEqual(result.eval_report["execution_status"]["status"], "success")

    def test_stub_generation_is_summarizable(self) -> None:
        generated = StubLLMClient().generate_code("", None)
        selection_plan = json.loads(generated.raw_text)
        parameter_payload = extract_parameter_payload(StubLLMClient().generate_code("inferring primitive call parameters", None).raw_text)
        merged_plan = merge_inferred_parameters(selection_plan, parameter_payload)
        plan = parse_plan(merged_plan, ["add_flat"], series_length=8)
        code = compile_plan_to_code(plan, series_length=8)
        series = execute_generated_code(code, ["add_flat"])
        summary = summarize_series(series)
        self.assertEqual(summary["length"], 8)
        self.assertIn("mean", summary)

    def test_parse_plan_and_execute_plan(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "Stable at 2.0 with a small local bump to 2.5.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=3.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 2.0},
                    "semantic_role": "flat base",
                    "semantic_layer": "global_scaffold",
                    "priority": 0,
                    "protected_constraints": [],
                    "confidence": 1.0,
                    "source_text": "Stable at 2.0",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_spike",
                    "parameters": {"timestep": 3, "amplitude": 0.5},
                    "semantic_role": "small local bump",
                    "semantic_layer": "local_event",
                    "priority": 1,
                    "protected_constraints": [],
                    "confidence": 0.9,
                    "source_text": "small local bump to 2.5",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_flat", "add_spike"], series_length=8)
        series = execute_plan(plan, "Stable at 2.0 with a small local bump to 2.5.", series_length=8)
        self.assertTrue(np.allclose(series[[0, 1, 5, 6, 7]], 2.0))
        self.assertEqual(series[3], 2.5)

    def test_execute_plan_allows_initial_additive_step_on_zero_baseline(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "A gentle upward trend from 0.0 to 3.0.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=4.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "additive",
                    "primitive": "add_trend",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "start_offset": 0.0, "end_offset": 3.0},
                    "semantic_role": "global upward trend",
                    "semantic_layer": "segment_structure",
                    "priority": 0,
                    "protected_constraints": [],
                    "confidence": 0.9,
                    "source_text": "gentle upward trend",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_trend"], series_length=8)
        series = execute_plan(plan, payload["input_description"], series_length=8)
        self.assertEqual(series[0], 0.0)
        self.assertAlmostEqual(series[-1], 3.0)

    def test_extract_json_payload_normalizes_string_unresolved_semantics(self) -> None:
        raw = json.dumps(
            {
                "version": "1.0",
                "input_description": "Ambiguous decline with unsupported semantics.",
                "registry_validation": {
                    "status": "valid",
                    "schema_path": "schemas/primitive.schema.json",
                    "target_path": "atomic_timeseries/registry.json",
                    "errors": [],
                },
                "global_context": _global_context(8, min_value=-1.0, max_value=10.0),
                "steps": [
                    {
                        "id": "step_1",
                        "kind": "unresolved",
                        "description": "fluctuating but gradual decline after timestep 15",
                        "semantic_role": "declining noisy tail",
                        "confidence": 0.6,
                        "source_text": "fluctuating but gradual decline after timestep 15",
                        "reason": "No single provided primitive captures this semantic exactly.",
                        "needs_library_extension": True,
                    }
                ],
                "unresolved_semantics": ["fluctuating but gradual decline after timestep 15"],
                "needs_library_extension": True,
                "assumptions": [],
                "notes": "",
            }
        )
        payload = extract_json_payload(raw)
        self.assertEqual(
            payload["unresolved_semantics"],
            [
                {
                    "id": "step_1",
                    "description": "fluctuating but gradual decline after timestep 15",
                    "source_text": "fluctuating but gradual decline after timestep 15",
                    "reason": "No single provided primitive captures this semantic exactly.",
                }
            ],
        )

    def test_parse_plan_allows_partial_output_when_requested(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "A partial plan with an executable flat segment and one unsupported detail.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=2.0),
            "steps": [
                {
                    "id": "step_1",
                    "kind": "primitive",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 1.0},
                    "semantic_role": "flat executable subset",
                    "effect_type": "overwrite",
                    "confidence": 0.9,
                    "source_text": "flat around one",
                    "needs_library_extension": False,
                },
                {
                    "id": "unresolved_1",
                    "kind": "unresolved",
                    "description": "unsupported fine-grained domain constraint",
                    "semantic_role": "unsupported detail",
                    "confidence": 0.4,
                    "source_text": "unsupported detail",
                    "reason": "No registered primitive covers this detail.",
                    "needs_library_extension": True,
                },
            ],
            "unresolved_semantics": [
                {
                    "id": "unresolved_1",
                    "description": "unsupported fine-grained domain constraint",
                    "source_text": "unsupported detail",
                    "reason": "No registered primitive covers this detail.",
                }
            ],
            "needs_library_extension": True,
            "assumptions": [],
        }

        with self.assertRaisesRegex(RuntimeError, "requires a library extension"):
            parse_plan(payload, ["add_flat"], series_length=8)

        plan = parse_plan(payload, ["add_flat"], series_length=8, allow_partial_output=True)
        self.assertEqual([step.primitive for step in plan.primitive_steps], ["add_flat"])
        self.assertTrue(plan.needs_library_extension)
        self.assertEqual(len(plan.unresolved_semantics), 1)
        np.testing.assert_allclose(execute_plan(plan, payload["input_description"], series_length=8), np.ones(8))

    def test_parse_plan_rejects_effect_type_mismatch(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "Invalid plan.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=-1.0, max_value=10.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_trend",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "slope": 1.0},
                    "semantic_role": "wrong effect type",
                    "semantic_layer": "segment_structure",
                    "priority": 0,
                    "protected_constraints": [],
                    "confidence": 1.0,
                    "source_text": "invalid",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        with self.assertRaisesRegex(RuntimeError, "effect_type"):
            parse_plan(payload, ["add_trend"], series_length=8)

    def test_parse_plan_rejects_plural_peak_caption_collapsed_to_single_event(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series has several peaks around the middle and late portions.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=5.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 2.0},
                    "semantic_role": "baseline",
                    "semantic_layer": "segment_structure",
                    "priority": 0,
                    "protected_constraints": [],
                    "confidence": 1.0,
                    "source_text": "baseline",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_peak",
                    "parameters": {"timestep": 4, "amplitude": 1.0},
                    "semantic_role": "single peak",
                    "semantic_layer": "local_event",
                    "priority": 1,
                    "protected_constraints": [],
                    "confidence": 0.8,
                    "source_text": "peak around the middle",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        with self.assertRaisesRegex(RuntimeError, "multiple peaks"):
            parse_plan(payload, ["add_flat", "add_peak"], series_length=8)

    def test_parse_plan_accepts_plural_peak_caption_when_multiple_events_are_present(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series has several peaks around the middle and late portions.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=5.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 2.0},
                    "semantic_role": "baseline",
                    "semantic_layer": "segment_structure",
                    "priority": 0,
                    "protected_constraints": [],
                    "confidence": 1.0,
                    "source_text": "baseline",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_peak",
                    "parameters": {"timestep": 3, "amplitude": 0.7},
                    "semantic_role": "middle peak",
                    "semantic_layer": "local_event",
                    "priority": 1,
                    "protected_constraints": [],
                    "confidence": 0.8,
                    "source_text": "peak around the middle",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_3",
                    "effect_type": "additive",
                    "primitive": "add_peak",
                    "parameters": {"timestep": 6, "amplitude": 0.9},
                    "semantic_role": "late peak",
                    "semantic_layer": "local_event",
                    "priority": 2,
                    "protected_constraints": [],
                    "confidence": 0.8,
                    "source_text": "peak around the late portion",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_flat", "add_peak"], series_length=8)
        self.assertEqual([step.primitive for step in plan.primitive_steps], ["add_flat", "add_peak", "add_peak"])

    def test_parse_plan_populates_semantic_layer_defaults(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "Flat baseline with late noise.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=3.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 2.0},
                    "semantic_role": "flat base",
                    "confidence": 1.0,
                    "source_text": "flat baseline",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_noise",
                    "parameters": {"start_timestep": 4, "end_timestep": 7, "noise_scale": 0.1, "random_seed": 7},
                    "semantic_role": "late noise",
                    "confidence": 0.8,
                    "source_text": "late noise",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        payload = extract_json_payload(json.dumps(payload))
        plan = parse_plan(payload, ["add_flat", "add_noise"], series_length=8)
        self.assertEqual(plan.primitive_steps[0].semantic_layer, "segment_structure")
        self.assertEqual(plan.primitive_steps[1].semantic_layer, "texture")

    def test_parse_plan_preserves_inferred_constraints(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "A gradual decline after timestep 2.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=-2.0, max_value=2.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "additive",
                    "primitive": "add_trend",
                    "parameters": {"start_timestep": 2, "end_timestep": 7, "slope": -0.1},
                    "semantic_role": "declining tail",
                    "semantic_layer": "segment_structure",
                    "priority": 0,
                    "protected_constraints": [],
                    "semantic_cues": [
                        {"kind": "decline", "source_text": "gradual decline", "confidence": 0.9},
                    ],
                    "inferred_constraints": [
                        {"kind": "slope_sign", "value": "negative", "source_text": "decline", "confidence": 0.9},
                        {"kind": "monotonicity", "value": "nonincreasing", "source_text": "gradual decline", "confidence": 0.7},
                    ],
                    "confidence": 0.9,
                    "source_text": "gradual decline after timestep 2",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_trend"], series_length=8)
        self.assertEqual(
            plan.primitive_steps[0].inferred_constraints,
            [
                {"kind": "slope_sign", "value": "negative", "source_text": "decline", "confidence": 0.9},
                {"kind": "monotonicity", "value": "nonincreasing", "source_text": "gradual decline", "confidence": 0.7},
            ],
        )
        self.assertEqual(
            plan.primitive_steps[0].semantic_cues,
            [
                {"kind": "decline", "source_text": "gradual decline", "confidence": 0.9},
            ],
        )

    def test_parse_plan_allows_initial_flat_scaffold_outside_candidate_set_for_additive_only_candidates(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "A sharp spike reaches 3.5 at timestep 4.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=4.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 1.0},
                    "semantic_role": "implicit scaffold",
                    "confidence": 0.8,
                    "source_text": "implicit baseline",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_spike",
                    "parameters": {"timestep": 4, "target_value": 3.5},
                    "semantic_role": "spike",
                    "confidence": 0.9,
                    "source_text": "sharp spike reaches 3.5",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_spike"], series_length=8)
        self.assertEqual([step.primitive for step in plan.primitive_steps], ["add_flat", "add_spike"])

    def test_ordered_steps_runs_texture_before_precise_anchor_steps(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "A spike reaches 3.5 with slight background noise.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=-1.0, max_value=4.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "additive",
                    "primitive": "add_spike",
                    "parameters": {"timestep": 4, "target_value": 3.5},
                    "semantic_role": "spike",
                    "semantic_layer": "local_event",
                    "priority": 1,
                    "confidence": 0.9,
                    "source_text": "spike reaches 3.5",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_noise",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "noise_scale": 0.01, "random_seed": 7},
                    "semantic_role": "background noise",
                    "semantic_layer": "texture",
                    "priority": 0,
                    "confidence": 0.8,
                    "source_text": "slight background noise",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_spike", "add_noise"], series_length=8)
        ordered = _ordered_steps(plan)
        self.assertEqual([step.primitive for step in ordered], ["add_noise", "add_spike"])

    def test_extract_json_payload_defaults_semantic_cues_and_inferred_constraints(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "Flat baseline.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=0.0, max_value=3.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 2.0},
                    "semantic_role": "flat base",
                    "confidence": 1.0,
                    "source_text": "flat baseline",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        normalized = extract_json_payload(json.dumps(payload))
        self.assertEqual(normalized["steps"][0]["semantic_cues"], [])
        self.assertEqual(normalized["steps"][0]["inferred_constraints"], [])

    def test_extract_json_payload_derives_semantic_cues_from_legacy_inferred_constraints(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "A gradual decline after timestep 2.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=-2.0, max_value=2.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "additive",
                    "primitive": "add_trend",
                    "parameters": {"start_timestep": 2, "end_timestep": 7, "slope": -0.1},
                    "semantic_role": "declining tail",
                    "confidence": 1.0,
                    "source_text": "gradual decline",
                    "inferred_constraints": ["negative slope"],
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        normalized = extract_json_payload(json.dumps(payload))
        self.assertEqual(
            normalized["steps"][0]["semantic_cues"],
            [{"kind": "decline", "source_text": "negative slope"}],
        )
        self.assertEqual(
            normalized["steps"][0]["inferred_constraints"],
            [{"kind": "slope_sign", "value": "negative", "source_text": "negative slope", "confidence": 0.8}],
        )

    def test_execute_plan_allows_decline_semantic_cue_metadata_without_enforcing_parameter_sign(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "A decline after timestep 2.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=9.8, max_value=10.2),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "additive",
                    "primitive": "add_trend",
                    "parameters": {"start_timestep": 2, "end_timestep": 7, "slope": 0.1},
                    "semantic_role": "declining tail",
                    "semantic_cues": [{"kind": "decline", "source_text": "decline", "confidence": 0.9}],
                    "confidence": 0.9,
                    "source_text": "decline",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_trend"], series_length=8)
        series = execute_plan(plan, payload["input_description"], series_length=8)
        self.assertEqual(series.shape[0], 8)

    def test_execute_plan_allows_plateau_semantic_cue_metadata_without_overwrite_shape_enforcement(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "A late plateau.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=9.8, max_value=10.2),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_ramp",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "start_value": 0.0, "end_value": 1.0},
                    "semantic_role": "claimed plateau",
                    "semantic_cues": [{"kind": "plateau", "source_text": "plateau", "confidence": 0.9}],
                    "confidence": 0.9,
                    "source_text": "plateau",
                    "needs_library_extension": False,
                }
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_ramp"], series_length=8)
        series = execute_plan(plan, payload["input_description"], series_length=8)
        self.assertEqual(series.shape[0], 8)

    def test_execute_plan_allows_light_noise_without_budget_enforcement(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series stays near 10.0 with only light noise and peaks near 10.2.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=9.8, max_value=10.2),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 10.0},
                    "semantic_role": "baseline",
                    "semantic_cues": [{"kind": "plateau", "source_text": "stays near 10.0", "confidence": 0.9}],
                    "confidence": 1.0,
                    "source_text": "stays near 10.0",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_noise",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "noise_scale": 0.1, "random_seed": 7},
                    "semantic_role": "light noise",
                    "semantic_cues": [{"kind": "light_noise", "source_text": "only light noise", "confidence": 0.9}],
                    "confidence": 0.9,
                    "source_text": "only light noise",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_flat", "add_noise"], series_length=8)
        series = execute_plan(plan, payload["input_description"], series_length=8)
        self.assertEqual(series.shape[0], 8)

    def test_execute_plan_allows_light_noise_within_budget(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series stays near 10.0 with only light noise and peaks near 10.2.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=9.8, max_value=10.2),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 10.0},
                    "semantic_role": "baseline",
                    "semantic_cues": [{"kind": "plateau", "source_text": "stays near 10.0", "confidence": 0.9}],
                    "confidence": 1.0,
                    "source_text": "stays near 10.0",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_noise",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "noise_scale": 0.01, "random_seed": 7},
                    "semantic_role": "light noise",
                    "semantic_cues": [{"kind": "light_noise", "source_text": "only light noise", "confidence": 0.9}],
                    "confidence": 0.9,
                    "source_text": "only light noise",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_flat", "add_noise"], series_length=8)
        series = execute_plan(plan, payload["input_description"], series_length=8)
        self.assertEqual(series.shape[0], 8)

    def test_execute_plan_allows_light_volatility_without_budget_enforcement(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series stays near 19.0 with only small fluctuations and peaks near 19.4.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=19.0, max_value=21.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 19.0},
                    "semantic_role": "baseline",
                    "confidence": 1.0,
                    "source_text": "stays near 19.0",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "overwrite",
                    "primitive": "add_volatility",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "volatility_scale": 1.0},
                    "semantic_role": "small fluctuations",
                    "semantic_cues": [{"kind": "light_noise", "source_text": "small fluctuations", "confidence": 0.9}],
                    "confidence": 0.9,
                    "source_text": "small fluctuations",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_flat", "add_volatility"], series_length=8)
        series = execute_plan(plan, payload["input_description"], series_length=8)
        self.assertEqual(series.shape[0], 8)

    def test_execute_plan_allows_oscillation_without_amplitude_budget_enforcement(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series oscillates mildly around 61.0 and peaks near 61.4.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=60.8, max_value=61.4),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 61.0},
                    "semantic_role": "baseline",
                    "confidence": 1.0,
                    "source_text": "around 61.0",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_seasonality",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "amplitude": 0.5, "period": 4.0, "phase": 0.0, "baseline": 0.0, "waveform": "sine"},
                    "semantic_role": "mild oscillation",
                    "semantic_cues": [{"kind": "oscillation", "source_text": "oscillates mildly", "confidence": 0.9}, {"kind": "light_noise", "source_text": "oscillates mildly", "confidence": 0.9}],
                    "confidence": 0.9,
                    "source_text": "oscillates mildly",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_flat", "add_seasonality"], series_length=8)
        series = execute_plan(plan, payload["input_description"], series_length=8)
        self.assertEqual(series.shape[0], 8)

    def test_execute_plan_allows_directional_change_without_budget_enforcement(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "The series rises from 19.0 to about 21.0 with a sustained climb.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=19.0, max_value=21.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 19.0},
                    "semantic_role": "baseline",
                    "confidence": 1.0,
                    "source_text": "19.0",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_trend",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "slope": 0.5},
                    "semantic_role": "sustained climb",
                    "semantic_cues": [{"kind": "increase", "source_text": "sustained climb", "confidence": 0.9}],
                    "confidence": 0.9,
                    "source_text": "sustained climb",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_flat", "add_trend"], series_length=8)
        series = execute_plan(plan, payload["input_description"], series_length=8)
        self.assertEqual(series.shape[0], 8)

    def test_protected_constant_constraint_can_fail_on_later_step(self) -> None:
        payload = {
            "version": "1.0",
            "input_description": "Flat baseline then a conflicting noisy window.",
            "registry_validation": {
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            "global_context": _global_context(8, min_value=1.0, max_value=3.0),
            "steps": [
                {
                    "kind": "primitive",
                    "id": "step_1",
                    "effect_type": "overwrite",
                    "primitive": "add_flat",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "target_value": 2.0},
                    "semantic_role": "protected flat base",
                    "semantic_layer": "segment_structure",
                    "priority": 0,
                    "protected_constraints": [{"kind": "constant_segment", "start": 0, "end": 7, "value": 2.0, "tolerance": 1e-6}],
                    "confidence": 1.0,
                    "source_text": "flat baseline",
                    "needs_library_extension": False,
                },
                {
                    "kind": "primitive",
                    "id": "step_2",
                    "effect_type": "additive",
                    "primitive": "add_noise",
                    "parameters": {"start_timestep": 0, "end_timestep": 7, "noise_scale": 0.1, "random_seed": 7},
                    "semantic_role": "conflicting noise",
                    "semantic_layer": "texture",
                    "priority": 1,
                    "protected_constraints": [],
                    "confidence": 0.8,
                    "source_text": "conflicting noise",
                    "needs_library_extension": False,
                },
            ],
            "unresolved_semantics": [],
            "needs_library_extension": False,
        }
        plan = parse_plan(payload, ["add_flat", "add_noise"], series_length=8)
        with self.assertRaisesRegex(RuntimeError, "Protected constant segment violated"):
            execute_plan(plan, payload["input_description"], series_length=8)

    def test_evaluate_generated_code_flags_zero_series_output(self) -> None:
        code = """
import numpy as np
from atomic_timeseries import add_flat

series = np.zeros(8, dtype=np.float64)
"""
        evaluation = evaluate_generated_code(code, ["add_flat"], "The series remains stable at 7.5.", numeric_range=(7.0, 8.0))
        self.assertTrue(evaluation.contract_valid)
        self.assertTrue(evaluation.executed)
        self.assertIn("zero_series_output", evaluation.failure_categories)

    def test_build_eval_report_is_schema_valid(self) -> None:
        evaluation = evaluate_generated_code(
            "import numpy as np\nfrom atomic_timeseries import add_flat\nseries = np.zeros(8, dtype=np.float64)\nseries = add_flat(series, start_timestep=0, end_timestep=7, target_value=2.5)\n",
            ["add_flat"],
            "The series remains stable at 2.5.",
            numeric_range=(2.0, 3.0),
        )
        report = build_eval_report(
            caption="The series remains stable at 2.5.",
            registry_validation={
                "status": "valid",
                "schema_path": "schemas/primitive.schema.json",
                "target_path": "atomic_timeseries/registry.json",
                "errors": [],
            },
            plan_validation_status={
                "status": "valid",
                "schema_path": "schemas/plan.schema.json",
                "target_path": "<generated_plan>",
                "errors": [],
            },
            code_generation_status={
                "status": "success",
                "draft_generated": True,
                "final_generated": True,
                "artifacts": [],
                "errors": [],
            },
            execution_status={
                "status": "success",
                "signal_summary": summarize_series(evaluation.series),
                "errors": [],
            },
            evaluation=evaluation,
            unresolved_semantics=[],
            failure_reasons=[],
        )
        schema_check = validate_payload("eval_report", report)
        self.assertEqual(schema_check["status"], "valid")


if __name__ == "__main__":
    unittest.main()
