from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import atomic_timeseries as ats
from atomic_timeseries.metadata import PRIMITIVE_MACHINE_METADATA
from faithts_pipeline.hybrid_pipeline import (
    PlanExecutionError,
    PlanValidationError,
    compile_plan_to_code,
    execute_plan,
    extract_json_payload,
    parse_plan,
)
from schema_validator import validate_instance, validate_payload
from faithts.repair import repair_plan

# --- Path configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RETRIEVAL_REPORT = REPO_ROOT / "retrieval_coverage_report.jsonl"
DEFAULT_PRIMITIVE_CATALOG = REPO_ROOT / "primitive_catalog.json"
DEFAULT_PROMPT_TEMPLATE = Path(__file__).resolve().with_name("prompt_template.md")
DEFAULT_PARAMETER_PROMPT_TEMPLATE = Path(__file__).resolve().with_name("parameter_inference_template.md")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("workflow_results.jsonl")
DEFAULT_EVAL_REPORT_PATH = Path(__file__).resolve().with_name("workflow_eval_reports.jsonl")
DEFAULT_SERIES_LENGTH = 128
DEFAULT_MAX_RETRIEVED_PRIMITIVES = 10

# --- Execution safety configuration ---
ALLOWED_NUMPY_CALLS = {
    "array",
    "arange",
    "linspace",
    "zeros",
}
ALLOWED_NUMPY_ATTRIBUTES = ALLOWED_NUMPY_CALLS | {"float64"}
SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
}
PRIMITIVE_SIGNATURES = {
    name: inspect.signature(getattr(ats, name))
    for name in ats.__all__
    if name.startswith("add_")
}
FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "getattr",
    "delattr",
    "vars",
}
DISALLOWED_AST_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


# --- Exceptions ---
class ExternalLLMConfigurationError(RuntimeError):
    """External LLM is not configured correctly."""


class ExternalLLMRequestError(RuntimeError):
    """External LLM request failed."""


class UnsafeGeneratedCodeError(RuntimeError):
    """Generated code did not pass validation."""

    def __init__(self, message: str, *, category: str = "contract_invalid"):
        super().__init__(message)
        self.category = category


class GeneratedCodeExecutionError(RuntimeError):
    """Generated code failed during execution."""

    def __init__(self, message: str, *, category: str = "execution_failed"):
        super().__init__(message)
        self.category = category


class PromptRenderError(RuntimeError):
    """Prompt rendering failed."""


# --- Restricted import hook ---
def _restricted_import(name, globals_=None, locals_=None, fromlist=(), level=0):
    del globals_, locals_
    if level != 0:
        raise UnsafeGeneratedCodeError(f"Relative imports are not allowed: {name}")
    if name not in {"numpy", "atomic_timeseries"}:
        raise UnsafeGeneratedCodeError(f"Only numpy imports are allowed, got: {name}")
    return __import__(name, {}, {}, fromlist, level)


SAFE_BUILTINS["__import__"] = _restricted_import


# --- Data models ---
@dataclass(frozen=True)
class RetrievalExample:
    index: int
    caption: str
    candidate_primitives: list[str]
    retrieval_status: str
    orchestration_level_descriptors: list[str]
    raw_record: dict[str, Any]


@dataclass(frozen=True)
class PrimitiveSpec:
    function_name: str
    signature: str
    short_description: str
    required_parameters: list[str]
    optional_parameters: list[str]
    exact_effect_on_series: str
    effect_type: str
    mutability: str
    required_arg_patterns: list[list[str]]
    mutually_exclusive_args: list[list[str]]
    recommended_usage: list[str]
    forbidden_usage_patterns: list[str]
    typical_caption_cues: list[str]
    core_semantic_tags: list[str]
    category: str


@dataclass(frozen=True)
class RegistryPrimitive:
    name: str
    args: dict[str, str]
    required_args: list[str]
    defaults: dict[str, Any]
    constraints: dict[str, str]
    description: str


@dataclass(frozen=True)
class LLMGeneration:
    provider: str
    model: str
    raw_text: str
    response_id: str | None = None
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkflowResult:
    example_index: int
    caption: str
    candidate_primitives: list[str]
    retrieved_primitives: list[str]
    retrieval_status: str
    orchestration_level_descriptors: list[str]
    llm_provider: str
    llm_model: str
    llm_response_id: str | None
    prompt: str
    structured_plan: dict[str, Any]
    generated_code: str
    contract_valid: bool
    executed: bool
    numerically_plausible: bool
    semantically_plausible: bool
    usable_success: bool
    failure_categories: list[str]
    series_summary: dict[str, Any]
    eval_report: dict[str, Any]


@dataclass(frozen=True)
class CodeEvaluation:
    contract_valid: bool
    executed: bool
    numerically_plausible: bool
    semantically_plausible: bool
    usable_success: bool
    failure_categories: list[str]
    series: np.ndarray | None = None


def _failure_reason(stage: str, code: str, message: str, related_step_ids: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "code": code,
        "message": message,
    }
    if related_step_ids:
        payload["related_step_ids"] = related_step_ids
    return payload


def _alignment_scores(evaluation: CodeEvaluation) -> dict[str, Any]:
    if evaluation.usable_success:
        score = 1.0
    elif evaluation.executed and evaluation.contract_valid:
        score = 0.5
    elif evaluation.contract_valid:
        score = 0.25
    else:
        score = 0.0
    return {
        "overall": score,
        "description_to_plan": score,
        "plan_to_code": score,
        "code_to_signal": score if evaluation.executed else 0.0,
    }


def build_eval_report(
    *,
    caption: str,
    registry_validation: dict[str, Any],
    plan_validation_status: dict[str, Any],
    code_generation_status: dict[str, Any],
    execution_status: dict[str, Any],
    evaluation: CodeEvaluation,
    unresolved_semantics: list[dict[str, Any]],
    failure_reasons: list[dict[str, Any]],
    notes: str = "",
) -> dict[str, Any]:
    normalized_execution_status = dict(execution_status)
    signal_summary = normalized_execution_status.get("signal_summary")
    if signal_summary is not None:
        normalized_execution_status["signal_summary"] = {
            key: signal_summary[key]
            for key in ["length", "min", "max", "mean", "std"]
            if key in signal_summary
        }
    report = {
        "version": "1.0",
        "input_description": caption,
        "registry_validation": {
            "status": registry_validation["status"],
            "schema_path": registry_validation["schema_path"],
            "target_path": registry_validation["target_path"],
            "errors": list(registry_validation["errors"]),
        },
        "plan_validation_status": {
            "status": plan_validation_status["status"],
            "schema_path": plan_validation_status["schema_path"],
            "target_path": plan_validation_status["target_path"],
            "errors": list(plan_validation_status["errors"]),
        },
        "code_generation_status": code_generation_status,
        "execution_status": normalized_execution_status,
        "semantic_alignment_score": _alignment_scores(evaluation),
        "unresolved_semantics": unresolved_semantics,
        "failure_reasons": failure_reasons,
        "notes": notes,
    }
    schema_check = validate_payload("eval_report", report, target_path="<generated_eval_report>")
    if schema_check["status"] != "valid":
        joined = "; ".join(schema_check["errors"])
        raise RuntimeError(f"Generated eval report failed schema validation: {joined}")
    return report


# --- Gemini configuration ---
@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str
    temperature: float = 0.1
    max_output_tokens: int = 2048

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("PROTOTYPE_CODEGEN_API_KEY")
        )
        model = os.getenv("PROTOTYPE_CODEGEN_MODEL", "gemini-2.0-flash")
        temperature = float(os.getenv("PROTOTYPE_CODEGEN_TEMPERATURE", "0.1"))
        max_output_tokens = int(os.getenv("PROTOTYPE_CODEGEN_MAX_OUTPUT_TOKENS", "4096"))
        if not api_key:
            raise ExternalLLMConfigurationError(
                "Missing Gemini API key. Set GOOGLE_API_KEY, GEMINI_API_KEY, or PROTOTYPE_CODEGEN_API_KEY."
            )
        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )


# Backward-compatible aliases for existing imports/callers.
OpenAICompatibleConfig = GeminiConfig


class LLMClient(Protocol):
    def generate_code(self, prompt: str, example: RetrievalExample) -> LLMGeneration: ...


def _maybe_import_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ExternalLLMConfigurationError(
            "google-genai is not installed. Install the Gemini SDK to use this workflow."
        ) from exc
    return genai, types


class GeminiLLMClient:
    def __init__(self, config: GeminiConfig):
        self.config = config
        self._types = None
        self.client = None
        self._use_rest = False

    def _ensure_client(self) -> None:
        if self._use_rest or (self.client is not None and self._types is not None):
            return
        try:
            genai, types = _maybe_import_genai()
            self._types = types
            self.client = genai.Client(api_key=self.config.api_key)
        except ExternalLLMConfigurationError:
            self._use_rest = True
        except Exception as exc:
            raise ExternalLLMConfigurationError(f"Failed to initialize Gemini client: {exc}") from exc

    def _generate_content_rest(self, prompt: str) -> LLMGeneration:
        endpoint_model = urllib.parse.quote(self.config.model, safe="")
        query = urllib.parse.urlencode({"key": self.config.api_key})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{endpoint_model}:generateContent?{query}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    raw_response = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = ExternalLLMRequestError(f"Gemini REST request failed: HTTP {exc.code}: {body}")
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise last_error from exc
            except Exception as exc:
                last_error = ExternalLLMRequestError(f"Gemini REST request failed: {exc}")
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
        else:
            assert last_error is not None
            raise last_error

        candidates = raw_response.get("candidates") or []
        if not candidates:
            raise ExternalLLMRequestError(f"Gemini returned no candidates: {raw_response}")
        first = candidates[0]
        parts = first.get("content", {}).get("parts") or []
        raw_text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
        if not raw_text:
            raise ExternalLLMRequestError("Gemini returned an empty response.")
        return LLMGeneration(
            provider="google-gemini-rest",
            model=self.config.model,
            raw_text=raw_text,
            response_id=raw_response.get("responseId"),
            finish_reason=str(first.get("finishReason") or "STOP"),
            raw_response=raw_response,
        )

    def generate_code(self, prompt: str, example: RetrievalExample) -> LLMGeneration:
        del example
        self._ensure_client()
        if self._use_rest:
            return self._generate_content_rest(prompt)
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                ),
            )
        except Exception as exc:
            raise ExternalLLMRequestError(f"Gemini request failed: {exc}") from exc

        raw_text = getattr(response, "text", None)
        if not raw_text:
            raise ExternalLLMRequestError("Gemini returned an empty response.")

        response_id = getattr(response, "response_id", None) or getattr(response, "name", None)
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", None) or "")

        raw_response: dict[str, Any] | None = None
        to_json = getattr(response, "model_dump", None)
        if callable(to_json):
            raw_response = to_json(mode="json")

        return LLMGeneration(
            provider="google-gemini",
            model=self.config.model,
            raw_text=raw_text,
            response_id=response_id,
            finish_reason=finish_reason or "STOP",
            raw_response=raw_response,
        )


# Backward-compatible alias for existing imports/callers.
OpenAICompatibleLLMClient = GeminiLLMClient


# --- Loading and prompt building ---
def _retrieval_example_from_record(record: dict[str, Any], index: int | None = None) -> RetrievalExample:
    record_index = record.get("index", index)
    if record_index is None:
        raise ValueError("Retrieval record is missing an index.")
    return RetrievalExample(
        index=int(record_index),
        caption=record["caption"],
        candidate_primitives=list(record["candidate_primitives"]),
        retrieval_status=record["retrieval_status"],
        orchestration_level_descriptors=list(record.get("orchestration_level_descriptors", [])),
        raw_record=record,
    )


def _load_retrieval_examples_from_captions() -> list[RetrievalExample]:
    import coverage_audit

    catalog = coverage_audit.build_catalog()
    examples: list[RetrievalExample] = []
    for index, caption in coverage_audit.parse_captions():
        record = coverage_audit.evaluate_caption(caption, catalog)
        record["index"] = index
        examples.append(_retrieval_example_from_record(record))
    return examples


def load_retrieval_examples(path: Path = DEFAULT_RETRIEVAL_REPORT) -> list[RetrievalExample]:
    if not path.exists():
        if path == DEFAULT_RETRIEVAL_REPORT:
            return _load_retrieval_examples_from_captions()
        raise FileNotFoundError(f"Retrieval report not found: {path}")

    examples: list[RetrievalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(_retrieval_example_from_record(json.loads(line)))
    return examples


def load_primitive_specs(path: Path = DEFAULT_PRIMITIVE_CATALOG) -> dict[str, PrimitiveSpec]:
    raw_specs = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["function_name"]: PrimitiveSpec(
            function_name=item["function_name"],
            signature=item["signature"],
            category=item["category"],
            short_description=item["short_description"],
            required_parameters=[entry["name"] for entry in item["required_parameters"]],
            optional_parameters=[entry["name"] for entry in item["optional_parameters"]],
            exact_effect_on_series=item["exact_effect_on_series"],
            effect_type=item["effect_type"],
            mutability=item["mutability"],
            required_arg_patterns=[list(group) for group in item["required_arg_patterns"]],
            mutually_exclusive_args=[list(group) for group in item["mutually_exclusive_args"]],
            recommended_usage=list(item["recommended_usage"]),
            forbidden_usage_patterns=list(item["forbidden_usage_patterns"]),
            typical_caption_cues=list(item["typical_caption_cues"]),
            core_semantic_tags=list(item["core_semantic_tags"]),
        )
        for item in raw_specs
    }


def load_registry_primitives(path: Path | None = None) -> dict[str, RegistryPrimitive]:
    registry_path = path or (REPO_ROOT / "atomic_timeseries" / "registry.json")
    raw_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return {
        item["name"]: RegistryPrimitive(
            name=item["name"],
            args=dict(item.get("args", {})),
            required_args=list(item.get("required_args", [])),
            defaults=dict(item.get("defaults", {})),
            constraints=dict(item.get("constraints", {})),
            description=item.get("description", ""),
        )
        for item in raw_registry
    }


def retrieve_primitives_for_caption(
    caption: str,
    specs: dict[str, PrimitiveSpec],
    retrieval_hints: list[str] | None = None,
    max_candidates: int = DEFAULT_MAX_RETRIEVED_PRIMITIVES,
) -> list[PrimitiveSpec]:
    text = caption.lower()
    hint_set = set(retrieval_hints or [])
    scored: list[tuple[float, PrimitiveSpec]] = []

    event_terms = ("peak", "trough", "dip", "spike", "outlier")
    texture_terms = ("noise", "fluctuation", "fluctuations", "volatile", "volatility", "oscillation", "oscillatory", "seasonal", "repeating")
    structural_terms = ("change", "shift", "transition", "regime", "break", "plateau", "flattening", "levels off")
    trend_terms = ("trend", "decline", "decrease", "increase", "climb", "rise", "fall", "plateau", "flat", "stable")

    def bucket_for_spec(spec: PrimitiveSpec) -> str:
        if spec.category == "local_event":
            return "local_event"
        if spec.category in {"stochastic", "seasonal"}:
            return "texture"
        if spec.category == "structural":
            return "structural"
        return "trend"

    for spec in specs.values():
        score = 0.0
        if spec.function_name in hint_set:
            score += 4.0
        for cue in spec.typical_caption_cues:
            if cue.lower() in text:
                score += 2.0
        for tag in spec.core_semantic_tags:
            normalized = tag.replace("_", " ")
            if normalized in text:
                score += 0.5
        if spec.effect_type == "overwrite" and any(word in text for word in ["stable", "flat", "constant", "level", "plateau"]):
            score += 0.25
        if spec.effect_type == "additive" and any(word in text for word in ["peak", "dip", "spike", "fluctuation", "noise", "shift", "trend"]):
            score += 0.25
        if spec.function_name == "add_plateau" and any(word in text for word in ["plateau", "levels off", "final plateau", "settles into a plateau"]):
            score += 2.0
        if spec.category == "stochastic" and any(word in text for word in texture_terms):
            score += 1.25
        if spec.category == "seasonal" and any(word in text for word in ("oscillation", "oscillatory", "seasonal", "repeating", "periodic", "cyclic")):
            score += 1.5
        if spec.category == "structural" and any(word in text for word in structural_terms):
            score += 0.75
        if spec.category == "trend" and any(word in text for word in trend_terms):
            score += 0.5
        if score > 0:
            scored.append((score, spec))

    scored.sort(key=lambda item: (-item[0], item[1].function_name))
    scored_specs = [spec for _, spec in scored]
    effective_max_candidates = max(max_candidates, len([name for name in retrieval_hints or [] if name in specs]))

    selected: list[PrimitiveSpec] = []
    selected_names: set[str] = set()

    def extend_with_specs(candidates: list[PrimitiveSpec], limit: int | None = None) -> None:
        added = 0
        for spec in candidates:
            if spec.function_name in selected_names:
                continue
            if len(selected) >= effective_max_candidates:
                return
            selected.append(spec)
            selected_names.add(spec.function_name)
            added += 1
            if limit is not None and added >= limit:
                return

    hinted_specs = [specs[name] for name in retrieval_hints or [] if name in specs]
    hinted_specs.sort(key=lambda spec: next(((-score, item.function_name) for score, item in scored if item.function_name == spec.function_name), (0.0, spec.function_name)))
    extend_with_specs(hinted_specs)

    bucket_targets: list[tuple[str, int]] = [("trend", 2)]
    if any(term in text for term in structural_terms):
        bucket_targets.append(("structural", 1))
    if any(term in text for term in event_terms):
        event_budget = 2 if sum(term in text for term in event_terms) > 1 else 1
        bucket_targets.append(("local_event", event_budget))
    if any(term in text for term in texture_terms):
        bucket_targets.append(("texture", 1))

    for bucket, target_count in bucket_targets:
        bucket_specs = [spec for spec in scored_specs if bucket_for_spec(spec) == bucket]
        current_count = sum(1 for spec in selected if bucket_for_spec(spec) == bucket)
        if current_count < target_count:
            extend_with_specs(bucket_specs, limit=target_count - current_count)

    extend_with_specs(scored_specs)

    if not selected and hint_set:
        selected = [specs[name] for name in retrieval_hints or [] if name in specs][:max_candidates]
    if not selected:
        fallback_names = [
            "add_flat",
            "add_plateau",
            "add_ramp",
            "add_trend",
            "add_peak",
            "add_trough",
            "add_change_point",
            "add_noise",
        ]
        selected = [specs[name] for name in fallback_names if name in specs][:max_candidates]
    return selected


def select_examples(examples: list[RetrievalExample], indices: list[int] | None) -> list[RetrievalExample]:
    if not indices:
        return examples
    wanted = set(indices)
    selected = [example for example in examples if example.index in wanted]
    missing = sorted(wanted - {example.index for example in selected})
    if missing:
        raise ValueError(f"Requested example indices were not found: {missing}")
    return selected


def build_prompt(
    example: RetrievalExample,
    retrieved_specs: list[PrimitiveSpec],
    template_path: Path = DEFAULT_PROMPT_TEMPLATE,
    series_length: int = DEFAULT_SERIES_LENGTH,
    numeric_range: tuple[float, float] | None = None,
) -> str:
    try:
        template = template_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptRenderError(f"Failed to read prompt template: {exc}") from exc

    spec_blocks: list[str] = []
    if not retrieved_specs:
        raise PromptRenderError("No retrieved primitive specs were provided to the planner.")

    for spec in retrieved_specs:
        spec_blocks.append(
            "\n".join(
                [
                    f"- {spec.function_name}",
                    f"  signature: {spec.signature}",
                    f"  category: {spec.category}",
                    f"  description: {spec.short_description}",
                    f"  effect_type: {spec.effect_type}",
                    f"  mutability: {spec.mutability}",
                    f"  effect: {spec.exact_effect_on_series}",
                    f"  required_arg_patterns: {spec.required_arg_patterns}",
                    f"  mutually_exclusive_args: {spec.mutually_exclusive_args}",
                ]
            )
        )

    try:
        return template.format(
            series_length=series_length,
            caption=example.caption,
            candidate_primitives=", ".join(spec.function_name for spec in retrieved_specs),
            primitive_specs="\n\n".join(spec_blocks),
        )
    except KeyError as exc:
        raise PromptRenderError(f"Prompt template placeholder is missing: {exc}") from exc


def build_parameter_inference_prompt(
    *,
    example: RetrievalExample,
    selection_plan: dict[str, Any],
    registry_primitives: dict[str, RegistryPrimitive],
    template_path: Path = DEFAULT_PARAMETER_PROMPT_TEMPLATE,
    series_length: int = DEFAULT_SERIES_LENGTH,
    numeric_range: tuple[float, float] | None = None,
) -> str:
    try:
        template = template_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptRenderError(f"Failed to read parameter inference template: {exc}") from exc

    step_blocks: list[str] = []
    for step in selection_plan.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            continue
        primitive_name = str(step["primitive"])
        registry_primitive = registry_primitives.get(
            primitive_name,
            RegistryPrimitive(
                name=primitive_name,
                args={},
                required_args=[],
                defaults={},
                constraints={},
                description="",
            ),
        )
        machine_metadata = PRIMITIVE_MACHINE_METADATA[primitive_name]
        step_blocks.append(
            json.dumps(
                {
                    "id": step["id"],
                    "primitive": primitive_name,
                    "current_parameters": step.get("parameters", {}),
                    "effect_type": step["effect_type"],
                    "semantic_role": step["semantic_role"],
                    "source_text": step["source_text"],
                    "semantic_layer": step.get("semantic_layer"),
                    "priority": step.get("priority"),
                    "semantic_cues": step.get("semantic_cues", []),
                    "inferred_constraints": step.get("inferred_constraints", []),
                    "args": registry_primitive.args,
                    "required_args": registry_primitive.required_args,
                    "defaults": registry_primitive.defaults,
                    "constraints": registry_primitive.constraints,
                    "required_arg_patterns": machine_metadata["required_arg_patterns"],
                    "mutually_exclusive_args": machine_metadata["mutually_exclusive_args"],
                },
                ensure_ascii=True,
                indent=2,
            )
        )

    try:
        return template.format(
            series_length=series_length,
            caption=example.caption,
            selection_plan=json.dumps(selection_plan, ensure_ascii=True, indent=2),
            primitive_step_specs="\n\n".join(step_blocks),
        )
    except KeyError as exc:
        raise PromptRenderError(f"Parameter inference template placeholder is missing: {exc}") from exc


def extract_parameter_payload(text: str) -> dict[str, dict[str, Any]]:
    payload = extract_json_payload(text)
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise PlanValidationError("Parameter inference output must contain a `steps` array.", category="plan_invalid")

    parameters_by_step: dict[str, dict[str, Any]] = {}
    for item in steps:
        if not isinstance(item, dict):
            raise PlanValidationError("Parameter inference steps must be JSON objects.", category="plan_invalid")
        step_id = item.get("id")
        primitive = item.get("primitive")
        parameters = item.get("parameters")
        if not isinstance(step_id, str) or not step_id:
            raise PlanValidationError("Parameter inference step is missing a valid `id`.", category="plan_invalid")
        if not isinstance(primitive, str) or not primitive:
            raise PlanValidationError(
                f"Parameter inference step `{step_id}` is missing a valid `primitive`.",
                category="plan_invalid",
            )
        if not isinstance(parameters, dict):
            raise PlanValidationError(
                f"Parameter inference step `{step_id}` must include an object `parameters` field.",
                category="plan_invalid",
            )
        parameters_by_step[step_id] = {"primitive": primitive, "parameters": parameters}
    return parameters_by_step


def merge_inferred_parameters(
    selection_plan: dict[str, Any],
    inferred_parameters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged_plan = dict(selection_plan)
    merged_steps: list[dict[str, Any]] = []

    for step in selection_plan.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            merged_steps.append(step)
            continue
        step_id = str(step["id"])
        if step_id not in inferred_parameters:
            raise PlanValidationError(
                f"Parameter inference did not return parameters for step `{step_id}`.",
                category="plan_invalid",
            )
        inferred = inferred_parameters[step_id]
        if inferred["primitive"] != step["primitive"]:
            raise PlanValidationError(
                f"Parameter inference changed primitive for step `{step_id}` from `{step['primitive']}` to `{inferred['primitive']}`.",
                category="plan_invalid",
            )
        merged_step = dict(step)
        merged_parameters = dict(step.get("parameters", {}))
        merged_parameters.update(dict(inferred["parameters"]))
        merged_parameters = {key: value for key, value in merged_parameters.items() if value is not None}
        merged_step["parameters"] = merged_parameters
        if step["primitive"] == "add_noise" and "random_seed" not in merged_step["parameters"]:
            seed = merged_plan.get("global_context", {}).get("seed")
            merged_step["parameters"]["random_seed"] = 7 if seed is None else int(seed)
        merged_steps.append(merged_step)

    merged_plan["steps"] = merged_steps
    return merged_plan


def normalize_plan_global_context(
    plan_payload: dict[str, Any],
    *,
    series_length: int,
    numeric_range: tuple[float, float],
) -> dict[str, Any]:
    normalized = dict(plan_payload)
    context = dict(normalized.get("global_context", {}))
    context["length"] = int(series_length)
    context["numeric_range"] = {
        "min": float(numeric_range[0]),
        "max": float(numeric_range[1]),
    }
    context.setdefault("dt", None)
    context.setdefault("seed", None)
    normalized["global_context"] = context
    return normalized


def normalize_step_parameters(plan_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(plan_payload)
    raw_steps = normalized.get("steps", [])
    if not isinstance(raw_steps, list):
        return normalized

    cleaned_steps: list[dict[str, Any] | Any] = []
    for step in raw_steps:
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            cleaned_steps.append(step)
            continue
        cleaned_step = dict(step)
        params = cleaned_step.get("parameters", {})
        if isinstance(params, dict):
            cleaned_step["parameters"] = {key: value for key, value in params.items() if value is not None}
        cleaned_steps.append(cleaned_step)
    normalized["steps"] = cleaned_steps
    return normalized


def extend_background_steps_for_additive_overlays(
    plan_payload: dict[str, Any],
    *,
    series_length: int,
) -> dict[str, Any]:
    normalized = dict(plan_payload)
    raw_steps = normalized.get("steps", [])
    if not isinstance(raw_steps, list):
        return normalized

    steps: list[dict[str, Any]] = []
    for step in raw_steps:
        steps.append(dict(step) if isinstance(step, dict) else step)

    overlay_markers = ("added on top", "superimposed", "around that level")
    background_primitives = {"add_flat", "add_trend", "add_ramp"}
    baseline_source_markers = (
        "baseline",
        "stable",
        "stays close",
        "remains fairly stable",
        "starts from",
        "begins near",
        "initial plateau",
        "plateau",
    )

    def _step_end(step: dict[str, Any]) -> int | None:
        params = step.get("parameters")
        if not isinstance(params, dict):
            return None
        value = params.get("end_timestep")
        return int(value) if isinstance(value, (int, float)) else None

    def _step_start(step: dict[str, Any]) -> int | None:
        params = step.get("parameters")
        if not isinstance(params, dict):
            return None
        for key in ("start_timestep", "anchor_timestep", "timestep"):
            value = params.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return None

    def _looks_like_persistent_baseline(step: dict[str, Any]) -> bool:
        if step.get("kind") != "primitive":
            return False
        if step.get("primitive") not in {"add_flat", "add_ramp"}:
            return False
        if _step_start(step) != 0:
            return False
        source_text = str(step.get("source_text", "")).lower()
        semantic_role = str(step.get("semantic_role", "")).lower()
        combined = f"{source_text} {semantic_role}"
        return any(marker in combined for marker in baseline_source_markers)

    # If the description introduces an initial baseline-like overwrite step and later
    # steps only modify subwindows or add structure on top, keep that baseline active
    # through the horizon unless another overwrite explicitly replaces portions of it.
    for step in steps:
        if not isinstance(step, dict):
            continue
        if not _looks_like_persistent_baseline(step):
            continue
        params = step.get("parameters")
        if not isinstance(params, dict):
            continue
        end = _step_end(step)
        if end is None or end >= series_length - 1:
            continue
        if any(
            isinstance(later, dict)
            and later.get("kind") == "primitive"
            and _step_start(later) is not None
            and _step_start(later) > end
            for later in steps
        ):
            params["end_timestep"] = series_length - 1

    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            continue
        source_text = str(step.get("source_text", "")).lower()
        if not any(marker in source_text for marker in overlay_markers):
            continue
        if step.get("effect_type") != "additive":
            continue
        overlay_start = _step_start(step)
        if overlay_start is None:
            continue

        for prior_index in range(index - 1, -1, -1):
            prior = steps[prior_index]
            if not isinstance(prior, dict) or prior.get("kind") != "primitive":
                continue
            if prior.get("primitive") not in background_primitives:
                continue
            prior_params = prior.get("parameters")
            if not isinstance(prior_params, dict):
                continue
            prior_end = _step_end(prior)
            if prior_end is None or prior_end < overlay_start - 1:
                continue

            truncated_by_later_overwrite = False
            for later in steps[index + 1 :]:
                if not isinstance(later, dict) or later.get("kind") != "primitive":
                    continue
                if later.get("effect_type") != "overwrite":
                    continue
                later_start = _step_start(later)
                if later_start is not None and later_start > overlay_start:
                    truncated_by_later_overwrite = True
                    break

            if not truncated_by_later_overwrite:
                prior_params["end_timestep"] = series_length - 1
            break

    normalized["steps"] = steps
    return normalized


# --- Response and code extraction helpers ---
def extract_text_from_openai_compatible_response(payload: dict[str, Any]) -> tuple[str, str | None]:
    text_parts: list[str] = []
    finish_reason: str | None = None
    for item in payload.get("output", []):
        finish_reason = finish_reason or item.get("finish_reason")
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_parts.append(content["text"])
    return "\n".join(text_parts).strip(), finish_reason


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


# --- Code validation and execution ---
def _child_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _is_series_reference(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "series"
    if isinstance(node, ast.Subscript):
        return _is_series_reference(node.value)
    if isinstance(node, ast.Attribute):
        return _is_series_reference(node.value)
    return False


def _collect_call_arguments(call: ast.Call, primitive_name: str) -> dict[str, ast.AST]:
    params = list(PRIMITIVE_SIGNATURES[primitive_name].parameters.values())
    call_args: dict[str, ast.AST] = {}
    for param, value in zip(params, call.args):
        call_args[param.name] = value
    for keyword in call.keywords:
        if keyword.arg is None:
            raise UnsafeGeneratedCodeError(
                "Starred keyword expansion is not allowed in generated code.",
                category="contract_invalid",
            )
        call_args[keyword.arg] = keyword.value
    return call_args


def _validate_primitive_call(call: ast.Call, primitive_name: str) -> None:
    metadata = PRIMITIVE_MACHINE_METADATA[primitive_name]
    call_args = _collect_call_arguments(call, primitive_name)
    provided = set(call_args)

    for arg_group in metadata["mutually_exclusive_args"]:
        overlap = provided.intersection(arg_group)
        if len(overlap) > 1:
            joined = ", ".join(sorted(overlap))
            raise UnsafeGeneratedCodeError(
                f"{primitive_name} received mutually exclusive arguments: {joined}",
                category="invalid_primitive_args",
            )

    required_patterns = metadata["required_arg_patterns"]
    if required_patterns and not any(set(pattern).issubset(provided) for pattern in required_patterns):
        readable = " or ".join(" + ".join(pattern) for pattern in required_patterns)
        raise UnsafeGeneratedCodeError(
            f"{primitive_name} requires one of these argument patterns: {readable}",
            category="invalid_primitive_args",
        )

    if metadata["effect_type"] == "additive":
        for arg_name in metadata.get("setter_like_args", []):
            if arg_name in call_args and _is_series_reference(call_args[arg_name]):
                raise UnsafeGeneratedCodeError(
                    f"{primitive_name} uses `series[...]` as a setter-style value for `{arg_name}`.",
                    category="additive_as_setter",
                )


def _validate_ast(tree: ast.AST, allowed_primitives: set[str]) -> None:
    parents = _child_parent_map(tree)
    saw_numpy_import = False
    saw_primitive_import = False
    saw_primitive_call = False

    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_AST_NODES):
            raise UnsafeGeneratedCodeError(
                f"Unsupported Python construct in generated code: {type(node).__name__}",
                category="contract_invalid",
            )

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeGeneratedCodeError(
                f"Forbidden name used in generated code: {node.id}",
                category="contract_invalid",
            )

        if isinstance(node, ast.Import):
            if len(node.names) != 1:
                raise UnsafeGeneratedCodeError(
                    "Only `import numpy as np` is allowed.",
                    category="illegal_import",
                )
            alias = node.names[0]
            if alias.name != "numpy" or alias.asname != "np":
                raise UnsafeGeneratedCodeError(
                    "Only `import numpy as np` is allowed.",
                    category="illegal_import",
                )
            saw_numpy_import = True

        if isinstance(node, ast.ImportFrom):
            if node.module != "atomic_timeseries":
                raise UnsafeGeneratedCodeError(
                    f"Only `from atomic_timeseries import ...` is allowed, got: {node.module}",
                    category="illegal_import",
                )
            for alias in node.names:
                if alias.asname is not None:
                    raise UnsafeGeneratedCodeError(
                        "Aliasing primitives is not allowed.",
                        category="primitive_shadowing",
                    )
                if alias.name not in allowed_primitives:
                    raise UnsafeGeneratedCodeError(
                        f"Imported primitive is not in the candidate set: {alias.name}",
                        category="contract_invalid",
                    )
            saw_primitive_import = True

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in allowed_primitives:
                    raise UnsafeGeneratedCodeError(
                        f"Primitive name shadowing is not allowed: {target.id}",
                        category="primitive_shadowing",
                    )

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr not in ALLOWED_NUMPY_ATTRIBUTES:
                raise UnsafeGeneratedCodeError(
                    f"Disallowed numpy API usage: np.{node.attr}",
                    category="unsupported_numpy_api",
                )

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if not isinstance(func.value, ast.Name) or func.value.id != "np":
                    raise UnsafeGeneratedCodeError(
                        "Only whitelisted `np.*` calls are allowed.",
                        category="unsupported_numpy_api",
                    )
                if func.attr not in ALLOWED_NUMPY_CALLS:
                    raise UnsafeGeneratedCodeError(
                        f"Disallowed numpy call: {func.attr}",
                        category="unsupported_numpy_api",
                    )
                continue

            if isinstance(func, ast.Name):
                if func.id in SAFE_BUILTINS:
                    continue
                if func.id in allowed_primitives:
                    saw_primitive_call = True
                    parent = parents.get(node)
                    if not (
                        isinstance(parent, ast.Assign)
                        and len(parent.targets) == 1
                        and isinstance(parent.targets[0], ast.Name)
                        and parent.targets[0].id == "series"
                        and parent.value is node
                    ):
                        raise UnsafeGeneratedCodeError(
                            f"Primitive call `{func.id}(...)` must be assigned back to `series`.",
                            category="unassigned_primitive_call",
                        )
                    _validate_primitive_call(node, func.id)
                    continue
                raise UnsafeGeneratedCodeError(
                    f"Disallowed function call: {func.id}",
                    category="contract_invalid",
                )

    if not saw_numpy_import:
        raise UnsafeGeneratedCodeError(
            "Generated code must include `import numpy as np`.",
            category="illegal_import",
        )
    if saw_primitive_call and not saw_primitive_import:
        raise UnsafeGeneratedCodeError(
            "Generated code must import at least one primitive from `atomic_timeseries`.",
            category="illegal_import",
        )


def execute_generated_code(code: str, candidate_primitives: list[str]) -> np.ndarray:
    stripped = strip_code_fences(code)
    try:
        tree = ast.parse(stripped, mode="exec")
    except SyntaxError as exc:
        raise UnsafeGeneratedCodeError(
            f"Generated code is not valid Python: {exc}",
            category="contract_invalid",
        ) from exc

    allowed_primitives = set(candidate_primitives)
    _validate_ast(tree, allowed_primitives)

    exec_globals = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "numpy": np,
    }
    exec_locals: dict[str, Any] = {}

    try:
        exec(compile(tree, filename="<generated>", mode="exec"), exec_globals, exec_locals)
    except Exception as exc:
        message = str(exc)
        category = "execution_failed"
        if "Provide either" in message or "Provide slope" in message or "Provide end_value" in message:
            category = "invalid_primitive_args"
        raise GeneratedCodeExecutionError(
            f"Generated code failed during execution: {exc}",
            category=category,
        ) from exc

    series = exec_locals.get("series", exec_globals.get("series"))
    if series is None:
        raise GeneratedCodeExecutionError(
            "Generated code did not assign a `series` variable.",
            category="execution_failed",
        )
    if not isinstance(series, np.ndarray):
        raise GeneratedCodeExecutionError(
            "Generated `series` is not a numpy.ndarray.",
            category="execution_failed",
        )
    if series.ndim != 1:
        raise GeneratedCodeExecutionError(
            "Generated `series` must be one-dimensional.",
            category="execution_failed",
        )
    if not np.issubdtype(series.dtype, np.number):
        raise GeneratedCodeExecutionError(
            "Generated `series` must be numeric.",
            category="execution_failed",
        )
    return series.astype(float, copy=False)


def summarize_series(series: np.ndarray) -> dict[str, Any]:
    if series.ndim != 1:
        raise ValueError("Series summary expects a one-dimensional array.")
    return {
        "length": int(series.shape[0]),
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "first_values": [float(value) for value in series[: min(8, len(series))]],
        "last_values": [float(value) for value in series[-min(8, len(series)) :]],
    }


def _extract_caption_decimal_values(caption: str) -> list[float]:
    return [float(match) for match in re.findall(r"\d+\.\d+", caption)]


def _extract_final_value_hint(caption: str) -> float | None:
    patterns = [
        r"(?:last value|ends at|ending at|end[s]?\s+(?:at|near)|final value(?: of)?|stabilizes at)\s+(\d+\.\d+)",
        r"(?:value\s+)(\d+\.\d+)\s*$",
    ]
    lowered = caption.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return float(match.group(1))
    return None


def evaluate_generated_code(
    code: str,
    candidate_primitives: list[str],
    caption: str,
    numeric_range: tuple[float, float] | None = None,
) -> CodeEvaluation:
    try:
        series = execute_generated_code(code, candidate_primitives)
    except UnsafeGeneratedCodeError as exc:
        return CodeEvaluation(
            contract_valid=False,
            executed=False,
            numerically_plausible=False,
            semantically_plausible=False,
            usable_success=False,
            failure_categories=[exc.category],
        )
    except GeneratedCodeExecutionError as exc:
        return CodeEvaluation(
            contract_valid=True,
            executed=False,
            numerically_plausible=False,
            semantically_plausible=False,
            usable_success=False,
            failure_categories=[exc.category],
        )

    failure_categories: list[str] = []
    summary = summarize_series(series)
    decimals = _extract_caption_decimal_values(caption)

    if decimals and not np.allclose(decimals, 0.0) and np.allclose(series, 0.0):
        failure_categories.append("zero_series_output")

    if np.nanmax(np.abs(series)) > 10_000:
        failure_categories.append("numeric_range_explosion")

    if "stable" in caption.lower() or "minimal fluctuation" in caption.lower() or "low volatility" in caption.lower():
        scale = max(abs(summary["mean"]), 1.0)
        if summary["std"] > max(0.25, 0.1 * scale):
            failure_categories.append("excessive_variance")

    final_value_hint = _extract_final_value_hint(caption)
    if final_value_hint is not None:
        tail_mean = float(np.mean(series[-8:]))
        if abs(tail_mean - final_value_hint) > max(0.25, abs(final_value_hint) * 0.05):
            failure_categories.append("endpoint_mismatch")

    numerically_plausible = not any(
        category in failure_categories
        for category in ["zero_series_output", "numeric_range_explosion", "excessive_variance"]
    )
    semantically_plausible = numerically_plausible and "endpoint_mismatch" not in failure_categories
    usable_success = numerically_plausible and semantically_plausible

    return CodeEvaluation(
        contract_valid=True,
        executed=True,
        numerically_plausible=numerically_plausible,
        semantically_plausible=semantically_plausible,
        usable_success=usable_success,
        failure_categories=failure_categories,
        series=series,
    )


# --- Workflow orchestration ---
def build_llm_client(provider: str = "gemini") -> LLMClient:
    normalized = provider.lower()
    if normalized in {"gemini", "google-gemini", "openai-compatible"}:
        return GeminiLLMClient(GeminiConfig.from_env())
    raise ValueError(f"Unsupported LLM provider: {provider}")


def run_workflow_for_example(
    example: RetrievalExample,
    specs: dict[str, PrimitiveSpec],
    llm_client: LLMClient,
    template_path: Path = DEFAULT_PROMPT_TEMPLATE,
    series_length: int = DEFAULT_SERIES_LENGTH,
    numeric_range: tuple[float, float] | None = None,
    max_candidates: int = DEFAULT_MAX_RETRIEVED_PRIMITIVES,
) -> WorkflowResult:
    registry_validation = validate_instance("primitive", REPO_ROOT / "atomic_timeseries" / "registry.json")
    if registry_validation["status"] != "valid":
        raise UnsafeGeneratedCodeError(
            f"Primitive registry failed validation: {', '.join(registry_validation['errors'])}",
            category="contract_invalid",
        )
    registry_primitives = load_registry_primitives()
    retrieved_specs = retrieve_primitives_for_caption(
        example.caption,
        specs,
        retrieval_hints=example.candidate_primitives,
        max_candidates=max_candidates,
    )
    retrieved_names = [spec.function_name for spec in retrieved_specs]
    prompt = build_prompt(
        example=example,
        retrieved_specs=retrieved_specs,
        template_path=template_path,
        series_length=series_length,
        numeric_range=numeric_range,
    )
    selection_generation = llm_client.generate_code(prompt, example)
    try:
        resolved_numeric_range = numeric_range
        if resolved_numeric_range is None:
            raw_numeric_range = example.raw_record.get("numeric_range") if isinstance(example.raw_record, dict) else None
            if isinstance(raw_numeric_range, dict) and {"min", "max"}.issubset(raw_numeric_range):
                resolved_numeric_range = (float(raw_numeric_range["min"]), float(raw_numeric_range["max"]))
        selection_plan_payload = extract_json_payload(selection_generation.raw_text)
        if resolved_numeric_range is None:
            raise PromptRenderError("Workflow input must include an explicit numeric_range.")
        selection_plan_payload = normalize_step_parameters(selection_plan_payload)
        selection_plan_payload = normalize_plan_global_context(
            selection_plan_payload,
            series_length=series_length,
            numeric_range=resolved_numeric_range,
        )
        parameter_prompt = build_parameter_inference_prompt(
            example=example,
            selection_plan=selection_plan_payload,
            registry_primitives=registry_primitives,
            series_length=series_length,
            numeric_range=resolved_numeric_range,
        )
        parameter_generation = llm_client.generate_code(parameter_prompt, example)
        inferred_parameters = extract_parameter_payload(parameter_generation.raw_text)
        plan_payload = merge_inferred_parameters(selection_plan_payload, inferred_parameters)
        plan_payload = extend_background_steps_for_additive_overlays(
            plan_payload,
            series_length=series_length,
        )
        plan_payload = normalize_step_parameters(plan_payload)
        plan_payload = normalize_plan_global_context(
            plan_payload,
            series_length=series_length,
            numeric_range=resolved_numeric_range,
        )
        plan_payload = repair_plan(plan_payload, series_length=series_length).plan
        plan_validation_status = validate_payload("plan", plan_payload, target_path="<generated_plan>")
        plan = parse_plan(plan_payload, retrieved_names, series_length, numeric_range=resolved_numeric_range)
    except PlanValidationError as exc:
        raise UnsafeGeneratedCodeError(str(exc), category=exc.category) from exc
    if plan_validation_status["status"] != "valid":
        raise UnsafeGeneratedCodeError(
            f"Generated plan failed schema validation: {', '.join(plan_validation_status['errors'])}",
            category="plan_invalid",
        )
    generated_code = compile_plan_to_code(plan, series_length)
    try:
        executed_series = execute_plan(plan, example.caption, series_length)
    except PlanValidationError as exc:
        raise UnsafeGeneratedCodeError(str(exc), category=exc.category) from exc
    except PlanExecutionError as exc:
        raise GeneratedCodeExecutionError(str(exc), category=exc.category) from exc

    plan_primitive_names = [step.primitive for step in plan.primitive_steps]
    allowed_code_primitives = sorted(set(retrieved_names) | set(plan_primitive_names))
    evaluation = evaluate_generated_code(
        generated_code,
        allowed_code_primitives,
        example.caption,
        numeric_range=(
            float(plan.global_context["numeric_range"]["min"]),
            float(plan.global_context["numeric_range"]["max"]),
        ),
    )
    if evaluation.executed and evaluation.series is not None and not np.allclose(
        executed_series,
        evaluation.series,
        equal_nan=True,
    ):
        raise GeneratedCodeExecutionError(
            "Compiled code result diverged from direct plan execution.",
            category="execution_failed",
        )
    unresolved_semantics = [
        {
            "id": item["id"],
            "description": item["description"],
            "reason": item["reason"],
            "related_step_ids": item.get("related_step_ids", []),
        }
        for item in plan_payload.get("unresolved_semantics", [])
    ]
    failure_reasons = [
        _failure_reason("evaluation", category, category.replace("_", " "))
        for category in evaluation.failure_categories
    ]
    eval_report = build_eval_report(
        caption=example.caption,
        registry_validation=registry_validation,
        plan_validation_status=plan_validation_status,
        code_generation_status={
            "status": "success",
            "draft_generated": True,
            "final_generated": True,
            "artifacts": [],
            "errors": [],
        },
        execution_status={
            "status": "success",
            "signal_summary": summarize_series(executed_series),
            "errors": [],
        },
        evaluation=evaluation,
        unresolved_semantics=unresolved_semantics,
        failure_reasons=failure_reasons,
    )
    return WorkflowResult(
        example_index=example.index,
        caption=example.caption,
        candidate_primitives=list(example.candidate_primitives),
        retrieved_primitives=retrieved_names,
        retrieval_status=example.retrieval_status,
        orchestration_level_descriptors=list(example.orchestration_level_descriptors),
        llm_provider=selection_generation.provider,
        llm_model=selection_generation.model,
        llm_response_id=selection_generation.response_id,
        prompt=prompt,
        structured_plan=plan_payload,
        generated_code=generated_code,
        contract_valid=evaluation.contract_valid,
        executed=evaluation.executed,
        numerically_plausible=evaluation.numerically_plausible,
        semantically_plausible=evaluation.semantically_plausible,
        usable_success=evaluation.usable_success,
        failure_categories=list(evaluation.failure_categories),
        series_summary=summarize_series(executed_series),
        eval_report=eval_report,
    )


def _write_results(path: Path, results: list[WorkflowResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _write_eval_reports(path: Path, results: list[WorkflowResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.eval_report, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-retrieval code generation with Gemini.")
    parser.add_argument(
        "--provider",
        default="gemini",
        help="LLM provider name. `gemini` is the default. `openai-compatible` is accepted as an alias.",
    )
    parser.add_argument(
        "--retrieval-report",
        type=Path,
        default=DEFAULT_RETRIEVAL_REPORT,
        help="Path to retrieval_coverage_report.jsonl",
    )
    parser.add_argument(
        "--primitive-catalog",
        type=Path,
        default=DEFAULT_PRIMITIVE_CATALOG,
        help="Path to primitive_catalog.json",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Path to the prompt template markdown file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save workflow results as JSONL",
    )
    parser.add_argument(
        "--eval-report-output",
        type=Path,
        default=DEFAULT_EVAL_REPORT_PATH,
        help="Where to save evaluation reports as JSONL",
    )
    parser.add_argument(
        "--series-length",
        type=int,
        default=DEFAULT_SERIES_LENGTH,
        help="Target length of the generated series",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        help="Optional example indices to run. If omitted, run all examples.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_RETRIEVED_PRIMITIVES,
        help="Maximum number of retrieved primitives to expose to the planner before schema validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    examples = load_retrieval_examples(args.retrieval_report)
    specs = load_primitive_specs(args.primitive_catalog)
    selected_examples = select_examples(examples, args.indices)
    llm_client = build_llm_client(args.provider)

    results = [
        run_workflow_for_example(
            example=example,
            specs=specs,
            llm_client=llm_client,
            template_path=args.prompt_template,
            series_length=args.series_length,
            max_candidates=args.max_candidates,
        )
        for example in selected_examples
    ]
    _write_results(args.output, results)
    _write_eval_reports(args.eval_report_output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
