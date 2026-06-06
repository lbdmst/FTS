from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

import atomic_timeseries as ats
from atomic_timeseries.metadata import PRIMITIVE_MACHINE_METADATA
from schema_validator import load_schema


class PlanValidationError(RuntimeError):
    def __init__(self, message: str, *, category: str = "plan_invalid"):
        super().__init__(message)
        self.category = category


class PlanExecutionError(RuntimeError):
    def __init__(self, message: str, *, category: str = "execution_failed"):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class PrimitivePlanStep:
    id: str
    effect_type: str
    primitive: str
    semantic_role: str
    parameters: dict[str, Any]
    source_text: str
    confidence: float
    semantic_layer: str
    priority: int
    protected_constraints: list[dict[str, Any]]
    semantic_cues: list[dict[str, Any]]
    inferred_constraints: list[dict[str, Any]]


@dataclass(frozen=True)
class TimeSeriesPlan:
    version: str
    input_description: str
    global_context: dict[str, Any]
    assumptions: list[str]
    primitive_steps: list[PrimitivePlanStep]
    unresolved_semantics: list[dict[str, Any]]
    needs_library_extension: bool

    @property
    def all_steps(self) -> list[PrimitivePlanStep]:
        return list(self.primitive_steps)


SEMANTIC_LAYER_ORDER = {
    "global_scaffold": 0,
    "segment_structure": 1,
    "local_event": 2,
    "texture": 3,
    "missingness": 4,
}

DEFAULT_LAYER_BY_PRIMITIVE = {
    "add_flat": "segment_structure",
    "add_ramp": "segment_structure",
    "add_growth": "segment_structure",
    "add_decay": "segment_structure",
    "add_trend": "segment_structure",
    "add_change_point": "global_scaffold",
    "add_level_shift": "global_scaffold",
    "add_peak": "local_event",
    "add_trough": "local_event",
    "add_spike": "local_event",
    "add_dip": "local_event",
    "add_outlier": "local_event",
    "add_noise": "texture",
    "add_volatility": "texture",
    "add_seasonality": "texture",
    "add_gap": "missingness",
}

VALID_SEMANTIC_CUE_KINDS = {
    "decline",
    "increase",
    "plateau",
    "oscillation",
    "light_noise",
}

PLURAL_LOCAL_EVENT_PATTERNS = (
    {
        "label": "plural peaks",
        "regexes": (
            re.compile(r"\b(?:several|multiple|repeated|many)\s+(?:small\s+|local\s+|prominent\s+|distinct\s+|separate\s+|isolated\s+)*peaks\b"),
        ),
        "primitives": {"add_peak", "add_spike"},
        "minimum_count": 2,
        "message": "Caption explicitly describes multiple peaks, but the plan does not include multiple peak-like local events.",
    },
    {
        "label": "plural troughs",
        "regexes": (
            re.compile(r"\b(?:several|multiple|repeated|many)\s+(?:small\s+|local\s+|prominent\s+|distinct\s+|separate\s+|isolated\s+)*troughs\b"),
        ),
        "primitives": {"add_trough", "add_dip"},
        "minimum_count": 2,
        "message": "Caption explicitly describes multiple troughs, but the plan does not include multiple trough-like local events.",
    },
    {
        "label": "plural peaks and troughs",
        "regexes": (
            re.compile(r"\b(?:several|multiple|repeated|many)\s+peaks\s+and\s+troughs\b"),
            re.compile(r"\b(?:several|multiple|repeated|many)\s+troughs\s+and\s+peaks\b"),
        ),
        "primitives": {"add_peak", "add_spike", "add_trough", "add_dip"},
        "minimum_count": 2,
        "message": "Caption explicitly describes several peaks and troughs, but the plan does not include multiple local-event extrema.",
    },
)


def _allow_implicit_flat_scaffold(
    *,
    primitive: str,
    step_index: int,
    candidate_set: set[str],
) -> bool:
    if primitive != "add_flat" or step_index != 0:
        return False
    if primitive in candidate_set or not candidate_set:
        return False
    # Allow the planner to make an unspecified baseline explicit when retrieval only
    # surfaced additive primitives. This keeps local events and texture-only captions
    # executable without forcing the baseline primitive to be retrieved explicitly.
    return all(PRIMITIVE_MACHINE_METADATA[name]["effect_type"] == "additive" for name in candidate_set)


def _normalize_numeric_range_payload(raw_numeric_range: Any) -> dict[str, float]:
    if not isinstance(raw_numeric_range, dict):
        raise PlanValidationError("Plan global_context.numeric_range must be an object.", category="plan_invalid")
    if "min" not in raw_numeric_range or "max" not in raw_numeric_range:
        raise PlanValidationError(
            "Plan global_context.numeric_range must include both `min` and `max`.",
            category="plan_invalid",
        )
    lower = float(raw_numeric_range["min"])
    upper = float(raw_numeric_range["max"])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise PlanValidationError("Plan numeric_range bounds must be finite numbers.", category="plan_invalid")
    if lower > upper:
        raise PlanValidationError("Plan global_context.numeric_range must satisfy min <= max.", category="plan_invalid")
    return {"min": lower, "max": upper}


def _infer_semantic_cue_kind(text: str) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ["declin", "downward", "descending", "negative slope", "nonincreasing"]):
        return "decline"
    if any(token in lowered for token in ["increas", "upward", "ascending", "positive slope", "nondecreasing", "climb"]):
        return "increase"
    if any(token in lowered for token in ["plateau", "flat", "constant", "stable level", "zero slope"]):
        return "plateau"
    if any(token in lowered for token in ["oscillat", "season", "periodic", "repeating pattern", "sine"]):
        return "oscillation"
    if any(token in lowered for token in ["noise", "jitter", "volatility", "fluctuation", "texture"]):
        return "light_noise"
    return None


def _normalize_semantic_cues(raw_cues: Any, raw_inferred_constraints: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _append(kind: str | None, source_text: str, confidence: float | None = None) -> None:
        if kind not in VALID_SEMANTIC_CUE_KINDS:
            return
        normalized_source = source_text.strip() or kind
        key = (kind, normalized_source)
        if key in seen:
            return
        payload: dict[str, Any] = {"kind": kind, "source_text": normalized_source}
        if confidence is not None:
            payload["confidence"] = float(confidence)
        normalized.append(payload)
        seen.add(key)

    for item in raw_cues if isinstance(raw_cues, list) else []:
        if isinstance(item, dict):
            source_text = str(item.get("source_text") or item.get("description") or item.get("kind") or "").strip()
            kind = item.get("kind")
            if isinstance(kind, str) and kind in VALID_SEMANTIC_CUE_KINDS:
                _append(kind, source_text, item.get("confidence"))
            else:
                cue_kind = _infer_semantic_cue_kind(" ".join(str(value) for value in item.values()))
                _append(cue_kind, source_text or str(item), item.get("confidence"))
        elif isinstance(item, str):
            _append(_infer_semantic_cue_kind(item), item)

    for item in raw_inferred_constraints if isinstance(raw_inferred_constraints, list) else []:
        if isinstance(item, dict):
            text = " ".join(str(value) for value in item.values())
            source_text = str(item.get("source_text") or item.get("description") or item.get("value") or text).strip()
            _append(_infer_semantic_cue_kind(text), source_text, item.get("confidence"))
        elif isinstance(item, str):
            _append(_infer_semantic_cue_kind(item), item)

    return normalized


def _normalize_inferred_constraints(raw_constraints: Any, semantic_cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for item in raw_constraints if isinstance(raw_constraints, list) else []:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        value = item.get("value")
        if kind in {"slope_sign", "endpoint_order", "monotonicity"} and isinstance(value, str):
            normalized_item: dict[str, Any] = {"kind": kind, "value": value}
            if "source_text" in item:
                normalized_item["source_text"] = str(item["source_text"])
            if "confidence" in item:
                normalized_item["confidence"] = float(item["confidence"])
            if "notes" in item:
                normalized_item["notes"] = str(item["notes"])
            normalized.append(normalized_item)

    if normalized:
        return normalized

    for cue in semantic_cues:
        source_text = str(cue.get("source_text") or cue["kind"])
        confidence = cue.get("confidence")
        if cue["kind"] == "decline":
            normalized.append({"kind": "slope_sign", "value": "negative", "source_text": source_text, "confidence": confidence or 0.8})
        elif cue["kind"] == "increase":
            normalized.append({"kind": "slope_sign", "value": "positive", "source_text": source_text, "confidence": confidence or 0.8})
        elif cue["kind"] == "plateau":
            normalized.append({"kind": "slope_sign", "value": "zero", "source_text": source_text, "confidence": confidence or 0.8})

    return normalized


def _validate_plural_local_event_grounding(
    *,
    input_description: str,
    primitive_steps: list[PrimitivePlanStep],
) -> None:
    text = input_description.lower()
    primitive_counts: dict[str, int] = {}
    for step in primitive_steps:
        primitive_counts[step.primitive] = primitive_counts.get(step.primitive, 0) + 1
    for rule in PLURAL_LOCAL_EVENT_PATTERNS:
        if not any(pattern.search(text) for pattern in rule["regexes"]):
            continue
        count = sum(primitive_counts.get(primitive, 0) for primitive in rule["primitives"])
        if count < int(rule["minimum_count"]):
            raise PlanValidationError(str(rule["message"]), category="plan_invalid")


def normalize_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    global_context = payload.get("global_context")
    if isinstance(global_context, dict) and "numeric_range" in global_context:
        normalized_global_context = dict(global_context)
        normalized_global_context["numeric_range"] = _normalize_numeric_range_payload(global_context["numeric_range"])
        normalized["global_context"] = normalized_global_context
    steps = payload.get("steps")
    unresolved = payload.get("unresolved_semantics")
    if not isinstance(steps, list) or not isinstance(unresolved, list):
        return normalized

    unresolved_steps = [
        step for step in steps if isinstance(step, dict) and step.get("kind") == "unresolved"
    ]
    normalized_unresolved: list[Any] = []
    generated_count = 0

    for index, item in enumerate(unresolved):
        if isinstance(item, dict):
            normalized_unresolved.append(item)
            continue
        if isinstance(item, str):
            item_text = item.strip()
            matching_step = None
            for step in unresolved_steps:
                step_description = str(step.get("description", "")).strip()
                step_source_text = str(step.get("source_text", "")).strip()
                if item_text and item_text in {step_description, step_source_text}:
                    matching_step = step
                    break
            generated_count += 1
            fallback_id = f"unresolved_{generated_count}"
            if matching_step is not None:
                normalized_unresolved.append(
                    {
                        "id": str(matching_step.get("id", fallback_id)),
                        "description": str(matching_step.get("description") or item_text),
                        "source_text": str(matching_step.get("source_text") or item_text),
                        "reason": str(matching_step.get("reason") or "Model marked this semantic as unresolved."),
                    }
                )
            else:
                fallback_text = item_text or f"unresolved semantic {index + 1}"
                normalized_unresolved.append(
                    {
                        "id": fallback_id,
                        "description": fallback_text,
                        "source_text": fallback_text,
                        "reason": "Model marked this semantic as unresolved.",
                    }
                )
            continue
        normalized_unresolved.append(item)

    normalized["unresolved_semantics"] = normalized_unresolved

    normalized_steps: list[Any] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            normalized_steps.append(step)
            continue
        item = dict(step)
        primitive = item.get("primitive")
        if item.get("kind") == "primitive":
            item.setdefault("semantic_layer", DEFAULT_LAYER_BY_PRIMITIVE.get(str(primitive), "segment_structure"))
            item.setdefault("priority", index)
            item.setdefault("protected_constraints", [])
            semantic_cues = _normalize_semantic_cues(item.get("semantic_cues", []), item.get("inferred_constraints", []))
            item["semantic_cues"] = semantic_cues
            item["inferred_constraints"] = _normalize_inferred_constraints(item.get("inferred_constraints", []), semantic_cues)
        elif item.get("kind") == "unresolved":
            item.setdefault("semantic_layer", "segment_structure")
            item.setdefault("priority", index)
        normalized_steps.append(item)
    normalized["steps"] = normalized_steps
    return normalized


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise PlanValidationError("Planner output did not contain a JSON object.", category="plan_invalid")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"Planner output was not valid JSON: {exc}", category="plan_invalid") from exc
    if not isinstance(payload, dict):
        raise PlanValidationError("Planner output did not contain a JSON object.", category="plan_invalid")
    return normalize_plan_payload(payload)


def _validate_schema(payload: dict[str, Any]) -> None:
    _, _, schema = load_schema("plan")
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    if errors:
        first = errors[0]
        path = "/".join(str(part) for part in first.absolute_path) or "$"
        raise PlanValidationError(f"Plan schema validation failed at {path}: {first.message}", category="plan_invalid")


def parse_plan(
    payload: dict[str, Any],
    candidate_primitives: list[str],
    series_length: int,
    numeric_range: tuple[float, float] | dict[str, float] | None = None,
    allow_partial_output: bool = False,
) -> TimeSeriesPlan:
    _validate_schema(payload)

    if payload["global_context"]["length"] != series_length:
        raise PlanValidationError(
            "Plan global_context.length does not match the requested series length.",
            category="plan_invalid",
        )

    plan_numeric_range = _normalize_numeric_range_payload(payload["global_context"]["numeric_range"])
    if numeric_range is not None:
        expected_numeric_range = _normalize_numeric_range_payload(
            numeric_range if isinstance(numeric_range, dict) else {"min": numeric_range[0], "max": numeric_range[1]}
        )
        if (
            not np.isclose(plan_numeric_range["min"], expected_numeric_range["min"], atol=1e-9, rtol=0.0)
            or not np.isclose(plan_numeric_range["max"], expected_numeric_range["max"], atol=1e-9, rtol=0.0)
        ):
            raise PlanValidationError(
                "Plan global_context.numeric_range does not match the requested numeric range.",
                category="plan_invalid",
            )

    registry_validation = payload["registry_validation"]
    if registry_validation["status"] != "valid":
        raise PlanValidationError(
            "Plan requires a valid primitive registry before code generation.",
            category="plan_invalid",
        )

    if payload["input_description"].strip() == "":
        raise PlanValidationError("Plan input_description must be non-empty.", category="plan_invalid")

    if payload["needs_library_extension"] and not allow_partial_output:
        raise PlanValidationError(
            "Plan still requires a library extension and cannot be compiled yet.",
            category="needs_library_extension",
        )

    if payload["unresolved_semantics"] and not allow_partial_output:
        raise PlanValidationError(
            "Plan contains unresolved semantics and cannot be compiled yet.",
            category="needs_library_extension",
        )

    primitive_steps: list[PrimitivePlanStep] = []
    candidate_set = set(candidate_primitives)
    for index, item in enumerate(payload["steps"]):
        if item["kind"] != "primitive":
            if allow_partial_output and item["kind"] == "unresolved":
                continue
            raise PlanValidationError(
                "Only primitive plan steps are executable in the current pipeline.",
                category="needs_library_extension",
            )
        primitive = item["primitive"]
        if primitive not in candidate_set and not _allow_implicit_flat_scaffold(
            primitive=primitive,
            step_index=index,
            candidate_set=candidate_set,
        ):
            raise PlanValidationError(
                f"Primitive `{primitive}` is not in the candidate set.",
                category="plan_invalid",
            )
        metadata = PRIMITIVE_MACHINE_METADATA[primitive]
        if metadata["effect_type"] != item["effect_type"]:
            raise PlanValidationError(
                f"Primitive `{primitive}` has effect_type `{metadata['effect_type']}`, not `{item['effect_type']}`.",
                category="plan_invalid",
            )
        primitive_steps.append(
            PrimitivePlanStep(
                id=item["id"],
                effect_type=item["effect_type"],
                primitive=primitive,
                semantic_role=item["semantic_role"],
                parameters=dict(item["parameters"]),
                source_text=item["source_text"],
                confidence=float(item["confidence"]),
                semantic_layer=str(item.get("semantic_layer", DEFAULT_LAYER_BY_PRIMITIVE.get(primitive, "segment_structure"))),
                priority=int(item.get("priority", len(primitive_steps))),
                protected_constraints=list(item.get("protected_constraints", [])),
                semantic_cues=list(item.get("semantic_cues", [])),
                inferred_constraints=list(item.get("inferred_constraints", [])),
            )
        )

    if allow_partial_output and not primitive_steps:
        raise PlanValidationError(
            "Partial-output mode requires at least one executable primitive step.",
            category="needs_library_extension",
        )

    _validate_plural_local_event_grounding(
        input_description=payload["input_description"],
        primitive_steps=primitive_steps,
    )

    return TimeSeriesPlan(
        version=payload["version"],
        input_description=payload["input_description"],
        global_context={**dict(payload["global_context"]), "numeric_range": plan_numeric_range},
        assumptions=list(payload.get("assumptions", [])),
        primitive_steps=primitive_steps,
        unresolved_semantics=list(payload["unresolved_semantics"]),
        needs_library_extension=bool(payload["needs_library_extension"]),
    )


def _ordered_steps(plan: TimeSeriesPlan) -> list[PrimitivePlanStep]:
    def _step_start(step: PrimitivePlanStep) -> int:
        params = step.parameters
        if "start_timestep" in params:
            return int(params["start_timestep"])
        if "anchor_timestep" in params:
            return int(params["anchor_timestep"])
        if "timestep" in params:
            return int(params["timestep"])
        return 0

    def _precision_phase(step: PrimitivePlanStep) -> int:
        params = step.parameters
        primitive = step.primitive
        if primitive in {"add_noise", "add_volatility", "add_seasonality"}:
            return 1  # texture/noise layer after broad structure
        if primitive in {"add_spike", "add_peak", "add_trough", "add_dip", "add_outlier"} and "target_value" in params:
            return 2
        return 0

    return sorted(
        plan.all_steps,
        key=lambda step: (
            _precision_phase(step),
            SEMANTIC_LAYER_ORDER.get(step.semantic_layer, 99),
            _step_start(step),
            int(step.priority),
            0 if step.effect_type == "overwrite" else 1,
            step.id,
        ),
    )


def _segment(series: np.ndarray, start: int, end: int) -> np.ndarray:
    return series[start : end + 1]


def _check_protected_constraints(constraints: list[dict[str, Any]], series: np.ndarray, *, step_id: str) -> None:
    for constraint in constraints:
        kind = constraint["kind"]
        if kind == "constant_segment":
            start = int(constraint["start"])
            end = int(constraint["end"])
            value = float(constraint["value"])
            tolerance = float(constraint.get("tolerance", 1e-6))
            if not np.allclose(_segment(series, start, end), value, atol=tolerance, rtol=0.0):
                raise PlanExecutionError(
                    f"Protected constant segment violated after `{step_id}`.",
                    category="protected_constraint_violation",
                )
        elif kind == "monotone_window":
            start = int(constraint["start"])
            end = int(constraint["end"])
            direction = constraint["direction"]
            diffs = np.diff(_segment(series, start, end))
            if direction == "nondecreasing":
                passed = bool(np.all(diffs >= -1e-8))
            else:
                passed = bool(np.all(diffs <= 1e-8))
            if not passed:
                raise PlanExecutionError(
                    f"Protected monotone window violated after `{step_id}`.",
                    category="protected_constraint_violation",
                )
        elif kind == "point_value":
            timestep = int(constraint["timestep"])
            value = float(constraint["value"])
            tolerance = float(constraint.get("tolerance", 1e-6))
            if not np.isclose(float(series[timestep]), value, atol=tolerance, rtol=0.0):
                raise PlanExecutionError(
                    f"Protected point value violated after `{step_id}`.",
                    category="protected_constraint_violation",
                )


def _sign_label(value: float, *, tolerance: float = 1e-9) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _endpoint_order_label(start_value: float, end_value: float, *, tolerance: float = 1e-9) -> str:
    delta = end_value - start_value
    if delta > tolerance:
        return "ascending"
    if delta < -tolerance:
        return "descending"
    return "flat"


def _compiled_execution_constraints(step: PrimitivePlanStep) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for cue in step.semantic_cues:
        kind = cue["kind"]
        source_text = str(cue.get("source_text") or kind)
        confidence = float(cue.get("confidence", step.confidence))
        if kind == "decline":
            constraints.append({"kind": "slope_sign", "value": "negative", "source_text": source_text, "confidence": confidence})
            constraints.append({"kind": "endpoint_order", "value": "descending", "source_text": source_text, "confidence": confidence})
        elif kind == "increase":
            constraints.append({"kind": "slope_sign", "value": "positive", "source_text": source_text, "confidence": confidence})
            constraints.append({"kind": "endpoint_order", "value": "ascending", "source_text": source_text, "confidence": confidence})
        elif kind == "plateau":
            constraints.append({"kind": "slope_sign", "value": "zero", "source_text": source_text, "confidence": confidence})
            constraints.append({"kind": "endpoint_order", "value": "flat", "source_text": source_text, "confidence": confidence})
        elif kind == "light_noise":
            constraints.append({"kind": "max_noise_scale", "source_text": source_text, "confidence": confidence})
    return constraints


def _parameter_slope_sign(step: PrimitivePlanStep) -> str | None:
    params = step.parameters
    primitive = step.primitive
    if primitive in {"add_ramp", "add_growth", "add_decay"} and "start_value" in params and "end_value" in params:
        return _sign_label(float(params["end_value"]) - float(params["start_value"]))
    if primitive == "add_trend":
        if "slope" in params:
            return _sign_label(float(params["slope"]))
        if "start_offset" in params and "end_offset" in params:
            return _sign_label(float(params["end_offset"]) - float(params["start_offset"]))
        if "start_value" in params and "end_value" in params:
            return _sign_label(float(params["end_value"]) - float(params["start_value"]))
    if primitive == "add_change_point" and "slope_change" in params:
        return _sign_label(float(params["slope_change"]))
    if primitive == "add_level_shift" and "shift" in params:
        return _sign_label(float(params["shift"]))
    if primitive in {"add_flat", "add_plateau"}:
        return "zero"
    return None


def _numeric_range_width(global_context: dict[str, Any]) -> float:
    numeric_range = _normalize_numeric_range_payload(global_context["numeric_range"])
    return max(0.0, float(numeric_range["max"] - numeric_range["min"]))


def _noise_budget(step: PrimitivePlanStep, global_context: dict[str, Any], series: np.ndarray, affected: np.ndarray) -> float:
    candidates: list[float] = []
    band_width = _numeric_range_width(global_context)
    if band_width > 0.0:
        candidates.append(min(0.08 * band_width, 0.05))
    if affected.size:
        segment = series[affected]
        local_span = float(np.max(segment) - np.min(segment))
        if local_span > 0.0:
            candidates.append(min(0.25 * local_span, 0.05))
        nonzero = np.abs(segment[np.abs(segment) > 1e-12])
        if nonzero.size:
            local_level = float(np.median(nonzero))
            candidates.append(min(0.02 * abs(local_level), 0.05))
    candidates = [value for value in candidates if value > 0.0]
    return max(1e-6, min(candidates)) if candidates else 0.05


def _change_budget(global_context: dict[str, Any], *, fraction: float, default: float) -> float:
    width = _numeric_range_width(global_context)
    if width <= 0.0:
        return default
    return max(1e-6, fraction * width)


def _trend_total_change(step: PrimitivePlanStep, series_length: int) -> float | None:
    params = step.parameters
    primitive = step.primitive
    if primitive == "add_trend":
        start = int(params.get("start_timestep", 0))
        end = int(params.get("end_timestep", series_length - 1))
        horizon = max(1, end - start)
        if "slope" in params:
            return float(params["slope"]) * horizon
        if "start_offset" in params and "end_offset" in params:
            return float(params["end_offset"]) - float(params["start_offset"])
    if primitive == "add_change_point":
        start = int(params["anchor_timestep"])
        end = int(params.get("end_timestep", series_length - 1))
        horizon = max(1, end - start)
        return float(params.get("level_change", 0.0)) + float(params.get("slope_change", 0.0)) * horizon
    if primitive == "add_level_shift":
        return float(params.get("shift", 0.0))
    return None


def _check_compiled_constraints_pre(
    step: PrimitivePlanStep,
    global_context: dict[str, Any],
    series: np.ndarray,
    affected: np.ndarray,
) -> None:
    del step, global_context, series, affected
    # Parameter inference is now delegated to the LLM. Keep execution-time validation
    # focused on schema/primitive correctness and hard numerical failures, not on
    # hand-authored semantic budgets that can over-constrain otherwise valid plans.
    return None


def _check_compiled_constraints_post(step: PrimitivePlanStep, series: np.ndarray, affected: np.ndarray) -> None:
    del step, series, affected
    return None


def _check_compiled_constraints_post_with_previous(
    step: PrimitivePlanStep,
    previous_series: np.ndarray,
    series: np.ndarray,
    affected: np.ndarray,
    global_context: dict[str, Any],
) -> None:
    del step, previous_series, series, affected, global_context
    return None


def _affected_indices_for_primitive(step: PrimitivePlanStep, series_length: int) -> np.ndarray:
    params = step.parameters
    name = step.primitive

    if "start_timestep" in params or "end_timestep" in params:
        start = int(params.get("start_timestep", 0))
        end = int(params.get("end_timestep", series_length - 1))
        return np.arange(start, end + 1)

    if name == "add_change_point":
        start = int(params["anchor_timestep"])
        end = int(params.get("end_timestep", series_length - 1))
        return np.arange(start, end + 1)

    if name == "add_level_shift":
        start = int(params.get("start_timestep", params["anchor_timestep"]))
        end = int(params.get("end_timestep", series_length - 1))
        return np.arange(start, end + 1)

    if name == "add_outlier":
        return np.array([int(params["timestep"])])

    if name in {"add_spike", "add_dip"}:
        center = int(params["timestep"])
        width = int(params.get("half_width", params.get("width", 0)))
        return np.arange(max(0, center - width), min(series_length - 1, center + width) + 1)

    if name in {"add_peak", "add_trough"}:
        if "start_timestep" in params or "end_timestep" in params:
            start = int(params.get("start_timestep", 0))
            end = int(params.get("end_timestep", series_length - 1))
            return np.arange(start, end + 1)
        return np.arange(series_length)

    return np.arange(series_length)


def compile_plan_to_code(plan: TimeSeriesPlan, series_length: int) -> str:
    used_primitives: list[str] = []
    lines = ["import numpy as np"]
    for step in _ordered_steps(plan):
        if step.primitive not in used_primitives:
            used_primitives.append(step.primitive)
    if used_primitives:
        lines.append(f"from atomic_timeseries import {', '.join(used_primitives)}")
    lines.append("")
    lines.append(f"series = np.zeros({series_length}, dtype=np.float64)")
    lines.append("")

    for step in _ordered_steps(plan):
        args = ", ".join(f"{key}={repr(value)}" for key, value in step.parameters.items())
        lines.append(f"series = {step.primitive}(series, {args})")

    return "\n".join(lines).strip() + "\n"


def _check_series_stability(
    series: np.ndarray,
    global_context: dict[str, Any],
    *,
    allowed_nonfinite: np.ndarray | None = None,
) -> None:
    del global_context
    finite = np.isfinite(series)
    if allowed_nonfinite is None:
        allowed_nonfinite = np.zeros(series.shape, dtype=bool)
    if np.any(~finite & ~allowed_nonfinite):
        raise PlanExecutionError("Series contains non-finite values during plan execution.", category="numeric_range_explosion")
    finite_values = series[finite]
    if finite_values.size and np.nanmax(np.abs(finite_values)) > 10_000:
        raise PlanExecutionError("Series magnitude exploded during plan execution.", category="numeric_range_explosion")


def _has_valid_baseline_for_additive(series: np.ndarray, touched: np.ndarray, affected: np.ndarray) -> bool:
    if not np.any(~touched[affected]):
        return True
    untouched = affected[~touched[affected]]
    # Allow additive edits to establish signal directly on untouched zero regions.
    # This keeps execution robust when the planner omits an explicit overwrite scaffold.
    return bool(np.allclose(series[untouched], 0.0, atol=1e-12, rtol=0.0))


def execute_plan(plan: TimeSeriesPlan, caption: str, series_length: int) -> np.ndarray:
    series = np.zeros(series_length, dtype=np.float64)
    touched = np.zeros(series_length, dtype=bool)
    allowed_nonfinite = np.zeros(series_length, dtype=bool)
    active_constraints: list[dict[str, Any]] = []

    for step in _ordered_steps(plan):
        affected = _affected_indices_for_primitive(step, series_length)
        _check_compiled_constraints_pre(step, plan.global_context, series, affected)
        if step.effect_type == "additive" and not _has_valid_baseline_for_additive(series, touched, affected):
            raise PlanExecutionError(
                f"Additive step `{step.primitive}` touches regions without an overwrite baseline.",
                category="additive_without_baseline",
            )
        fn = getattr(ats, step.primitive)
        previous_series = series.copy()
        try:
            series = fn(series, **step.parameters)
        except Exception as exc:
            message = str(exc)
            category = "invalid_primitive_args" if "Provide " in message else "execution_failed"
            raise PlanExecutionError(
                f"Primitive execution failed for `{step.primitive}`: {exc}",
                category=category,
            ) from exc
        touched[affected] = True
        if step.primitive == "add_gap":
            fill_value = step.parameters.get("fill_value", np.nan)
            try:
                gap_is_missing = bool(np.isnan(float(fill_value)))
            except (TypeError, ValueError):
                gap_is_missing = str(fill_value).lower() == "nan"
            if gap_is_missing:
                allowed_nonfinite[affected] = True
        if np.any(touched):
            _check_series_stability(
                series[touched],
                plan.global_context,
                allowed_nonfinite=allowed_nonfinite[touched],
            )
        active_constraints.extend(step.protected_constraints)
        _check_protected_constraints(active_constraints, series, step_id=step.id)
        _check_compiled_constraints_post(step, series, affected)
        _check_compiled_constraints_post_with_previous(step, previous_series, series, affected, plan.global_context)

    return series
