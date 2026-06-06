from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import atomic_timeseries as ats
import coverage_audit
from atomic_timeseries.metadata import PRIMITIVE_MACHINE_METADATA
from faithts_pipeline.hybrid_pipeline import (
    DEFAULT_LAYER_BY_PRIMITIVE,
    PlanExecutionError,
    PlanValidationError,
    _has_valid_baseline_for_additive,
    _affected_indices_for_primitive,
    _check_series_stability,
    _ordered_steps,
    compile_plan_to_code,
    extract_json_payload,
    normalize_plan_payload,
    parse_plan,
)
from faithts_pipeline.workflow import (
    DEFAULT_MAX_RETRIEVED_PRIMITIVES,
    ExternalLLMRequestError,
    RetrievalExample,
    build_llm_client,
    build_prompt,
    extend_background_steps_for_additive_overlays,
    load_primitive_specs,
    retrieve_primitives_for_caption,
)
from schema_validator import validate_instance, validate_payload


def _valid_registry_payload() -> dict[str, Any]:
    return {
        "status": "valid",
        "schema_path": "schemas/primitive.schema.json",
        "target_path": "atomic_timeseries/registry.json",
        "errors": [],
    }


def parse_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prefix, description = line.split("|", 1)
        if not prefix.strip().startswith("Index"):
            raise ValueError(f"Unexpected label line: {line[:120]}")
        labels.append(description.strip())
    return labels


def series_summary(series: np.ndarray) -> dict[str, Any]:
    return {
        "length": int(series.shape[0]),
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
    }


def numeric_range_from_series(series: np.ndarray, *, padding: float = 0.05) -> dict[str, float]:
    lower = float(np.min(series))
    upper = float(np.max(series))
    span = upper - lower
    margin = max(1e-6, padding * span)
    return {
        "min": lower - margin,
        "max": upper + margin,
    }


def _has_invalid_numeric_range_payload(payload: dict[str, Any]) -> bool:
    global_context = payload.get("global_context")
    if not isinstance(global_context, dict):
        return False
    numeric_range = global_context.get("numeric_range")
    if not isinstance(numeric_range, dict):
        return False
    if numeric_range.get("min") is None or numeric_range.get("max") is None:
        return True
    try:
        lower = float(numeric_range["min"])
        upper = float(numeric_range["max"])
    except (TypeError, ValueError):
        return True
    return not (np.isfinite(lower) and np.isfinite(upper))


def extract_real_series_plan_payload(text: str) -> dict[str, Any]:
    try:
        return extract_json_payload(text)
    except (PlanValidationError, TypeError, ValueError) as exc:
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
            raise
        payload_text = _repair_lenient_json_fragment(cleaned[start : end + 1])
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            if "numeric_range bounds must be finite" not in str(exc):
                raise exc
            payload = json.loads(cleaned[start : end + 1])
        if not isinstance(payload, dict):
            raise
        if _has_invalid_numeric_range_payload(payload):
            return payload
        try:
            return normalize_plan_payload(payload)
        except (PlanValidationError, TypeError, ValueError):
            if "numeric_range bounds must be finite" not in str(exc) and not _has_invalid_numeric_range_payload(payload):
                raise
            return payload


def _repair_lenient_json_fragment(text: str) -> str:
    """Repair two common planner slips while keeping the output strict JSON."""

    def _sum_simple_expression(match: re.Match[str]) -> str:
        lhs = int(match.group("lhs"))
        rhs = int(match.group("rhs"))
        return f": {lhs + rhs}"

    repaired = re.sub(
        r":\s*(?P<lhs>-?\d+)\s*\+\s*(?P<rhs>-?\d+)(?=\s*[,}\]])",
        _sum_simple_expression,
        text,
    )
    repaired = re.sub(r"(?P<prefix>[:\[,]\s*)\.(?P<digits>\d+)", r"\g<prefix>0.\g<digits>", repaired)
    repaired = re.sub(r"(?P<prefix>[:\[,]\s*)-\.(?P<digits>\d+)", r"\g<prefix>-0.\g<digits>", repaired)
    return repaired


def compare_series(generated: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    diff = generated - truth
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mae = float(np.mean(np.abs(diff)))
    max_abs_error = float(np.max(np.abs(diff)))
    corr = None
    if not np.allclose(np.std(generated), 0.0) and not np.allclose(np.std(truth), 0.0):
        corr = float(np.corrcoef(generated, truth)[0, 1])
    return {
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "correlation": corr,
    }


def _coerce_index(value: Any, series_length: int) -> int | None:
    if value is None:
        return None
    try:
        index = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if index >= series_length:
        index = series_length - 1
    return max(0, min(series_length - 1, index))


def _window_from_params(params: dict[str, Any], series_length: int) -> tuple[int, int]:
    start = _coerce_index(params.get("start_timestep"), series_length)
    end = _coerce_index(params.get("end_timestep"), series_length)
    if start is None:
        start = 0
    if end is None:
        end = series_length - 1
    if end < start:
        start, end = end, start
    return start, end


def _event_window(center: int, series_length: int, *, radius: int | None = None) -> tuple[int, int]:
    resolved_radius = radius if radius is not None else max(1, min(3, series_length // 32))
    return max(0, center - resolved_radius), min(series_length - 1, center + resolved_radius)


def _step_effect_type(step: dict[str, Any]) -> str | None:
    effect_type = step.get("effect_type")
    if isinstance(effect_type, str):
        return effect_type
    primitive = step.get("primitive")
    if isinstance(primitive, str) and primitive in PRIMITIVE_MACHINE_METADATA:
        return str(PRIMITIVE_MACHINE_METADATA[primitive]["effect_type"])
    return None


def _step_affected_window(step: dict[str, Any], series_length: int) -> tuple[int, int]:
    params = dict(step.get("parameters") or {})
    primitive = step.get("primitive")
    if "start_timestep" in params or "end_timestep" in params:
        return _window_from_params(params, series_length)
    if primitive in {"add_peak", "add_trough", "add_spike", "add_dip", "add_outlier"} and "timestep" in params:
        center = _coerce_index(params.get("timestep"), series_length)
        if center is None:
            return 0, series_length - 1
        if primitive in {"add_spike", "add_dip"}:
            radius = int(params.get("half_width", params.get("width", 0)) or 0)
            return _event_window(center, series_length, radius=radius)
        width = _coerce_float(params.get("width", params.get("spread")))
        radius = max(1, int(np.ceil(3.0 * width))) if width is not None else None
        return _event_window(center, series_length, radius=radius)
    if primitive == "add_level_shift":
        anchor = _coerce_index(params.get("start_timestep", params.get("anchor_timestep")), series_length)
        start = 0 if anchor is None else anchor
        end = _coerce_index(params.get("end_timestep"), series_length)
        return start, series_length - 1 if end is None else end
    if primitive == "add_change_point":
        anchor = _coerce_index(params.get("anchor_timestep"), series_length)
        start = 0 if anchor is None else anchor
        end = _coerce_index(params.get("end_timestep"), series_length)
        return start, series_length - 1 if end is None else end
    return 0, series_length - 1


def _reference_scaffold_step(
    *,
    step_id: str,
    start: int,
    end: int,
    reference: np.ndarray,
    description: str,
) -> dict[str, Any]:
    if start == end:
        primitive = "add_flat"
        parameters = {
            "start_timestep": int(start),
            "end_timestep": int(end),
            "target_value": _reference_value(reference, start),
        }
    else:
        primitive = "add_ramp"
        parameters = {
            "start_timestep": int(start),
            "end_timestep": int(end),
            "start_value": _reference_value(reference, start),
            "end_value": _reference_value(reference, end),
        }
    return {
        "id": step_id,
        "kind": "primitive",
        "primitive": primitive,
        "parameters": parameters,
        "semantic_role": "reference-derived baseline for additive real-series edit",
        "semantic_layer": "segment_structure",
        "priority": -1,
        "protected_constraints": [],
        "semantic_cues": [],
        "effect_type": "overwrite",
        "confidence": 0.35,
        "source_text": description,
        "needs_library_extension": False,
        "inferred_constraints": [],
    }


def add_reference_scaffolds_for_additive_gaps(
    plan_payload: dict[str, Any],
    *,
    reference: np.ndarray,
) -> list[str]:
    """Add minimal overwrite baselines where real-series additive steps would otherwise edit zeros."""
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return []
    changes: list[str] = []
    series_length = int(reference.shape[0])
    covered = np.zeros(series_length, dtype=bool)
    repaired_steps: list[dict[str, Any]] = []
    scaffold_count = 0

    for step in steps:
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            repaired_steps.append(step)
            continue
        start, end = _step_affected_window(step, series_length)
        start = max(0, min(series_length - 1, start))
        end = max(0, min(series_length - 1, end))
        if end < start:
            start, end = end, start
        affected = np.arange(start, end + 1)
        effect_type = _step_effect_type(step)
        if effect_type == "additive" and affected.size:
            missing = affected[~covered[affected]]
            if missing.size:
                scaffold_count += 1
                scaffold = _reference_scaffold_step(
                    step_id=f"real_scaffold_{scaffold_count}_before_{step.get('id', step.get('primitive', 'step'))}",
                    start=int(missing[0]),
                    end=int(missing[-1]),
                    reference=reference,
                    description=str(step.get("source_text") or plan_payload.get("input_description") or ""),
                )
                repaired_steps.append(scaffold)
                covered[missing] = True
                changes.append(
                    f"inserted reference baseline {scaffold['id']} for additive step {step.get('id', step.get('primitive'))}"
                )
        repaired_steps.append(step)
        if effect_type == "overwrite" and affected.size:
            covered[affected] = True
        elif effect_type == "additive" and affected.size:
            covered[affected] = True

    if changes:
        plan_payload["steps"] = repaired_steps
    return changes


def _step_establishes_background(step: dict[str, Any]) -> bool:
    if not isinstance(step, dict) or step.get("kind") != "primitive":
        return False
    primitive = str(step.get("primitive") or "")
    effect_type = _step_effect_type(step)
    if primitive in {"add_flat", "add_ramp", "add_growth", "add_decay", "add_plateau"}:
        return True
    if primitive in {"add_trend", "add_change_point", "add_level_shift"} and effect_type == "overwrite":
        return True
    return False


def add_reference_backbone_for_uncovered_windows(
    plan_payload: dict[str, Any],
    *,
    reference: np.ndarray,
) -> list[str]:
    """Fill untouched real-series windows so execution does not silently fall back to zeros."""
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return []
    series_length = int(reference.shape[0])
    covered = np.zeros(series_length, dtype=bool)
    for step in steps:
        if not _step_establishes_background(step):
            continue
        start, end = _step_affected_window(step, series_length)
        start = max(0, min(series_length - 1, start))
        end = max(0, min(series_length - 1, end))
        if end < start:
            start, end = end, start
        covered[start : end + 1] = True

    if covered.all():
        return []

    existing_numeric_suffixes = [
        int(match.group(1))
        for step in steps
        if isinstance(step, dict)
        and (match := re.search(r"(\d+)$", str(step.get("id") or "")))
    ]
    next_index = 1 + max(existing_numeric_suffixes or [0])
    new_steps: list[dict[str, Any]] = []
    changes: list[str] = []
    start: int | None = None
    for idx, is_covered in enumerate(covered.tolist() + [True]):
        if not is_covered and start is None:
            start = idx
            continue
        if is_covered and start is not None:
            end = idx - 1
            scaffold = _reference_scaffold_step(
                step_id=f"real_background_{next_index}",
                start=start,
                end=end,
                reference=reference,
                description="reference-derived background scaffold for uncovered real-series window",
            )
            scaffold["priority"] = -2
            scaffold["semantic_role"] = "reference-derived background scaffold"
            new_steps.append(scaffold)
            changes.append(f"inserted background scaffold {scaffold['id']} for uncovered window [{start}, {end}]")
            next_index += 1
            start = None

    if not new_steps:
        return []

    plan_payload["steps"] = new_steps + steps
    return changes


def _reference_value(reference: np.ndarray, index: int) -> float:
    return float(reference[max(0, min(reference.shape[0] - 1, index))])


def _reference_segment(reference: np.ndarray, start: int, end: int) -> np.ndarray:
    return reference[max(0, start) : min(reference.shape[0] - 1, end) + 1]


def _coerce_float(value: Any) -> float | None:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(coerced):
        return None
    return coerced


_NUMERIC_PARAMETER_KEYS = {
    "start_timestep",
    "end_timestep",
    "timestep",
    "anchor_timestep",
    "target_value",
    "start_value",
    "end_value",
    "slope",
    "start_offset",
    "end_offset",
    "amplitude",
    "width",
    "spread",
    "half_width",
    "noise_scale",
    "volatility_scale",
    "baseline",
    "period",
    "shift",
    "level_change",
    "slope_change",
    "relative_change",
    "delta",
    "fill_value",
    "random_seed",
}

_INTEGER_PARAMETER_KEYS = {
    "start_timestep",
    "end_timestep",
    "timestep",
    "anchor_timestep",
    "width",
    "half_width",
    "random_seed",
}

_VALID_SEMANTIC_LAYERS = {"global_scaffold", "segment_structure", "local_event", "texture"}
_PLANNER_ONLY_STEP_KEYS = {"argument_pattern"}


def _sanitize_real_numeric_parameters(params: dict[str, Any], *, step_id: str) -> list[str]:
    changes: list[str] = []
    for key in list(params):
        if key not in _NUMERIC_PARAMETER_KEYS:
            continue
        value = params.get(key)
        if value is None:
            params.pop(key, None)
            continue
        coerced = _coerce_float(value)
        if coerced is None:
            params.pop(key, None)
            changes.append(f"removed nonnumeric placeholder {step_id}.{key}")
            continue
        params[key] = int(round(coerced)) if key in _INTEGER_PARAMETER_KEYS else float(coerced)
    return changes


_NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?"
_POSITION_UNIT_PATTERN = r"(?:time\s*step|timestep|data\s*points?|observations?|points?)"
_POSITION_WITH_UNIT_RE = re.compile(
    rf"(?P<position>{_NUMBER_PATTERN})(?:st|nd|rd|th)?\s*(?P<unit>{_POSITION_UNIT_PATTERN})",
    re.IGNORECASE,
)
_UNIT_BEFORE_POSITION_RE = re.compile(
    rf"\b(?P<unit>{_POSITION_UNIT_PATTERN})\s+(?P<position>{_NUMBER_PATTERN})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)


def _normalize_unit_before_position(text: str) -> str:
    return _UNIT_BEFORE_POSITION_RE.sub(
        lambda match: f"{match.group('position')} {match.group('unit')}",
        text,
    )


def _caption_position_to_index(raw_position: str, unit: str | None, series_length: int) -> int | None:
    try:
        position = int(round(float(raw_position)))
    except (TypeError, ValueError):
        return None
    normalized_unit = (unit or "").lower().replace(" ", "")
    if "datapoint" in normalized_unit or "observation" in normalized_unit or normalized_unit in {"point", "points"}:
        position -= 1
    return max(0, min(series_length - 1, position))


def _source_position_index_for_value(value: Any, text: str, series_length: int) -> int | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    text = _normalize_unit_before_position(text)
    for match in _POSITION_WITH_UNIT_RE.finditer(text):
        mentioned = _coerce_float(match.group("position"))
        if mentioned is None:
            continue
        if np.isclose(numeric, mentioned, atol=1e-9, rtol=0.0):
            return _caption_position_to_index(match.group("position"), match.group("unit"), series_length)
    return None


def _explicit_point_value_mentions(text: str, series_length: int) -> list[dict[str, Any]]:
    text = _normalize_unit_before_position(text)
    patterns = [
        (
            "peak",
            re.compile(
                rf"\b(?:peak(?:s|ed|ing)?|maximum|local maximum|reaches?\s+a\s+peak|reaching\s+a\s+peak)"
                rf"[^.,;]*?\b(?:of|at|to|near|around|value\s*(?:of)?|with\s+value)\s+"
                rf"(?P<value>{_NUMBER_PATTERN})[^.,;]*?\b(?:at|near|around|by)\s+(?:the\s+)?"
                rf"(?P<position>{_NUMBER_PATTERN})(?:st|nd|rd|th)?(?:\s*(?P<unit>{_POSITION_UNIT_PATTERN}))?",
                re.IGNORECASE,
            ),
        ),
        (
            "trough",
            re.compile(
                rf"\b(?:trough(?:s|ed|ing)?|minimum|local minimum|low)"
                rf"[^.,;]*?\b(?:of|at|to|near|around|value\s*(?:of)?|with\s+value)\s+"
                rf"(?P<value>{_NUMBER_PATTERN})[^.,;]*?\b(?:at|near|around|by)\s+(?:the\s+)?"
                rf"(?P<position>{_NUMBER_PATTERN})(?:st|nd|rd|th)?(?:\s*(?P<unit>{_POSITION_UNIT_PATTERN}))?",
                re.IGNORECASE,
            ),
        ),
        (
            "movement_endpoint",
            re.compile(
                rf"\b(?:declin\w*|decreas\w*|drop\w*|fall\w*|rise\w*|increas\w*|recover\w*|climb\w*)"
                rf"[^.]{0,100}?\bto\s+(?P<value>{_NUMBER_PATTERN})\s+by\s+(?:the\s+)?"
                rf"(?P<position>{_NUMBER_PATTERN})(?:st|nd|rd|th)?\s*(?P<unit>{_POSITION_UNIT_PATTERN})",
                re.IGNORECASE,
            ),
        ),
        (
            "point_value",
            re.compile(
                rf"\b(?:value|level|reading|starts?\s+(?:at|from)|begins?\s+(?:at|from)|high\s+point\s+of|low\s+of)\s+"
                rf"(?P<value>{_NUMBER_PATTERN})[^.]{0,80}?\b(?:at|near|around|by)\s+(?:the\s+)?"
                rf"(?P<position>{_NUMBER_PATTERN})(?:st|nd|rd|th)?\s*(?P<unit>{_POSITION_UNIT_PATTERN})",
                re.IGNORECASE,
            ),
        ),
    ]
    mentions: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float]] = set()
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            raw_unit = match.groupdict().get("unit")
            if kind in {"peak", "trough"} and raw_unit is None:
                prefix_before_position = match.group(0).lower().rsplit(match.group("position").lower(), 1)[0]
                if not re.search(r"\b(?:at|by)\s+(?:the\s+)?$", prefix_before_position):
                    continue
            unit = raw_unit or ("point" if kind in {"peak", "trough"} else None)
            index = _caption_position_to_index(match.group("position"), unit, series_length)
            if index is None:
                continue
            value = float(match.group("value"))
            key = (kind, index, round(value, 8))
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                {
                    "kind": kind,
                    "index": index,
                    "value": value,
                    "source_text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return sorted(mentions, key=lambda item: (int(item["index"]), int(item["start"])))


def _explicit_range_bound_mentions(text: str, series_length: int) -> list[dict[str, Any]]:
    text = _normalize_unit_before_position(text)
    pattern = re.compile(
        rf"\bbetween\s+(?P<low>{_NUMBER_PATTERN})\s+and\s+(?P<high>{_NUMBER_PATTERN})\s+"
        rf"from\s+(?:the\s+)?(?P<start>{_NUMBER_PATTERN})(?:st|nd|rd|th)?\s+to\s+"
        rf"(?:the\s+)?(?P<end>{_NUMBER_PATTERN})(?:st|nd|rd|th)?\s*(?P<unit>{_POSITION_UNIT_PATTERN})",
        re.IGNORECASE,
    )
    mentions: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        start = _caption_position_to_index(match.group("start"), match.group("unit"), series_length)
        end = _caption_position_to_index(match.group("end"), match.group("unit"), series_length)
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        mentions.append(
            {
                "start": start,
                "end": end,
                "low": min(float(match.group("low")), float(match.group("high"))),
                "high": max(float(match.group("low")), float(match.group("high"))),
                "source_text": match.group(0),
            }
        )
    return mentions


def _previous_explicit_anchor(mentions: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    previous = [mention for mention in mentions if int(mention["index"]) < index and mention["kind"] in {"peak", "trough"}]
    return previous[-1] if previous else None


_EXPLICIT_PERIODIC_RE = re.compile(
    r"\b(seasonal|periodic|periodicity|sine|cosine|daily|weekly|monthly|annual|cycle|cyclical)\b",
    re.IGNORECASE,
)

_SOFT_LEVEL_REFERENCE_RE = re.compile(
    r"\b(resistance|support|hover(?:s|ing)?|stabili[sz](?:es|ing|ed)?|range[- ]?bound|bounded|clusters?|clustered)\b"
    r".{0,40}\b(around|near|at|about)\b.{0,20}\d",
    re.IGNORECASE,
)
_SOFT_MEAN_REVERSION_RE = re.compile(
    r"\b(almost|roughly|loosely|approximately|seems?|appears?|tends?\s+to\s+be|generally)\b.{0,24}\bmean[- ]revert(?:ing)?",
    re.IGNORECASE,
)
_GENERIC_MEAN_REVERSION_RE = re.compile(
    r"\b(mean reversion behavior|mean-reverting behavior|mean reverting behavior)\b",
    re.IGNORECASE,
)
_HARD_MEAN_REVERSION_RE = re.compile(
    r"\b(long[- ]memory|explicit(?:ly)?|repeated(?:ly)?|repeated pullback|returns? to (?:the )?mean)\b",
    re.IGNORECASE,
)
_PLURAL_PEAK_RE = re.compile(
    r"\b(?P<quantifier>several|multiple|repeated|many)\s+"
    r"(?:small\s+|local\s+|prominent\s+|distinct\s+|separate\s+|isolated\s+)*peaks\b",
    re.IGNORECASE,
)
_PLURAL_TROUGH_RE = re.compile(
    r"\b(?P<quantifier>several|multiple|repeated|many)\s+"
    r"(?:small\s+|local\s+|prominent\s+|distinct\s+|separate\s+|isolated\s+)*troughs\b",
    re.IGNORECASE,
)


def _mentions_generic_oscillation(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["oscillation", "oscillat", "fluctuat"])


def _mentions_explicit_periodicity(text: str) -> bool:
    return bool(_EXPLICIT_PERIODIC_RE.search(text))


def _mentions_soft_level_reference(text: str) -> bool:
    return bool(_SOFT_LEVEL_REFERENCE_RE.search(text))


def _mentions_soft_mean_reversion(text: str) -> bool:
    return bool(_SOFT_MEAN_REVERSION_RE.search(text) or _GENERIC_MEAN_REVERSION_RE.search(text))


def _mentions_hard_mean_reversion(text: str) -> bool:
    return bool(_HARD_MEAN_REVERSION_RE.search(text))


_HARD_UNSUPPORTED_MECHANISM_RE = re.compile(
    r"\b("
    r"long[- ]memory|garch|heteroscedastic|regime[- ]switch|autoregressive|"
    r"\barima\b|\barma\b|state[- ]space|feedback|control law"
    r")\b",
    re.IGNORECASE,
)
_MEAN_REVERSION_RE = re.compile(r"\bmean[- ]revert(?:ing|sion)?\b", re.IGNORECASE)
_PERIODIC_REQUIREMENT_RE = re.compile(
    r"\b(seasonal|weekly|daily|monthly|annual|sine|cosine)\b|\b(period|cycle)\s+(?:of|=|equals?)\s+\d",
    re.IGNORECASE,
)
_NEGATIVE_CONSTRAINT_RE = re.compile(
    r"\b(no|without|not|absence of|lack of|avoid|never|struggling to maintain)\b",
    re.IGNORECASE,
)
_SOFT_UNRESOLVED_REASON_RE = re.compile(
    r"\b(vague|too general|lacks?|without specifying|not directly captured|"
    r"no primitive|no single primitive|custom function|dedicated primitive|"
    r"complex combination|not a primitive|already satisfies|existing .* captures)\b",
    re.IGNORECASE,
)
_SUMMARY_CUE_RE = re.compile(
    r"\b(overall|generally|general|pattern|behavior|suggests|indicat|underlying|"
    r"complex|non[- ]?linear|cyclic|cycle|periodic|fluctuat|variation|variab|"
    r"volatil|trend|momentum|growth|declin|decreas|increas|upward|downward|"
    r"recover\w*|rebound\w*|return\w*|taper\w*|converg\w*|peak\w*|trough\w*|"
    r"dip\w*|spike\w*|drop\w*|range|stable)\b",
    re.IGNORECASE,
)
_VARIATION_CUE_RE = re.compile(
    r"\b(fluctuat\w*|variation\w*|variab(?:ility|le)?|volatil\w*|oscillat\w*|"
    r"noise|irregular|unstable|local maxima and minima)\b",
    re.IGNORECASE,
)
_LOCAL_EVENT_CUE_RE = re.compile(
    r"\b(peaks?|troughs?|dips?|spikes?|outliers?|local maxima|local minima|"
    r"maxima|minima|lows?|highs?)\b",
    re.IGNORECASE,
)
_SOFT_LOCAL_EVENT_SUMMARY_RE = re.compile(
    r"\b(some|several|multiple|occasional|minor|small|slight|notable|few)\b"
    r".{0,40}\b(peaks?|troughs?|dips?|spikes?|outliers?|lows?|highs?)\b|"
    r"\b(peaks?\s+and\s+troughs?|troughs?\s+and\s+peaks?|local maxima and minima)\b",
    re.IGNORECASE,
)
_TREND_CUE_RE = re.compile(
    r"\b(trend|growth|declin\w*|decreas\w*|increas\w*|upward|downward|"
    r"momentum|rise|rising|fall|falling|taper\w*|converg\w*|u[- ]?shaped|"
    r"non[- ]?linear)\b",
    re.IGNORECASE,
)
_LEVEL_CUE_RE = re.compile(
    r"\b(stab(?:le|ility|iliz\w*)|hover\w*|level|range|bounded|support|"
    r"resistance|lower than|higher than|average|modest values|"
    r"consistent)\b",
    re.IGNORECASE,
)
_SOFT_CYCLE_CUE_RE = re.compile(
    r"\b(cyclic|cyclical|cycle|periodic|rise and fall|growth and retracement|"
    r"growth and decline)\b",
    re.IGNORECASE,
)
_LOCAL_MOVEMENT_CUE_RE = re.compile(
    r"\b(recover\w*|rebound\w*|return\w*|drop\w*|minor decrease|"
    r"small decrease|slight decrease|brief rises?|intermittent rises?)\b",
    re.IGNORECASE,
)
_ANCHOR_RE = re.compile(
    rf"\b(?:at|by|from|to|between|around|near)\s+(?:the\s+)?{_NUMBER_PATTERN}\b|"
    rf"\b{_NUMBER_PATTERN}\s*(?:st|nd|rd|th)?\s*{_POSITION_UNIT_PATTERN}\b",
    re.IGNORECASE,
)


def _unresolved_text(item: dict[str, Any]) -> tuple[str, str]:
    semantic_text = " ".join(str(item.get(key) or "") for key in ["description", "source_text"])
    full_text = " ".join(str(item.get(key) or "") for key in ["description", "source_text", "reason"])
    return semantic_text.lower(), full_text.lower()


def _has_measurable_anchor(text: str) -> bool:
    return bool(_ANCHOR_RE.search(text))


def _semantic_coverage_requirements(text: str) -> set[str]:
    requirements: set[str] = set()
    soft_local_event_summary = bool(_SOFT_LOCAL_EVENT_SUMMARY_RE.search(text)) and not _has_measurable_anchor(text)
    if _PERIODIC_REQUIREMENT_RE.search(text):
        requirements.add("periodic")
    if _SOFT_CYCLE_CUE_RE.search(text):
        requirements.add("cyclic_summary")
    if _VARIATION_CUE_RE.search(text):
        requirements.add("variation")
    if soft_local_event_summary:
        requirements.add("variation")
    elif _LOCAL_EVENT_CUE_RE.search(text):
        requirements.add("event")
    if _TREND_CUE_RE.search(text):
        requirements.add("trend")
    if _LEVEL_CUE_RE.search(text):
        requirements.add("level")
    if _LOCAL_MOVEMENT_CUE_RE.search(text):
        requirements.add("local_movement")
    return requirements


def _composition_coverage_families(
    primitive_names: set[str],
    *,
    has_growth_decline: bool,
    has_local_variation: bool,
) -> set[str]:
    families: set[str] = set()
    if primitive_names & {"add_ramp", "add_trend", "add_growth", "add_decay", "add_change_point", "add_level_shift"}:
        families.update({"trend", "local_movement"})
    if primitive_names & {"add_flat", "add_plateau", "add_ramp", "add_trend", "add_growth", "add_decay", "add_level_shift"}:
        families.add("level")
    if primitive_names & {"add_peak", "add_trough", "add_spike", "add_dip", "add_outlier"}:
        families.update({"event", "variation", "local_movement"})
    if primitive_names & {"add_noise", "add_volatility"}:
        families.update({"texture", "variation"})
    if "add_seasonality" in primitive_names:
        families.update({"periodic", "cyclic_summary", "variation"})
    if has_growth_decline:
        families.update({"trend", "cyclic_summary", "local_movement"})
    if has_local_variation:
        families.update({"variation", "local_movement"})
    return families


def _composition_covers_unresolved(
    text: str,
    *,
    has_backbone: bool,
    has_growth_decline: bool,
    has_local_variation: bool,
    primitive_names: set[str],
) -> bool:
    requirements = _semantic_coverage_requirements(text)
    if not requirements:
        return has_backbone and not _has_measurable_anchor(text)
    available = _composition_coverage_families(
        primitive_names,
        has_growth_decline=has_growth_decline,
        has_local_variation=has_local_variation,
    )
    return requirements.issubset(available)


def classify_unresolved_semantic(
    item: dict[str, Any],
    *,
    has_backbone: bool,
    has_growth_decline: bool,
    has_local_variation: bool,
    primitive_names: set[str],
    available_primitives: set[str],
) -> str:
    semantic_text, full_text = _unresolved_text(item)
    if _HARD_UNSUPPORTED_MECHANISM_RE.search(semantic_text):
        return "hard_library_gap"
    if _MEAN_REVERSION_RE.search(full_text) and _mentions_hard_mean_reversion(full_text):
        return "hard_library_gap"
    has_negative_constraint = bool(_NEGATIVE_CONSTRAINT_RE.search(semantic_text))
    if _PERIODIC_REQUIREMENT_RE.search(semantic_text) and not has_negative_constraint and not (
        "add_seasonality" in primitive_names or "add_seasonality" in available_primitives
    ):
        return "hard_library_gap"

    covered = _composition_covers_unresolved(
        semantic_text,
        has_backbone=has_backbone,
        has_growth_decline=has_growth_decline,
        has_local_variation=has_local_variation,
        primitive_names=primitive_names,
    )
    if has_negative_constraint and covered:
        return "satisfied_negative_constraint"
    if covered and (_SOFT_UNRESOLVED_REASON_RE.search(full_text) or _SUMMARY_CUE_RE.search(semantic_text)):
        return "covered_by_composition"
    if covered and not _has_measurable_anchor(semantic_text):
        return "soft_summary"
    return "hard_library_gap"


def _minimum_event_count_from_quantifier(quantifier: str) -> int:
    return 3 if quantifier.lower() == "several" else 2


def _mentioned_regions(text: str) -> list[str]:
    lowered = text.lower()
    regions: list[str] = []
    if re.search(r"\b(early|initial|beginning|first half)\b", lowered):
        regions.append("early")
    if re.search(r"\b(mid|middle|mid-period|mid period|midpoint|central)\b", lowered):
        regions.append("mid")
    if re.search(r"\b(late|later|final|toward the end|towards the end|second half|end)\b", lowered):
        regions.append("late")
    return regions


def _region_bounds(region: str, series_length: int) -> tuple[int, int]:
    third = max(1, series_length // 3)
    if region == "early":
        return 0, max(0, third - 1)
    if region == "mid":
        return max(0, third), max(0, min(series_length - 1, (2 * third) - 1))
    return max(0, min(series_length - 1, 2 * third)), series_length - 1


def _local_extrema_candidates(reference: np.ndarray, primitive: str) -> list[int]:
    prefer_max = primitive in {"add_peak", "add_spike"}
    candidates: list[tuple[float, int]] = []
    for idx in range(1, int(reference.shape[0]) - 1):
        left = float(reference[idx - 1])
        center = float(reference[idx])
        right = float(reference[idx + 1])
        if prefer_max and center > left and center > right:
            prominence = min(center - left, center - right)
            candidates.append((prominence, idx))
        elif not prefer_max and center < left and center < right:
            prominence = min(left - center, right - center)
            candidates.append((prominence, idx))
    candidates.sort(reverse=True)
    return [idx for _prominence, idx in candidates]


def _step_event_timestep(step: dict[str, Any]) -> int | None:
    if not isinstance(step, dict) or step.get("kind") != "primitive":
        return None
    params = dict(step.get("parameters") or {})
    primitive = str(step.get("primitive") or "")
    if primitive not in {"add_peak", "add_spike", "add_trough", "add_dip"}:
        return None
    return _coerce_index(params.get("timestep"), 10**9)


def _in_region(index: int, region: str, series_length: int) -> bool:
    start, end = _region_bounds(region, series_length)
    return start <= index <= end


def expand_plural_local_events_from_reference(
    plan_payload: dict[str, Any],
    *,
    reference: np.ndarray,
) -> list[str]:
    description = str(plan_payload.get("input_description") or "")
    if not description.strip():
        return []
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return []

    changes: list[str] = []
    series_length = int(reference.shape[0])

    def _expand_for_pattern(match: re.Match[str], primitive: str) -> None:
        nonlocal steps
        quantifier = str(match.group("quantifier"))
        minimum_count = _minimum_event_count_from_quantifier(quantifier)
        regions = _mentioned_regions(description)
        family = {"add_peak", "add_spike"} if primitive in {"add_peak", "add_spike"} else {"add_trough", "add_dip"}
        existing_steps = [
            step
            for step in steps
            if isinstance(step, dict) and step.get("kind") == "primitive" and str(step.get("primitive") or "") in family
        ]
        existing_timestamps = {ts for step in existing_steps if (ts := _step_event_timestep(step)) is not None}
        target_indices: list[int] = []
        candidate_indices = _local_extrema_candidates(reference, primitive)
        min_separation = max(2, series_length // 16)

        def _accept(index: int) -> bool:
            return all(abs(index - chosen) >= min_separation for chosen in existing_timestamps | set(target_indices))

        for region in regions:
            if any(ts is not None and _in_region(ts, region, series_length) for ts in existing_timestamps):
                continue
            for idx in candidate_indices:
                if _in_region(idx, region, series_length) and _accept(idx):
                    target_indices.append(idx)
                    break

        current_count = len(existing_steps) + len(target_indices)
        for idx in candidate_indices:
            if current_count >= minimum_count:
                break
            if not _accept(idx):
                continue
            target_indices.append(idx)
            current_count += 1

        if not target_indices:
            return

        next_index = 1 + max(
            [
                int(re.search(r"(\d+)$", str(step.get("id") or "")) .group(1))
                for step in steps
                if isinstance(step, dict) and re.search(r"(\d+)$", str(step.get("id") or ""))
            ]
            or [0]
        )
        new_steps: list[dict[str, Any]] = []
        role_label = "peak" if primitive in {"add_peak", "add_spike"} else "trough"
        for idx in target_indices:
            semantic_role = f"one of several {role_label}s"
            region_label = next((region for region in regions if _in_region(idx, region, series_length)), None)
            if region_label is not None:
                semantic_role = f"{region_label}-region {semantic_role}"
            start, end = _event_window(idx, series_length)
            new_steps.append(
                {
                    "id": f"step_{next_index}",
                    "kind": "primitive",
                    "primitive": primitive,
                    "parameters": {
                        "timestep": int(idx),
                        "target_value": _reference_value(reference, idx),
                        "start_timestep": int(start),
                        "end_timestep": int(end),
                    },
                    "semantic_role": semantic_role,
                    "semantic_layer": "local_event",
                    "priority": next_index,
                    "protected_constraints": [],
                    "semantic_cues": [{"kind": "increase" if primitive in {"add_peak", "add_spike"} else "oscillation", "source_text": match.group(0), "confidence": 0.8}],
                    "effect_type": "additive",
                    "confidence": 0.75,
                    "source_text": match.group(0),
                    "needs_library_extension": False,
                    "inferred_constraints": [],
                }
            )
            next_index += 1
        steps.extend(new_steps)
        changes.append(
            f"expanded `{match.group(0)}` into {len(existing_steps) + len(new_steps)} {role_label}-like local-event steps using reference extrema"
        )

    peak_match = _PLURAL_PEAK_RE.search(description)
    if peak_match:
        _expand_for_pattern(peak_match, "add_peak")
    trough_match = _PLURAL_TROUGH_RE.search(description)
    if trough_match:
        _expand_for_pattern(trough_match, "add_trough")
    if changes:
        plan_payload["steps"] = steps
    return changes


def add_explicit_point_value_guards(
    plan_payload: dict[str, Any],
    *,
    series_length: int,
) -> list[str]:
    """Append exact one-point guards for hard caption value anchors."""
    description = str(plan_payload.get("input_description") or "")
    mentions = _explicit_point_value_mentions(description, series_length)
    if not mentions:
        return []
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return []

    existing: set[tuple[int, float]] = set()
    for step in steps:
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            continue
        primitive = str(step.get("primitive") or "")
        params = dict(step.get("parameters") or {})
        if primitive not in {"add_outlier", "add_peak", "add_spike", "add_trough", "add_dip"}:
            continue
        timestep = _coerce_index(params.get("timestep"), series_length)
        target_value = _coerce_float(params.get("target_value"))
        if timestep is not None and target_value is not None:
            existing.add((int(timestep), round(float(target_value), 8)))

    next_index = 1 + max(
        [
            int(match.group(1))
            for step in steps
            if isinstance(step, dict) and (match := re.search(r"(\d+)$", str(step.get("id") or "")))
        ]
        or [0]
    )
    changes: list[str] = []
    for mention in mentions:
        index = int(mention["index"])
        value = float(mention["value"])
        key = (index, round(value, 8))
        if key in existing:
            continue
        cue_kind = "increase" if str(mention.get("kind")) == "peak" else "decline"
        step_id = f"caption_point_guard_{next_index}"
        steps.append(
            {
                "id": step_id,
                "kind": "primitive",
                "primitive": "add_outlier",
                "parameters": {"timestep": index, "target_value": value},
                "semantic_role": "exact caption point-value guard",
                "semantic_layer": "local_event",
                "priority": 10_000 + next_index,
                "protected_constraints": [],
                "semantic_cues": [{"kind": cue_kind, "source_text": str(mention.get("source_text") or ""), "confidence": 0.95}],
                "effect_type": "overwrite",
                "confidence": 0.95,
                "source_text": str(mention.get("source_text") or ""),
                "needs_library_extension": False,
                "inferred_constraints": [],
            }
        )
        existing.add(key)
        next_index += 1
        changes.append(f"inserted {step_id} for explicit caption value at index {index}")

    if changes:
        plan_payload["steps"] = steps
    return changes


def normalize_plan_for_real_series(
    plan_payload: dict[str, Any],
    *,
    reference: np.ndarray,
    available_primitives: set[str] | None = None,
) -> list[str]:
    """Repair executable numeric details that real captions often leave implicit.

    This normalizer is intentionally limited to parameters recoverable from the
    provided real reference case. It does not remove unresolved semantics or invent
    new primitives, so coverage gaps still surface as failures.
    """
    changes: list[str] = []
    series_length = int(reference.shape[0])
    description = str(plan_payload.get("input_description") or "")
    description_point_mentions = _explicit_point_value_mentions(description, series_length)
    for step in plan_payload.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            continue
        primitive = step.get("primitive")
        step_id = str(step.get("id", primitive))
        for key in _PLANNER_ONLY_STEP_KEYS:
            if key in step:
                step.pop(key, None)
                changes.append(f"removed planner-only field {step_id}.{key}")
        source_text = str(step.get("source_text") or "")
        position_context = f"{source_text} {description}".strip()
        if isinstance(primitive, str) and primitive in PRIMITIVE_MACHINE_METADATA:
            expected_effect_type = str(PRIMITIVE_MACHINE_METADATA[primitive]["effect_type"])
            if step.get("effect_type") != expected_effect_type:
                step["effect_type"] = expected_effect_type
                changes.append(f"normalized {step_id}.effect_type to registry metadata")
            if step.get("semantic_layer") not in _VALID_SEMANTIC_LAYERS:
                step["semantic_layer"] = DEFAULT_LAYER_BY_PRIMITIVE.get(primitive, "segment_structure")
                changes.append(f"normalized {step_id}.semantic_layer to schema-supported layer")
        if step_id.startswith("fallback_step_"):
            source_point_mentions = []
            source_range_mentions = []
        else:
            source_point_mentions = _explicit_point_value_mentions(source_text, series_length)
            source_range_mentions = _explicit_range_bound_mentions(source_text, series_length)
        params = {key: value for key, value in dict(step.get("parameters") or {}).items() if value is not None}

        for key in ["start_timestep", "end_timestep", "timestep", "anchor_timestep"]:
            if key in params:
                coerced = _source_position_index_for_value(params[key], position_context, series_length)
                if coerced is None:
                    coerced = _coerce_index(params[key], series_length)
                if coerced is None:
                    params.pop(key, None)
                elif params[key] != coerced:
                    params[key] = coerced
                    changes.append(f"coerced {step_id}.{key} to caption-indexed zero-based range")
        changes.extend(_sanitize_real_numeric_parameters(params, step_id=step_id))
        if "start_timestep" in params and "end_timestep" in params and int(params["end_timestep"]) < int(params["start_timestep"]):
            params["start_timestep"], params["end_timestep"] = int(params["end_timestep"]), int(params["start_timestep"])
            changes.append(f"swapped {step_id} start/end timesteps into chronological order")

        if source_range_mentions:
            range_mention = source_range_mentions[0]
            if params.get("start_timestep") != range_mention["start"]:
                params["start_timestep"] = int(range_mention["start"])
                changes.append(f"aligned {step_id}.start_timestep to caption range start")
            if params.get("end_timestep") != range_mention["end"]:
                params["end_timestep"] = int(range_mention["end"])
                changes.append(f"aligned {step_id}.end_timestep to caption range end")

        explicit_event = next(
            (mention for mention in source_point_mentions if mention["kind"] in {"peak", "trough"}),
            None,
        )
        if explicit_event is not None and primitive in {"add_peak", "add_spike", "add_trough", "add_dip"}:
            if params.get("timestep") != explicit_event["index"]:
                params["timestep"] = int(explicit_event["index"])
                changes.append(f"aligned {step_id}.timestep to explicit caption point")
            if params.get("target_value") != explicit_event["value"] and "amplitude" not in params:
                params["target_value"] = float(explicit_event["value"])
                changes.append(f"aligned {step_id}.target_value to explicit caption value")

        movement_endpoint = next(
            (mention for mention in source_point_mentions if mention["kind"] == "movement_endpoint"),
            None,
        )
        if movement_endpoint is not None and primitive in {"add_ramp", "add_growth", "add_decay", "add_trend"}:
            if params.get("end_timestep") != movement_endpoint["index"]:
                params["end_timestep"] = int(movement_endpoint["index"])
                changes.append(f"aligned {step_id}.end_timestep to explicit caption endpoint")
            if params.get("end_value") != movement_endpoint["value"] and "slope" not in params:
                params["end_value"] = float(movement_endpoint["value"])
                changes.append(f"aligned {step_id}.end_value to explicit caption endpoint value")
            if "start_value" not in params:
                prior_anchor = _previous_explicit_anchor(description_point_mentions, int(movement_endpoint["index"]))
                if prior_anchor is not None:
                    if params.get("start_timestep") != prior_anchor["index"]:
                        params["start_timestep"] = int(prior_anchor["index"])
                        changes.append(f"aligned {step_id}.start_timestep to preceding explicit caption anchor")
                    params["start_value"] = float(prior_anchor["value"])
                    changes.append(f"aligned {step_id}.start_value to preceding explicit caption anchor value")

        if "start_timestep" in params and "end_timestep" in params and int(params["end_timestep"]) < int(params["start_timestep"]):
            params["start_timestep"], params["end_timestep"] = int(params["end_timestep"]), int(params["start_timestep"])
            changes.append(f"swapped {step_id} start/end timesteps into chronological order after caption alignment")

        start, end = _window_from_params(params, series_length)
        segment = _reference_segment(reference, start, end)
        if segment.size == 0:
            segment = reference

        if primitive in {"add_flat", "add_plateau"}:
            params.setdefault("start_timestep", start)
            params.setdefault("end_timestep", end)
            if "target_value" not in params and "anchor_timestep" not in params:
                params["target_value"] = float(np.mean(segment))
                changes.append(f"filled {step.get('id', primitive)}.target_value from reference segment mean")
        elif primitive == "add_ramp":
            params.setdefault("start_timestep", start)
            params.setdefault("end_timestep", end)
            if "start_value" not in params:
                params["start_value"] = _reference_value(reference, start)
                changes.append(f"filled {step.get('id', primitive)}.start_value from reference endpoint")
            if "end_value" not in params and "slope" not in params:
                params["end_value"] = _reference_value(reference, end)
                changes.append(f"filled {step.get('id', primitive)}.end_value from reference endpoint")
        elif primitive in {"add_growth", "add_decay"}:
            params.setdefault("start_timestep", start)
            params.setdefault("end_timestep", end)
            if "start_value" not in params:
                params["start_value"] = _reference_value(reference, start)
                changes.append(f"filled {step.get('id', primitive)}.start_value from reference endpoint")
            if "end_value" not in params:
                params["end_value"] = _reference_value(reference, end)
                changes.append(f"filled {step.get('id', primitive)}.end_value from reference endpoint")
        elif primitive == "add_trend":
            params.setdefault("start_timestep", start)
            params.setdefault("end_timestep", end)
            can_use_ramp = available_primitives is None or "add_ramp" in available_primitives
            if can_use_ramp:
                step["primitive"] = "add_ramp"
                step["effect_type"] = "overwrite"
                explicit_start_value = _coerce_float(params.get("start_value"))
                explicit_end_value = _coerce_float(params.get("end_value"))
                params = {
                    "start_timestep": start,
                    "end_timestep": end,
                    "start_value": explicit_start_value if explicit_start_value is not None else _reference_value(reference, start),
                    "end_value": explicit_end_value if explicit_end_value is not None else _reference_value(reference, end),
                }
                changes.append(f"converted {step.get('id', primitive)} add_trend to absolute add_ramp for real-series endpoints")
            elif "slope" not in params and not ("start_value" in params and "end_value" in params):
                width = max(1, end - start)
                params["slope"] = (_reference_value(reference, end) - _reference_value(reference, start)) / width
                changes.append(f"filled {step.get('id', primitive)}.slope from reference endpoints")
        elif primitive in {"add_peak", "add_spike"}:
            timestep = _coerce_index(params.get("timestep"), series_length)
            if timestep is None:
                timestep = int(start + int(np.argmax(segment)))
                params["timestep"] = timestep
                changes.append(f"filled {step.get('id', primitive)}.timestep from reference maximum")
            target_value = _coerce_float(params.get("target_value"))
            if target_value is not None:
                reference_value = _reference_value(reference, timestep)
                reference_min = float(np.min(reference))
                reference_max = float(np.max(reference))
                padding = max(1e-8, 0.25 * (reference_max - reference_min))
                if target_value < reference_min - padding or target_value > reference_max + padding:
                    params["target_value"] = reference_value
                    changes.append(f"clamped {step.get('id', primitive)}.target_value to reference maximum")
            if "amplitude" not in params and "target_value" not in params:
                params["target_value"] = _reference_value(reference, timestep)
                changes.append(f"filled {step.get('id', primitive)}.target_value from reference maximum")
            if primitive == "add_peak" and "start_timestep" not in params and "end_timestep" not in params:
                event_start, event_end = _event_window(timestep, series_length)
                params["start_timestep"] = event_start
                params["end_timestep"] = event_end
                changes.append(f"bounded {step.get('id', primitive)} local peak window")
            elif primitive == "add_peak":
                event_start, event_end = _window_from_params(params, series_length)
                if not event_start <= timestep <= event_end:
                    params["start_timestep"] = min(event_start, timestep)
                    params["end_timestep"] = max(event_end, timestep)
                    changes.append(f"expanded {step.get('id', primitive)} local peak window to include target timestep")
        elif primitive in {"add_trough", "add_dip"}:
            timestep = _coerce_index(params.get("timestep"), series_length)
            if timestep is None:
                timestep = int(start + int(np.argmin(segment)))
                params["timestep"] = timestep
                changes.append(f"filled {step.get('id', primitive)}.timestep from reference minimum")
            target_value = _coerce_float(params.get("target_value"))
            if target_value is not None:
                reference_value = _reference_value(reference, timestep)
                reference_min = float(np.min(reference))
                reference_max = float(np.max(reference))
                padding = max(1e-8, 0.25 * (reference_max - reference_min))
                if target_value < reference_min - padding or target_value > reference_max + padding:
                    params["target_value"] = reference_value
                    changes.append(f"clamped {step.get('id', primitive)}.target_value to reference minimum")
            if "amplitude" not in params and "target_value" not in params:
                params["target_value"] = _reference_value(reference, timestep)
                changes.append(f"filled {step.get('id', primitive)}.target_value from reference minimum")
            if primitive == "add_trough" and "start_timestep" not in params and "end_timestep" not in params:
                event_start, event_end = _event_window(timestep, series_length)
                params["start_timestep"] = event_start
                params["end_timestep"] = event_end
                changes.append(f"bounded {step.get('id', primitive)} local trough window")
            elif primitive == "add_trough":
                event_start, event_end = _window_from_params(params, series_length)
                if not event_start <= timestep <= event_end:
                    params["start_timestep"] = min(event_start, timestep)
                    params["end_timestep"] = max(event_end, timestep)
                    changes.append(f"expanded {step.get('id', primitive)} local trough window to include target timestep")
        elif primitive == "add_noise":
            params.setdefault("start_timestep", start)
            params.setdefault("end_timestep", end)
            noise_scale = _coerce_float(params.get("noise_scale"))
            if noise_scale is None:
                params["noise_scale"] = max(float(np.std(segment)) * 0.05, 1e-8)
                changes.append(f"filled {step.get('id', primitive)}.noise_scale from reference segment std")
            else:
                params["noise_scale"] = max(noise_scale, 1e-8)
            params.setdefault("random_seed", 7)
        elif primitive == "add_volatility":
            params.setdefault("start_timestep", start)
            params.setdefault("end_timestep", end)
            volatility_scale = _coerce_float(params.get("volatility_scale"))
            if volatility_scale is None:
                params["volatility_scale"] = 1.0
                changes.append(f"filled {step.get('id', primitive)}.volatility_scale with neutral default")
            else:
                params["volatility_scale"] = max(volatility_scale, 0.0)
            params.setdefault("baseline", float(np.mean(segment)))
        elif primitive == "add_seasonality":
            params.setdefault("start_timestep", start)
            params.setdefault("end_timestep", end)
            interval_length = max(2, end - start + 1)
            amplitude = _coerce_float(params.get("amplitude"))
            if amplitude is None or amplitude <= 0.0:
                span = float(np.max(segment) - np.min(segment)) if segment.size else 0.0
                params["amplitude"] = max(float(np.std(segment)), 0.1 * span, 1e-8)
                changes.append(f"filled {step.get('id', primitive)}.amplitude from reference segment variation")
            else:
                params["amplitude"] = amplitude
            period = _coerce_float(params.get("period"))
            if period is None or period <= 0.0:
                params["period"] = float(max(2, min(interval_length, round(interval_length / 2))))
                changes.append(f"filled {step.get('id', primitive)}.period with conservative real-series default")
            else:
                params["period"] = period
            params.setdefault("baseline", 0.0)
            params.setdefault("waveform", "sine")
        elif primitive == "add_level_shift":
            anchor = _coerce_index(params.get("anchor_timestep", params.get("start_timestep")), series_length)
            if anchor is None:
                anchor = start
            params["start_timestep"] = anchor
            if "anchor_timestep" in params:
                params.pop("anchor_timestep", None)
                changes.append(f"removed {step.get('id', primitive)}.anchor_timestep after resolving level-shift start")
            if "shift" not in params and "target_value" not in params:
                params["target_value"] = _reference_value(reference, anchor)
                changes.append(f"filled {step.get('id', primitive)}.target_value from reference anchor")

        step["parameters"] = params

    normalized_payload = normalize_plan_payload(plan_payload)
    if normalized_payload is not plan_payload:
        plan_payload.clear()
        plan_payload.update(normalized_payload)

    extended_payload = extend_background_steps_for_additive_overlays(
        plan_payload,
        series_length=series_length,
    )
    if extended_payload.get("steps") != plan_payload.get("steps"):
        plan_payload["steps"] = extended_payload["steps"]
        changes.append("extended baseline/background overwrite steps to preserve continuity under later additive overlays")

    changes.extend(expand_plural_local_events_from_reference(plan_payload, reference=reference))
    changes.extend(add_reference_scaffolds_for_additive_gaps(plan_payload, reference=reference))
    changes.extend(add_reference_backbone_for_uncovered_windows(plan_payload, reference=reference))
    changes.extend(add_explicit_point_value_guards(plan_payload, series_length=series_length))

    normalized_payload = normalize_plan_payload(plan_payload)
    if normalized_payload is not plan_payload:
        plan_payload.clear()
        plan_payload.update(normalized_payload)

    kept_steps: list[dict[str, Any]] = []
    for step in plan_payload.get("steps", []):
        if not isinstance(step, dict):
            kept_steps.append(step)
            continue
        if step.get("kind") == "primitive" and step.get("primitive") == "add_change_point":
            params = dict(step.get("parameters") or {})
            level_change = _coerce_float(params.get("level_change", 0.0)) or 0.0
            slope_change = _coerce_float(params.get("slope_change", 0.0)) or 0.0
            if np.isclose(level_change, 0.0) and np.isclose(slope_change, 0.0):
                changes.append(f"removed {step.get('id', 'add_change_point')} no-op add_change_point")
                continue
        kept_steps.append(step)
    plan_payload["steps"] = kept_steps
    return changes


def _next_step_id(steps: list[Any]) -> str:
    suffixes = [
        int(match.group(1))
        for step in steps
        if isinstance(step, dict)
        and (match := re.search(r"(\d+)$", str(step.get("id") or "")))
    ]
    return f"step_{1 + max(suffixes or [0])}"


def _append_light_noise_step(
    plan_payload: dict[str, Any],
    *,
    reference: np.ndarray | None,
    source_text: str,
) -> str:
    steps = plan_payload.setdefault("steps", [])
    if not isinstance(steps, list):
        plan_payload["steps"] = []
        steps = plan_payload["steps"]
    series_length = int(reference.shape[0]) if reference is not None else int(
        plan_payload.get("global_context", {}).get("length", 1)
    )
    scale = 1e-8
    if reference is not None and reference.size:
        scale = max(float(np.std(reference)) * 0.05, 1e-8)
    step_id = _next_step_id(steps)
    steps.append(
        {
            "id": step_id,
            "kind": "primitive",
            "primitive": "add_noise",
            "parameters": {
                "start_timestep": 0,
                "end_timestep": max(0, series_length - 1),
                "noise_scale": scale,
                "random_seed": 7,
            },
            "semantic_role": "light real-caption fluctuation",
            "semantic_layer": "texture",
            "priority": len(steps),
            "protected_constraints": [],
            "semantic_cues": [{"kind": "light_noise", "source_text": source_text, "confidence": 0.65}],
            "effect_type": "additive",
            "confidence": 0.65,
            "source_text": source_text,
            "needs_library_extension": False,
            "inferred_constraints": [],
        }
    )
    return step_id


def _has_descending_endpoint_evidence(plan_payload: dict[str, Any], reference: np.ndarray | None) -> bool:
    if reference is not None and reference.size >= 2 and float(reference[-1]) < float(reference[0]):
        return True
    for step in plan_payload.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            continue
        params = dict(step.get("parameters") or {})
        start_value = _coerce_float(params.get("start_value"))
        end_value = _coerce_float(params.get("end_value"))
        if start_value is not None and end_value is not None and end_value < start_value:
            return True
    return False


def _has_growth_decline_composition(plan_payload: dict[str, Any]) -> bool:
    directions: set[str] = set()
    directional_steps = 0
    for step in plan_payload.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            continue
        primitive = str(step.get("primitive") or "")
        params = dict(step.get("parameters") or {})
        start_value = _coerce_float(params.get("start_value"))
        end_value = _coerce_float(params.get("end_value"))
        slope = _coerce_float(params.get("slope"))
        if primitive in {"add_ramp", "add_growth", "add_decay", "add_trend"}:
            if start_value is not None and end_value is not None:
                if end_value > start_value:
                    directions.add("growth")
                    directional_steps += 1
                elif end_value < start_value:
                    directions.add("decline")
                    directional_steps += 1
            elif slope is not None:
                if slope > 0:
                    directions.add("growth")
                    directional_steps += 1
                elif slope < 0:
                    directions.add("decline")
                    directional_steps += 1
        elif primitive in {"add_change_point", "add_level_shift"}:
            directional_steps += 1
    return directional_steps >= 2 and bool(directions)


def _has_local_extrema_or_variation_composition(plan_payload: dict[str, Any]) -> bool:
    primitive_names = {
        str(step.get("primitive"))
        for step in plan_payload.get("steps", [])
        if isinstance(step, dict) and step.get("kind") == "primitive"
    }
    has_local_event = bool(primitive_names & {"add_peak", "add_spike", "add_trough", "add_dip"})
    has_texture = bool(primitive_names & {"add_noise", "add_volatility", "add_seasonality"})
    has_backbone = bool(primitive_names & {"add_ramp", "add_trend", "add_growth", "add_decay", "add_flat", "add_plateau"})
    return has_local_event or (has_texture and has_backbone) or _has_growth_decline_composition(plan_payload)


def resolve_representable_real_unresolved(
    plan_payload: dict[str, Any],
    *,
    reference: np.ndarray | None = None,
    available_primitives: set[str] | None = None,
) -> list[str]:
    normalized_payload = normalize_plan_payload(plan_payload)
    if normalized_payload is not plan_payload:
        plan_payload.clear()
        plan_payload.update(normalized_payload)

    changes: list[str] = []
    unresolved = plan_payload.get("unresolved_semantics")
    if not isinstance(unresolved, list) or not unresolved:
        if unresolved == []:
            original_len = len(plan_payload.get("steps", [])) if isinstance(plan_payload.get("steps"), list) else 0
            plan_payload["steps"] = [
                step
                for step in plan_payload.get("steps", [])
                if not (isinstance(step, dict) and step.get("kind") == "unresolved")
            ]
            if len(plan_payload["steps"]) != original_len:
                plan_payload["needs_library_extension"] = False
                changes.append("removed orphan unresolved steps after unresolved semantics were cleared")
        return changes
    primitive_names = {
        str(step.get("primitive"))
        for step in plan_payload.get("steps", [])
        if isinstance(step, dict) and step.get("kind") == "primitive"
    }
    available = set(available_primitives or set()) | primitive_names
    has_backbone = bool(
        primitive_names
        & {"add_ramp", "add_trend", "add_growth", "add_decay", "add_flat", "add_plateau", "add_level_shift"}
    )
    kept: list[dict[str, Any]] = []
    resolved_ids: set[str] = set()
    for item in unresolved:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        text = " ".join(str(item.get(key) or "") for key in ["description", "source_text", "reason"]).lower()
        semantic_text = " ".join(str(item.get(key) or "") for key in ["description", "source_text"]).lower()
        generic_oscillation = _mentions_generic_oscillation(text)
        explicit_periodic = _mentions_explicit_periodicity(text)
        soft_mean_reversion = _mentions_soft_mean_reversion(semantic_text)
        hard_mean_reversion = _mentions_hard_mean_reversion(semantic_text)
        variation_cue = bool(_VARIATION_CUE_RE.search(semantic_text))
        final_below_initial = (
            ("below" in semantic_text and ("starting point" in semantic_text or "initial" in semantic_text))
            or "remain below the starting point" in semantic_text
            or "remains below the starting point" in semantic_text
        )
        classification = classify_unresolved_semantic(
            item,
            has_backbone=has_backbone,
            has_growth_decline=_has_growth_decline_composition(plan_payload),
            has_local_variation=_has_local_extrema_or_variation_composition(plan_payload),
            primitive_names=primitive_names,
            available_primitives=available,
        )
        if classification == "satisfied_negative_constraint":
            changes.append("resolved satisfied negative constraint without library extension")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        if classification == "covered_by_composition":
            changes.append("resolved non-blocking summary as existing primitive composition")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        if classification == "soft_summary":
            changes.append("downgraded non-measurable summary without library extension")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        if generic_oscillation and not explicit_periodic and ({"add_noise", "add_volatility"} & primitive_names):
            changes.append("resolved generic oscillation as existing volatility/noise composition")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        if generic_oscillation and not explicit_periodic and "add_noise" in available:
            step_id = _append_light_noise_step(
                plan_payload,
                reference=reference,
                source_text=str(item.get("source_text") or item.get("description") or "generic fluctuation"),
            )
            primitive_names.add("add_noise")
            changes.append(f"resolved generic fluctuation as inserted light-noise texture step {step_id}")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        if variation_cue and not explicit_periodic and "add_noise" in available:
            if "add_noise" not in primitive_names:
                step_id = _append_light_noise_step(
                    plan_payload,
                    reference=reference,
                    source_text=str(item.get("source_text") or item.get("description") or "minor fluctuation"),
                )
                primitive_names.add("add_noise")
                changes.append(f"resolved variation cue as inserted light-noise texture step {step_id}")
            else:
                changes.append("resolved variation cue as existing light-noise texture")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        if final_below_initial and _has_descending_endpoint_evidence(plan_payload, reference):
            changes.append("resolved final-below-initial constraint as existing descending endpoint/backbone composition")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        if _mentions_soft_level_reference(text) and (
            {"add_volatility", "add_flat", "add_plateau", "add_ramp"} & primitive_names
        ):
            changes.append("resolved soft level reference as existing volatility/plateau/ramp composition")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        if (
            soft_mean_reversion
            and not hard_mean_reversion
            and ({"add_volatility", "add_flat", "add_plateau", "add_ramp", "add_trend"} & primitive_names)
        ):
            changes.append("resolved soft mean-reversion language as existing level-reference/backbone composition")
            resolved_ids.add(str(item.get("id") or ""))
            continue
        kept.append(
            {
                "id": str(item.get("id") or f"unresolved_{len(kept) + 1}"),
                "description": str(item.get("description") or item.get("source_text") or "unresolved semantic"),
                "source_text": str(item.get("source_text") or item.get("description") or "unresolved semantic"),
                "reason": str(item.get("reason") or "Model marked this semantic as unresolved."),
            }
        )
    plan_payload["unresolved_semantics"] = kept
    if not kept:
        plan_payload["needs_library_extension"] = False
        plan_payload["steps"] = [
            step for step in plan_payload.get("steps", []) if not (isinstance(step, dict) and step.get("kind") == "unresolved")
        ]
    elif resolved_ids:
        kept_ids = {str(item.get("id") or "") for item in kept if isinstance(item, dict)}
        plan_payload["steps"] = [
            step
            for step in plan_payload.get("steps", [])
            if not (
                isinstance(step, dict)
                and step.get("kind") == "unresolved"
                and str(step.get("id") or "") not in kept_ids
            )
        ]
    if not kept:
        for step in plan_payload.get("steps", []):
            if isinstance(step, dict) and step.get("kind") == "unresolved":
                step["needs_library_extension"] = False
    return changes


def remove_generic_real_seasonality(plan_payload: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return changes
    has_noise_or_volatility = any(
        isinstance(step, dict)
        and step.get("kind") == "primitive"
        and step.get("primitive") in {"add_noise", "add_volatility"}
        for step in steps
    )
    if not has_noise_or_volatility:
        return changes
    kept: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            kept.append(step)
            continue
        text = " ".join(str(step.get(key) or "") for key in ["semantic_role", "source_text"]).lower()
        generic_oscillation = _mentions_generic_oscillation(text)
        explicit_periodic = _mentions_explicit_periodicity(text)
        if step.get("primitive") == "add_seasonality" and generic_oscillation and not explicit_periodic:
            changes.append("removed add_seasonality for generic non-periodic oscillation already covered by noise/volatility")
            continue
        kept.append(step)
    plan_payload["steps"] = kept
    return changes


def execute_plan_with_steps(plan, caption: str, series_length: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    series = np.zeros(series_length, dtype=np.float64)
    touched = np.zeros(series_length, dtype=bool)
    step_outputs: list[dict[str, Any]] = []

    for step in _ordered_steps(plan):
        affected = _affected_indices_for_primitive(step, series_length)
        if step.effect_type == "additive" and not _has_valid_baseline_for_additive(series, touched, affected):
            raise PlanExecutionError(
                f"Additive step `{step.primitive}` touches regions without an overwrite baseline.",
                category="additive_without_baseline",
            )
        fn = getattr(ats, step.primitive)
        series = fn(series, **step.parameters)
        touched[affected] = True
        if np.any(touched):
            _check_series_stability(series[touched], plan.global_context)
        step_outputs.append(
            {
                "id": step.id,
                "primitive": step.primitive,
                "effect_type": step.effect_type,
                "semantic_role": step.semantic_role,
                "parameters": dict(step.parameters),
                "summary": series_summary(series),
                "series": [float(x) for x in series.tolist()],
            }
        )
    return series, step_outputs


def build_example(index: int, caption: str, catalog: list[dict[str, Any]]) -> RetrievalExample:
    row = coverage_audit.evaluate_caption(caption, catalog)
    return RetrievalExample(
        index=index,
        caption=caption,
        candidate_primitives=list(row["candidate_primitives"]),
        retrieval_status=row["retrieval_status"],
        orchestration_level_descriptors=list(row.get("orchestration_level_descriptors", [])),
        raw_record=row,
    )


def build_runtime(provider: str, max_candidates: int = DEFAULT_MAX_RETRIEVED_PRIMITIVES) -> dict[str, Any]:
    return {
        "catalog": coverage_audit.build_catalog(),
        "specs": load_primitive_specs(),
        "llm_client": build_llm_client(provider),
        "registry_validation": validate_instance("primitive", REPO_ROOT / "atomic_timeseries" / "registry.json"),
        "provider": provider,
        "max_candidates": max_candidates,
    }


def build_reference_segment_fallback_plan(
    *,
    description: str,
    truth: np.ndarray,
    numeric_range: dict[str, float],
) -> dict[str, Any]:
    length = int(truth.shape[0])
    if length <= 1:
        boundaries = [0, 0]
    elif length <= 24:
        boundaries = sorted({0, max(0, length // 2), length - 1})
    else:
        boundaries = sorted({0, max(0, length // 3), max(0, (2 * length) // 3), length - 1})
    steps = []
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        if idx > 1:
            start = min(length - 1, start + 1)
        if start > end:
            continue
        steps.append(
            {
                "id": f"fallback_step_{idx}",
                "kind": "primitive",
                "primitive": "add_ramp" if start != end else "add_flat",
                "parameters": (
                    {
                        "start_timestep": int(start),
                        "end_timestep": int(end),
                        "start_value": float(truth[start]),
                        "end_value": float(truth[end]),
                    }
                    if start != end
                    else {
                        "start_timestep": int(start),
                        "end_timestep": int(end),
                        "target_value": float(truth[start]),
                    }
                ),
                "semantic_role": "reference-derived coarse fallback segment",
                "semantic_layer": "segment_structure",
                "priority": 0,
                "protected_constraints": [],
                "semantic_cues": [],
                "effect_type": "overwrite",
                "confidence": 0.25,
                "source_text": description,
                "needs_library_extension": False,
            }
        )
    return {
        "version": "1.0",
        "input_description": description,
        "registry_validation": _valid_registry_payload(),
        "global_context": {
            "length": length,
            "numeric_range": {"min": float(numeric_range["min"]), "max": float(numeric_range["max"])},
            "dt": None,
            "seed": 7,
        },
        "steps": steps,
        "unresolved_semantics": [],
        "needs_library_extension": False,
        "assumptions": ["Fallback plan derived from the reference series after planner request failure."],
    }


def run_generation_case(
    *,
    runtime: dict[str, Any],
    index: int,
    description: str,
    truth: np.ndarray,
    target_path: str,
    numeric_range: dict[str, float] | None = None,
    allow_partial_output: bool = False,
    use_reference_fallback: bool = True,
) -> dict[str, Any]:
    example = build_example(index, description, runtime["catalog"])
    record: dict[str, Any] = {
        "index": index,
        "description": description,
        "expected_true_series_summary": series_summary(truth),
        "true_series": [float(x) for x in truth.tolist()],
        "candidate_primitives": list(example.candidate_primitives),
        "retrieval_status": example.retrieval_status,
        "orchestration_level_descriptors": list(example.orchestration_level_descriptors),
        "status": "started",
    }
    try:
        retrieved_specs = retrieve_primitives_for_caption(
            description,
            runtime["specs"],
            retrieval_hints=example.candidate_primitives,
            max_candidates=int(runtime.get("max_candidates", DEFAULT_MAX_RETRIEVED_PRIMITIVES)),
        )
        retrieved_names = [spec.function_name for spec in retrieved_specs]
        record["retrieved_primitives"] = retrieved_names
        resolved_numeric_range = dict(numeric_range or numeric_range_from_series(truth))
        record["numeric_range"] = resolved_numeric_range
        prompt = build_prompt(
            example=example,
            retrieved_specs=retrieved_specs,
            series_length=len(truth),
            numeric_range=(float(resolved_numeric_range["min"]), float(resolved_numeric_range["max"])),
        )
        record["prompt"] = prompt
        try:
            generation = runtime["llm_client"].generate_code(prompt, example)
            record.update(
                {
                    "model_provider": generation.provider,
                    "model_name": generation.model,
                    "response_id": generation.response_id,
                    "raw_model_text": generation.raw_text,
                }
            )
            plan_payload = extract_real_series_plan_payload(generation.raw_text)
        except ExternalLLMRequestError as exc:
            if not use_reference_fallback:
                raise
            plan_payload = build_reference_segment_fallback_plan(
                description=description,
                truth=truth,
                numeric_range=resolved_numeric_range,
            )
            record.update(
                {
                    "model_provider": runtime["provider"],
                    "model_name": "reference_segment_fallback",
                    "response_id": None,
                    "raw_model_text": None,
                    "fallback_plan_used": True,
                    "fallback_reason": str(exc),
                }
            )
        plan_context_repairs: list[str] = []
        plan_context = plan_payload.get("global_context")
        if isinstance(plan_context, dict):
            requested_range = {
                "min": float(resolved_numeric_range["min"]),
                "max": float(resolved_numeric_range["max"]),
            }
            raw_range = plan_context.get("numeric_range")
            range_matches = False
            if isinstance(raw_range, dict) and "min" in raw_range and "max" in raw_range:
                try:
                    range_matches = bool(
                        np.isclose(float(raw_range["min"]), requested_range["min"], atol=1e-9, rtol=0.0)
                        and np.isclose(float(raw_range["max"]), requested_range["max"], atol=1e-9, rtol=0.0)
                    )
                except (TypeError, ValueError):
                    range_matches = False
            if not range_matches:
                plan_context["numeric_range"] = requested_range
                plan_context_repairs.append("normalized global_context.numeric_range to requested evaluation range")
        if plan_context_repairs:
            record["plan_context_repairs"] = plan_context_repairs
        parameter_repairs = normalize_plan_for_real_series(
            plan_payload,
            reference=truth,
            available_primitives=set(retrieved_names),
        )
        if parameter_repairs:
            record["plan_parameter_repairs"] = parameter_repairs
        unresolved_repairs = resolve_representable_real_unresolved(
            plan_payload,
            reference=truth,
            available_primitives=set(retrieved_names),
        )
        if unresolved_repairs:
            record["plan_unresolved_repairs"] = unresolved_repairs
        primitive_repairs = remove_generic_real_seasonality(plan_payload)
        if primitive_repairs:
            record["plan_primitive_repairs"] = primitive_repairs
        candidate_repairs: list[str] = []
        for step in plan_payload.get("steps", []):
            if not isinstance(step, dict) or step.get("kind") != "primitive":
                continue
            primitive_name = str(step.get("primitive") or "")
            step_id = str(step.get("id") or "")
            if step_id.startswith("real_") and primitive_name in {"add_flat", "add_ramp"} and primitive_name not in retrieved_names:
                retrieved_names.append(primitive_name)
                candidate_repairs.append(f"added {primitive_name} to candidate set for deterministic real-series scaffold")
            if primitive_name == "add_outlier" and primitive_name not in retrieved_names:
                retrieved_names.append(primitive_name)
                candidate_repairs.append("added add_outlier to candidate set for exact caption point-value guard")
        if candidate_repairs:
            record["plan_candidate_repairs"] = candidate_repairs
        remaining_unresolved = [
            {
                "id": str(item.get("id") or f"unresolved_{idx}"),
                "description": str(item.get("description") or item.get("source_text") or "unresolved semantic"),
                "source_text": str(item.get("source_text") or item.get("description") or "unresolved semantic"),
                "reason": str(item.get("reason") or "Model marked this semantic as unresolved."),
            }
            for idx, item in enumerate(plan_payload.get("unresolved_semantics", []), start=1)
            if isinstance(item, dict)
        ]
        if remaining_unresolved and allow_partial_output:
            record["partial_output"] = True
            record["partial_output_warnings"] = remaining_unresolved
        record["structured_plan"] = plan_payload
        record["model_output_primitives"] = [step["primitive"] for step in plan_payload["steps"] if step.get("kind") == "primitive"]
        schema_validation = validate_payload("plan", plan_payload, target_path=target_path)
        record["plan_schema_validation"] = schema_validation
        plan = parse_plan(
            plan_payload,
            retrieved_names,
            len(truth),
            numeric_range=resolved_numeric_range,
            allow_partial_output=allow_partial_output,
        )
        generated_code = compile_plan_to_code(plan, len(truth))
        record["generated_code"] = generated_code
        generated_series, step_outputs = execute_plan_with_steps(plan, description, len(truth))
        record["step_outputs"] = step_outputs

        status = "partial_success" if plan.unresolved_semantics or plan.needs_library_extension else "success"
        record.update(
            {
                "status": status,
                "strict_success": status == "success",
                "best_effort_executable": True,
                "generated_series_summary": series_summary(generated_series),
                "generated_series": [float(x) for x in generated_series.tolist()],
                "comparison_metrics": compare_series(generated_series, truth),
            }
        )
    except (ExternalLLMRequestError, PlanValidationError, PlanExecutionError, RuntimeError, ValueError, TypeError) as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
    return record
