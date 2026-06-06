from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

from atomic_timeseries.metadata import PRIMITIVE_MACHINE_METADATA
from faithts_pipeline.hybrid_pipeline import DEFAULT_LAYER_BY_PRIMITIVE
from schema_validator import validate_payload


@dataclass(frozen=True)
class RepairResult:
    plan: dict[str, Any]
    iterations: int
    changes: list[str]
    schema_status: str


def _registry_validation() -> dict[str, Any]:
    return {
        "status": "valid",
        "schema_path": "schemas/primitive.schema.json",
        "target_path": "atomic_timeseries/registry.json",
        "errors": [],
    }


def _numeric_range(raw: Any) -> dict[str, float]:
    if isinstance(raw, dict) and "min" in raw and "max" in raw:
        return {"min": float(raw["min"]), "max": float(raw["max"])}
    return {"min": -1000.0, "max": 1000.0}


def _append_change(changes: list[str], message: str) -> None:
    if message not in changes:
        changes.append(message)


def _text_forbids_fresh_noise(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in [
            "rather than introducing fresh random noise",
            "without introducing fresh random noise",
            "not introducing fresh random noise",
            "no fresh random noise",
            "rather than adding fresh random noise",
        ]
    )


def _text_has_explicit_noise_scale(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\bscale\s+[-+]?\d", lowered) or re.search(r"\bnoise[_ -]?scale\s*[:=]?\s*[-+]?\d", lowered))


def _text_has_explicit_gap_fill(text: str) -> bool:
    lowered = text.lower()
    return "fill" in lowered or "filled" in lowered or "encoded as" in lowered


def _text_requests_missing_values(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ["missing", "data gap", "gap in the data", "gap"])


def _extract_source_timestep(text: str) -> int | None:
    match = re.search(r"\b(?:at|near|around)\s+timestep\s+(\d+)\b", text.lower())
    return int(match.group(1)) if match else None


def _extract_explicit_target_value(text: str) -> float | None:
    lowered = text.lower()
    patterns = [
        r"\breaches\s+([-+]?\d+(?:\.\d+)?)",
        r"\bto\s+([-+]?\d+(?:\.\d+)?)",
        r"\bset exactly to\s+([-+]?\d+(?:\.\d+)?)",
        r"\bvalue\s+([-+]?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return float(match.group(1))
    return None


def _extract_after_timestep(text: str) -> int | None:
    match = re.search(r"\bafter timestep\s+(\d+)\b", text.lower())
    return int(match.group(1)) if match else None


def _recover_workflow_failure_steps(plan: dict[str, Any], description: str, series_length: int) -> list[dict[str, Any]] | None:
    if not plan.get("needs_library_extension") or not plan.get("unresolved_semantics"):
        return None
    unresolved_text = " ".join(
        str(item.get("reason", "")) + " " + str(item.get("description", ""))
        for item in plan.get("unresolved_semantics", [])
        if isinstance(item, dict)
    ).lower()
    if "workflow failed" not in unresolved_text:
        return None

    lowered = description.lower()
    gap_match = re.search(r"\bfrom timestep\s+(\d+)\s+to\s+(\d+)\b", lowered)
    if gap_match and ("missing entirely" in lowered or "data is missing" in lowered or "missing interval" in lowered):
        start, end = int(gap_match.group(1)), int(gap_match.group(2))
        return [
            {
                "id": "step_1",
                "kind": "primitive",
                "primitive": "add_gap",
                "parameters": {"start_timestep": start, "end_timestep": end},
                "semantic_role": "missing interval",
                "semantic_layer": DEFAULT_LAYER_BY_PRIMITIVE.get("add_gap", "segment_structure"),
                "effect_type": PRIMITIVE_MACHINE_METADATA["add_gap"]["effect_type"],
                "confidence": 0.8,
                "source_text": description,
                "needs_library_extension": False,
            }
        ]

    start = _extract_after_timestep(description)
    if start is not None and "more volatile" in lowered and _text_forbids_fresh_noise(description):
        return [
            {
                "id": "step_1",
                "kind": "primitive",
                "primitive": "add_volatility",
                "parameters": {
                    "start_timestep": start,
                    "end_timestep": series_length - 1,
                    "volatility_scale": 2.0,
                    "baseline": 0.0,
                },
                "semantic_role": "wider swings around same center",
                "semantic_layer": DEFAULT_LAYER_BY_PRIMITIVE.get("add_volatility", "texture"),
                "effect_type": PRIMITIVE_MACHINE_METADATA["add_volatility"]["effect_type"],
                "confidence": 0.8,
                "source_text": description,
                "needs_library_extension": False,
            }
        ]
    return None


def _apply_semantic_normalization(repaired_steps: list[dict[str, Any]], description: str, changes: list[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    primitive_names = {str(step.get("primitive")) for step in repaired_steps if step.get("kind") == "primitive"}
    forbid_noise = _text_forbids_fresh_noise(description)
    after_timestep = _extract_after_timestep(description)

    for step in repaired_steps:
        if step.get("kind") != "primitive":
            normalized.append(step)
            continue
        primitive = str(step.get("primitive", ""))
        if primitive == "add_noise" and forbid_noise and "add_volatility" in primitive_names:
            _append_change(changes, "removed add_noise contradicted by fresh-random-noise negation")
            continue

        repaired_step = dict(step)
        params = dict(repaired_step.get("parameters", {}))
        source_text = str(repaired_step.get("source_text") or description)
        if after_timestep is not None and "start_timestep" in params and int(params["start_timestep"]) == after_timestep + 1:
            params["start_timestep"] = after_timestep
            repaired_step["parameters"] = params
            _append_change(changes, "aligned start_timestep with explicit after-timestep boundary")

        if primitive in {"add_spike", "add_dip", "add_peak", "add_trough", "add_outlier"} and "timestep" in params:
            source_timestep = _extract_source_timestep(source_text)
            if source_timestep is not None and int(params["timestep"]) != source_timestep:
                old_timestep = int(params["timestep"])
                params["timestep"] = source_timestep
                repaired_step["parameters"] = params
                _append_change(changes, f"aligned {primitive} timestep from {old_timestep} to explicit source timestep {source_timestep}")

        if primitive in {"add_spike", "add_dip", "add_peak", "add_trough", "add_outlier"} and "target_value" not in params:
            target = _extract_explicit_target_value(source_text)
            if target is not None:
                params.pop("amplitude", None)
                params.pop("relative_change", None)
                params.pop("delta", None)
                params["target_value"] = target
                repaired_step["parameters"] = params
                _append_change(changes, f"converted explicit {primitive} target claim to target_value")

        if primitive == "add_gap" and "fill_value" in params and _text_requests_missing_values(source_text) and not _text_has_explicit_gap_fill(source_text):
            params.pop("fill_value", None)
            repaired_step["parameters"] = params
            _append_change(changes, "removed implicit add_gap fill_value so missing values use the registry default")

        if (
            primitive == "add_noise"
            and "noise_scale" in params
            and float(params["noise_scale"]) > 0.5
            and not _text_has_explicit_noise_scale(description)
            and ("jitter" in description.lower() or "noisy" in description.lower())
        ):
            params["noise_scale"] = 0.2
            repaired_step["parameters"] = params
            _append_change(changes, "clamped vague jitter noise_scale to conservative default")

        if primitive in {"add_spike", "add_dip"} and "target_value" not in params and "amplitude" in params:
            numeric_range = repaired_step.get("_global_numeric_range")
            if isinstance(numeric_range, dict) and "min" in numeric_range and "max" in numeric_range:
                low = float(numeric_range["min"])
                high = float(numeric_range["max"])
                if 0.0 <= high - low <= 1.0 and ("impulse" in description.lower() or "spike" in description.lower()):
                    params.pop("amplitude", None)
                    params["target_value"] = (low + high) / 2.0
                    repaired_step["parameters"] = params
                    _append_change(changes, "converted narrow-range impulse amplitude to target_value")

        normalized.append(repaired_step)
    return normalized


def _text_requests_persistent_level_shift(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in [
            "jump",
            "jumps",
            "shift",
            "shifts",
            "level shift",
            "new level",
            "stays at that new level",
        ]
    )


def _level_shift_start(step: dict[str, Any]) -> int | None:
    params = step.get("parameters", {})
    if not isinstance(params, dict):
        return None
    raw = params.get("start_timestep", params.get("anchor_timestep"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _extendable_flat_baseline_index(
    steps: list[dict[str, Any]],
    *,
    shift_index: int,
    shift_start: int,
    series_length: int,
) -> int | None:
    for prev_index in range(shift_index - 1, -1, -1):
        prev = steps[prev_index]
        if not isinstance(prev, dict) or prev.get("kind") not in {None, "primitive"}:
            continue
        if prev.get("primitive") != "add_flat":
            continue
        params = prev.get("parameters", {})
        if not isinstance(params, dict) or "target_value" not in params:
            continue
        if "start_timestep" not in params or "end_timestep" not in params:
            continue
        try:
            start = int(params["start_timestep"])
            end = int(params["end_timestep"])
        except (TypeError, ValueError):
            continue
        if start == 0 and start <= shift_start and (
            end < series_length - 1 or prev.get("semantic_layer") != "global_scaffold"
        ):
            return prev_index
    return None


def _needs_level_shift_baseline_repair(plan: dict[str, Any], description: str, series_length: int) -> bool:
    if not _text_requests_persistent_level_shift(description):
        return False
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        return False
    typed_steps = [step for step in steps if isinstance(step, dict)]
    for index, step in enumerate(typed_steps):
        if step.get("primitive") != "add_level_shift":
            continue
        params = step.get("parameters", {})
        if not isinstance(params, dict) or "shift" not in params or "target_value" in params:
            continue
        shift_start = _level_shift_start(step)
        if shift_start is None:
            continue
        if _extendable_flat_baseline_index(
            typed_steps,
            shift_index=index,
            shift_start=shift_start,
            series_length=series_length,
        ) is not None:
            return True
    return False


def _is_persistent_flat_baseline_for_level_shift(
    step: dict[str, Any],
    *,
    step_index: int,
    steps: list[Any],
    description: str,
    series_length: int,
) -> bool:
    if not _text_requests_persistent_level_shift(description):
        return False
    if step.get("primitive") != "add_flat":
        return False
    params = step.get("parameters", {})
    if not isinstance(params, dict) or "target_value" not in params:
        return False
    try:
        start = int(params.get("start_timestep", 0))
        end = int(params.get("end_timestep", -1))
    except (TypeError, ValueError):
        return False
    if start != 0 or end != series_length - 1:
        return False
    for later in steps[step_index + 1 :]:
        if not isinstance(later, dict) or later.get("primitive") != "add_level_shift":
            continue
        later_params = later.get("parameters", {})
        if isinstance(later_params, dict) and "shift" in later_params and "target_value" not in later_params:
            return True
    return False


def _apply_level_shift_baseline_repair(
    repaired_steps: list[dict[str, Any]],
    description: str,
    series_length: int,
    changes: list[str],
) -> list[dict[str, Any]]:
    if not _text_requests_persistent_level_shift(description):
        return repaired_steps

    steps = [dict(step) if isinstance(step, dict) else step for step in repaired_steps]
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("primitive") != "add_level_shift":
            continue
        params = step.get("parameters", {})
        if not isinstance(params, dict) or "shift" not in params or "target_value" in params:
            continue
        shift_start = _level_shift_start(step)
        if shift_start is None:
            continue
        baseline_index = _extendable_flat_baseline_index(
            steps,
            shift_index=index,
            shift_start=shift_start,
            series_length=series_length,
        )
        if baseline_index is None:
            continue

        baseline = dict(steps[baseline_index])
        baseline_params = dict(baseline.get("parameters", {}))
        old_end = int(baseline_params["end_timestep"])
        baseline_params["end_timestep"] = series_length - 1
        baseline["parameters"] = baseline_params
        old_layer = baseline.get("semantic_layer")
        baseline["semantic_layer"] = "global_scaffold"
        role = str(baseline.get("semantic_role") or "")
        if "baseline" not in role.lower():
            baseline["semantic_role"] = "persistent baseline"
        steps[baseline_index] = baseline
        if old_end < series_length - 1:
            _append_change(
                changes,
                f"extended flat baseline before add_level_shift from end_timestep {old_end} to {series_length - 1}",
            )
        if old_layer != "global_scaffold":
            _append_change(changes, "promoted persistent flat baseline before add_level_shift to global_scaffold")
    return steps


def _text_requests_global_baseline(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ["first part", "first half", "initial segment", "early segment"]):
        return False
    return any(
        phrase in lowered
        for phrase in [
            "baseline is",
            "starts from a baseline",
            "stays around",
            "mostly centered at",
            "series is flat at",
            "the series is flat at",
            "begins flat",
            "starts flat",
        ]
    )


def _has_later_compositional_edit(steps: list[dict[str, Any]], index: int) -> bool:
    later_primitives = {
        "add_spike",
        "add_dip",
        "add_peak",
        "add_trough",
        "add_outlier",
        "add_seasonality",
        "add_noise",
        "add_volatility",
        "add_ramp",
        "add_trend",
        "add_growth",
        "add_decay",
        "add_change_point",
        "add_level_shift",
    }
    return any(
        isinstance(step, dict) and step.get("primitive") in later_primitives
        for step in steps[index + 1 :]
    )


def _needs_global_baseline_repair(plan: dict[str, Any], description: str, series_length: int) -> bool:
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        return False
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("primitive") != "add_flat":
            continue
        params = step.get("parameters", {})
        if not isinstance(params, dict) or "target_value" not in params:
            continue
        try:
            start = int(params.get("start_timestep", 0))
            end = int(params.get("end_timestep", -1))
        except (TypeError, ValueError):
            continue
        source_text = str(step.get("source_text") or description)
        if start == 0 and end < series_length - 1 and _text_requests_global_baseline(source_text) and _has_later_compositional_edit(steps, index):
            return True
    return False


def _apply_global_baseline_repair(
    repaired_steps: list[dict[str, Any]],
    description: str,
    series_length: int,
    changes: list[str],
) -> list[dict[str, Any]]:
    steps = [dict(step) if isinstance(step, dict) else step for step in repaired_steps]
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("primitive") != "add_flat":
            continue
        params = step.get("parameters", {})
        if not isinstance(params, dict) or "target_value" not in params:
            continue
        try:
            start = int(params.get("start_timestep", 0))
            end = int(params.get("end_timestep", -1))
        except (TypeError, ValueError):
            continue
        source_text = str(step.get("source_text") or description)
        if not (start == 0 and end < series_length - 1 and _text_requests_global_baseline(source_text)):
            continue
        if not _has_later_compositional_edit(steps, index):
            continue
        repaired = dict(step)
        repaired_params = dict(params)
        repaired_params["end_timestep"] = series_length - 1
        repaired["parameters"] = repaired_params
        repaired["semantic_layer"] = "global_scaffold"
        role = str(repaired.get("semantic_role") or "")
        if "baseline" not in role.lower():
            repaired["semantic_role"] = "global baseline"
        steps[index] = repaired
        _append_change(changes, f"extended global baseline add_flat from end_timestep {end} to {series_length - 1}")
    return steps


def _repair_unresolved_step(step: dict[str, Any], description: str, index: int, changes: list[str]) -> dict[str, Any]:
    semantic = step.get("description") or step.get("semantic_role") or step.get("primitive") or "unresolved semantic"
    repaired = {
        "id": str(step.get("id") or f"unresolved_{index}"),
        "kind": "unresolved",
        "description": str(semantic),
        "semantic_role": str(step.get("semantic_role") or semantic),
        "confidence": float(step.get("confidence", 0.5)),
        "source_text": str(step.get("source_text") or description),
        "reason": str(step.get("reason") or "No registered primitive covers this semantic."),
        "needs_library_extension": True,
    }
    _append_change(changes, f"converted step {index} to unresolved")
    return repaired


def _repair_primitive_step(step: dict[str, Any], description: str, index: int, changes: list[str]) -> dict[str, Any]:
    repaired = dict(step)
    primitive = str(repaired.get("primitive", ""))
    if primitive not in PRIMITIVE_MACHINE_METADATA:
        return _repair_unresolved_step(repaired, description, index, changes)

    repaired.setdefault("id", f"step_{index}")
    repaired["kind"] = "primitive"
    repaired.setdefault("parameters", {})
    repaired.setdefault("semantic_role", primitive.replace("add_", ""))
    repaired.setdefault("confidence", 0.5)
    repaired.setdefault("source_text", description)
    repaired["needs_library_extension"] = False

    expected_effect = PRIMITIVE_MACHINE_METADATA[primitive]["effect_type"]
    if repaired.get("effect_type") != expected_effect:
        repaired["effect_type"] = expected_effect
        _append_change(changes, f"fixed effect_type for {primitive}")

    expected_layer = DEFAULT_LAYER_BY_PRIMITIVE.get(primitive)
    if expected_layer is not None and repaired.get("semantic_layer") != expected_layer:
        repaired["semantic_layer"] = expected_layer
        _append_change(changes, f"fixed semantic_layer for {primitive}")

    params = dict(repaired.get("parameters", {}))
    if primitive == "add_noise" and "random_seed" not in params and "random_generator" not in params:
        params["random_seed"] = 7
        _append_change(changes, "added deterministic random_seed to add_noise")
    repaired["parameters"] = {key: value for key, value in params.items() if value is not None}

    for field in ["semantic_layer", "priority", "protected_constraints", "semantic_cues", "inferred_constraints", "depends_on", "notes"]:
        value = repaired.get(field)
        if field in repaired and (value is None or value == ""):
            repaired.pop(field)
    return repaired


def _repair_once(plan: dict[str, Any], series_length: int, changes: list[str]) -> dict[str, Any]:
    repaired = deepcopy(plan)
    description = str(repaired.get("input_description") or "time-series description")
    if repaired.get("input_description") != description:
        repaired["input_description"] = description
        _append_change(changes, "filled input_description")

    if not isinstance(repaired.get("registry_validation"), dict):
        repaired["registry_validation"] = _registry_validation()
        _append_change(changes, "filled registry_validation")

    global_context = dict(repaired.get("global_context") or {})
    if global_context.get("length") != series_length:
        global_context["length"] = int(series_length)
        _append_change(changes, "fixed global_context.length")
    global_context["numeric_range"] = _numeric_range(global_context.get("numeric_range"))
    global_context.setdefault("dt", None)
    global_context.setdefault("seed", None)
    repaired["global_context"] = global_context

    raw_steps = repaired.get("steps")
    if not isinstance(raw_steps, list):
        raw_steps = []
        _append_change(changes, "initialized empty steps")

    recovered_steps = _recover_workflow_failure_steps(repaired, description, int(series_length))
    if recovered_steps is not None:
        raw_steps = recovered_steps
        repaired["unresolved_semantics"] = []
        repaired["needs_library_extension"] = False
        _append_change(changes, "recovered supported primitive plan from workflow failure")

    repaired_steps: list[dict[str, Any]] = []
    unresolved_semantics: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps, start=1):
        if not isinstance(step, dict):
            unresolved = _repair_unresolved_step({"description": str(step)}, description, index, changes)
            repaired_steps.append(unresolved)
        elif step.get("kind") == "unresolved":
            unresolved = _repair_unresolved_step(step, description, index, changes)
            repaired_steps.append(unresolved)
        else:
            repaired_step = _repair_primitive_step(step, description, index, changes)
            if repaired_step.get("kind") == "primitive":
                repaired_step["_global_numeric_range"] = global_context["numeric_range"]
            repaired_steps.append(repaired_step)
        if repaired_steps[-1].get("kind") == "unresolved":
            unresolved_semantics.append(
                {
                    "id": str(repaired_steps[-1]["id"]),
                    "description": str(repaired_steps[-1]["description"]),
                    "source_text": str(repaired_steps[-1]["source_text"]),
                    "reason": str(repaired_steps[-1]["reason"]),
                }
            )

    existing_unresolved = repaired.get("unresolved_semantics")
    if isinstance(existing_unresolved, list):
        for item in existing_unresolved:
            if isinstance(item, dict):
                unresolved_semantics.append(
                    {
                        "id": str(item.get("id") or f"u{len(unresolved_semantics) + 1}"),
                        "description": str(item.get("description") or item.get("semantic") or "unresolved semantic"),
                        "source_text": str(item.get("source_text") or description),
                        "reason": str(item.get("reason") or "No registered primitive covers this semantic."),
                    }
                )
            elif isinstance(item, str):
                unresolved_semantics.append(
                    {
                        "id": f"u{len(unresolved_semantics) + 1}",
                        "description": item,
                        "source_text": description,
                        "reason": "No registered primitive covers this semantic.",
                    }
                )

    repaired_steps = _apply_semantic_normalization(repaired_steps, description, changes)
    repaired_steps = _apply_global_baseline_repair(repaired_steps, description, int(series_length), changes)
    repaired_steps = _apply_level_shift_baseline_repair(repaired_steps, description, int(series_length), changes)
    for step in repaired_steps:
        if isinstance(step, dict):
            step.pop("_global_numeric_range", None)

    # Deduplicate unresolved semantics while preserving order.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in unresolved_semantics:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        deduped.append(item)

    if not repaired_steps and deduped:
        for item in deduped:
            repaired_steps.append(
                {
                    "id": item["id"],
                    "kind": "unresolved",
                    "description": item["description"],
                    "semantic_role": item["description"],
                    "confidence": 0.5,
                    "source_text": item["source_text"],
                    "reason": item["reason"],
                    "needs_library_extension": True,
                }
            )

    repaired["steps"] = repaired_steps
    repaired["unresolved_semantics"] = deduped
    repaired["needs_library_extension"] = bool(repaired.get("needs_library_extension")) or bool(deduped)
    repaired.setdefault("version", "1.0")
    repaired.setdefault("assumptions", [])
    return repaired


def _needs_static_repair(plan: dict[str, Any], series_length: int) -> bool:
    if validate_payload("plan", plan)["status"] != "valid":
        return True
    if plan.get("global_context", {}).get("length") != series_length:
        return True
    description = str(plan.get("input_description") or "")
    if _recover_workflow_failure_steps(plan, description, series_length) is not None:
        return True
    if _needs_global_baseline_repair(plan, description, series_length):
        return True
    if _needs_level_shift_baseline_repair(plan, description, series_length):
        return True
    primitive_names = {str(step.get("primitive")) for step in plan.get("steps", []) if isinstance(step, dict)}
    if _text_forbids_fresh_noise(description) and "add_volatility" in primitive_names and "add_noise" in primitive_names:
        return True
    after_timestep = _extract_after_timestep(description)
    steps = plan.get("steps", [])
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            return True
        if step.get("kind") == "unresolved":
            continue
        primitive = step.get("primitive")
        if primitive not in PRIMITIVE_MACHINE_METADATA:
            return True
        if step.get("effect_type") != PRIMITIVE_MACHINE_METADATA[primitive]["effect_type"]:
            return True
        expected_layer = DEFAULT_LAYER_BY_PRIMITIVE.get(primitive)
        persistent_flat_baseline = _is_persistent_flat_baseline_for_level_shift(
            step,
            step_index=step_index,
            steps=steps,
            description=description,
            series_length=series_length,
        )
        if (
            expected_layer is not None
            and step.get("semantic_layer") != expected_layer
            and not (persistent_flat_baseline and step.get("semantic_layer") == "global_scaffold")
        ):
            return True
        params = step.get("parameters", {})
        if primitive == "add_noise" and "random_seed" not in params and "random_generator" not in params:
            return True
        if after_timestep is not None and "start_timestep" in params and int(params["start_timestep"]) == after_timestep + 1:
            return True
        source_text = str(step.get("source_text") or description)
        if primitive in {"add_spike", "add_dip", "add_peak", "add_trough", "add_outlier"} and "timestep" in params:
            source_timestep = _extract_source_timestep(source_text)
            if source_timestep is not None and int(params["timestep"]) != source_timestep:
                return True
        if primitive in {"add_spike", "add_dip", "add_peak", "add_trough", "add_outlier"} and "target_value" not in params:
            if _extract_explicit_target_value(source_text) is not None:
                return True
        if primitive == "add_gap" and "fill_value" in params and _text_requests_missing_values(source_text) and not _text_has_explicit_gap_fill(source_text):
            return True
        if (
            primitive == "add_noise"
            and "noise_scale" in params
            and float(params["noise_scale"]) > 0.5
            and not _text_has_explicit_noise_scale(description)
            and ("jitter" in description.lower() or "noisy" in description.lower())
        ):
            return True
        if primitive in {"add_spike", "add_dip"} and "target_value" not in params and "amplitude" in params:
            numeric_range = plan.get("global_context", {}).get("numeric_range")
            if isinstance(numeric_range, dict) and "min" in numeric_range and "max" in numeric_range:
                low = float(numeric_range["min"])
                high = float(numeric_range["max"])
                if 0.0 <= high - low <= 1.0 and ("impulse" in description.lower() or "spike" in description.lower()):
                    return True
    return False


def repair_plan(plan: dict[str, Any], *, series_length: int, max_iterations: int = 2) -> RepairResult:
    repaired = deepcopy(plan)
    changes: list[str] = []
    status = validate_payload("plan", repaired)["status"]
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        if not _needs_static_repair(repaired, series_length):
            break
        repaired = _repair_once(repaired, series_length, changes)
        iterations = iteration
        status = validate_payload("plan", repaired)["status"]
    return RepairResult(plan=repaired, iterations=iterations, changes=changes, schema_status=status)
