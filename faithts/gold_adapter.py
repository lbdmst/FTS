from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PRIMITIVE_TO_SEMANTIC = {
    "add_flat": ("flat", "check_flat"),
    "add_plateau": ("flat", "check_flat"),
    "add_growth": ("growth", "check_growth"),
    "add_decay": ("decay", "check_decay"),
    "add_ramp": ("linear_ramp", "check_ramp"),
    "add_trend": ("trend", "check_trend"),
    "add_spike": ("spike", "check_spike"),
    "add_dip": ("dip", "check_dip"),
    "add_outlier": ("outlier", "check_outlier"),
    "add_peak": ("peak", "check_peak"),
    "add_trough": ("trough", "check_trough"),
    "add_level_shift": ("level_shift", "check_level_shift"),
    "add_change_point": ("change_point", "check_change_point"),
    "add_seasonality": ("seasonality", "check_seasonality"),
    "add_noise": ("noise", "check_noise"),
    "add_volatility": ("volatility", "check_volatility"),
    "add_gap": ("gap", "check_gap"),
}

SEMANTICALLY_EQUIVALENT_PRIMITIVES = {
    "add_flat": ["add_flat", "add_plateau"],
    "add_plateau": ["add_flat", "add_plateau"],
}


def load_gold_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload in {path}")
        cases.extend(payload)
    return cases


def step_window(step: dict[str, Any], length: int) -> list[int] | None:
    params = step.get("parameters", {})
    primitive = step.get("primitive")
    if "start_timestep" in params and "end_timestep" in params:
        return [int(params["start_timestep"]), int(params["end_timestep"])]
    if "anchor_timestep" in params:
        return [int(params["anchor_timestep"]), int(params.get("end_timestep", length - 1))]
    if "timestep" in params:
        timestep = int(params["timestep"])
        if primitive in {"add_spike", "add_dip"}:
            width = int(params.get("half_width", params.get("width", 0)))
        else:
            width = int(round(float(params.get("width", params.get("spread", 0)))))
        return [max(0, timestep - width), min(length - 1, timestep + width)]
    if "start_timestep" in params or "end_timestep" in params:
        return [int(params.get("start_timestep", 0)), int(params.get("end_timestep", length - 1))]
    return None


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _numbers_in_text(text: str) -> list[float]:
    numbers: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        try:
            numbers.append(float(match.group(0)))
        except ValueError:
            continue
    return numbers


def _number_is_mentioned(value: Any, text: str, *, allow_magnitude: bool = False) -> bool:
    try:
        expected = float(value)
    except (TypeError, ValueError):
        return False
    tolerance = max(1e-6, abs(expected) * 1e-6)
    for number in _numbers_in_text(text):
        if abs(number - expected) <= tolerance:
            return True
        if allow_magnitude and abs(abs(number) - abs(expected)) <= tolerance:
            return True
    return False


def _clause_mentions_value(text: str, verb_pattern: str, cue: str, value: Any) -> bool:
    try:
        expected = float(value)
    except (TypeError, ValueError):
        return False
    lowered = text.lower()
    verb_match = re.search(rf"(?:{verb_pattern})", lowered, re.IGNORECASE)
    if verb_match is None:
        return False
    tail = lowered[verb_match.start() : verb_match.start() + 240]
    pattern = re.compile(rf"\b{cue}\s+({_NUMBER_RE.pattern})", re.IGNORECASE)
    for match in pattern.finditer(tail):
        try:
            actual = float(match.group(1))
        except ValueError:
            continue
        if abs(actual - expected) <= max(1e-6, abs(expected) * 1e-6):
            return True
    return False


def _parameter_is_explicit(primitive: str, key: str, value: Any, description: str) -> bool:
    lowered = description.lower()
    if key in {"start_timestep", "end_timestep", "timestep", "anchor_timestep"}:
        return _number_is_mentioned(value, description)
    if key in {"width", "half_width", "spread"}:
        return "width" in lowered or "spread" in lowered
    if key in {"growth_rate", "decay_rate"}:
        cue = key.replace("_", " ")
        return cue in lowered and _number_is_mentioned(value, description)
    if key == "random_seed":
        return "seed" in lowered and _number_is_mentioned(value, description)
    if key == "fill_value":
        return ("fill" in lowered or "filled" in lowered) and (
            str(value).lower() == "nan" or _number_is_mentioned(value, description)
        )
    if primitive == "add_decay" and key == "start_value":
        return _clause_mentions_value(description, r"decays?|decay", "from", value)
    if primitive == "add_decay" and key == "end_value":
        return _clause_mentions_value(description, r"decays?|decay", "to", value)
    if primitive == "add_growth" and key == "start_value":
        return _clause_mentions_value(description, r"grows?|growth", "from", value)
    if primitive == "add_growth" and key == "end_value":
        return _clause_mentions_value(description, r"grows?|growth", "to", value)
    if key in {"slope", "shift", "level_change", "slope_change", "relative_change", "delta"}:
        directional = any(
            cue in lowered
            for cue in ["down", "downward", "lower", "negative", "decline", "decrease", "drop", "up", "upward", "raise", "increase", "positive"]
        )
        return _number_is_mentioned(value, description, allow_magnitude=directional)
    if key in {"target_value", "start_value", "end_value", "amplitude", "period", "phase", "baseline", "noise_scale", "volatility_scale"}:
        return _number_is_mentioned(value, description)
    return True


def _feature_parameters(primitive: str, params: dict[str, Any], description: str) -> dict[str, Any]:
    explicit: dict[str, Any] = {}
    for key, value in params.items():
        if _parameter_is_explicit(primitive, key, value, description):
            explicit[key] = value
    return explicit


def _semantic_layer_for_gold_step(step: dict[str, Any], length: int) -> str | None:
    explicit = step.get("semantic_layer")
    if explicit:
        return str(explicit)
    if step.get("primitive") != "add_flat":
        return None
    params = step.get("parameters", {})
    role = str(step.get("semantic_role") or "").lower()
    if "baseline" not in role:
        return None
    try:
        start = int(params.get("start_timestep", 0))
        end = int(params.get("end_timestep", -1))
    except (TypeError, ValueError):
        return None
    if start == 0 and end == length - 1:
        return "global_scaffold"
    return None


def semantic_spec_from_gold_case(case: dict[str, Any]) -> dict[str, Any]:
    outline = case.get("expected_plan_outline", {})
    unresolved = outline.get("unresolved_semantics", [])
    unsupported_semantics = []
    for index, item in enumerate(unresolved, start=1):
        semantic = str(item.get("semantic") or item.get("description") or f"unsupported_{index}")
        unsupported_semantics.append(
            {
                "id": str(item.get("id") or f"u{index}"),
                "description": semantic,
                "reason": str(item.get("reason") or "No registered primitive covers this semantic."),
            }
        )

    features = []
    for index, step in enumerate(outline.get("steps", []), start=1):
        primitive = step["primitive"]
        if primitive not in PRIMITIVE_TO_SEMANTIC:
            continue
        semantic, checker = PRIMITIVE_TO_SEMANTIC[primitive]
        features.append(
            {
                "id": f"f{index}",
                "semantic": semantic,
                "expected_primitives": SEMANTICALLY_EQUIVALENT_PRIMITIVES.get(primitive, [primitive]),
                "window": step_window(step, int(case["length"])),
                "parameters": _feature_parameters(primitive, step.get("parameters", {}), case["description"]),
                "effect_type": step.get("effect_type"),
                "checker": checker,
                "required": True,
            }
        )

    for index, forbidden in enumerate(case.get("assertions", {}).get("must_not_retrieve", []), start=1):
        features.append(
            {
                "id": f"neg{index}",
                "semantic": "negation",
                "expected_primitives": [],
                "window": [0, int(case["length"]) - 1],
                "parameters": {"forbidden_primitives": [forbidden]},
                "effect_type": None,
                "checker": "check_negation",
                "required": True,
            }
        )

    return {
        "case_id": case["case_id"],
        "description": case["description"],
        "length": int(case["length"]),
        "features": features,
        "unsupported_semantics": unsupported_semantics,
        "numeric_range": case.get("numeric_range"),
        "metadata": {"category": case.get("category", "unknown")},
    }


def oracle_plan_from_gold_case(case: dict[str, Any]) -> dict[str, Any]:
    outline = case.get("expected_plan_outline", {})
    steps = []
    for index, step in enumerate(outline.get("steps", []), start=1):
        payload = {
            "id": f"step_{index}",
            "kind": "primitive",
            "primitive": step["primitive"],
            "parameters": dict(step.get("parameters", {})),
            "semantic_role": str(step.get("semantic_role") or step["primitive"]),
            "effect_type": str(step["effect_type"]),
            "confidence": 1.0,
            "source_text": case["description"],
            "needs_library_extension": False,
        }
        semantic_layer = _semantic_layer_for_gold_step(step, int(case["length"]))
        if semantic_layer is not None:
            payload["semantic_layer"] = semantic_layer
        steps.append(payload)

    unresolved_semantics = []
    for index, item in enumerate(outline.get("unresolved_semantics", []), start=1):
        semantic = str(item.get("semantic") or item.get("description") or f"unsupported_{index}")
        unresolved_semantics.append(
            {
                "id": str(item.get("id") or f"u{index}"),
                "description": semantic,
                "source_text": case["description"],
                "reason": str(item.get("reason") or "No registered primitive covers this semantic."),
            }
        )
        steps.append(
            {
                "id": str(item.get("id") or f"u{index}"),
                "kind": "unresolved",
                "description": semantic,
                "semantic_role": semantic,
                "confidence": 1.0,
                "source_text": case["description"],
                "reason": str(item.get("reason") or "No registered primitive covers this semantic."),
                "needs_library_extension": True,
            }
        )

    needs_extension = bool(outline.get("needs_library_extension")) or bool(unresolved_semantics)
    return {
        "version": "1.0",
        "input_description": case["description"],
        "registry_validation": {
            "status": "valid",
            "schema_path": "schemas/primitive.schema.json",
            "target_path": "atomic_timeseries/registry.json",
            "errors": [],
        },
        "global_context": {
            "length": int(case["length"]),
            "numeric_range": case.get("numeric_range") or {"min": -1000.0, "max": 1000.0},
            "dt": None,
            "seed": None,
        },
        "steps": steps,
        "unresolved_semantics": unresolved_semantics,
        "needs_library_extension": needs_extension,
        "assumptions": [],
    }


def is_core_evaluable(case: dict[str, Any]) -> bool:
    outline = case.get("expected_plan_outline", {})
    return bool(outline.get("steps")) or bool(outline.get("needs_library_extension")) or bool(outline.get("unresolved_semantics"))
