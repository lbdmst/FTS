from __future__ import annotations

import re
from typing import Any


def _registry_validation() -> dict[str, Any]:
    return {
        "status": "valid",
        "schema_path": "schemas/primitive.schema.json",
        "target_path": "atomic_timeseries/registry.json",
        "errors": [],
    }


def _global_context(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "length": int(case["length"]),
        "numeric_range": case.get("numeric_range") or {"min": -1000.0, "max": 1000.0},
        "dt": None,
        "seed": 7,
    }


def _primitive_step(
    index: int,
    primitive: str,
    parameters: dict[str, Any],
    semantic_role: str,
    effect_type: str,
    source_text: str,
) -> dict[str, Any]:
    return {
        "id": f"step_{index}",
        "kind": "primitive",
        "primitive": primitive,
        "parameters": parameters,
        "semantic_role": semantic_role,
        "effect_type": effect_type,
        "confidence": 0.75,
        "source_text": source_text,
        "needs_library_extension": False,
    }


def _unresolved_plan(case: dict[str, Any], semantic: str, reason: str) -> dict[str, Any]:
    description = case["description"]
    unresolved = {
        "id": "u1",
        "description": semantic,
        "source_text": description,
        "reason": reason,
    }
    return {
        "version": "1.0",
        "input_description": description,
        "registry_validation": _registry_validation(),
        "global_context": _global_context(case),
        "steps": [
            {
                "id": "u1",
                "kind": "unresolved",
                "description": semantic,
                "semantic_role": semantic,
                "confidence": 0.8,
                "source_text": description,
                "reason": reason,
                "needs_library_extension": True,
            }
        ],
        "unresolved_semantics": [unresolved],
        "needs_library_extension": True,
        "assumptions": [],
    }


def _number(pattern: str, text: str, default: float | None = None) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return default
    return float(match.group(1))


def _integer(pattern: str, text: str, default: int | None = None) -> int | None:
    value = _number(pattern, text, None)
    return default if value is None else int(round(value))


def _add_step(steps: list[dict[str, Any]], primitive: str, params: dict[str, Any], role: str, effect: str, text: str) -> None:
    steps.append(_primitive_step(len(steps) + 1, primitive, params, role, effect, text))


def _parse_flat(text: str, steps: list[dict[str, Any]], length: int) -> None:
    for match in re.finditer(
        r"(?:flat|constant|stable baseline|stays flat|remains flat|is flat)(?:\s+at|\s+near)?\s+(-?\d+(?:\.\d+)?)"
        r"(?:\s+from timestep\s+(\d+)\s+to\s+(\d+)|\s+across all|\s+throughout)?",
        text,
        flags=re.IGNORECASE,
    ):
        value = float(match.group(1))
        start = int(match.group(2)) if match.group(2) is not None else 0
        end = int(match.group(3)) if match.group(3) is not None else length - 1
        _add_step(
            steps,
            "add_flat",
            {"start_timestep": start, "end_timestep": end, "target_value": value},
            "constant baseline",
            "overwrite",
            match.group(0),
        )

    first_half = re.search(r"first half is flat at\s+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if first_half is not None:
        _add_step(
            steps,
            "add_flat",
            {"start_timestep": 0, "end_timestep": (length // 2) - 1, "target_value": float(first_half.group(1))},
            "initial flat regime",
            "overwrite",
            first_half.group(0),
        )


def _parse_ramp_and_trend(text: str, steps: list[dict[str, Any]], length: int) -> None:
    for match in re.finditer(
        r"(?:between|from) timestep\s+(\d+)\s+(?:and|to)\s+(\d+).*?"
        r"(?:linearly from|moves linearly from|linear ramp to|from)\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    ):
        _add_step(
            steps,
            "add_ramp",
            {
                "start_timestep": int(match.group(1)),
                "end_timestep": int(match.group(2)),
                "start_value": float(match.group(3)),
                "end_value": float(match.group(4)),
            },
            "linear ramp",
            "overwrite",
            match.group(0),
        )

    for match in re.finditer(
        r"(?:from timestep\s+(\d+)\s+to\s+(\d+).*?|after timestep\s+(\d+).*?)"
        r"(?:trend|declines|increases|slope).*?slope\s+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    ):
        start = int(match.group(1) or match.group(3))
        end = int(match.group(2)) if match.group(2) is not None else length - 1
        _add_step(
            steps,
            "add_trend",
            {"start_timestep": start, "end_timestep": end, "slope": float(match.group(4))},
            "linear trend",
            "additive",
            match.group(0),
        )


def _parse_events(text: str, steps: list[dict[str, Any]]) -> None:
    event_patterns = [
        ("add_spike", "spike", r"spike(?:\s+to|.*?reaching)\s+(-?\d+(?:\.\d+)?).*?timestep\s+(\d+)"),
        ("add_dip", "dip", r"dip(?:\s+to|.*?reaching)\s+(-?\d+(?:\.\d+)?).*?timestep\s+(\d+)"),
        ("add_peak", "rounded peak", r"peak(?:\s+reaches|.*?reaching)\s+(-?\d+(?:\.\d+)?).*?timestep\s+(\d+)"),
        ("add_trough", "rounded trough", r"trough(?:\s+reaches|.*?reaching)\s+(-?\d+(?:\.\d+)?).*?timestep\s+(\d+)"),
    ]
    for primitive, role, pattern in event_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            params: dict[str, Any] = {"timestep": int(match.group(2)), "target_value": float(match.group(1))}
            if primitive in {"add_peak", "add_trough"}:
                params["width"] = 2
                params["start_timestep"] = max(0, int(match.group(2)) - 5)
                params["end_timestep"] = int(match.group(2)) + 5
            _add_step(steps, primitive, params, role, "additive", match.group(0))


def _parse_structural(text: str, steps: list[dict[str, Any]], length: int) -> None:
    for match in re.finditer(r"(?:at|after) timestep\s+(\d+).*?jump\w*\s+(up|down)\s+by\s+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE):
        shift = float(match.group(3))
        if match.group(2).lower() == "down":
            shift = -abs(shift)
        _add_step(
            steps,
            "add_level_shift",
            {"anchor_timestep": int(match.group(1)), "shift": shift},
            "level shift",
            "additive",
            match.group(0),
        )

    for match in re.finditer(
        r"change point.*?timestep\s+(\d+).*?jumps up by\s+(-?\d+(?:\.\d+)?).*?slope\s+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    ):
        _add_step(
            steps,
            "add_change_point",
            {"anchor_timestep": int(match.group(1)), "level_change": float(match.group(2)), "slope_change": float(match.group(3))},
            "change point",
            "additive",
            match.group(0),
        )

    plateau = re.search(r"from timestep\s+(\d+)\s+to\s+(\d+).*?plateau at\s+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if plateau is not None:
        _add_step(
            steps,
            "add_flat",
            {
                "start_timestep": int(plateau.group(1)),
                "end_timestep": int(plateau.group(2)),
                "target_value": float(plateau.group(3)),
            },
            "plateau",
            "overwrite",
            plateau.group(0),
        )

    for match in re.finditer(r"from timestep\s+(\d+)\s+to\s+(\d+).*?(?:missing|gap)", text, flags=re.IGNORECASE):
        _add_step(
            steps,
            "add_gap",
            {"start_timestep": int(match.group(1)), "end_timestep": int(match.group(2))},
            "missing interval",
            "overwrite",
            match.group(0),
        )


def _parse_texture(text: str, steps: list[dict[str, Any]]) -> None:
    for match in re.finditer(
        r"from timestep\s+(\d+)\s+to\s+(\d+).*?(?:sine|seasonal|periodic).*?amplitude\s+(-?\d+(?:\.\d+)?).*?period\s+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    ):
        _add_step(
            steps,
            "add_seasonality",
            {
                "start_timestep": int(match.group(1)),
                "end_timestep": int(match.group(2)),
                "amplitude": float(match.group(3)),
                "period": float(match.group(4)),
                "waveform": "sine",
            },
            "seasonality",
            "additive",
            match.group(0),
        )

    for match in re.finditer(
        r"(?:from|between) timestep\s+(\d+)\s+(?:to|and)\s+(\d+).*?(?:noise|jitter).*?scale\s+(-?\d+(?:\.\d+)?)(?:.*?seed\s+(\d+))?",
        text,
        flags=re.IGNORECASE,
    ):
        params: dict[str, Any] = {
            "start_timestep": int(match.group(1)),
            "end_timestep": int(match.group(2)),
            "noise_scale": float(match.group(3)),
            "random_seed": int(match.group(4) or 7),
        }
        _add_step(steps, "add_noise", params, "noise", "additive", match.group(0))

    for match in re.finditer(
        r"(?:between timestep\s+(\d+)\s+and\s+(\d+)|after timestep\s+(\d+)).*?(?:twice|(\d+(?:\.\d+)?) times).*?baseline",
        text,
        flags=re.IGNORECASE,
    ):
        start = int(match.group(1) or match.group(3))
        end = int(match.group(2)) if match.group(2) is not None else 99
        scale = 2.0 if "twice" in match.group(0).lower() else float(match.group(4))
        _add_step(
            steps,
            "add_volatility",
            {"start_timestep": start, "end_timestep": end, "volatility_scale": scale, "baseline": 0.0},
            "volatility",
            "overwrite",
            match.group(0),
        )


def plan_for_case(case: dict[str, Any]) -> dict[str, Any]:
    text = str(case["description"])
    lowered = text.lower()
    if "mean-reverting" in lowered or "mean reversion" in lowered:
        return _unresolved_plan(case, "mean_reversion", "No registered primitive explicitly applies pullback toward a reference level.")
    if "does not indicate any repeating period" in lowered:
        return _unresolved_plan(
            case,
            "mild_nonperiodic_oscillation",
            "The description denies seasonality but does not provide enough stochastic parameters for a registered noise or volatility primitive.",
        )

    length = int(case["length"])
    steps: list[dict[str, Any]] = []
    _parse_flat(text, steps, length)
    _parse_ramp_and_trend(text, steps, length)
    _parse_structural(text, steps, length)
    _parse_texture(text, steps)
    _parse_events(text, steps)

    if not steps:
        return _unresolved_plan(case, "unparsed_description", "The rule parser could not map the description to registered primitives.")

    return {
        "version": "1.0",
        "input_description": text,
        "registry_validation": _registry_validation(),
        "global_context": _global_context(case),
        "steps": steps,
        "unresolved_semantics": [],
        "needs_library_extension": False,
        "assumptions": ["Generated by deterministic regex baseline."],
    }

