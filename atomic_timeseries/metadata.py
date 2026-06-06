from __future__ import annotations

from typing import Any


PRIMITIVE_MACHINE_METADATA: dict[str, dict[str, Any]] = {
    "add_trend": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["slope"], ["start_value", "end_value"], ["start_offset", "end_offset"]],
        "mutually_exclusive_args": [
            ["slope", "start_value"],
            ["slope", "end_value"],
            ["slope", "start_offset"],
            ["slope", "end_offset"],
            ["start_value", "start_offset"],
            ["end_value", "end_offset"],
        ],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use `slope` for relative drift and `start_value`/`end_value` or `start_offset`/`end_offset` only when the additive baseline is intentional.",
        ],
        "forbidden_usage_patterns": [
            "Do not treat `start_value`/`end_value` or `start_offset`/`end_offset` as overwrite-style setters on a nonzero baseline.",
            "Do not pass `slope` together with endpoint offsets.",
        ],
        "setter_like_args": ["start_value", "end_value", "start_offset", "end_offset"],
    },
    "add_ramp": {
        "effect_type": "overwrite",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["end_value"], ["slope"]],
        "mutually_exclusive_args": [["end_value", "slope"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use this for exact piecewise linear segments that should overwrite the interval.",
        ],
        "forbidden_usage_patterns": [
            "Do not pass both `end_value` and `slope` in the same call.",
        ],
        "setter_like_args": ["start_value", "end_value"],
    },
    "add_flat": {
        "effect_type": "overwrite",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["target_value"]],
        "mutually_exclusive_args": [],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for exact flat intervals and plateaus.",
        ],
        "forbidden_usage_patterns": [
            "Do not expect this to preserve within-interval variation.",
        ],
        "setter_like_args": ["target_value"],
    },
    "add_growth": {
        "effect_type": "overwrite",
        "mutability": "returns_new_series",
        "required_arg_patterns": [],
        "mutually_exclusive_args": [],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for curved overwrite-style growth segments.",
        ],
        "forbidden_usage_patterns": [
            "Do not use for additive growth on top of an existing segment.",
        ],
        "setter_like_args": ["start_value", "end_value"],
    },
    "add_decay": {
        "effect_type": "overwrite",
        "mutability": "returns_new_series",
        "required_arg_patterns": [],
        "mutually_exclusive_args": [],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for curved overwrite-style decay segments.",
        ],
        "forbidden_usage_patterns": [
            "Do not use for additive decay on top of an existing segment.",
        ],
        "setter_like_args": ["start_value", "end_value"],
    },
    "add_plateau": {
        "effect_type": "overwrite",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["target_value"], ["anchor_timestep"]],
        "mutually_exclusive_args": [["target_value", "anchor_timestep"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use `anchor_timestep` only to copy an already-correct level into another interval.",
        ],
        "forbidden_usage_patterns": [
            "Do not pass both `target_value` and `anchor_timestep`.",
        ],
        "setter_like_args": ["target_value"],
    },
    "add_change_point": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["level_change"], ["slope_change"]],
        "mutually_exclusive_args": [],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for post-anchor additive shifts rather than absolute level setting.",
        ],
        "forbidden_usage_patterns": [
            "Do not use as an overwrite-style plateau constructor.",
        ],
        "setter_like_args": [],
    },
    "add_seasonality": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["amplitude", "period"]],
        "mutually_exclusive_args": [],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use only when the caption truly describes repeating periodic structure.",
        ],
        "forbidden_usage_patterns": [
            "Do not use for irregular noise or one-off oscillations.",
        ],
        "setter_like_args": ["baseline"],
    },
    "add_peak": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["amplitude"], ["target_value"]],
        "mutually_exclusive_args": [["amplitude", "target_value"], ["width", "spread"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for local positive bumps, not for sustained plateaus.",
        ],
        "forbidden_usage_patterns": [
            "Do not pass both `amplitude` and `target_value`.",
        ],
        "setter_like_args": ["target_value"],
    },
    "add_trough": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["amplitude"], ["target_value"]],
        "mutually_exclusive_args": [["amplitude", "target_value"], ["width", "spread"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for local negative bumps, not for sustained lower regimes.",
        ],
        "forbidden_usage_patterns": [
            "Do not pass both `amplitude` and `target_value`.",
        ],
        "setter_like_args": ["target_value"],
    },
    "add_spike": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["amplitude"], ["target_value"]],
        "mutually_exclusive_args": [["amplitude", "target_value"], ["width", "half_width"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for sharp upward events at one timestep or a very narrow window.",
        ],
        "forbidden_usage_patterns": [
            "Do not pass both `amplitude` and `target_value`.",
        ],
        "setter_like_args": ["target_value"],
    },
    "add_dip": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["amplitude"], ["target_value"]],
        "mutually_exclusive_args": [["amplitude", "target_value"], ["width", "half_width"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for sharp downward events at one timestep or a very narrow window.",
        ],
        "forbidden_usage_patterns": [
            "Do not pass both `amplitude` and `target_value`.",
        ],
        "setter_like_args": ["target_value"],
    },
    "add_outlier": {
        "effect_type": "overwrite",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["target_value"], ["relative_change"], ["delta"]],
        "mutually_exclusive_args": [["target_value", "relative_change"], ["target_value", "delta"], ["relative_change", "delta"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Prefer `target_value` for exact one-point values and `delta` for explicit additive offsets.",
        ],
        "forbidden_usage_patterns": [
            "Do not pass multiple one-point change modes in the same call.",
        ],
        "setter_like_args": ["target_value"],
    },
    "add_level_shift": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["start_timestep", "shift"], ["start_timestep", "target_value"], ["anchor_timestep", "shift"], ["anchor_timestep", "target_value"]],
        "mutually_exclusive_args": [["shift", "target_value"], ["start_timestep", "anchor_timestep"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use for tail shifts relative to the existing baseline, not as an overwrite plateau.",
        ],
        "forbidden_usage_patterns": [
            "Do not pass both `shift` and `target_value`.",
        ],
        "setter_like_args": ["target_value"],
    },
    "add_gap": {
        "effect_type": "overwrite",
        "mutability": "returns_new_series",
        "required_arg_patterns": [],
        "mutually_exclusive_args": [],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use only for explicit missing intervals or deliberate fill values.",
        ],
        "forbidden_usage_patterns": [
            "Do not use for normal dips or troughs.",
        ],
        "setter_like_args": ["fill_value"],
    },
    "add_noise": {
        "effect_type": "additive",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["noise_scale"]],
        "mutually_exclusive_args": [["random_seed", "random_generator"]],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use only with an explicit `random_seed` when stochastic behavior is necessary.",
        ],
        "forbidden_usage_patterns": [
            "Do not omit `random_seed` in generated code.",
        ],
        "setter_like_args": [],
    },
    "add_volatility": {
        "effect_type": "overwrite",
        "mutability": "returns_new_series",
        "required_arg_patterns": [["volatility_scale"]],
        "mutually_exclusive_args": [],
        "recommended_usage": [
            "Assign the returned array back to `series`.",
            "Use to rescale existing deviations around a known baseline.",
        ],
        "forbidden_usage_patterns": [
            "Do not use on a flat zero baseline expecting it to create variation.",
        ],
        "setter_like_args": ["baseline"],
    },
}


def get_primitive_machine_metadata(name: str) -> dict[str, Any]:
    try:
        return PRIMITIVE_MACHINE_METADATA[name]
    except KeyError as exc:
        raise KeyError(f"Unknown primitive metadata request: {name}") from exc
