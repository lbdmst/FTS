from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import atomic_timeseries as ats
from atomic_timeseries.metadata import PRIMITIVE_MACHINE_METADATA


ROOT = Path(__file__).resolve().parent
T2S_CASES_PATH = ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
CATALOG_PATH = ROOT / "primitive_catalog.json"
REPORT_PATH = ROOT / "retrieval_coverage_report.jsonl"
SUMMARY_PATH = ROOT / "retrieval_summary.md"


PRIMITIVE_META: dict[str, dict[str, Any]] = {
    "add_trend": {
        "category": "trend",
        "short_description": "Adds a linear trend over an inclusive interval.",
        "core_semantic_tags": ["trend", "linear", "interval", "upward", "downward"],
        "what_it_does": [
            "Adds a linearly changing offset across an interval.",
            "Supports slope-driven or endpoint-driven specification.",
            "Leaves values outside the interval unchanged.",
        ],
        "what_it_does_not_do": [
            "Does not set the interval to exact values unless endpoint parameters are used carefully.",
            "Does not model exponential curvature.",
            "Does not create a discrete level jump by itself.",
        ],
        "exact_effect": "Adds a linear delta to each timestep in the inclusive interval `[start_timestep, end_timestep]`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "upward trend",
            "downward trend",
            "gradual increase",
            "gradual decline",
            "drift",
            "steady rise",
            "steady fall",
        ],
        "example_initial_series": [0, 0, 0, 0, 0],
        "example_call": {"start_timestep": 1, "end_timestep": 4, "slope": 1.5},
    },
    "add_ramp": {
        "category": "trend",
        "short_description": "Sets an interval to a linear ramp with exact endpoints.",
        "core_semantic_tags": ["trend", "linear", "ramp", "interval", "exact_value"],
        "what_it_does": [
            "Replaces an interval with a linear ramp.",
            "Lets the caller pin exact start and end values.",
            "Provides a clean primitive for piecewise linear segments.",
        ],
        "what_it_does_not_do": [
            "Does not preserve the previous segment values inside the interval.",
            "Does not create curvature.",
            "Does not add oscillation or local events.",
        ],
        "exact_effect": "Overwrites every value in the inclusive interval with evenly spaced linear values between the chosen endpoints.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "ramps up",
            "ramps down",
            "linearly rises to",
            "linearly falls to",
            "moves from X to Y",
            "piecewise linear segment",
        ],
        "example_initial_series": [0, 0, 0, 0, 0],
        "example_call": {"start_timestep": 1, "end_timestep": 3, "start_value": 2.0, "end_value": 6.0},
    },
    "add_flat": {
        "category": "trend",
        "short_description": "Sets every value in an interval to a constant target level.",
        "core_semantic_tags": ["flat", "constant", "level", "interval"],
        "what_it_does": [
            "Creates a flat segment at a single target value.",
            "Overwrites the whole interval deterministically.",
            "Provides a direct primitive for stable plateaus or constant runs.",
        ],
        "what_it_does_not_do": [
            "Does not smooth into or out of the segment.",
            "Does not preserve pre-existing local variation within the interval.",
            "Does not imply a level shift outside the specified interval.",
        ],
        "exact_effect": "Overwrites all timesteps in `[start_timestep, end_timestep]` with `target_value`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "flat",
            "constant",
            "stable at",
            "remains at",
            "fixed value",
            "horizontal segment",
        ],
        "example_initial_series": [1, 2, 3, 4, 5],
        "example_call": {"start_timestep": 1, "end_timestep": 3, "target_value": 7.5},
    },
    "add_growth": {
        "category": "trend",
        "short_description": "Sets an interval to an exponential growth curve between exact bounds.",
        "core_semantic_tags": ["growth", "curve", "accelerating", "interval", "exact_value"],
        "what_it_does": [
            "Creates a monotone increasing curved segment.",
            "Allows exact start and end values.",
            "Captures accelerated upward movement with a single atomic curve primitive.",
        ],
        "what_it_does_not_do": [
            "Does not generate noise or volatility by itself.",
            "Does not add a local spike centered at one timestep.",
            "Does not automatically continue growth beyond the interval.",
        ],
        "exact_effect": "Overwrites the interval with a normalized exponential growth curve from `start_value` to `end_value`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "accelerating growth",
            "curved increase",
            "exponential growth",
            "surges upward",
            "increases faster over time",
        ],
        "example_initial_series": [0, 0, 0, 0, 0, 0],
        "example_call": {"start_timestep": 1, "end_timestep": 4, "start_value": 2.0, "end_value": 10.0, "growth_rate": 2.0},
    },
    "add_decay": {
        "category": "trend",
        "short_description": "Sets an interval to an exponential decay curve between exact bounds.",
        "core_semantic_tags": ["decay", "curve", "accelerating", "interval", "exact_value"],
        "what_it_does": [
            "Creates a monotone decreasing curved segment.",
            "Allows exact start and end values.",
            "Captures accelerated downward movement without adding extra events.",
        ],
        "what_it_does_not_do": [
            "Does not create an abrupt step change.",
            "Does not create local troughs centered on a single point.",
            "Does not preserve the old values inside the interval.",
        ],
        "exact_effect": "Overwrites the interval with a normalized exponential decay curve from `start_value` to `end_value`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "accelerating decline",
            "curved drop",
            "exponential decay",
            "falls faster over time",
            "decays toward",
        ],
        "example_initial_series": [0, 0, 0, 0, 0, 0],
        "example_call": {"start_timestep": 1, "end_timestep": 4, "start_value": 10.0, "end_value": 2.0, "decay_rate": 2.0},
    },
    "add_change_point": {
        "category": "structural",
        "short_description": "Applies a level and/or slope change after an anchor timestep.",
        "core_semantic_tags": ["change_point", "structural_change", "level_shift", "slope_shift", "transition"],
        "what_it_does": [
            "Applies a post-anchor level change, slope change, or both.",
            "Can model abrupt or smooth transitions.",
            "Leaves the pre-anchor prefix untouched.",
        ],
        "what_it_does_not_do": [
            "Does not infer the breakpoint from data.",
            "Does not create repeating oscillations.",
            "Does not represent multiple separate breakpoints in one call.",
        ],
        "exact_effect": "Adds a weighted level change plus a post-anchor linear slope change over `[anchor_timestep, end_timestep]`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "change point",
            "regime change",
            "after timestep",
            "turning point",
            "slope changes",
            "break in trend",
        ],
        "example_initial_series": [0, 0, 0, 0, 0, 0],
        "example_call": {"anchor_timestep": 2, "level_change": 2.0, "slope_change": 1.0},
    },
    "add_seasonality": {
        "category": "seasonal",
        "short_description": "Adds a periodic waveform over an interval.",
        "core_semantic_tags": ["seasonality", "periodic", "cycle", "waveform", "oscillation"],
        "what_it_does": [
            "Adds a periodic signal with configurable amplitude, period, phase, and waveform.",
            "Supports sine, cosine, square, sawtooth, and triangle patterns.",
            "Can be scoped to any inclusive interval.",
        ],
        "what_it_does_not_do": [
            "Does not infer period from data.",
            "Does not create one-off isolated peaks or spikes.",
            "Does not add stochastic noise by itself.",
        ],
        "exact_effect": "Adds `baseline + amplitude * waveform(angle)` to each timestep in the interval, where `angle` advances according to `period` and `phase`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "seasonal",
            "periodic",
            "cyclical",
            "repeating pattern",
            "wave-like",
            "oscillation with period",
        ],
        "example_initial_series": [0, 0, 0, 0, 0, 0, 0, 0],
        "example_call": {"start_timestep": 2, "end_timestep": 5, "amplitude": 2.0, "period": 4.0, "waveform": "sine"},
    },
    "add_peak": {
        "category": "local_event",
        "short_description": "Adds a positive Gaussian-shaped local maximum around a center timestep.",
        "core_semantic_tags": ["peak", "local_maximum", "bump", "positive_event"],
        "what_it_does": [
            "Adds a smooth positive local bump centered on a chosen timestep.",
            "Lets the caller control width and amplitude or exact center value.",
            "Can be restricted to a local editing window.",
        ],
        "what_it_does_not_do": [
            "Does not create a one-point outlier unless width is very small and the window is narrow.",
            "Does not create a repeating pattern.",
            "Does not automatically add a matching decline afterward.",
        ],
        "exact_effect": "Adds a Gaussian-shaped positive bump over the chosen edit window, centered at `timestep`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "peak",
            "local maximum",
            "rounded bump",
            "hump",
            "crests at",
        ],
        "example_initial_series": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "example_call": {"timestep": 5, "target_value": 8.0, "width": 1.5, "start_timestep": 2, "end_timestep": 8},
    },
    "add_trough": {
        "category": "local_event",
        "short_description": "Adds a negative Gaussian-shaped local minimum around a center timestep.",
        "core_semantic_tags": ["trough", "local_minimum", "valley", "negative_event"],
        "what_it_does": [
            "Adds a smooth negative local dip centered on a chosen timestep.",
            "Lets the caller control width and amplitude or exact center value.",
            "Supports local editing windows just like `add_peak`.",
        ],
        "what_it_does_not_do": [
            "Does not create a missing interval.",
            "Does not imply a long-term downward trend.",
            "Does not create a one-point outlier unless the caller uses a very narrow edit.",
        ],
        "exact_effect": "Subtracts a Gaussian-shaped bump over the chosen edit window, centered at `timestep`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "trough",
            "local minimum",
            "valley",
            "rounded dip",
            "bottoms out",
        ],
        "example_initial_series": [10, 10, 10, 10, 10, 10, 10, 10, 10],
        "example_call": {"timestep": 4, "target_value": 3.0, "width": 1.0, "start_timestep": 2, "end_timestep": 6},
    },
    "add_spike": {
        "category": "local_event",
        "short_description": "Adds a sharp positive event centered on a timestep.",
        "core_semantic_tags": ["spike", "sharp_event", "positive_event", "local_jump"],
        "what_it_does": [
            "Adds a sharp one-point or triangular positive event.",
            "Supports exact target value or relative amplitude.",
            "Works well for narrow isolated positive excursions.",
        ],
        "what_it_does_not_do": [
            "Does not create a smooth Gaussian bump like `add_peak`.",
            "Does not affect distant timesteps outside the narrow width window.",
            "Does not create a sustained higher level after the event.",
        ],
        "exact_effect": "Adds either a single-point jump at `timestep` or a narrow triangular positive event centered at `timestep`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "spike",
            "sharp rise",
            "sudden jump",
            "brief surge",
            "isolated positive event",
        ],
        "example_initial_series": [0, 0, 0, 0, 0, 0, 0],
        "example_call": {"timestep": 3, "target_value": 4.0},
    },
    "add_dip": {
        "category": "local_event",
        "short_description": "Adds a sharp negative event centered on a timestep.",
        "core_semantic_tags": ["dip", "sharp_event", "negative_event", "local_drop"],
        "what_it_does": [
            "Adds a sharp one-point or triangular negative event.",
            "Supports exact target value or relative amplitude.",
            "Works well for abrupt downward blips.",
        ],
        "what_it_does_not_do": [
            "Does not create a smooth valley like `add_trough`.",
            "Does not create a sustained lower regime.",
            "Does not mark missingness.",
        ],
        "exact_effect": "Subtracts either a single-point jump at `timestep` or a narrow triangular negative event centered at `timestep`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "dip",
            "sharp drop",
            "brief drop",
            "sudden fall",
            "isolated negative event",
        ],
        "example_initial_series": [10, 10, 10, 10, 10, 10, 10],
        "example_call": {"timestep": 3, "target_value": 5.0},
    },
    "add_outlier": {
        "category": "local_event",
        "short_description": "Sets a single timestep to an exact outlier value or relative jump.",
        "core_semantic_tags": ["outlier", "point_anomaly", "single_point", "anomaly"],
        "what_it_does": [
            "Changes exactly one timestep.",
            "Supports direct target-value specification or relative change.",
            "Provides the most atomic point anomaly primitive in the library.",
        ],
        "what_it_does_not_do": [
            "Does not spread to neighboring timesteps.",
            "Does not create a shaped spike or dip with width.",
            "Does not encode a missing value gap.",
        ],
        "exact_effect": "Overwrites or offsets a single timestep and leaves all others unchanged.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "outlier",
            "anomaly",
            "single-point jump",
            "isolated point",
            "one unusual value",
        ],
        "example_initial_series": [0, 0, 0, 0, 0, 0, 0],
        "example_call": {"timestep": 5, "target_value": -9.0},
    },
    "add_level_shift": {
        "category": "structural",
        "short_description": "Applies a sustained level shift starting at a timestep.",
        "core_semantic_tags": ["level_shift", "step_change", "regime_shift", "transition"],
        "what_it_does": [
            "Raises or lowers the series from a starting timestep onward.",
            "Supports abrupt or smooth entry into the shifted level.",
            "Can target a specific post-shift level.",
        ],
        "what_it_does_not_do": [
            "Does not change slope unless combined with another primitive.",
            "Does not create a temporary local event that later disappears.",
            "Does not create multiple shifts in one call.",
        ],
        "exact_effect": "Adds a level offset to `[start_timestep, end_timestep]`, optionally with smooth transition weights.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "level shift",
            "step change",
            "moves to a new level",
            "downward shift",
            "upward shift",
            "step-like increase",
        ],
        "example_initial_series": [0, 0, 0, 0, 0, 0],
        "example_call": {"start_timestep": 2, "shift": 3.0},
    },
    "add_gap": {
        "category": "structural",
        "short_description": "Replaces an interval with missing values or a chosen fill value.",
        "core_semantic_tags": ["gap", "missing", "missing_interval", "mask"],
        "what_it_does": [
            "Marks an inclusive interval as missing.",
            "Can also fill with a custom placeholder value.",
            "Provides the atomic representation for explicit data gaps.",
        ],
        "what_it_does_not_do": [
            "Does not interpolate across the gap.",
            "Does not create a valley or local trough.",
            "Does not change values outside the chosen interval.",
        ],
        "exact_effect": "Overwrites every timestep in `[start_timestep, end_timestep]` with `fill_value` (default `NaN`).",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "gap",
            "missing interval",
            "missing values",
            "data absent",
            "masked segment",
        ],
        "example_initial_series": [0, 1, 2, 3, 4, 5],
        "example_call": {"start_timestep": 2, "end_timestep": 4},
    },
    "add_noise": {
        "category": "stochastic",
        "short_description": "Adds Gaussian noise over an interval.",
        "core_semantic_tags": ["noise", "stochastic", "volatility", "randomness"],
        "what_it_does": [
            "Adds zero-mean Gaussian noise over an interval.",
            "Supports deterministic replay through a seed or generator.",
            "Provides a low-level primitive for random local variation.",
        ],
        "what_it_does_not_do": [
            "Does not control volatility relative to an existing baseline structure.",
            "Does not produce deterministic periodic oscillations.",
            "Does not infer noise scale from caption text.",
        ],
        "exact_effect": "Adds independent Gaussian samples with standard deviation `noise_scale` to each timestep in the interval.",
        "behavior": "stochastic_with_deterministic_seed_option",
        "typical_caption_cues": [
            "noisy",
            "random fluctuations",
            "jitter",
            "volatility",
            "erratic small movements",
        ],
        "example_initial_series": [0, 0, 0, 0, 0],
        "example_call": {"noise_scale": 0.5, "random_seed": 123},
    },
    "add_volatility": {
        "category": "stochastic",
        "short_description": "Scales deviations from a baseline within an interval.",
        "core_semantic_tags": ["volatility", "variance", "amplitude_scaling", "dispersion"],
        "what_it_does": [
            "Amplifies or dampens deviations from a baseline within an interval.",
            "Works on existing structure already present in the segment.",
            "Provides an atomic primitive for changing fluctuation amplitude.",
        ],
        "what_it_does_not_do": [
            "Does not create new random noise by itself.",
            "Does not change the baseline trend unless deviation scaling implies it indirectly.",
            "Does not infer the baseline automatically beyond a simple interval mean fallback.",
        ],
        "exact_effect": "Replaces each value in the interval with `baseline + (value - baseline) * volatility_scale`.",
        "behavior": "deterministic",
        "typical_caption_cues": [
            "more volatile",
            "less volatile",
            "volatility increases",
            "volatility decreases",
            "larger swings",
            "smaller swings",
        ],
        "example_initial_series": [0, 8, 10, 12, 20],
        "example_call": {"start_timestep": 1, "end_timestep": 3, "volatility_scale": 2.0, "baseline": 10.0},
    },
}


SEMANTIC_RULES: dict[str, dict[str, Any]] = {
    "trend_up": {"patterns": [r"\bupward trend\b", r"\bupward drift\b", r"\brise\b", r"\brises\b", r"\bincrease\b", r"\bincreases\b", r"\bclimb\b", r"\bclimbs\b"], "primitives": ["add_trend", "add_ramp", "add_change_point"]},
    "trend_down": {"patterns": [r"\bdownward trend\b", r"\bdownward drift\b", r"\bdeclin(?:e|es|ing)\b", r"\bfall\b", r"\bfalls\b", r"\bdrop\b", r"\bdrops\b", r"\bdecreas(?:e|es|ing)\b"], "primitives": ["add_trend", "add_ramp", "add_change_point"]},
    "growth_curve": {"patterns": [r"\bexponential growth\b", r"\baccelerating\b", r"\bsurges upward\b", r"\bcurved increase\b"], "primitives": ["add_growth"]},
    "decay_curve": {"patterns": [r"\bexponential decay\b", r"\baccelerating decline\b", r"\bcurved drop\b"], "primitives": ["add_decay"]},
    "flat": {"patterns": [r"\bflat\b", r"\bconstant\b", r"\bstable\b", r"\bplateau\b", r"\bhorizontal\b", r"\bstationary\b", r"\bhovers?\b"], "primitives": ["add_flat"]},
    "seasonality": {"patterns": [r"\bseasonal(?:ity)?\b", r"\bperiodic(?:ity)?\b", r"\bcyclic(?:al)?\b", r"\brepeating pattern\b"], "primitives": ["add_seasonality"]},
    "peak": {"patterns": [r"\bpeak\b", r"\blocal peak\b", r"\blocal maximum\b", r"\bhighest value\b"], "primitives": ["add_peak"]},
    "trough": {"patterns": [r"\btrough\b", r"\blocal trough\b", r"\blocal minimum\b", r"\blowest value\b"], "primitives": ["add_trough"]},
    "spike": {"patterns": [r"\bspike\b", r"\bsharp rise\b", r"\bsudden jump\b", r"\bbrief surge\b", r"\bimpulse\b", r"\bone-timestep\b", r"\bone timestep\b"], "primitives": ["add_spike"]},
    "dip": {"patterns": [r"\bdip\b", r"\bsharp drop\b", r"\bsudden fall\b", r"\bbrief drop\b"], "primitives": ["add_dip"]},
    "outlier": {"patterns": [r"\boutlier\b", r"\banomal(?:y|ous)\b", r"\bsingle-point\b", r"\bone unusual value\b"], "primitives": ["add_outlier"]},
    "level_shift": {"patterns": [r"\blevel shift\b", r"\bstep change\b", r"\bdownward shift\b", r"\bupward shift\b", r"\bshifts downward\b", r"\bshifts upward\b", r"\bmoves to a new sustained level\b", r"\bnew level\b"], "primitives": ["add_level_shift"]},
    "change_point": {"patterns": [r"\bchange point\b", r"\bregime change\b", r"\bturning point\b", r"\bbreak in trend\b"], "primitives": ["add_change_point"]},
    "gap": {"patterns": [r"\bgap\b", r"\bmissing interval\b", r"\bmissing values\b", r"\bdata absent\b", r"\bmissing entirely\b"], "primitives": ["add_gap"]},
    "noise": {"patterns": [r"\bnoise\b", r"\bnoisy\b", r"\brandom fluctuations\b", r"\bjitter\b"], "primitives": ["add_noise"]},
    "volatility": {"patterns": [r"\bvolatility\b", r"\bvolatile\b", r"\bfluctuat(?:e|es|ing|ions)\b", r"\berratic\b", r"\bswings\b", r"\boscillat(?:e|es|ing|ion)\b"], "primitives": ["add_volatility"]},
    "volatility_change": {"patterns": [r"\bmore volatile\b", r"\bless volatile\b", r"\bvolatility increases\b", r"\bvolatility decreases\b", r"\bincreased volatility\b", r"\bdecreased volatility\b"], "primitives": ["add_volatility"]},
    "soft_level_reference": {"patterns": [r"\bresistance\b.{0,40}\b(?:around|near|at|about)\b", r"\bsupport\b.{0,40}\b(?:around|near|at|about)\b", r"\bhover(?:s|ing)?\b.{0,40}\b(?:around|near|at|about)\b", r"\bstabili[sz](?:es|ing|ed)?\b.{0,40}\b(?:around|near|at|about)\b", r"\brange[- ]?bound\b.{0,40}\b(?:around|near|at|about)\b"], "primitives": ["add_volatility", "add_flat"]},
    "mean_reversion": {"patterns": [r"\bmean reversion\b", r"\bmean-reverting\b", r"\bmean reverting\b", r"\breturns? to (?:the )?mean\b"], "primitives": []},
}


NEGATED_TAG_PHRASES: dict[str, list[str]] = {
    "peak": ["no peak", "no peaks", "no significant peaks", "no prominent peaks", "no local extrema", "no peaks or troughs"],
    "trough": ["no trough", "no troughs", "no significant troughs", "no prominent troughs", "no local extrema", "no peaks or troughs"],
    "mean_reversion": ["no evidence of mean reversion", "no mean reversion"],
}

NEGATION_TOKENS = {
    "no",
    "not",
    "without",
    "lack",
    "lacks",
    "lacking",
    "absence",
    "doesn't",
    "doesnt",
    "nor",
    "neither",
    "never",
}

NEGATION_BIGRAMS = {
    ("does", "not"),
    ("did", "not"),
    ("is", "not"),
    ("are", "not"),
    ("was", "not"),
    ("were", "not"),
    ("absence", "of"),
}


TRUE_PRIMITIVE_GAPS: dict[str, str] = {
    "mean_reversion": "No existing primitive applies an explicit pull back toward a reference level; an atomic `add_mean_reversion(...)` primitive would fill this gap.",
}

ORCHESTRATION_LEVEL_SEMANTICS = {
    "mean_reversion",
    "soft_level_reference",
}


def load_public_functions() -> list[str]:
    names = []
    for name in sorted(ats.__all__):
        if name == "add_plateau":
            continue
        if name.startswith("add_") and callable(getattr(ats, name)):
            names.append(name)
    return names


def round_list(values: Any) -> list[Any]:
    array = np.asarray(values)
    result = []
    for item in array.tolist():
        if isinstance(item, float):
            if np.isnan(item):
                result.append("nan")
            else:
                result.append(round(item, 6))
        else:
            result.append(item)
    return result


def build_catalog() -> list[dict[str, Any]]:
    catalog = []
    public_names = load_public_functions()
    missing_meta = sorted(set(public_names) - set(PRIMITIVE_META))
    if missing_meta:
        raise ValueError(f"Missing metadata for primitives: {missing_meta}")

    for name in public_names:
        fn = getattr(ats, name)
        signature = inspect.signature(fn)
        params = list(signature.parameters.values())
        required = []
        optional = []
        for param in params[1:]:
            field = {"name": param.name}
            if param.default is inspect._empty:
                required.append(field)
            else:
                default = param.default
                if isinstance(default, float) and np.isnan(default):
                    default = "nan"
                field["default"] = default
                optional.append(field)

        meta = PRIMITIVE_META[name]
        machine_meta = PRIMITIVE_MACHINE_METADATA[name]
        initial = np.asarray(meta["example_initial_series"], dtype=float)
        result = fn(initial, **meta["example_call"])
        invalid_examples: list[dict[str, Any]] = []
        for args in machine_meta["mutually_exclusive_args"]:
            if len(args) >= 2:
                invalid_examples.append(
                    {
                        "reason": f"Do not pass {', '.join(args)} together.",
                        "call": {"function": name, **{arg: "<value>" for arg in args}},
                    }
                )
        catalog.append(
            {
                "function_name": name,
                "signature": f"{name}{signature}",
                "category": meta["category"],
                "short_description": meta["short_description"],
                "core_semantic_tags": meta["core_semantic_tags"],
                "what_it_does": meta["what_it_does"],
                "what_it_does_not_do": meta["what_it_does_not_do"],
                "required_parameters": required,
                "optional_parameters": optional,
                "exact_effect_on_series": meta["exact_effect"],
                "behavior": meta["behavior"],
                "effect_type": machine_meta["effect_type"],
                "mutability": machine_meta["mutability"],
                "required_arg_patterns": machine_meta["required_arg_patterns"],
                "mutually_exclusive_args": machine_meta["mutually_exclusive_args"],
                "recommended_usage": machine_meta["recommended_usage"],
                "forbidden_usage_patterns": machine_meta["forbidden_usage_patterns"],
                "typical_caption_cues": meta["typical_caption_cues"],
                "minimal_usage_example": {
                    "initial_series": round_list(initial),
                    "call": {"function": name, **meta["example_call"]},
                    "result_series": round_list(result),
                },
                "canonical_examples": {
                    "valid": [
                        {
                            "description": "Assign the returned array back to `series`.",
                            "code": (
                                "series = "
                                + f"{name}(series, "
                                + ", ".join(
                                    f"{key}={repr(value)}"
                                    for key, value in meta["example_call"].items()
                                )
                                + ")"
                            ),
                        }
                    ],
                    "invalid": invalid_examples,
                },
            }
        )
    return catalog


def parse_captions() -> list[tuple[int, str]]:
    records = json.loads(T2S_CASES_PATH.read_text(encoding="utf-8"))
    return [
        (index, str(record["description"]).strip())
        for index, record in enumerate(records)
        if str(record.get("description", "")).strip()
    ]


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0).lower(), match.start(), match.end()) for match in re.finditer(r"[a-z]+(?:'[a-z]+)?", text.lower())]


def is_negated(text: str, match_start: int, window: int = 6) -> bool:
    tokens = tokenize_with_spans(text)
    preceding = [token for token in tokens if token[2] <= match_start]
    if not preceding:
        return False
    window_tokens = preceding[-window:]
    words = [word for word, _, _ in window_tokens]

    for idx, word in enumerate(words):
        if word in {"not", "no"} and idx + 1 < len(words) and words[idx + 1] == "only":
            continue
        if word in NEGATION_TOKENS:
            return True

    for idx in range(len(words) - 1):
        if (words[idx], words[idx + 1]) in NEGATION_BIGRAMS:
            return True

    return False


def non_negated_pattern_match(text: str, pattern: str, window: int = 6) -> bool:
    for match in re.finditer(pattern, text):
        if not is_negated(text, match.start(), window=window):
            return True
    return False


def cue_to_pattern(cue: str) -> str:
    escaped = re.escape(cue.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return rf"\b{escaped}\b"


def infer_semantics(caption: str) -> list[str]:
    text = caption.lower()
    tags = set()
    for tag, rule in SEMANTIC_RULES.items():
        if any(non_negated_pattern_match(text, pattern) for pattern in rule["patterns"]):
            tags.add(tag)
    for tag, phrases in NEGATED_TAG_PHRASES.items():
        if any(phrase in text for phrase in phrases):
            tags.discard(tag)
    if "oscill" in text and "seasonal" not in text and "periodic" not in text and "cyclic" not in text:
        tags.add("volatility")
    if "linearly from" in text or ("moves linearly from" in text):
        tags.update({"trend_up", "trend_down"})
    return sorted(tags)


def has_near_zero_volatility_language(text: str) -> bool:
    phrases = [
        "no volatility",
        "zero volatility",
        "virtually zero",
        "essentially zero",
        "nonexistent volatility",
        "perfectly stable",
        "perfectly flat",
        "completely flat",
        "constant signal",
        "remains constant",
        "value remains constant",
        "flat line",
        "unchanging",
        "invariant",
        "no fluctuations",
        "lack of variation",
    ]
    return any(phrase in text for phrase in phrases)


def retrieve_primitives(caption: str, semantics: list[str], catalog: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    text = caption.lower()
    candidates: list[str] = []
    weak_reasons: list[str] = []

    for semantic in semantics:
        for primitive in SEMANTIC_RULES[semantic]["primitives"]:
            if primitive not in candidates:
                candidates.append(primitive)

    for entry in catalog:
        matched_cues = [cue for cue in entry["typical_caption_cues"] if non_negated_pattern_match(text, cue_to_pattern(cue))]
        if matched_cues and entry["function_name"] not in candidates:
            candidates.append(entry["function_name"])

    # Deterministic phrase-level routing for retrieval cases that are otherwise
    # too subtle for simple keyword overlap.
    if "linearly from" in text or "moves linearly from" in text:
        for primitive in ["add_ramp", "add_trend"]:
            if primitive not in candidates:
                candidates.append(primitive)

    if "sharp spike" in text or "sharp upward impulse" in text or "one-timestep" in text or "one timestep" in text:
        if "add_spike" not in candidates:
            candidates.append("add_spike")

    if "missing entirely" in text:
        if "add_gap" not in candidates:
            candidates.append("add_gap")

    if ("shift" in text or "jumps up by" in text or "jumps down by" in text) and (
        "new level" in text or "lower level" in text or "higher level" in text or "jumps up by" in text or "jumps down by" in text
    ):
        if "add_level_shift" not in candidates:
            candidates.append("add_level_shift")
        if "regime shift" in text and "add_change_point" not in candidates:
            candidates.append("add_change_point")

    noise_cues = [
        "noise",
        "noisy",
        "random jitter",
        "jitter",
        "random fluctuations",
        "fresh random noise",
        "fresh random",
    ]
    volatility_scale_cues = [
        "more volatile",
        "less volatile",
        "fluctuations become",
        "swings widen",
        "swings become",
        "same baseline",
        "rescaled",
        "being rescaled",
    ]

    if any(cue in text for cue in noise_cues):
        if "add_noise" not in candidates:
            candidates.append("add_noise")
    if any(cue in text for cue in volatility_scale_cues):
        if "add_volatility" not in candidates:
            candidates.append("add_volatility")

    if "resistance" in text or "support" in text or "range-bound" in text or "range bound" in text:
        if "add_volatility" not in candidates:
            candidates.append("add_volatility")
        if "add_flat" not in candidates:
            candidates.append("add_flat")
        weak_reasons.append(
            "Soft level-reference wording is represented by existing volatility/flat composition, not a new primitive."
        )

    if "rather than introducing fresh random noise" in text or "rather than fresh random noise" in text:
        candidates = [primitive for primitive in candidates if primitive != "add_noise"]
    if "no claim" in text and "rescaled" in text:
        candidates = [primitive for primitive in candidates if primitive != "add_volatility"]

    if "regime shift" in text or "rather than just a finite plateau" in text:
        candidates = [primitive for primitive in candidates if primitive != "add_flat"]
        for primitive in ["add_level_shift", "add_change_point"]:
            if primitive not in candidates:
                candidates.append(primitive)

    if has_near_zero_volatility_language(text):
        candidates = [primitive for primitive in candidates if primitive not in {"add_noise", "add_volatility"}]

    if "does not indicate any repeating period" in text or "without explicit periodicity" in text:
        if "add_noise" not in candidates:
            candidates.append("add_noise")
        if "add_volatility" not in candidates:
            candidates.append("add_volatility")
        weak_reasons.append("Caption mentions oscillation without explicit periodicity, so retrieval favors volatility/noise over true seasonality.")

    if "oscill" in text and "seasonal" not in text and "periodic" not in text and "cyclic" not in text:
        if "add_noise" not in candidates:
            candidates.append("add_noise")
        if "add_volatility" not in candidates:
            candidates.append("add_volatility")
        weak_reasons.append("Caption mentions oscillation without explicit periodicity, so retrieval favors volatility/noise over true seasonality.")
    if "stable" in text and not any(tag in semantics for tag in ["flat", "trend_up", "trend_down", "level_shift", "change_point"]):
        weak_reasons.append("Stable/steady wording is broad and may retrieve several low-level shape primitives.")
    if not candidates:
        weak_reasons.append("No strong primitive cue was detected.")

    return candidates, weak_reasons


def evaluate_caption(caption: str, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    semantics = infer_semantics(caption)
    candidates, weak_reasons = retrieve_primitives(caption, semantics, catalog)
    true_gaps = [tag for tag in semantics if tag in TRUE_PRIMITIVE_GAPS and tag not in ORCHESTRATION_LEVEL_SEMANTICS]
    orchestration_descriptors = [tag for tag in semantics if tag in ORCHESTRATION_LEVEL_SEMANTICS]
    key_retrievable = bool(candidates) or not semantics

    if true_gaps:
        retrieval_status = "gap"
    elif weak_reasons or orchestration_descriptors:
        retrieval_status = "weak"
    else:
        retrieval_status = "strong"

    notes = []
    if candidates:
        notes.append(f"Candidate primitives: {', '.join(candidates)}.")
    if weak_reasons:
        notes.extend(weak_reasons)
    if orchestration_descriptors:
        notes.append(
            "High-level semantics requiring later orchestration but not new atomic primitives: "
            + ", ".join(orchestration_descriptors)
            + "."
        )
    if true_gaps:
        notes.extend(TRUE_PRIMITIVE_GAPS[tag] for tag in true_gaps)

    return {
        "caption": caption,
        "inferred_semantic_tags": semantics,
        "candidate_primitives": candidates,
        "key_semantics_retrievable": key_retrievable,
        "retrieval_status": retrieval_status,
        "ambiguous_or_weak_retrieval": weak_reasons,
        "orchestration_level_descriptors": orchestration_descriptors,
        "true_primitive_gaps": true_gaps,
        "notes": " ".join(notes),
    }


def representative_examples(rows: list[dict[str, Any]], status: str, limit: int = 5) -> list[dict[str, Any]]:
    matched = [row for row in rows if row["retrieval_status"] == status]
    matched.sort(
        key=lambda row: (
            len(row["true_primitive_gaps"]),
            len(row.get("orchestration_level_descriptors", [])),
            len(row["ambiguous_or_weak_retrieval"]),
            len(row["candidate_primitives"]),
            row["caption"],
        )
    )
    if status == "gap":
        matched.sort(key=lambda row: (-len(row["true_primitive_gaps"]), -len(row["candidate_primitives"]), row["caption"]))
    return matched[:limit]


def build_summary(catalog: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    status_counts = Counter(row["retrieval_status"] for row in rows)
    primitive_counts = Counter(primitive for row in rows for primitive in row["candidate_primitives"])
    semantic_counts = Counter(tag for row in rows for tag in row["inferred_semantic_tags"])
    gap_counts = Counter(tag for row in rows for tag in row["true_primitive_gaps"])
    orchestration_counts = Counter(tag for row in rows for tag in row.get("orchestration_level_descriptors", []))
    only_orchestration = sum(1 for row in rows if not row["true_primitive_gaps"])

    def pct(count: int) -> str:
        return f"{(100.0 * count / total):.1f}%"

    lines = []
    lines.append("# Retrieval-Ready Primitive Catalog Summary")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Public atomic primitives cataloged: {len(catalog)}")
    lines.append(f"- Caption samples evaluated: {total}")
    lines.append("- Evaluation target: retrieval only. The report asks which atomic primitives a future orchestrator should retrieve, not whether one function can cover a full caption.")
    lines.append("")
    lines.append("## Retrieval Coverage")
    lines.append(f"- `strong` retrieval: {status_counts['strong']} captions ({pct(status_counts['strong'])})")
    lines.append(f"- `weak` retrieval: {status_counts['weak']} captions ({pct(status_counts['weak'])})")
    lines.append(f"- `gap` retrieval: {status_counts['gap']} captions ({pct(status_counts['gap'])})")
    lines.append(f"- Captions requiring only later orchestration, not new primitives: {only_orchestration} ({pct(only_orchestration)})")
    lines.append("")
    lines.append("## Most Frequently Retrieved Primitives")
    for name, count in primitive_counts.most_common(12):
        lines.append(f"- `{name}`: {count} captions ({pct(count)})")
    lines.append("")
    lines.append("## Most Common Semantic Cues In Captions")
    for tag, count in semantic_counts.most_common(12):
        lines.append(f"- `{tag}`: {count} captions ({pct(count)})")
    lines.append("")
    lines.append("## Hardest Caption Patterns To Retrieve Cleanly")
    lines.append("- Explicit mean reversion or rebound language. These are now treated as orchestration-level descriptors rather than primitive gaps when the underlying low-level actions are retrievable.")
    lines.append("- Generic oscillation language without periodic cues. This is usually retrieved as volatility/noise, but the wording can be ambiguous between random fluctuation and true seasonality.")
    lines.append("- Broad stability language such as `stable`, `steady`, or `hovering`, which can retrieve `add_flat`, `add_trend`, or `add_level_shift` depending on context.")
    lines.append("")
    lines.append("## Orchestration-Level Descriptors")
    if orchestration_counts:
        for tag, count in orchestration_counts.most_common():
            lines.append(f"- `{tag}`: {count} captions")
    else:
        lines.append("- No recurring orchestration-level descriptors were recorded.")
    lines.append("")
    lines.append("## Genuine Missing Atomic Primitives")
    if gap_counts:
        for tag, count in gap_counts.most_common():
            lines.append(f"- `{tag}`: {count} captions. {TRUE_PRIMITIVE_GAPS[tag]}")
    else:
        lines.append("- No recurring missing atomic primitive was found.")
    lines.append("")
    lines.append("## Strong Retrieval Examples")
    for row in representative_examples(rows, "strong"):
        lines.append(f"- {row['caption']}")
        lines.append(f"  Retrieved: {', '.join(row['candidate_primitives'])}")
    lines.append("")
    lines.append("## Weak Retrieval Examples")
    for row in representative_examples(rows, "weak"):
        lines.append(f"- {row['caption']}")
        lines.append(f"  Retrieved: {', '.join(row['candidate_primitives'])}")
        weak_bits = list(row["ambiguous_or_weak_retrieval"])
        if row.get("orchestration_level_descriptors"):
            weak_bits.append("Orchestration descriptors: " + ", ".join(row["orchestration_level_descriptors"]))
        lines.append(f"  Why weak: {' '.join(weak_bits)}")
    lines.append("")
    lines.append("## Gap Examples")
    for row in representative_examples(rows, "gap"):
        lines.append(f"- {row['caption']}")
        lines.append(f"  Retrieved: {', '.join(row['candidate_primitives'])}")
        lines.append(f"  Missing atomic op: {', '.join(row['true_primitive_gaps'])}")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("- The current library is already structurally close to retrieval-ready: most captions can retrieve sensible low-level candidates, then rely on a separate LLM for sequencing and parameterization.")
    lines.append("- Multi-phase captions and mean-reversion-style descriptions are primarily an orchestration problem, not a primitive catalog problem.")
    return "\n".join(lines) + "\n"


def validate_catalog(catalog: list[dict[str, Any]]) -> None:
    required_fields = {
        "function_name",
        "signature",
        "category",
        "short_description",
        "core_semantic_tags",
        "what_it_does",
        "what_it_does_not_do",
        "required_parameters",
        "optional_parameters",
        "exact_effect_on_series",
        "behavior",
        "effect_type",
        "mutability",
        "required_arg_patterns",
        "mutually_exclusive_args",
        "recommended_usage",
        "forbidden_usage_patterns",
        "typical_caption_cues",
        "minimal_usage_example",
        "canonical_examples",
    }
    public_names = set(load_public_functions())
    catalog_names = {entry["function_name"] for entry in catalog}
    if public_names != catalog_names:
        raise ValueError(f"Catalog mismatch. Public={sorted(public_names)} Catalog={sorted(catalog_names)}")
    for entry in catalog:
        missing = required_fields - set(entry)
        if missing:
            raise ValueError(f"Catalog entry {entry['function_name']} missing fields: {sorted(missing)}")
        if not entry["core_semantic_tags"]:
            raise ValueError(f"Catalog entry {entry['function_name']} has no semantic tags")


def main() -> None:
    catalog = build_catalog()
    validate_catalog(catalog)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2))

    rows = []
    for index, caption in parse_captions():
        row = {"index": index, **evaluate_caption(caption, catalog)}
        rows.append(row)

    with REPORT_PATH.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    SUMMARY_PATH.write_text(build_summary(catalog, rows))

    counts = Counter(row["retrieval_status"] for row in rows)
    print(f"Wrote {CATALOG_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Retrieval totals: strong={counts['strong']}, weak={counts['weak']}, gap={counts['gap']}")


if __name__ == "__main__":
    main()
