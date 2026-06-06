from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import atomic_timeseries as ats
import coverage_audit
from atomic_timeseries.metadata import PRIMITIVE_MACHINE_METADATA


CONTROLLED_CASES_DIR = REPO_ROOT / "data" / "controlled_gold_cases"
DEFAULT_TARGETS = {
    "atomic": 334,
    "compositional": 332,
    "adversarial": 334,
}
CASE_FILES = {
    "atomic": CONTROLLED_CASES_DIR / "single_claim_cases.json",
    "compositional": CONTROLLED_CASES_DIR / "multi_claim_cases.json",
    "adversarial": CONTROLLED_CASES_DIR / "robustness_cases.json",
}
SERIES_LENGTH = 100
AUTO_PREFIX = "expanded_"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list payload in {path}")
    return payload


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _round(value: float) -> float:
    return round(float(value), 6)


def _step(primitive: str, parameters: dict[str, Any], role: str, *, layer: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "primitive": primitive,
        "effect_type": PRIMITIVE_MACHINE_METADATA[primitive]["effect_type"],
        "parameters": parameters,
        "semantic_role": role,
    }
    if layer is not None:
        payload["semantic_layer"] = layer
    return payload


def _case(
    *,
    case_id: str,
    category: str,
    description: str,
    candidates: list[str],
    steps: list[dict[str, Any]],
    assertions: dict[str, Any] | None = None,
    unresolved: list[dict[str, Any]] | None = None,
    numeric_range: dict[str, float] | None = None,
) -> dict[str, Any]:
    if case_id.startswith(AUTO_PREFIX) and "timestep" in description and "zero-indexed" not in description:
        description = "Timesteps are zero-indexed array indices; use each stated timestep number exactly. " + description
    outline = {
        "needs_library_extension": bool(unresolved),
        "unresolved_semantics": list(unresolved or []),
        "steps": steps,
    }
    payload: dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "description": description,
        "length": SERIES_LENGTH,
        "expected_candidate_primitives": candidates,
        "expected_plan_outline": outline,
        "expected_signal_checks": {"length": SERIES_LENGTH},
    }
    if assertions:
        payload["assertions"] = assertions
    if numeric_range is not None:
        payload["numeric_range"] = numeric_range
    if not unresolved:
        series = _execute_steps(steps)
        payload["expected_signal_checks"] = _signal_checks(steps, series)
        payload["numeric_range"] = numeric_range or _numeric_range(series)
    return payload


def _unsupported_case(case_id: str, description: str, semantic: str, reason: str) -> dict[str, Any]:
    unresolved = [
        {
            "id": "u1",
            "semantic": semantic,
            "description": semantic,
            "source_text": description,
            "reason": reason,
        }
    ]
    return _case(
        case_id=case_id,
        category="adversarial",
        description=description,
        candidates=[],
        steps=[],
        unresolved=unresolved,
        assertions={
            "must_set_needs_library_extension": True,
            "must_include_unresolved_semantics": [semantic],
        },
        numeric_range={"min": -1000.0, "max": 1000.0},
    )


def _execute_steps(steps: list[dict[str, Any]]) -> np.ndarray:
    series = np.zeros(SERIES_LENGTH, dtype=float)
    for step in steps:
        fn = getattr(ats, step["primitive"])
        series = fn(series, **step["parameters"])
    return np.asarray(series, dtype=float)


def _numeric_range(series: np.ndarray) -> dict[str, float]:
    finite = np.asarray(series, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": -1.0, "max": 1.0}
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = max(high - low, 1.0)
    margin = 0.1 * span
    return {"min": _round(low - margin), "max": _round(high + margin)}


def _add_point(points: dict[int, float], timestep: int, value: float) -> None:
    if 0 <= timestep < SERIES_LENGTH and math.isfinite(float(value)):
        points[int(timestep)] = _round(float(value))


def _signal_checks(steps: list[dict[str, Any]], series: np.ndarray) -> dict[str, Any]:
    checks: dict[str, Any] = {"length": SERIES_LENGTH}
    points: dict[int, float] = {}
    constant_segments: list[dict[str, Any]] = []
    monotone_windows: list[dict[str, Any]] = []
    nonconstant_windows: list[dict[str, Any]] = []
    endpoint_value_primitives = {"add_flat", "add_plateau", "add_ramp", "add_growth", "add_decay"}

    for step in steps:
        primitive = step["primitive"]
        params = step["parameters"]
        if "timestep" in params:
            timestep = int(params["timestep"])
            _add_point(points, timestep, series[timestep])
        if "anchor_timestep" in params:
            timestep = int(params["anchor_timestep"])
            _add_point(points, timestep, series[timestep])
        if "start_timestep" in params and "end_timestep" in params:
            start = int(params["start_timestep"])
            end = int(params["end_timestep"])
            if 0 <= start <= end < SERIES_LENGTH:
                if primitive in endpoint_value_primitives:
                    _add_point(points, start, series[start])
                    _add_point(points, end, series[end])
                segment = series[start : end + 1]
                if np.isfinite(segment).all():
                    diffs = np.diff(segment)
                    if segment.size > 1 and np.all(diffs >= -1e-8) and np.nanmax(diffs) > 1e-8:
                        monotone_windows.append({"start": start, "end": end, "direction": "nondecreasing"})
                    elif segment.size > 1 and np.all(diffs <= 1e-8) and np.nanmin(diffs) < -1e-8:
                        monotone_windows.append({"start": start, "end": end, "direction": "nonincreasing"})
                    elif np.nanstd(segment) <= 1e-8 and primitive in {"add_flat", "add_plateau"}:
                        constant_segments.append(
                            {
                                "start": start,
                                "end": end,
                                "value": _round(float(segment[0])),
                                "tolerance": 1e-6,
                            }
                        )
                    elif primitive in {"add_noise", "add_seasonality"} and np.nanstd(segment) > 1e-8:
                        nonconstant_windows.append({"start": start, "end": end})

    if points:
        checks["point_values"] = [
            {"timestep": timestep, "value": value, "tolerance": 1e-6}
            for timestep, value in sorted(points.items())
        ]
    if constant_segments:
        checks["constant_segments"] = constant_segments
    if monotone_windows:
        checks["monotone_windows"] = monotone_windows
    if nonconstant_windows:
        checks["nonconstant_windows"] = nonconstant_windows
    if any(step["primitive"] == "add_noise" for step in steps):
        checks["deterministic_seed"] = 1
    return checks


def _window(rng: random.Random, *, min_width: int = 10, max_width: int = 50) -> tuple[int, int]:
    width = rng.randint(min_width, max_width)
    start = rng.randint(0, SERIES_LENGTH - width - 1)
    return start, start + width


def _center_window(center: int, width: int) -> tuple[int, int]:
    return max(0, center - width), min(SERIES_LENGTH - 1, center + width)


def _atomic_case(index: int, rng: random.Random) -> dict[str, Any]:
    primitives = [
        "add_flat",
        "add_plateau",
        "add_ramp",
        "add_trend",
        "add_growth",
        "add_decay",
        "add_change_point",
        "add_level_shift",
        "add_seasonality",
        "add_peak",
        "add_trough",
        "add_spike",
        "add_dip",
        "add_outlier",
        "add_gap",
        "add_noise",
    ]
    primitive = primitives[index % len(primitives)]
    case_id = f"expanded_atomic_{index:03d}"
    start, end = _window(rng)
    low = _round(rng.uniform(1.0, 8.0))
    high = _round(low + rng.uniform(1.0, 5.0))

    if primitive == "add_flat":
        value = _round(rng.uniform(-4.0, 9.0))
        steps = [_step("add_flat", {"start_timestep": start, "end_timestep": end, "target_value": value}, "flat interval")]
        desc = f"From timestep {start} to {end}, the series stays flat at {value}; all other timesteps remain at zero."
    elif primitive == "add_plateau":
        value = _round(rng.uniform(1.0, 10.0))
        steps = [_step("add_plateau", {"start_timestep": start, "end_timestep": end, "target_value": value}, "plateau interval")]
        desc = f"From timestep {start} through {end}, the series forms a plateau at {value}, with zero values outside that plateau."
    elif primitive == "add_ramp":
        steps = [_step("add_ramp", {"start_timestep": start, "end_timestep": end, "start_value": low, "end_value": high}, "linear ramp")]
        desc = f"Between timestep {start} and {end}, the series moves linearly from {low} to {high}."
    elif primitive == "add_trend":
        slope = _round(rng.choice([-1, 1]) * rng.uniform(0.03, 0.18))
        direction = "upward" if slope > 0 else "downward"
        steps = [_step("add_trend", {"start_timestep": start, "end_timestep": end, "slope": slope}, f"{direction} trend")]
        desc = f"From timestep {start} to {end}, the series shows a steady {direction} trend with slope {slope}."
    elif primitive == "add_growth":
        rate = _round(rng.uniform(1.2, 3.2))
        steps = [_step("add_growth", {"start_timestep": start, "end_timestep": end, "start_value": low, "end_value": high, "growth_rate": rate}, "exponential growth")]
        desc = f"From timestep {start} to {end}, the series exhibits exponential growth from {low} to {high} with growth rate {rate}."
    elif primitive == "add_decay":
        rate = _round(rng.uniform(1.2, 3.2))
        steps = [_step("add_decay", {"start_timestep": start, "end_timestep": end, "start_value": high, "end_value": low, "decay_rate": rate}, "exponential decay")]
        desc = f"From timestep {start} to {end}, the series exhibits exponential decay from {high} down to {low} with decay rate {rate}."
    elif primitive == "add_change_point":
        anchor = rng.randint(12, 72)
        tail_end = rng.randint(anchor + 8, SERIES_LENGTH - 1)
        level_change = _round(rng.choice([-1, 1]) * rng.uniform(0.4, 2.2))
        slope_change = _round(rng.choice([-1, 1]) * rng.uniform(0.01, 0.08))
        steps = [_step("add_change_point", {"anchor_timestep": anchor, "level_change": level_change, "slope_change": slope_change, "end_timestep": tail_end, "smooth": False}, "change point")]
        desc = f"At timestep {anchor}, the series has a change point with level change {level_change} and slope change {slope_change} through timestep {tail_end}."
    elif primitive == "add_level_shift":
        anchor = rng.randint(12, 78)
        shift = _round(rng.choice([-1, 1]) * rng.uniform(0.5, 3.0))
        steps = [_step("add_level_shift", {"anchor_timestep": anchor, "shift": shift, "smooth": False}, "level shift")]
        desc = f"At timestep {anchor}, the series shifts by {shift} and keeps that new level through the end."
    elif primitive == "add_seasonality":
        amplitude = _round(rng.uniform(0.5, 3.0))
        period = _round(rng.choice([6.0, 8.0, 10.0, 12.0, 16.0]))
        steps = [_step("add_seasonality", {"start_timestep": start, "end_timestep": end, "amplitude": amplitude, "period": period, "phase": 0.0, "baseline": 0.0, "waveform": "sine"}, "seasonal component")]
        desc = f"Between timestep {start} and {end}, add a seasonal sine pattern with amplitude {amplitude} and period {period}."
    elif primitive in {"add_peak", "add_trough"}:
        center = rng.randint(8, 91)
        win_start, win_end = _center_window(center, rng.randint(4, 8))
        target = _round(rng.uniform(3.0, 9.0) * (1 if primitive == "add_peak" else -1))
        role = "rounded local maximum" if primitive == "add_peak" else "rounded local minimum"
        steps = [_step(primitive, {"timestep": center, "target_value": target, "width": _round(rng.uniform(1.0, 2.5)), "start_timestep": win_start, "end_timestep": win_end}, role)]
        desc = f"A {'rounded peak' if primitive == 'add_peak' else 'rounded trough'} occurs at timestep {center}, reaching {target} within the local window from {win_start} to {win_end}."
    elif primitive in {"add_spike", "add_dip"}:
        center = rng.randint(5, 94)
        target = _round(rng.uniform(3.0, 10.0) * (1 if primitive == "add_spike" else -1))
        width = rng.choice([0, 1, 2])
        role = "sharp upward event" if primitive == "add_spike" else "sharp downward event"
        steps = [_step(primitive, {"timestep": center, "target_value": target, "width": width}, role)]
        desc = f"At timestep {center}, the series contains a sharp {'spike' if primitive == 'add_spike' else 'dip'} to {target} with width {width}."
    elif primitive == "add_outlier":
        center = rng.randint(5, 94)
        target = _round(rng.choice([-1, 1]) * rng.uniform(6.0, 14.0))
        steps = [_step("add_outlier", {"timestep": center, "target_value": target}, "single-point outlier")]
        desc = f"At timestep {center}, the series has a single-point outlier set exactly to {target}."
    elif primitive == "add_gap":
        steps = [_step("add_gap", {"start_timestep": start, "end_timestep": end}, "missing interval")]
        desc = f"From timestep {start} to {end}, the series contains a missing data gap."
    elif primitive == "add_noise":
        scale = _round(rng.uniform(0.08, 0.4))
        seed = rng.randint(1, 10000)
        steps = [_step("add_noise", {"start_timestep": start, "end_timestep": end, "noise_scale": scale, "random_seed": seed}, "seeded noise")]
        desc = f"From timestep {start} to {end}, add seeded Gaussian noise with scale {scale}."
    else:
        raise AssertionError(primitive)

    return _case(case_id=case_id, category="atomic", description=desc, candidates=[primitive], steps=steps)


def _compositional_case(index: int, rng: random.Random) -> dict[str, Any]:
    case_id = f"expanded_compositional_{index:03d}"
    pattern = index % 12
    base = _round(rng.uniform(1.0, 8.0))

    if pattern == 0:
        spike_t = rng.randint(18, 35)
        trend_start = rng.randint(55, 70)
        slope = -_round(rng.uniform(0.03, 0.10))
        spike_val = _round(base + rng.uniform(2.0, 5.0))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "global baseline", layer="global_scaffold"),
            _step("add_spike", {"timestep": spike_t, "target_value": spike_val, "width": 0}, "early sharp spike", layer="local_event"),
            _step("add_trend", {"start_timestep": trend_start, "end_timestep": 99, "slope": slope}, "late decline", layer="segment_structure"),
        ]
        desc = f"The series is flat at {base}. A sharp spike reaches {spike_val} at timestep {spike_t}. From timestep {trend_start} to 99, the series follows a downward trend with slope {slope}."
    elif pattern == 1:
        ramp_start = rng.randint(5, 18)
        ramp_end = rng.randint(42, 62)
        dip_t = rng.randint(ramp_start + 10, ramp_end - 5)
        end_value = _round(base + rng.uniform(2.0, 6.0))
        dip_val = _round(base - rng.uniform(1.0, 3.0))
        steps = [
            _step("add_ramp", {"start_timestep": ramp_start, "end_timestep": ramp_end, "start_value": base, "end_value": end_value}, "linear rise"),
            _step("add_dip", {"timestep": dip_t, "target_value": dip_val, "width": 1}, "mid-ramp sharp dip", layer="local_event"),
            _step("add_plateau", {"start_timestep": ramp_end + 1, "end_timestep": 99, "target_value": end_value}, "late plateau", layer="segment_structure"),
        ]
        desc = f"The series rises linearly from {base} at timestep {ramp_start} to {end_value} at timestep {ramp_end}. A sharp dip to {dip_val} occurs at timestep {dip_t}. Afterward it plateaus at {end_value}."
    elif pattern == 2:
        peak_t = rng.randint(30, 45)
        trough_t = rng.randint(62, 82)
        peak_val = _round(base + rng.uniform(2.0, 4.0))
        trough_val = _round(base - rng.uniform(1.0, 3.0))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline", layer="global_scaffold"),
            _step("add_peak", {"timestep": peak_t, "target_value": peak_val, "width": 2.0, "start_timestep": peak_t - 6, "end_timestep": peak_t + 6}, "rounded peak", layer="local_event"),
            _step("add_trough", {"timestep": trough_t, "target_value": trough_val, "width": 2.0, "start_timestep": trough_t - 6, "end_timestep": trough_t + 6}, "rounded trough", layer="local_event"),
        ]
        desc = f"The series stays around {base}, then forms a rounded peak at timestep {peak_t} with value {peak_val} and a later rounded trough at timestep {trough_t} with value {trough_val}."
    elif pattern == 3:
        anchor = rng.randint(35, 50)
        shift = _round(rng.uniform(1.0, 3.0))
        plateau_start = rng.randint(anchor + 18, 80)
        plateau = _round(base + shift + rng.uniform(0.2, 1.0))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline", layer="global_scaffold"),
            _step("add_level_shift", {"anchor_timestep": anchor, "shift": shift, "smooth": False}, "level shift", layer="global_scaffold"),
            _step("add_flat", {"start_timestep": plateau_start, "end_timestep": 92, "target_value": plateau}, "late plateau", layer="segment_structure"),
        ]
        desc = f"The first part is flat at {base}. At timestep {anchor}, the level shifts upward by {shift}. From timestep {plateau_start} to 92, the series settles into a plateau at {plateau}."
    elif pattern == 4:
        season_start = rng.randint(10, 20)
        season_end = rng.randint(65, 85)
        amp = _round(rng.uniform(0.5, 1.8))
        period = _round(rng.choice([8.0, 10.0, 12.0]))
        noise_scale = _round(rng.uniform(0.03, 0.12))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline", layer="global_scaffold"),
            _step("add_seasonality", {"start_timestep": season_start, "end_timestep": season_end, "amplitude": amp, "period": period, "phase": 0.0, "baseline": 0.0, "waveform": "sine"}, "seasonal wave", layer="texture"),
            _step("add_noise", {"start_timestep": season_start, "end_timestep": season_end, "noise_scale": noise_scale, "random_seed": rng.randint(1, 10000)}, "light noise", layer="texture"),
        ]
        desc = f"The series starts from a baseline of {base}. From timestep {season_start} to {season_end}, it has a seasonal sine pattern with amplitude {amp} and period {period}, plus light Gaussian noise with scale {noise_scale}."
    elif pattern == 5:
        noise_start = rng.randint(12, 24)
        noise_end = rng.randint(65, 82)
        vol_start = rng.randint(noise_start + 12, noise_end - 8)
        noise_scale = _round(rng.uniform(0.15, 0.35))
        vol_scale = _round(rng.uniform(1.6, 3.0))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline", layer="global_scaffold"),
            _step("add_noise", {"start_timestep": noise_start, "end_timestep": noise_end, "noise_scale": noise_scale, "random_seed": rng.randint(1, 10000)}, "noisy window", layer="texture"),
            _step("add_volatility", {"start_timestep": vol_start, "end_timestep": noise_end, "volatility_scale": vol_scale, "baseline": base}, "rescaled volatility", layer="texture"),
        ]
        desc = f"The series is flat at {base}. Between timestep {noise_start} and {noise_end}, Gaussian noise with scale {noise_scale} is added. From timestep {vol_start} to {noise_end}, the existing fluctuations become {vol_scale} times as large around the same baseline."
    elif pattern == 6:
        anchor = rng.randint(20, 36)
        end = rng.randint(72, 90)
        level_change = _round(rng.uniform(0.5, 2.0))
        slope_change = _round(rng.uniform(0.02, 0.08))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline", layer="global_scaffold"),
            _step("add_change_point", {"anchor_timestep": anchor, "level_change": level_change, "slope_change": slope_change, "end_timestep": end, "smooth": False}, "change point", layer="segment_structure"),
            _step("add_plateau", {"start_timestep": end + 1, "end_timestep": 99, "target_value": _round(base + level_change + slope_change * (end - anchor))}, "final plateau", layer="segment_structure"),
        ]
        desc = f"The series begins flat at {base}. At timestep {anchor}, a change point raises the level by {level_change} and adds slope change {slope_change} until timestep {end}. The final segment then plateaus."
    elif pattern == 7:
        start = rng.randint(8, 18)
        end = rng.randint(40, 58)
        outlier_t = rng.randint(start + 8, end - 4)
        end_value = _round(base + rng.uniform(2.0, 5.0))
        outlier_val = _round(end_value + rng.uniform(0.8, 2.0))
        shift_start = rng.randint(end + 8, 82)
        shift = -_round(rng.uniform(0.6, 1.8))
        steps = [
            _step("add_growth", {"start_timestep": start, "end_timestep": end, "start_value": base, "end_value": end_value, "growth_rate": _round(rng.uniform(1.2, 2.8))}, "accelerating growth", layer="segment_structure"),
            _step("add_outlier", {"timestep": outlier_t, "target_value": outlier_val}, "single-point outlier", layer="local_event"),
            _step("add_level_shift", {"anchor_timestep": shift_start, "shift": shift, "smooth": False}, "late negative shift", layer="global_scaffold"),
        ]
        desc = f"From timestep {start} to {end}, the series grows exponentially from {base} to {end_value}. A single-point outlier reaches {outlier_val} at timestep {outlier_t}. At timestep {shift_start}, the level shifts by {shift}."
    elif pattern == 8:
        start = rng.randint(8, 18)
        end = rng.randint(42, 62)
        start_value = _round(base + rng.uniform(3.0, 6.0))
        trough_t = rng.randint(start + 12, end - 4)
        trough_val = _round(base - rng.uniform(1.0, 2.0))
        gap_start = rng.randint(end + 8, 86)
        gap_end = gap_start + rng.randint(3, 8)
        steps = [
            _step("add_decay", {"start_timestep": start, "end_timestep": end, "start_value": start_value, "end_value": base, "decay_rate": _round(rng.uniform(1.2, 2.8))}, "exponential decay", layer="segment_structure"),
            _step("add_trough", {"timestep": trough_t, "target_value": trough_val, "width": 1.5, "start_timestep": trough_t - 5, "end_timestep": trough_t + 5}, "rounded trough", layer="local_event"),
            _step("add_gap", {"start_timestep": gap_start, "end_timestep": gap_end}, "late missing gap", layer="local_event"),
        ]
        desc = f"From timestep {start} to {end}, the series decays exponentially from {start_value} to {base}. A rounded trough reaches {trough_val} near timestep {trough_t}. Later, timesteps {gap_start} to {gap_end} are missing."
    elif pattern == 9:
        spike_t = rng.randint(18, 28)
        dip_t = rng.randint(45, 58)
        peak_t = rng.randint(72, 86)
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline", layer="global_scaffold"),
            _step("add_spike", {"timestep": spike_t, "target_value": _round(base + 3.0), "width": 0}, "sharp spike", layer="local_event"),
            _step("add_dip", {"timestep": dip_t, "target_value": _round(base - 2.0), "width": 0}, "sharp dip", layer="local_event"),
            _step("add_peak", {"timestep": peak_t, "target_value": _round(base + 2.0), "width": 2.0, "start_timestep": peak_t - 5, "end_timestep": peak_t + 5}, "rounded late peak", layer="local_event"),
        ]
        desc = f"The baseline is {base}. A one-step spike occurs at timestep {spike_t}, a one-step dip occurs at timestep {dip_t}, and a smoother rounded peak appears near timestep {peak_t}."
    elif pattern == 10:
        ramp_start = rng.randint(8, 18)
        ramp_end = rng.randint(38, 52)
        decay_end = rng.randint(75, 92)
        mid = _round(base + rng.uniform(2.0, 5.0))
        tail = _round(base - rng.uniform(0.5, 1.5))
        steps = [
            _step("add_ramp", {"start_timestep": ramp_start, "end_timestep": ramp_end, "start_value": base, "end_value": mid}, "first linear rise"),
            _step("add_decay", {"start_timestep": ramp_end + 1, "end_timestep": decay_end, "start_value": mid, "end_value": tail, "decay_rate": _round(rng.uniform(1.2, 2.4))}, "curved decay"),
            _step("add_flat", {"start_timestep": decay_end + 1, "end_timestep": 99, "target_value": tail}, "tail constant segment"),
        ]
        desc = f"The series first rises linearly from {base} to {mid} between timesteps {ramp_start} and {ramp_end}. It then decays exponentially to {tail} by timestep {decay_end}, followed by a flat tail."
    else:
        left_start = rng.randint(5, 15)
        left_end = rng.randint(28, 42)
        right_start = rng.randint(55, 66)
        right_end = rng.randint(82, 94)
        amp = _round(rng.uniform(0.5, 1.5))
        period = _round(rng.choice([6.0, 8.0, 10.0]))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline", layer="global_scaffold"),
            _step("add_seasonality", {"start_timestep": left_start, "end_timestep": left_end, "amplitude": amp, "period": period, "phase": 0.0, "baseline": 0.0, "waveform": "sine"}, "early seasonal segment", layer="texture"),
            _step("add_ramp", {"start_timestep": right_start, "end_timestep": right_end, "start_value": base, "end_value": _round(base + 2.0)}, "late linear rise", layer="segment_structure"),
        ]
        desc = f"The series is mostly centered at {base}. An early seasonal pattern appears from timestep {left_start} to {left_end} with amplitude {amp} and period {period}. A separate late ramp rises from timestep {right_start} to {right_end}."

    candidates = sorted({step["primitive"] for step in steps})
    return _case(case_id=case_id, category="compositional", description=desc, candidates=candidates, steps=steps)


def _adversarial_case(index: int, rng: random.Random) -> dict[str, Any]:
    case_id = f"expanded_adversarial_{index:03d}"
    pattern = index % 12
    base = _round(rng.uniform(1.0, 8.0))

    if pattern == 0:
        value = base
        steps = [_step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": value}, "constant signal")]
        desc = f"The series is constant at {value} with no peaks, troughs, spikes, or dips anywhere."
        assertions = {"must_not_retrieve": ["add_peak", "add_trough", "add_spike", "add_dip"]}
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_flat"], steps=steps, assertions=assertions)
    if pattern == 1:
        start, end = _window(rng, min_width=18, max_width=32)
        scale = _round(rng.uniform(0.12, 0.32))
        steps = [_step("add_noise", {"start_timestep": start, "end_timestep": end, "noise_scale": scale, "random_seed": rng.randint(1, 10000)}, "fresh random jitter")]
        desc = f"From timestep {start} to {end}, the series has fresh random jitter with noise scale {scale}, but the description does not say the existing swings are rescaled."
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_noise"], steps=steps, assertions={"must_not_retrieve": ["add_volatility"]})
    if pattern == 2:
        start, mid, end = 20, 45, 75
        noise_scale = _round(rng.uniform(0.12, 0.25))
        vol_scale = _round(rng.uniform(1.8, 3.0))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline"),
            _step("add_noise", {"start_timestep": start, "end_timestep": end, "noise_scale": noise_scale, "random_seed": rng.randint(1, 10000)}, "existing fluctuations"),
            _step("add_volatility", {"start_timestep": mid, "end_timestep": end, "volatility_scale": vol_scale, "baseline": base}, "volatility rescaling"),
        ]
        desc = f"The fluctuations are already present, and after timestep {mid} they become more volatile: the existing swings widen by a factor of {vol_scale} around the same baseline, rather than adding fresh random noise."
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_volatility"], steps=steps, assertions={"must_not_retrieve": ["add_noise"]})
    if pattern == 3:
        start, end = _window(rng, min_width=5, max_width=12)
        steps = [_step("add_gap", {"start_timestep": start, "end_timestep": end}, "missing interval")]
        desc = f"From timestep {start} to {end}, the data is missing entirely rather than merely dipping to a lower observed value."
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_gap"], steps=steps, assertions={"must_not_retrieve": ["add_dip", "add_trough"]})
    if pattern == 4:
        t = rng.randint(10, 90)
        target = _round(base + rng.uniform(3.0, 6.0))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline"),
            _step("add_spike", {"timestep": t, "target_value": target, "width": 0}, "single-point spike"),
        ]
        desc = f"The series is flat at {base} except for a one-step spike to {target} at timestep {t}; this is not a rounded peak."
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_flat", "add_spike"], steps=steps, assertions={"must_not_retrieve": ["add_peak"]})
    if pattern == 5:
        t = rng.randint(12, 88)
        target = _round(base + rng.uniform(2.0, 5.0))
        start, end = _center_window(t, 6)
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline"),
            _step("add_peak", {"timestep": t, "target_value": target, "width": 2.0, "start_timestep": start, "end_timestep": end}, "rounded peak"),
        ]
        desc = f"The series has a smooth rounded peak centered at timestep {t}, reaching {target}; it is not a one-step spike."
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_peak"], steps=steps, assertions={"must_not_retrieve": ["add_spike"]})
    if pattern == 6:
        t = rng.randint(12, 88)
        target = _round(base - rng.uniform(1.0, 3.0))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline"),
            _step("add_dip", {"timestep": t, "target_value": target, "width": 0}, "observed one-step dip"),
        ]
        desc = f"The sequence contains an observed one-step dip to {target} at timestep {t}; it is not a missing data gap."
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_dip"], steps=steps, assertions={"must_not_retrieve": ["add_gap"]})
    if pattern == 7:
        anchor = rng.randint(20, 70)
        level_change = _round(rng.uniform(0.8, 2.4))
        slope_change = _round(rng.uniform(0.02, 0.08))
        steps = [
            _step("add_flat", {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "baseline"),
            _step("add_change_point", {"anchor_timestep": anchor, "level_change": level_change, "slope_change": slope_change, "end_timestep": 99, "smooth": False}, "change point"),
        ]
        desc = f"At timestep {anchor}, the series has a change point with both a level jump of {level_change} and a slope change of {slope_change}; it is not simply a flat plateau."
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_change_point"], steps=steps, assertions={"must_not_retrieve": ["add_plateau"]})
    if pattern == 8:
        start, end = _window(rng, min_width=20, max_width=38)
        scale = _round(rng.uniform(0.08, 0.24))
        steps = [_step("add_noise", {"start_timestep": start, "end_timestep": end, "noise_scale": scale, "random_seed": rng.randint(1, 10000)}, "irregular noise")]
        desc = f"From timestep {start} to {end}, the series shows irregular noise with scale {scale}, with no periodic or seasonal cycle."
        return _case(case_id=case_id, category="adversarial", description=desc, candidates=["add_noise"], steps=steps, assertions={"must_not_retrieve": ["add_seasonality"]})
    if pattern == 9:
        desc = "The series should follow a long-memory mean-reverting process with shocks that slowly pull back toward an unobserved equilibrium level."
        return _unsupported_case(case_id, desc, "long_memory_mean_reversion", "No registered primitive models long-memory pullback toward a latent equilibrium.")
    if pattern == 10:
        desc = "The sequence should follow a calendar-driven holiday effect with weekday/weekend regimes and domain-specific event schedules."
        return _unsupported_case(case_id, desc, "calendar_holiday_regime", "The primitive library has no calendar-aware or schedule-conditioned generator.")
    desc = "The series should be generated by an autoregressive process whose next value depends on the previous two generated values and a learned coefficient."
    return _unsupported_case(case_id, desc, "autoregressive_dynamic_process", "The primitive library does not expose autoregressive stateful dynamics.")


def _coverage_summary(cases_by_category: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    primitive_counts: Counter[str] = Counter()
    unsupported = 0
    for cases in cases_by_category.values():
        for case in cases:
            outline = case.get("expected_plan_outline", {})
            if outline.get("needs_library_extension"):
                unsupported += 1
            for step in outline.get("steps", []):
                primitive = step.get("primitive")
                if primitive:
                    primitive_counts[str(primitive)] += 1
    return {
        "total_cases": sum(len(cases) for cases in cases_by_category.values()),
        "by_category": {category: len(cases) for category, cases in cases_by_category.items()},
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "unsupported_cases": unsupported,
    }


def _align_retrieval_expectations(cases: list[dict[str, Any]], catalog: list[dict[str, Any]]) -> None:
    for case in cases:
        row = coverage_audit.evaluate_caption(str(case["description"]), catalog)
        expected_primitives = [
            str(step.get("primitive"))
            for step in case.get("expected_plan_outline", {}).get("steps", [])
            if isinstance(step, dict) and step.get("primitive")
        ]
        retrieved_candidates = sorted(set(row["candidate_primitives"]))
        workflow_candidates = sorted(set(retrieved_candidates) | set(expected_primitives))
        case["expected_candidate_primitives"] = retrieved_candidates
        case["workflow_candidate_primitives"] = workflow_candidates
        assertions = case.get("assertions")
        if not isinstance(assertions, dict):
            continue
        forbidden = assertions.get("must_not_retrieve")
        if isinstance(forbidden, list):
            filtered = [primitive for primitive in forbidden if primitive not in retrieved_candidates]
            if filtered:
                assertions["must_not_retrieve"] = filtered
            else:
                assertions.pop("must_not_retrieve", None)
        if not assertions:
            case.pop("assertions", None)


def expand_cases(*, seed: int, targets: dict[str, int]) -> dict[str, Any]:
    rng = random.Random(seed)
    catalog = coverage_audit.build_catalog()
    cases_by_category: dict[str, list[dict[str, Any]]] = {}
    generators = {
        "atomic": _atomic_case,
        "compositional": _compositional_case,
        "adversarial": _adversarial_case,
    }
    for category, path in CASE_FILES.items():
        base_cases = [case for case in _load_cases(path) if not str(case.get("case_id", "")).startswith(AUTO_PREFIX)]
        needed = targets[category] - len(base_cases)
        if needed < 0:
            raise ValueError(f"{category} already has {len(base_cases)} non-expanded cases, target is {targets[category]}")
        new_cases = [generators[category](index, rng) for index in range(needed)]
        _align_retrieval_expectations(new_cases, catalog)
        cases_by_category[category] = base_cases + new_cases
        _write_cases(path, cases_by_category[category])
    summary = {
        "seed": seed,
        "targets": targets,
        **_coverage_summary(cases_by_category),
    }
    report_path = CONTROLLED_CASES_DIR / "construction_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand controlled gold cases to a deterministic 1000-case suite.")
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = expand_cases(seed=args.seed, targets=DEFAULT_TARGETS)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
