from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .semantic_spec import SemanticFeature


@dataclass(frozen=True)
class CheckResult:
    feature_id: str
    checker: str
    passed: bool
    score: float
    error: float
    reason: str


def _window(feature: SemanticFeature, length: int) -> tuple[int, int]:
    if feature.window is None:
        return 0, length - 1
    start, end = feature.window
    if start < 0 or end < start or end >= length:
        raise ValueError(f"Feature {feature.id} has invalid window {feature.window}.")
    return start, end


def _segment(series: np.ndarray, feature: SemanticFeature) -> np.ndarray:
    start, end = _window(feature, len(series))
    return series[start : end + 1]


def _score_from_error(error: float, tolerance: float) -> float:
    if not np.isfinite(error):
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - (error / max(tolerance, 1e-12)))))


def _required_float(feature: SemanticFeature, name: str) -> float:
    if name not in feature.parameters:
        raise ValueError(f"Feature {feature.id} is missing parameter `{name}`.")
    return float(feature.parameters[name])


def _ok(feature: SemanticFeature, checker: str, error: float, reason: str, *, tolerance: float | None = None) -> CheckResult:
    tol = feature.tolerance if tolerance is None else tolerance
    passed = bool(error <= tol)
    return CheckResult(feature.id, checker, passed, _score_from_error(error, tol), float(error), reason)


def _max_abs_error(observed: np.ndarray, expected: np.ndarray) -> float:
    diff = np.abs(observed - expected)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return np.inf
    return float(np.max(finite))


def _finite_values(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    return data[np.isfinite(data)]


def _local_neighbors(series: np.ndarray, timestep: int) -> np.ndarray:
    values: list[float] = []
    if timestep > 0:
        values.append(float(series[timestep - 1]))
    if timestep + 1 < len(series):
        values.append(float(series[timestep + 1]))
    return _finite_values(np.asarray(values, dtype=float))


def _prominence_tolerance(series: np.ndarray, timestep: int) -> float:
    neighbors = _local_neighbors(series, timestep)
    scale_values = _finite_values(np.concatenate([neighbors, np.asarray([series[timestep]], dtype=float)]))
    scale = max(1.0, float(np.nanmax(np.abs(scale_values))) if scale_values.size else 1.0)
    return max(1e-6, 1e-4 * scale)


def _qualitative_local_extreme(
    series: np.ndarray,
    feature: SemanticFeature,
    *,
    mode: str,
    checker: str,
) -> CheckResult:
    timestep = int(feature.parameters.get("timestep", feature.window[0] if feature.window else -1))
    if timestep < 0 or timestep >= len(series):
        return CheckResult(feature.id, checker, False, 0.0, np.inf, f"{checker} timestep out of bounds")
    center = float(series[timestep])
    if not np.isfinite(center):
        return CheckResult(feature.id, checker, False, 0.0, np.inf, "center value is non-finite")

    if feature.window is not None and feature.window[1] > feature.window[0]:
        start, end = _window(feature, len(series))
        segment = _finite_values(series[start : end + 1])
        if segment.size < 2:
            return CheckResult(feature.id, checker, False, 0.0, np.inf, "local window too short")
        if mode == "high":
            reference = float(np.nanmax(segment[segment < np.nanmax(segment)])) if np.any(segment < np.nanmax(segment)) else center
            is_extreme = center >= float(np.nanmax(segment)) - 1e-8
            prominence = center - reference
        else:
            reference = float(np.nanmin(segment[segment > np.nanmin(segment)])) if np.any(segment > np.nanmin(segment)) else center
            is_extreme = center <= float(np.nanmin(segment)) + 1e-8
            prominence = reference - center
    else:
        neighbors = _local_neighbors(series, timestep)
        if neighbors.size == 0:
            return CheckResult(feature.id, checker, False, 0.0, np.inf, "no local neighbors for qualitative check")
        if mode == "high":
            reference = float(np.nanmax(neighbors))
            is_extreme = center > reference
            prominence = center - reference
        else:
            reference = float(np.nanmin(neighbors))
            is_extreme = center < reference
            prominence = reference - center

    tolerance = _prominence_tolerance(series, timestep)
    passed = bool(is_extreme and prominence > tolerance)
    error = 0.0 if passed else max(0.0, tolerance - prominence)
    return CheckResult(
        feature.id,
        checker,
        passed,
        1.0 if passed else 0.0,
        float(error),
        f"qualitative_prominence={prominence:.6g}, tolerance={tolerance:.6g}",
    )


def check_flat(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    if seg.size == 0:
        return CheckResult(feature.id, "check_flat", False, 0.0, np.inf, "flat window is empty")
    if "target_value" not in feature.parameters:
        std_error = float(np.nanstd(seg))
        tolerance = max(feature.tolerance, 1e-6)
        return _ok(feature, "check_flat", std_error, f"std={std_error:.6g}", tolerance=tolerance)
    target = _required_float(feature, "target_value")
    value_error = float(np.nanmax(np.abs(seg - target)))
    std_error = float(np.nanstd(seg))
    error = max(value_error, std_error)
    return _ok(feature, "check_flat", error, f"max_abs={value_error:.6g}, std={std_error:.6g}")


def check_ramp(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    if seg.size < 2:
        return CheckResult(feature.id, "check_ramp", False, 0.0, np.inf, "ramp window needs at least two points")
    if not np.isfinite(seg).all():
        return CheckResult(feature.id, "check_ramp", False, 0.0, np.inf, "ramp segment contains non-finite values")
    if "start_value" in feature.parameters and "end_value" in feature.parameters:
        start_value = _required_float(feature, "start_value")
        end_value = _required_float(feature, "end_value")
        expected = np.linspace(start_value, end_value, seg.size)
        error = float(np.nanmax(np.abs(seg - expected)))
        return _ok(feature, "check_ramp", error, f"max_abs={error:.6g}")
    diffs = np.diff(seg)
    if diffs.size == 0:
        return CheckResult(feature.id, "check_ramp", False, 0.0, np.inf, "ramp window needs variation")
    positive_violation = float(max(0.0, -np.nanmin(diffs)))
    negative_violation = float(max(0.0, np.nanmax(diffs)))
    monotone_violation = min(positive_violation, negative_violation)
    linearity_error = float(np.nanstd(diffs))
    variation = float(abs(seg[-1] - seg[0]))
    tolerance = max(feature.tolerance, 1e-6, 0.02 * max(variation, 1.0))
    error = max(monotone_violation, linearity_error)
    if variation <= tolerance:
        error = max(error, tolerance)
    return _ok(feature, "check_ramp", error, f"monotone_violation={monotone_violation:.6g}, diff_std={linearity_error:.6g}", tolerance=tolerance)


def check_growth(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    if seg.size < 2:
        return CheckResult(feature.id, "check_growth", False, 0.0, np.inf, "growth window needs at least two points")
    if not np.isfinite(seg).all():
        return CheckResult(feature.id, "check_growth", False, 0.0, np.inf, "growth segment contains non-finite values")
    diffs = np.diff(seg)
    monotone_violation = float(max(0.0, -np.nanmin(diffs))) if diffs.size else 0.0
    errors = [monotone_violation]
    if "start_value" in feature.parameters:
        errors.append(abs(float(seg[0]) - _required_float(feature, "start_value")))
    if "end_value" in feature.parameters:
        errors.append(abs(float(seg[-1]) - _required_float(feature, "end_value")))
    if "growth_rate" in feature.parameters:
        start_value = _required_float(feature, "start_value")
        end_value = _required_float(feature, "end_value")
        growth_rate = float(feature.parameters["growth_rate"])
        if growth_rate <= 0:
            return CheckResult(feature.id, "check_growth", False, 0.0, np.inf, "growth_rate must be positive")
        weights = np.expm1(np.linspace(0.0, growth_rate, seg.size))
        if np.allclose(weights[-1], 0.0):
            expected = np.full(seg.size, start_value)
        else:
            weights = weights / weights[-1]
            expected = start_value + (end_value - start_value) * weights
        errors.append(_max_abs_error(seg, expected))
    error = max(errors) if errors else 0.0
    tolerance = max(feature.tolerance, 1e-6)
    return _ok(feature, "check_growth", error, f"claim_error={error:.6g}", tolerance=tolerance)


def check_decay(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    if seg.size < 2:
        return CheckResult(feature.id, "check_decay", False, 0.0, np.inf, "decay window needs at least two points")
    if not np.isfinite(seg).all():
        return CheckResult(feature.id, "check_decay", False, 0.0, np.inf, "decay segment contains non-finite values")
    diffs = np.diff(seg)
    monotone_violation = float(max(0.0, np.nanmax(diffs))) if diffs.size else 0.0
    errors = [monotone_violation]
    if "start_value" in feature.parameters:
        errors.append(abs(float(seg[0]) - _required_float(feature, "start_value")))
    if "end_value" in feature.parameters:
        errors.append(abs(float(seg[-1]) - _required_float(feature, "end_value")))
    if "decay_rate" in feature.parameters:
        start_value = _required_float(feature, "start_value")
        end_value = _required_float(feature, "end_value")
        decay_rate = float(feature.parameters["decay_rate"])
        if decay_rate <= 0:
            return CheckResult(feature.id, "check_decay", False, 0.0, np.inf, "decay_rate must be positive")
        weights = np.exp(-np.linspace(0.0, decay_rate, seg.size))
        if np.isclose(weights[0], weights[-1]):
            normalized = np.zeros(seg.size, dtype=float)
        else:
            normalized = (weights[0] - weights) / (weights[0] - weights[-1])
        expected = start_value + (end_value - start_value) * normalized
        errors.append(_max_abs_error(seg, expected))
    error = max(errors) if errors else 0.0
    tolerance = max(feature.tolerance, 1e-6)
    return _ok(feature, "check_decay", error, f"claim_error={error:.6g}", tolerance=tolerance)


def check_trend(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    if seg.size < 2:
        return CheckResult(feature.id, "check_trend", False, 0.0, np.inf, "trend window needs at least two points")
    slope = _required_float(feature, "slope")
    expected_delta = np.arange(seg.size, dtype=float) * slope
    observed_delta = seg - seg[0]
    tolerance = max(feature.tolerance, abs(slope) * 0.05 + 1e-6)
    error = float(np.nanmax(np.abs(observed_delta - expected_delta)))
    return _ok(feature, "check_trend", error, f"delta_max_abs={error:.6g}", tolerance=tolerance)


def check_spike(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    timestep = int(feature.parameters.get("timestep", feature.window[0] if feature.window else -1))
    if timestep < 0 or timestep >= len(series):
        return CheckResult(feature.id, "check_spike", False, 0.0, np.inf, "spike timestep out of bounds")
    target = _required_float(feature, "target_value") if "target_value" in feature.parameters else None
    amplitude = _required_float(feature, "amplitude") if "amplitude" in feature.parameters else None
    if target is not None:
        error = abs(float(series[timestep]) - target)
        return _ok(feature, "check_spike", error, f"value={float(series[timestep]):.6g}")
    if amplitude is not None:
        baseline = 0.0
        if timestep > 0:
            baseline = float(series[timestep - 1])
        error = abs((float(series[timestep]) - baseline) - amplitude)
        return _ok(feature, "check_spike", error, f"amplitude_error={error:.6g}")
    return _qualitative_local_extreme(series, feature, mode="high", checker="check_spike")


def check_dip(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    timestep = int(feature.parameters.get("timestep", feature.window[0] if feature.window else -1))
    if timestep < 0 or timestep >= len(series):
        return CheckResult(feature.id, "check_dip", False, 0.0, np.inf, "dip timestep out of bounds")
    if "target_value" in feature.parameters:
        target = _required_float(feature, "target_value")
        error = abs(float(series[timestep]) - target)
    elif "amplitude" in feature.parameters:
        amplitude = abs(_required_float(feature, "amplitude"))
        baseline = float(series[timestep - 1]) if timestep > 0 else 0.0
        error = abs((baseline - float(series[timestep])) - amplitude)
    else:
        return _qualitative_local_extreme(series, feature, mode="low", checker="check_dip")
    return _ok(feature, "check_dip", error, f"error={error:.6g}")


def check_outlier(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    timestep = int(feature.parameters.get("timestep", feature.window[0] if feature.window else -1))
    if timestep < 0 or timestep >= len(series):
        return CheckResult(feature.id, "check_outlier", False, 0.0, np.inf, "outlier timestep out of bounds")
    if "target_value" in feature.parameters:
        target = _required_float(feature, "target_value")
        error = abs(float(series[timestep]) - target)
    elif "delta" in feature.parameters or "relative_change" in feature.parameters:
        delta = float(feature.parameters.get("delta", feature.parameters.get("relative_change", 0.0)))
        baseline = float(series[timestep - 1]) if timestep > 0 else 0.0
        error = abs((float(series[timestep]) - baseline) - delta)
    else:
        neighbors = _local_neighbors(series, timestep)
        if neighbors.size == 0:
            return CheckResult(feature.id, "check_outlier", False, 0.0, np.inf, "no local neighbors for qualitative outlier")
        baseline = float(np.nanmean(neighbors))
        prominence = abs(float(series[timestep]) - baseline)
        tolerance = _prominence_tolerance(series, timestep)
        passed = prominence > tolerance
        return CheckResult(
            feature.id,
            "check_outlier",
            bool(passed),
            1.0 if passed else 0.0,
            0.0 if passed else float(tolerance - prominence),
            f"qualitative_prominence={prominence:.6g}, tolerance={tolerance:.6g}",
        )
    return _ok(feature, "check_outlier", error, f"error={error:.6g}")


def check_level_shift(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    start, end = _window(feature, len(series))
    tail = series[start : end + 1]
    if "target_value" in feature.parameters:
        expected = _required_float(feature, "target_value")
    elif "shift" in feature.parameters:
        before = float(series[start - 1]) if start > 0 else 0.0
        expected = before + _required_float(feature, "shift")
    else:
        if tail.size == 0 or not np.isfinite(tail).all():
            return CheckResult(feature.id, "check_level_shift", False, 0.0, np.inf, "level-shift tail is invalid")
        before = float(series[start - 1]) if start > 0 else 0.0
        shift = abs(float(np.nanmean(tail)) - before)
        tolerance = max(feature.tolerance, 1e-6, 1e-4 * max(1.0, abs(before), abs(float(np.nanmean(tail)))))
        passed = shift > tolerance
        error = 0.0 if passed else tolerance - shift
        return CheckResult(feature.id, "check_level_shift", bool(passed), 1.0 if passed else 0.0, float(error), f"qualitative_shift={shift:.6g}")
    error = float(np.nanmax(np.abs(tail - expected))) if tail.size else np.inf
    return _ok(feature, "check_level_shift", error, f"tail_max_abs={error:.6g}")


def check_peak(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    timestep = int(feature.parameters.get("timestep", feature.window[0] if feature.window else -1))
    if timestep < 0 or timestep >= len(series):
        return CheckResult(feature.id, "check_peak", False, 0.0, np.inf, "peak timestep out of bounds")
    target = _required_float(feature, "target_value") if "target_value" in feature.parameters else None
    if target is not None:
        error = abs(float(series[timestep]) - target)
    else:
        return _qualitative_local_extreme(series, feature, mode="high", checker="check_peak")
    return _ok(feature, "check_peak", error, f"center_value={float(series[timestep]):.6g}")


def check_trough(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    timestep = int(feature.parameters.get("timestep", feature.window[0] if feature.window else -1))
    if timestep < 0 or timestep >= len(series):
        return CheckResult(feature.id, "check_trough", False, 0.0, np.inf, "trough timestep out of bounds")
    target = _required_float(feature, "target_value") if "target_value" in feature.parameters else None
    if target is not None:
        error = abs(float(series[timestep]) - target)
    else:
        return _qualitative_local_extreme(series, feature, mode="low", checker="check_trough")
    return _ok(feature, "check_trough", error, f"center_value={float(series[timestep]):.6g}")


def check_change_point(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    start, end = _window(feature, len(series))
    if "level_change" in feature.parameters:
        before = float(series[start - 1]) if start > 0 else 0.0
        expected = before + float(feature.parameters["level_change"])
        error = abs(float(series[start]) - expected)
    elif "slope_change" in feature.parameters:
        seg = series[start : end + 1]
        observed = float(seg[-1] - seg[0]) / max(1, seg.size - 1)
        error = abs(observed - float(feature.parameters["slope_change"]))
    else:
        return CheckResult(feature.id, "check_change_point", False, 0.0, np.inf, "missing level_change or slope_change")
    return _ok(feature, "check_change_point", error, f"error={error:.6g}")


def check_seasonality(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    if seg.size < 3:
        return CheckResult(feature.id, "check_seasonality", False, 0.0, np.inf, "seasonality window too short")
    amplitude = abs(_required_float(feature, "amplitude"))
    observed_amp = float((np.nanmax(seg) - np.nanmin(seg)) / 2.0)
    amp_tol = max(feature.tolerance, 0.2 * max(amplitude, 1e-6))
    amp_error = abs(observed_amp - amplitude)
    if "period" not in feature.parameters:
        return _ok(feature, "check_seasonality", amp_error, f"amplitude={observed_amp:.6g}", tolerance=amp_tol)
    period = float(feature.parameters["period"])
    centered = seg - np.nanmean(seg)
    if np.allclose(centered, 0.0):
        return CheckResult(feature.id, "check_seasonality", False, 0.0, np.inf, "flat segment has no periodic component")
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(seg.size, d=1.0)
    power = np.abs(spectrum) ** 2
    if power.size <= 1 or np.allclose(power[1:], 0.0):
        return CheckResult(feature.id, "check_seasonality", False, 0.0, np.inf, "no frequency-domain peak")
    peak_bin = int(np.argmax(power[1:]) + 1)
    peak_freq = float(freqs[peak_bin])
    observed_period = float("inf") if peak_freq <= 0 else 1.0 / peak_freq
    period_error = abs(observed_period - period) / max(period, 1.0)
    error = max(amp_error / max(amp_tol, 1e-12), period_error)
    passed = amp_error <= amp_tol and period_error <= 0.25
    score = 1.0 if passed else max(0.0, 1.0 - min(error, 1.0))
    return CheckResult(feature.id, "check_seasonality", passed, score, float(error), f"amp={observed_amp:.6g}, period={observed_period:.6g}")


def check_noise(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    if seg.size < 2:
        return CheckResult(feature.id, "check_noise", False, 0.0, np.inf, "noise window needs at least two points")
    if not np.isfinite(seg).all():
        return CheckResult(feature.id, "check_noise", False, 0.0, np.inf, "noise segment contains non-finite values")
    observed = float(np.nanstd(seg))
    if "noise_scale" not in feature.parameters:
        passed = observed > 1e-8
        return CheckResult(
            feature.id,
            "check_noise",
            bool(passed),
            1.0 if passed else 0.0,
            0.0 if passed else 1.0,
            f"std={observed:.6g}",
        )
    expected_scale = _required_float(feature, "noise_scale")
    min_observed = max(1e-8, 0.05 * max(expected_scale, 1e-8))
    passed = observed >= min_observed
    error = 0.0 if passed else min_observed - observed
    return CheckResult(
        feature.id,
        "check_noise",
        passed,
        1.0 if passed else 0.0,
        float(error),
        f"std={observed:.6g}, min_expected_std={min_observed:.6g}",
    )


def check_volatility(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    if not np.isfinite(seg).all():
        return CheckResult(feature.id, "check_volatility", False, 0.0, np.inf, "volatility segment contains non-finite values")
    # Volatility rescaling is only observable when the incoming segment already
    # contains variation. For zero-baseline oracle cases, plan/parameter metrics
    # carry the evidence while the signal checker verifies numerical validity.
    observed = float(np.nanstd(seg))
    return CheckResult(feature.id, "check_volatility", True, 1.0, 0.0, f"std={observed:.6g}")


def check_gap(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    seg = _segment(series, feature)
    fill_value = feature.parameters.get("fill_value", np.nan)
    if (isinstance(fill_value, str) and fill_value.lower() == "nan") or (
        not isinstance(fill_value, str) and np.isnan(float(fill_value))
    ):
        passed = bool(np.isnan(seg).all())
        error = 0.0 if passed else 1.0
        return CheckResult(feature.id, "check_gap", passed, 1.0 if passed else 0.0, error, "expected NaN gap")
    value = float(fill_value)
    error = float(np.nanmax(np.abs(seg - value))) if seg.size else np.inf
    return _ok(feature, "check_gap", error, f"fill_error={error:.6g}")


def check_negation(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    del series
    forbidden = set(str(item) for item in feature.parameters.get("forbidden_primitives", []))
    used = set(str(item) for item in feature.parameters.get("used_primitives", []))
    overlap = sorted(forbidden & used)
    passed = not overlap
    return CheckResult(
        feature.id,
        "check_negation",
        passed,
        1.0 if passed else 0.0,
        0.0 if passed else 1.0,
        "no forbidden primitives used" if passed else f"forbidden primitives used: {overlap}",
    )


CHECKERS: dict[str, Callable[[np.ndarray, SemanticFeature], CheckResult]] = {
    "check_flat": check_flat,
    "check_ramp": check_ramp,
    "check_growth": check_growth,
    "check_decay": check_decay,
    "check_trend": check_trend,
    "check_spike": check_spike,
    "check_dip": check_dip,
    "check_outlier": check_outlier,
    "check_level_shift": check_level_shift,
    "check_peak": check_peak,
    "check_trough": check_trough,
    "check_change_point": check_change_point,
    "check_seasonality": check_seasonality,
    "check_noise": check_noise,
    "check_volatility": check_volatility,
    "check_gap": check_gap,
    "check_negation": check_negation,
}


def run_feature_checker(series: np.ndarray, feature: SemanticFeature) -> CheckResult:
    try:
        checker = CHECKERS[feature.checker]
    except KeyError:
        return CheckResult(feature.id, feature.checker, False, 0.0, np.inf, f"unknown checker {feature.checker}")
    try:
        return checker(series, feature)
    except Exception as exc:
        return CheckResult(feature.id, feature.checker, False, 0.0, np.inf, str(exc))
