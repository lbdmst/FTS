from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._base import (
    FloatArray,
    _as_float_array,
    _interval_positions,
    _interval_slice,
    _normalize_interval,
    _normalize_timestep,
    _transition_weights,
)


def add_trend(
    series: ArrayLike,
    start_timestep: int | None = None,
    end_timestep: int | None = None,
    *,
    slope: float | None = None,
    start_value: float | None = None,
    end_value: float | None = None,
    start_offset: float | None = None,
    end_offset: float | None = None,
) -> FloatArray:
    """Return a new series with an additive linear trend over an inclusive interval.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    positions = _interval_positions(start, end) - start
    if (start_value is not None or end_value is not None) and (start_offset is not None or end_offset is not None):
        raise ValueError("Provide either start_value/end_value or start_offset/end_offset, not both")
    if start_offset is not None:
        start_value = float(start_offset)
    if end_offset is not None:
        end_value = float(end_offset)
    if slope is not None and (start_value is not None or end_value is not None):
        raise ValueError("Provide either slope or both start_value and end_value, not both modes")
    if slope is not None:
        delta = positions * float(slope)
    elif start_value is not None and end_value is not None:
        delta = np.linspace(float(start_value), float(end_value), end - start + 1)
    else:
        raise ValueError("Provide slope or both start_value and end_value")
    result[_interval_slice(start, end)] += delta
    return result


def add_ramp(
    series: ArrayLike,
    start_timestep: int,
    end_timestep: int,
    *,
    start_value: float | None = None,
    end_value: float | None = None,
    slope: float | None = None,
) -> FloatArray:
    """Return a new series with an interval overwritten by a linear ramp.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    length = end - start + 1
    if start_value is None:
        start_value = float(result[start])
    if end_value is not None and slope is not None:
        raise ValueError("Provide either end_value or slope, not both")
    if end_value is None:
        if slope is None:
            raise ValueError("Provide end_value or slope")
        end_value = float(start_value) + float(slope) * (length - 1)
    result[_interval_slice(start, end)] = np.linspace(float(start_value), float(end_value), length)
    return result


def add_flat(
    series: ArrayLike,
    start_timestep: int,
    end_timestep: int,
    *,
    target_value: float,
) -> FloatArray:
    """Return a new series with every timestep in the interval overwritten.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    result[_interval_slice(start, end)] = float(target_value)
    return result


def add_growth(
    series: ArrayLike,
    start_timestep: int,
    end_timestep: int,
    *,
    start_value: float | None = None,
    end_value: float | None = None,
    growth_rate: float = 1.0,
) -> FloatArray:
    """Return a new series with an interval overwritten by an exponential growth curve.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    if growth_rate <= 0:
        raise ValueError("growth_rate must be positive")
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    length = end - start + 1
    start_value = float(result[start]) if start_value is None else float(start_value)
    end_value = float(result[end]) if end_value is None else float(end_value)
    weights = np.expm1(np.linspace(0.0, float(growth_rate), length))
    if np.allclose(weights[-1], 0.0):
        curve = np.full(length, start_value)
    else:
        weights = weights / weights[-1]
        curve = start_value + (end_value - start_value) * weights
    result[_interval_slice(start, end)] = curve
    return result


def add_decay(
    series: ArrayLike,
    start_timestep: int,
    end_timestep: int,
    *,
    start_value: float | None = None,
    end_value: float | None = None,
    decay_rate: float = 1.0,
) -> FloatArray:
    """Return a new series with an interval overwritten by an exponential decay curve.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    if decay_rate <= 0:
        raise ValueError("decay_rate must be positive")
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    length = end - start + 1
    start_value = float(result[start]) if start_value is None else float(start_value)
    end_value = float(result[end]) if end_value is None else float(end_value)
    weights = np.exp(-np.linspace(0.0, float(decay_rate), length))
    if np.isclose(weights[0], weights[-1]):
        normalized = np.zeros(length, dtype=float)
    else:
        normalized = (weights[0] - weights) / (weights[0] - weights[-1])
    curve = start_value + (end_value - start_value) * normalized
    result[_interval_slice(start, end)] = curve
    return result


def add_plateau(
    series: ArrayLike,
    start_timestep: int,
    end_timestep: int,
    *,
    target_value: float | None = None,
    anchor_timestep: int | None = None,
) -> FloatArray:
    """Return a new series with an interval overwritten by a flat plateau.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    if target_value is not None and anchor_timestep is not None:
        raise ValueError("Provide either target_value or anchor_timestep, not both")
    if target_value is None:
        if anchor_timestep is None:
            raise ValueError("Provide target_value or anchor_timestep")
        anchor = _normalize_timestep(result, anchor_timestep)
        target_value = float(result[anchor])
    return add_flat(result, start_timestep=start, end_timestep=end, target_value=float(target_value))


def add_change_point(
    series: ArrayLike,
    anchor_timestep: int,
    *,
    level_change: float = 0.0,
    slope_change: float = 0.0,
    end_timestep: int | None = None,
    smooth: bool = False,
) -> FloatArray:
    """Return a new series with an additive post-anchor level and/or slope change.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    start, end = _normalize_interval(result, anchor_timestep, end_timestep)
    if level_change == 0.0 and slope_change == 0.0:
        raise ValueError("Provide a non-zero level_change or slope_change")
    length = end - start + 1
    weights = _transition_weights(length, smooth=smooth)
    positions = np.arange(length, dtype=float)
    result[_interval_slice(start, end)] += weights * float(level_change) + positions * float(slope_change)
    return result
