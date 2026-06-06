from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._base import (
    FloatArray,
    _as_float_array,
    _interval_slice,
    _normalize_interval,
    _normalize_timestep,
    _resolve_amplitude,
)


def _gaussian_bump(
    positions: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    if width <= 0:
        raise ValueError("width must be positive")
    return np.exp(-0.5 * ((positions - center) / float(width)) ** 2)


def _validate_window_contains_center(start: int, end: int, center: int) -> None:
    if not start <= center <= end:
        raise ValueError("The local edit interval must include the target timestep")


def add_peak(
    series: ArrayLike,
    timestep: int,
    *,
    amplitude: float | None = None,
    target_value: float | None = None,
    width: float = 1.0,
    spread: float | None = None,
    start_timestep: int | None = None,
    end_timestep: int | None = None,
) -> FloatArray:
    """Return a new series with an additive positive Gaussian-shaped local maximum.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    center = _normalize_timestep(result, timestep)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    _validate_window_contains_center(start, end, center)
    if spread is not None:
        if width != 1.0:
            raise ValueError("Provide either width or spread, not both")
        width = float(spread)
    positions = np.arange(start, end + 1, dtype=float)
    bump = _gaussian_bump(positions, center=float(center), width=float(width))
    amp = _resolve_amplitude(result[center], amplitude=amplitude, target_value=target_value)
    result[_interval_slice(start, end)] += amp * bump
    return result


def add_trough(
    series: ArrayLike,
    timestep: int,
    *,
    amplitude: float | None = None,
    target_value: float | None = None,
    width: float = 1.0,
    spread: float | None = None,
    start_timestep: int | None = None,
    end_timestep: int | None = None,
) -> FloatArray:
    """Return a new series with an additive negative Gaussian-shaped local minimum.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    center = _normalize_timestep(result, timestep)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    _validate_window_contains_center(start, end, center)
    if spread is not None:
        if width != 1.0:
            raise ValueError("Provide either width or spread, not both")
        width = float(spread)
    positions = np.arange(start, end + 1, dtype=float)
    bump = _gaussian_bump(positions, center=float(center), width=float(width))
    amp = _resolve_amplitude(result[center], amplitude=amplitude, target_value=target_value)
    result[_interval_slice(start, end)] -= abs(amp) * bump
    return result


def add_spike(
    series: ArrayLike,
    timestep: int,
    *,
    amplitude: float | None = None,
    target_value: float | None = None,
    width: int = 0,
    half_width: int | None = None,
) -> FloatArray:
    """Return a new series with an additive sharp positive event.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    center = _normalize_timestep(result, timestep)
    amp = _resolve_amplitude(result[center], amplitude=amplitude, target_value=target_value)
    if half_width is not None:
        if width != 0:
            raise ValueError("Provide either width or half_width, not both")
        width = int(half_width)
    if width <= 0:
        result[center] += amp
        return result
    start = max(0, center - width)
    end = min(len(result) - 1, center + width)
    triangle = 1.0 - np.abs(np.arange(start, end + 1) - center) / float(width + 1)
    result[_interval_slice(start, end)] += amp * triangle
    return result


def add_dip(
    series: ArrayLike,
    timestep: int,
    *,
    amplitude: float | None = None,
    target_value: float | None = None,
    width: int = 0,
    half_width: int | None = None,
) -> FloatArray:
    """Return a new series with an additive sharp negative event.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    center = _normalize_timestep(result, timestep)
    amp = _resolve_amplitude(result[center], amplitude=amplitude, target_value=target_value)
    if half_width is not None:
        if width != 0:
            raise ValueError("Provide either width or half_width, not both")
        width = int(half_width)
    if width <= 0:
        result[center] -= abs(amp)
        return result
    start = max(0, center - width)
    end = min(len(result) - 1, center + width)
    triangle = 1.0 - np.abs(np.arange(start, end + 1) - center) / float(width + 1)
    result[_interval_slice(start, end)] -= abs(amp) * triangle
    return result


def add_outlier(
    series: ArrayLike,
    timestep: int,
    *,
    target_value: float | None = None,
    relative_change: float | None = None,
    delta: float | None = None,
) -> FloatArray:
    """Return a new series with a one-point outlier edit.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    index = _normalize_timestep(result, timestep)
    if relative_change is not None and delta is not None:
        raise ValueError("Provide either relative_change or delta, not both")
    if delta is not None:
        relative_change = float(delta)
    if target_value is not None and relative_change is not None:
        raise ValueError("Provide either target_value or relative_change, not both")
    if target_value is not None:
        result[index] = float(target_value)
    elif relative_change is not None:
        result[index] += float(relative_change)
    else:
        raise ValueError("Provide target_value or relative_change")
    return result
