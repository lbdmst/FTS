from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._base import (
    FloatArray,
    _as_float_array,
    _coerce_rng,
    _interval_slice,
    _normalize_interval,
    _safe_mean,
)


def add_noise(
    series: ArrayLike,
    start_timestep: int | None = None,
    end_timestep: int | None = None,
    *,
    noise_scale: float,
    random_seed: int | None = None,
    random_generator: np.random.Generator | None = None,
) -> FloatArray:
    """Return a new series with additive Gaussian noise over an interval.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    rng = _coerce_rng(random_seed=random_seed, random_generator=random_generator)
    result[_interval_slice(start, end)] += rng.normal(0.0, float(noise_scale), size=end - start + 1)
    return result


def add_volatility(
    series: ArrayLike,
    start_timestep: int,
    end_timestep: int,
    *,
    volatility_scale: float,
    baseline: float | None = None,
) -> FloatArray:
    """Return a new series with interval deviations rescaled around a baseline.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    if volatility_scale < 0:
        raise ValueError("volatility_scale must be non-negative")
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    segment = result[_interval_slice(start, end)]
    reference = float(baseline) if baseline is not None else _safe_mean(segment, fallback=0.0)
    result[_interval_slice(start, end)] = reference + (segment - reference) * float(volatility_scale)
    return result
