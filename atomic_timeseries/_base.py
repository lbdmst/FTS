from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit


FloatArray = NDArray[np.float64]


def _as_float_array(series: ArrayLike) -> FloatArray:
    array = np.asarray(series, dtype=float)
    if array.ndim != 1:
        raise ValueError("series must be one-dimensional")
    return array.astype(float, copy=True)


def _normalize_interval(
    series: FloatArray,
    start_timestep: int | None = None,
    end_timestep: int | None = None,
) -> tuple[int, int]:
    start = 0 if start_timestep is None else int(start_timestep)
    end = len(series) - 1 if end_timestep is None else int(end_timestep)
    if start < 0 or end < 0:
        raise ValueError("timesteps must be non-negative")
    if start >= len(series) or end >= len(series):
        raise ValueError("timesteps must fall within the series length")
    if end < start:
        raise ValueError("end_timestep must be greater than or equal to start_timestep")
    return start, end


def _normalize_timestep(series: FloatArray, timestep: int) -> int:
    index = int(timestep)
    if index < 0 or index >= len(series):
        raise ValueError("timestep must fall within the series length")
    return index


def _interval_slice(start: int, end: int) -> slice:
    return slice(start, end + 1)


def _interval_positions(start: int, end: int) -> FloatArray:
    return np.arange(start, end + 1, dtype=float)


def _resolve_amplitude(
    current_value: float,
    amplitude: float | None = None,
    target_value: float | None = None,
) -> float:
    if amplitude is not None and target_value is not None:
        raise ValueError("Provide either amplitude or target_value, not both")
    if target_value is not None:
        return float(target_value) - float(current_value)
    if amplitude is None:
        raise ValueError("One of amplitude or target_value must be provided")
    return float(amplitude)


def _coerce_rng(
    random_seed: int | None = None,
    random_generator: np.random.Generator | None = None,
) -> np.random.Generator:
    if random_seed is not None and random_generator is not None:
        raise ValueError("Provide either random_seed or random_generator, not both")
    if random_generator is not None:
        return random_generator
    return np.random.default_rng(random_seed)


def _abrupt_weights(length: int) -> FloatArray:
    if length <= 0:
        raise ValueError("length must be positive")
    return np.ones(length, dtype=float)


def _smooth_step_weights(length: int, steepness: float = 10.0) -> FloatArray:
    if length <= 1:
        return np.array([1.0], dtype=float)
    x = np.linspace(-1.0, 1.0, num=length, dtype=float)
    weights = expit(steepness * x)
    return (weights - weights[0]) / (weights[-1] - weights[0])


def _transition_weights(length: int, smooth: bool) -> FloatArray:
    return _smooth_step_weights(length) if smooth else _abrupt_weights(length)


def _safe_mean(values: Iterable[float], fallback: float) -> float:
    data = np.asarray(list(values), dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return float(fallback)
    return float(np.mean(finite))
