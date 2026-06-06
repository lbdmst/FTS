from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from ._base import FloatArray, _as_float_array, _interval_positions, _interval_slice, _normalize_interval


def add_seasonality(
    series: ArrayLike,
    start_timestep: int | None = None,
    end_timestep: int | None = None,
    *,
    amplitude: float,
    period: float,
    phase: float = 0.0,
    baseline: float = 0.0,
    waveform: str = "sine",
) -> FloatArray:
    """Return a new series with an additive periodic waveform over an interval.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    positions = _interval_positions(start, end)
    angle = (2.0 * np.pi * (positions - start) / float(period)) + float(phase)
    waveforms = {
        "sine": np.sin(angle),
        "cosine": np.cos(angle),
        "square": signal.square(angle),
        "sawtooth": signal.sawtooth(angle),
        "triangle": signal.sawtooth(angle, width=0.5),
    }
    if waveform not in waveforms:
        raise ValueError(f"Unsupported waveform: {waveform}")
    result[_interval_slice(start, end)] += float(baseline) + float(amplitude) * waveforms[waveform]
    return result
