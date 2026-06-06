from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._base import (
    FloatArray,
    _as_float_array,
    _interval_slice,
    _normalize_interval,
    _transition_weights,
)


def add_level_shift(
    series: ArrayLike,
    start_timestep: int | None = None,
    *,
    anchor_timestep: int | None = None,
    shift: float | None = None,
    target_value: float | None = None,
    end_timestep: int | None = None,
    smooth: bool = False,
) -> FloatArray:
    """Return a new series with an additive level shift over a tail interval.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    if start_timestep is not None and anchor_timestep is not None:
        raise ValueError("Provide either start_timestep or anchor_timestep, not both")
    if anchor_timestep is not None:
        start_timestep = int(anchor_timestep)
    if start_timestep is None:
        raise ValueError("Provide start_timestep or anchor_timestep")
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    if shift is not None and target_value is not None:
        raise ValueError("Provide either shift or target_value, not both")
    if target_value is not None:
        shift = float(target_value) - float(result[start])
    if shift is None:
        raise ValueError("Provide shift or target_value")
    weights = _transition_weights(end - start + 1, smooth=smooth)
    result[_interval_slice(start, end)] += float(shift) * weights
    return result


def add_gap(
    series: ArrayLike,
    start_timestep: int,
    end_timestep: int,
    *,
    fill_value: float = np.nan,
) -> FloatArray:
    """Return a new series with an interval overwritten by a gap or fill value.

    This primitive is immutable: always assign the returned array back to `series`.
    """
    result = _as_float_array(series)
    start, end = _normalize_interval(result, start_timestep, end_timestep)
    result[_interval_slice(start, end)] = float(fill_value)
    return result
