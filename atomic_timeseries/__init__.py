"""Atomic primitives for deterministic time-series construction.

All public `add_*` functions are immutable:
- each call returns a new one-dimensional float array
- no primitive modifies the input series in place
- callers must always assign the return value back, for example:
  `series = add_flat(series, start_timestep=0, end_timestep=9, target_value=3.0)`
"""

from .events import add_dip, add_outlier, add_peak, add_spike, add_trough
from .metadata import PRIMITIVE_MACHINE_METADATA, get_primitive_machine_metadata
from .seasonal import add_seasonality
from .stochastic import add_noise, add_volatility
from .structural import add_gap, add_level_shift
from .trend import (
    add_change_point,
    add_decay,
    add_flat,
    add_growth,
    add_plateau,
    add_ramp,
    add_trend,
)

__all__ = [
    "add_change_point",
    "add_decay",
    "add_dip",
    "add_flat",
    "add_gap",
    "add_growth",
    "add_level_shift",
    "add_noise",
    "add_outlier",
    "add_peak",
    "add_plateau",
    "add_ramp",
    "add_seasonality",
    "add_spike",
    "add_trend",
    "add_trough",
    "add_volatility",
    "PRIMITIVE_MACHINE_METADATA",
    "get_primitive_machine_metadata",
]
