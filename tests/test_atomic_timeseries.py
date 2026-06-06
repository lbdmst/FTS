import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from atomic_timeseries import (
    PRIMITIVE_MACHINE_METADATA,
    add_change_point,
    add_decay,
    add_dip,
    add_flat,
    add_gap,
    add_growth,
    add_level_shift,
    add_noise,
    add_outlier,
    add_peak,
    add_plateau,
    add_ramp,
    add_seasonality,
    add_spike,
    add_trend,
    add_trough,
    add_volatility,
)


class AtomicTimeSeriesTests(unittest.TestCase):
    def test_add_trend_exact_interval(self) -> None:
        series = np.zeros(6)
        result = add_trend(series, start_timestep=1, end_timestep=4, slope=2.0)
        assert_array_equal(result, np.array([0.0, 0.0, 2.0, 4.0, 6.0, 0.0]))

    def test_add_trend_accepts_offset_aliases(self) -> None:
        series = np.zeros(6)
        result = add_trend(series, start_timestep=1, end_timestep=4, start_offset=1.0, end_offset=4.0)
        assert_array_equal(result, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.0]))

    def test_add_ramp_sets_exact_endpoints(self) -> None:
        series = np.zeros(5)
        result = add_ramp(series, start_timestep=1, end_timestep=3, start_value=2.0, end_value=6.0)
        assert_array_equal(result, np.array([0.0, 2.0, 4.0, 6.0, 0.0]))

    def test_add_flat_sets_exact_values(self) -> None:
        series = np.arange(6, dtype=float)
        result = add_flat(series, start_timestep=2, end_timestep=4, target_value=7.5)
        assert_array_equal(result, np.array([0.0, 1.0, 7.5, 7.5, 7.5, 5.0]))

    def test_add_growth_and_decay_keep_exact_bounds(self) -> None:
        series = np.zeros(6)
        growth = add_growth(series, start_timestep=1, end_timestep=4, start_value=2.0, end_value=10.0, growth_rate=2.0)
        decay = add_decay(series, start_timestep=1, end_timestep=4, start_value=10.0, end_value=2.0, decay_rate=2.0)
        self.assertEqual(growth[1], 2.0)
        self.assertEqual(growth[4], 10.0)
        self.assertEqual(decay[1], 10.0)
        self.assertEqual(decay[4], 2.0)
        self.assertTrue(np.all(np.diff(growth[1:5]) >= 0))
        self.assertTrue(np.all(np.diff(decay[1:5]) <= 0))

    def test_add_peak_hits_exact_target_at_center(self) -> None:
        series = np.zeros(11)
        result = add_peak(series, timestep=5, target_value=8.0, width=1.5, start_timestep=2, end_timestep=8)
        self.assertEqual(result[5], 8.0)
        self.assertGreater(result[4], 0.0)
        self.assertGreater(result[6], 0.0)

    def test_add_trough_and_dip_are_negative_local_edits(self) -> None:
        series = np.full(9, 10.0)
        trough = add_trough(series, timestep=4, target_value=3.0, width=1.0, start_timestep=2, end_timestep=6)
        dip = add_dip(series, timestep=4, target_value=5.0)
        self.assertEqual(trough[4], 3.0)
        self.assertEqual(dip[4], 5.0)
        self.assertTrue(np.all(trough[[0, 1, 7, 8]] == 10.0))

    def test_add_spike_and_outlier_exact_timestep(self) -> None:
        series = np.zeros(7)
        spike = add_spike(series, timestep=3, target_value=4.0)
        outlier = add_outlier(series, timestep=5, target_value=-9.0)
        assert_array_equal(spike, np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]))
        self.assertEqual(outlier[5], -9.0)

    def test_add_level_shift_abrupt_and_smooth(self) -> None:
        series = np.zeros(6)
        abrupt = add_level_shift(series, start_timestep=2, shift=3.0)
        smooth = add_level_shift(series, start_timestep=2, shift=3.0, smooth=True)
        assert_array_equal(abrupt, np.array([0.0, 0.0, 3.0, 3.0, 3.0, 3.0]))
        self.assertTrue(0.0 <= smooth[2] < smooth[3] < smooth[4] < smooth[5] <= 3.0)

    def test_add_level_shift_accepts_anchor_alias(self) -> None:
        series = np.zeros(6)
        shifted = add_level_shift(series, anchor_timestep=2, shift=3.0)
        assert_array_equal(shifted, np.array([0.0, 0.0, 3.0, 3.0, 3.0, 3.0]))

    def test_add_change_point_combines_level_and_slope(self) -> None:
        series = np.zeros(6)
        result = add_change_point(series, anchor_timestep=2, level_change=2.0, slope_change=1.0)
        assert_array_equal(result, np.array([0.0, 0.0, 2.0, 3.0, 4.0, 5.0]))

    def test_add_seasonality_respects_interval(self) -> None:
        series = np.zeros(8)
        result = add_seasonality(series, start_timestep=2, end_timestep=5, amplitude=2.0, period=4.0, waveform="sine")
        assert_allclose(result[:2], 0.0)
        assert_allclose(result[6:], 0.0)
        assert_allclose(result[2:6], np.array([0.0, 2.0, 0.0, -2.0]), atol=1e-7)

    def test_add_volatility_scales_deviations_from_baseline(self) -> None:
        series = np.array([0.0, 8.0, 10.0, 12.0, 20.0])
        result = add_volatility(series, start_timestep=1, end_timestep=3, volatility_scale=2.0, baseline=10.0)
        assert_array_equal(result, np.array([0.0, 6.0, 10.0, 14.0, 20.0]))

    def test_add_gap_sets_missing_interval(self) -> None:
        series = np.arange(6, dtype=float)
        result = add_gap(series, start_timestep=2, end_timestep=4)
        self.assertTrue(np.isnan(result[2:5]).all())
        assert_array_equal(result[[0, 1, 5]], np.array([0.0, 1.0, 5.0]))

    def test_add_plateau_uses_anchor_value(self) -> None:
        series = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        result = add_plateau(series, start_timestep=1, end_timestep=3, anchor_timestep=4)
        assert_array_equal(result, np.array([0.0, 8.0, 8.0, 8.0, 8.0]))

    def test_overlap_safe_local_edits_compose_predictably(self) -> None:
        series = np.zeros(9)
        updated = add_peak(series, timestep=4, amplitude=5.0, width=1.0, start_timestep=2, end_timestep=6)
        updated = add_spike(updated, timestep=4, amplitude=3.0)
        self.assertEqual(updated[4], 8.0)
        self.assertGreater(updated[3], 0.0)
        self.assertGreater(updated[5], 0.0)

    def test_seeded_noise_is_deterministic(self) -> None:
        series = np.zeros(5)
        first = add_noise(series, noise_scale=0.5, random_seed=123)
        second = add_noise(series, noise_scale=0.5, random_seed=123)
        assert_allclose(first, second)

    def test_local_event_alias_parameters_match_existing_behavior(self) -> None:
        series = np.zeros(9)
        peak = add_peak(series, timestep=4, amplitude=3.0, spread=1.0, start_timestep=2, end_timestep=6)
        spike = add_spike(series, timestep=4, amplitude=3.0, half_width=1)
        trough = add_trough(np.full(9, 5.0), timestep=4, target_value=2.0, spread=1.0, start_timestep=2, end_timestep=6)
        self.assertEqual(peak[4], 3.0)
        self.assertEqual(spike[4], 3.0)
        self.assertEqual(trough[4], 2.0)
        self.assertGreater(spike[3], 0.0)

    def test_add_outlier_accepts_delta_alias(self) -> None:
        series = np.full(5, 2.0)
        result = add_outlier(series, timestep=2, delta=-0.5)
        assert_array_equal(result, np.array([2.0, 2.0, 1.5, 2.0, 2.0]))

    def test_public_metadata_marks_immutability(self) -> None:
        self.assertEqual(PRIMITIVE_MACHINE_METADATA["add_flat"]["mutability"], "returns_new_series")
        self.assertEqual(PRIMITIVE_MACHINE_METADATA["add_trend"]["effect_type"], "additive")

    def test_add_plateau_rejects_ambiguous_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "either target_value or anchor_timestep"):
            add_plateau(np.zeros(5), start_timestep=1, end_timestep=3, target_value=2.0, anchor_timestep=4)

    def test_add_ramp_rejects_end_value_and_slope_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "either end_value or slope"):
            add_ramp(np.zeros(5), start_timestep=1, end_timestep=3, start_value=0.0, end_value=2.0, slope=1.0)

    def test_add_trend_rejects_mixed_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "either slope or both start_value and end_value"):
            add_trend(np.zeros(5), start_timestep=0, end_timestep=4, slope=1.0, start_value=0.0, end_value=4.0)


if __name__ == "__main__":
    unittest.main()
