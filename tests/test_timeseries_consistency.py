import unittest

import numpy as np

from evals.timeseries_consistency import evaluate_description_series_consistency


class TimeseriesConsistencyTests(unittest.TestCase):
    def test_upward_trend_and_change_point_score_high_for_matching_series(self) -> None:
        series = np.concatenate([np.zeros(50), np.linspace(0.0, 4.0, 50)])
        description = (
            "The time series shows an upward trend. "
            "After timestep 50 there is a clear shift and the series climbs."
        )
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["aspect_scores"]["overall_trend_up"], 0.8)
        self.assertGreaterEqual(report["aspect_scores"]["change_point_alignment"], 0.8)
        self.assertGreaterEqual(report["consistency_score"], 0.8)
        self.assertGreaterEqual(report["confidence_score"], 0.7)

    def test_downward_description_scores_poorly_on_upward_series(self) -> None:
        series = np.linspace(0.0, 5.0, 100)
        description = "The signal exhibits a clear downward trend."
        report = evaluate_description_series_consistency(description, series)
        self.assertLessEqual(report["aspect_scores"]["overall_trend_down"], 0.35)
        self.assertEqual(report["summary_verdict"], "inconsistent")

    def test_peak_and_trough_claims_use_timing_and_value_alignment(self) -> None:
        x = np.linspace(-2.0, 2.0, 101)
        series = 10.0 - x**2
        description = "A peak is observed at timestep 50 with value 10.0."
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["aspect_scores"]["peak_alignment"], 0.95)
        self.assertEqual(report["matched_claims"][0]["name"], "peak_alignment")

    def test_low_volatility_claim_scores_high_on_nearly_constant_series(self) -> None:
        series = np.full(100, 3.0)
        series[40:45] += 0.001
        description = "The series remains stable with very low volatility and a flat profile."
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["aspect_scores"]["volatility_low"], 0.9)
        self.assertGreaterEqual(report["aspect_scores"]["overall_trend_flat_or_stable"], 0.9)
        self.assertEqual(report["summary_verdict"], "consistent")

    def test_data_point_claims_are_one_based_and_value_checked(self) -> None:
        series = np.array([16.672, 17.938, 18.501, 19.416, 18.0, 17.0, 16.0, 15.476], dtype=float)
        description = (
            "The time series starts at a peak of 19.416 at the 4.0 data point, "
            "then exhibits a decline to 15.476 by the 8.0 data point."
        )
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["aspect_scores"]["point_value_3"], 0.95)
        self.assertGreaterEqual(report["aspect_scores"]["point_value_7"], 0.95)

        mismatched = series.copy()
        mismatched[7] = 16.7
        mismatch_report = evaluate_description_series_consistency(description, mismatched)
        self.assertLessEqual(mismatch_report["aspect_scores"]["point_value_7"], 0.4)
        self.assertTrue(
            any(claim["name"] == "point_value_7" for claim in mismatch_report["mismatched_claims"])
        )

    def test_data_point_range_bounds_are_checked(self) -> None:
        series = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 4.5, 4.2, 4.1], dtype=float)
        description = "Values fluctuate between 4.0 and 5.0 from the 4.0 to 6.0 data points."
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["aspect_scores"]["range_bounds_3_5"], 0.95)

    def test_weak_directional_trend_matches_when_direction_is_correct(self) -> None:
        series = np.linspace(0.599, 0.643, 96)
        description = "The time series data demonstrates a general upward trend in values."
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["aspect_scores"]["overall_trend_up"], 0.6)
        self.assertTrue(any(claim["name"] == "overall_trend_up" for claim in report["matched_claims"]))

    def test_supported_broad_trend_is_consistent_without_hard_score_cutoff(self) -> None:
        series = np.linspace(1.0, 1.1, 24)
        description = "The series shows an overall upward trend."
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["caption_correctness_score"], 0.6)
        self.assertLess(report["caption_correctness_score"], 0.85)
        self.assertEqual(report["summary_verdict"], "consistent")

    def test_supported_minor_fluctuation_summary_is_consistent(self) -> None:
        series = np.array([1.0, 1.02, 0.99, 1.01] * 6, dtype=float)
        description = "The series shows minor fluctuations."
        report = evaluate_description_series_consistency(description, series)
        self.assertIn("fluctuation_minor", report["aspect_scores"])
        self.assertTrue(any(claim["name"] == "fluctuation_minor" for claim in report["matched_claims"]))
        self.assertEqual(report["summary_verdict"], "consistent")

    def test_local_stable_range_does_not_become_overall_flat_claim(self) -> None:
        series = np.concatenate([np.linspace(2.0, 1.0, 20), np.linspace(1.0, 3.0, 40), np.full(36, 2.5)])
        description = (
            "The time series exhibits a fluctuating trend with initial decreases, followed by growth and recovery phases. "
            "It rebounds to a stable range before ending with minor fluctuations."
        )
        report = evaluate_description_series_consistency(description, series)
        self.assertNotIn("overall_trend_flat_or_stable", report["aspect_scores"])

    def test_peak_point_value_claim_accepts_peaking_at_data_point_wording(self) -> None:
        series = np.linspace(14.0, 16.0, 24)
        series[12] = 19.416
        description = "The series shows an upward trend, reaching a peak at 19.416 around the 13.0 data point."
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["aspect_scores"]["point_value_12"], 0.95)
        self.assertGreaterEqual(report["aspect_scores"]["peak_alignment"], 0.95)

    def test_unit_before_position_wording_is_supported(self) -> None:
        series = np.array([16.391, 15.0, 14.0, 13.718, 14.2, 19.416, 18.0, 17.0], dtype=float)
        description = (
            "The time series begins at a high point of 16.391 at data point 1.0, "
            "then rises until it peaks at 19.416 around point 6.0."
        )
        report = evaluate_description_series_consistency(description, series)
        self.assertGreaterEqual(report["aspect_scores"]["point_value_0"], 0.95)
        self.assertGreaterEqual(report["aspect_scores"]["point_value_5"], 0.95)
        self.assertGreaterEqual(report["aspect_scores"]["peak_alignment"], 0.95)

    def test_peak_position_without_value_is_not_treated_as_huge_value_anchor(self) -> None:
        series = np.array([0.013, 0.02, 0.03, 0.048, 0.03, 0.02, 0.01, 0.004], dtype=float)
        description = (
            "Values rise from 0.013 to 0.048 until reaching a peak at 10.0, "
            "after which a gradual decrease occurs, ending at 0.004."
        )
        report = evaluate_description_series_consistency(description, series)
        self.assertFalse(any(name.startswith("point_value_") for name in report["aspect_scores"]))

    def test_caption_correctness_penalizes_explicit_contradictions(self) -> None:
        series = np.linspace(0.0, 1.0, 16)
        description = "The signal has an upward trend and reaches value 10.0 at timestep 8."
        report = evaluate_description_series_consistency(description, series)
        self.assertLess(report["caption_correctness_score"], report["consistency_score"])
        self.assertGreater(report["correctness_details"]["contradicted_weight_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
