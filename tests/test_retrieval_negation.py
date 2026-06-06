import unittest

import coverage_audit


class RetrievalNegationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = coverage_audit.build_catalog()

    def test_negated_peak_trough_and_mean_reversion_do_not_tag(self) -> None:
        caption = (
            "The series is constant with no peaks or troughs, "
            "nor is there any evidence of mean reversion."
        )
        tags = coverage_audit.infer_semantics(caption)
        self.assertNotIn("peak", tags)
        self.assertNotIn("trough", tags)
        self.assertNotIn("mean_reversion", tags)

    def test_negated_trend_does_not_tag(self) -> None:
        caption = "There is no upward or downward trend and the signal remains flat."
        tags = coverage_audit.infer_semantics(caption)
        self.assertNotIn("trend_up", tags)
        self.assertNotIn("trend_down", tags)
        self.assertIn("flat", tags)

    def test_negated_cues_do_not_retrieve_primitives(self) -> None:
        caption = (
            "The provided time series exhibits a constant value with no peaks or troughs "
            "and no evidence of mean reversion."
        )
        row = coverage_audit.evaluate_caption(caption, self.catalog)
        self.assertNotIn("add_peak", row["candidate_primitives"])
        self.assertNotIn("add_trough", row["candidate_primitives"])
        self.assertNotIn("mean_reversion", row["true_primitive_gaps"])

    def test_noise_language_prefers_noise_over_volatility_scaling(self) -> None:
        caption = (
            "The last 30 timesteps become noisy with random jitter, "
            "but there is no claim that the existing swings are being rescaled."
        )
        row = coverage_audit.evaluate_caption(caption, self.catalog)
        self.assertIn("add_noise", row["candidate_primitives"])
        self.assertNotIn("add_volatility", row["candidate_primitives"])

    def test_oscillation_without_periodicity_is_weak_and_not_seasonal(self) -> None:
        caption = (
            "The signal oscillates mildly around a stable level, "
            "but the description does not indicate any repeating period or seasonal cycle."
        )
        row = coverage_audit.evaluate_caption(caption, self.catalog)
        self.assertIn("add_volatility", row["candidate_primitives"])
        self.assertIn("add_noise", row["candidate_primitives"])
        self.assertNotIn("add_seasonality", row["candidate_primitives"])
        self.assertEqual(row["retrieval_status"], "weak")

    def test_soft_level_reference_retrieves_existing_composition_not_gap(self) -> None:
        caption = (
            "The series recovers after an early decline, then fluctuates with "
            "resistance around 3500 before drifting lower."
        )
        row = coverage_audit.evaluate_caption(caption, self.catalog)
        self.assertIn("soft_level_reference", row["orchestration_level_descriptors"])
        self.assertIn("add_volatility", row["candidate_primitives"])
        self.assertIn("add_flat", row["candidate_primitives"])
        self.assertNotIn("soft_level_reference", row["true_primitive_gaps"])
        self.assertNotEqual(row["retrieval_status"], "gap")


if __name__ == "__main__":
    unittest.main()
