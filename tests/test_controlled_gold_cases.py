import unittest

from evals.run_controlled_gold_cases import CONTROLLED_CASES_DIR, run_cases
from faithts.gold_adapter import is_core_evaluable, load_gold_cases


class ControlledGoldCaseTests(unittest.TestCase):
    def test_gold_case_runner_loads_all_categories(self) -> None:
        category_to_path = {
            "atomic": CONTROLLED_CASES_DIR / "single_claim_cases.json",
            "compositional": CONTROLLED_CASES_DIR / "multi_claim_cases.json",
            "adversarial": CONTROLLED_CASES_DIR / "robustness_cases.json",
        }
        for category, path in category_to_path.items():
            self.assertTrue(path.exists())
            results = run_cases([path], category=category)
            self.assertTrue(results, f"No results loaded for {category}")
            self.assertTrue(all(result.case_id for result in results))
            self.assertTrue(all(result.category == category for result in results))

    def test_atomic_gold_cases_currently_pass(self) -> None:
        results = run_cases([CONTROLLED_CASES_DIR / "single_claim_cases.json"], category="atomic")
        failed = [result.case_id for result in results if not result.passed]
        self.assertFalse(failed, f"Atomic gold cases failed: {failed}")

    def test_compositional_gold_cases_currently_pass(self) -> None:
        results = run_cases([CONTROLLED_CASES_DIR / "multi_claim_cases.json"], category="compositional")
        failed = [result.case_id for result in results if not result.passed]
        self.assertFalse(failed, f"Compositional gold cases failed: {failed}")

    def test_adversarial_gold_cases_currently_pass(self) -> None:
        results = run_cases([CONTROLLED_CASES_DIR / "robustness_cases.json"], category="adversarial")
        failed = [result.case_id for result in results if not result.passed]
        self.assertFalse(failed, f"Adversarial gold cases failed: {failed}")

    def test_core_benchmark_gold_cases_are_evaluable(self) -> None:
        paths = [
            CONTROLLED_CASES_DIR / "single_claim_cases.json",
            CONTROLLED_CASES_DIR / "multi_claim_cases.json",
            CONTROLLED_CASES_DIR / "robustness_cases.json",
        ]
        cases = load_gold_cases(paths)
        skipped = [case["case_id"] for case in cases if not is_core_evaluable(case)]
        self.assertFalse(skipped, f"Core benchmark would skip cases: {skipped}")


if __name__ == "__main__":
    unittest.main()
