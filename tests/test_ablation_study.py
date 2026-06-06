from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.run_baseline_benchmark import CONTROLLED_CASES_DIR
from evals.run_ablation_study import render_markdown, run_ablation, run_ablations


class AblationStudyTests(unittest.TestCase):
    def test_full_oracle_ablation_passes_atomic_upper_bound(self) -> None:
        payload = run_ablation("full_oracle", category="atomic")
        expected_cases = sum(
            1
            for case in json.loads((CONTROLLED_CASES_DIR / "single_claim_cases.json").read_text(encoding="utf-8"))
            if case.get("category") == "atomic"
        )

        self.assertEqual(payload["aggregate"]["cases"], float(expected_cases))
        self.assertEqual(payload["aggregate"]["pass_rate"], 1.0)
        self.assertEqual(payload["failure_taxonomy"]["failed_cases"], 0)

    def test_no_schema_validation_fails_despite_semantic_signal(self) -> None:
        payload = run_ablation("no_schema_validation", category="atomic")

        self.assertEqual(payload["aggregate"]["csr"], 1.0)
        self.assertEqual(payload["aggregate"]["pass_rate"], 0.0)
        self.assertGreater(payload["failure_taxonomy"]["counts"]["schema_invalid"], 0)

    def test_no_registry_grounding_is_taxonomized(self) -> None:
        payload = run_ablation("no_registry_grounding", category="atomic")

        self.assertEqual(payload["aggregate"]["pass_rate"], 0.0)
        self.assertGreater(payload["failure_taxonomy"]["counts"]["wrong_primitive"], 0)

    def test_no_rejection_increases_unsupported_hallucination(self) -> None:
        payload = run_ablation("no_rejection", category="adversarial")

        self.assertEqual(payload["aggregate"]["hus"], 1.0)
        self.assertGreater(payload["failure_taxonomy"]["counts"]["unsupported_hallucination"], 0)

    def test_repair_challenge_shows_repair_gain(self) -> None:
        without_repair = run_ablation("repair_challenge_no_repair", category="atomic")
        with_repair = run_ablation("repair_challenge_with_repair", category="atomic")

        self.assertLess(without_repair["aggregate"]["pass_rate"], with_repair["aggregate"]["pass_rate"])
        self.assertGreater(with_repair["repair_summary"]["changed_cases"], 0)
        self.assertGreater(with_repair["repair_summary"]["total_iterations"], 0)

    def test_runner_serializes_json_and_markdown(self) -> None:
        payload = run_ablations(ablations=["full_oracle", "no_rejection"], category="atomic")
        markdown = render_markdown(payload)

        self.assertIn("| Variant | Cases |", markdown)
        self.assertIn("full_oracle", markdown)
        json.dumps(payload, allow_nan=False)

    def test_cli_style_outputs_are_writable_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = run_ablations(ablations=["full_oracle"], category="atomic")
            json_path = root / "ablations.json"
            md_path = root / "ablations.md"
            json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            md_path.write_text(render_markdown(payload), encoding="utf-8")

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.read_text(encoding="utf-8").startswith("# Ablation Study"))


if __name__ == "__main__":
    unittest.main()
