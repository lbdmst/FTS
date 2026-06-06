from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.render_claim_evidence import build_claim_evidence_report, render_markdown
from evals.run_ablation_study import run_ablations
from evals.run_baselines import run_methods
from evals.run_revision_risk_experiments import run_revision_risk_experiments


class ClaimEvidenceRendererTests(unittest.TestCase):
    def test_claim_evidence_marks_supported_and_pending_claims(self) -> None:
        baseline_payload = run_methods(methods=["oracle_program", "ours_full", "rule_parser"], category="atomic")
        ablation_payload = run_ablations(
            ablations=["full_oracle", "no_registry_grounding", "repair_challenge_no_repair", "repair_challenge_with_repair"],
            category="atomic",
        )

        report = build_claim_evidence_report(baseline_payload=baseline_payload, ablation_payload=ablation_payload)
        statuses = {item["claim_id"]: item["status"] for item in report["claims"]}

        self.assertEqual(statuses["H2_registry_grounding_matters"], "supported")
        self.assertEqual(statuses["H3_static_repair_improves_reliability"], "supported")
        self.assertEqual(statuses["H6_edit_locality_pending"], "pending")
        self.assertEqual(statuses["H8_public_text_generator_baseline_pending"], "pending")
        self.assertEqual(report["idea_evaluation"]["verdict"], "Accept with Revisions")

    def test_claim_evidence_uses_revision_risk_report(self) -> None:
        baseline_payload = run_methods(methods=["oracle_program", "ours_full", "rule_parser"], category="atomic")
        ablation_payload = run_ablations(
            ablations=["full_oracle", "no_registry_grounding", "repair_challenge_no_repair", "repair_challenge_with_repair"],
            category="atomic",
        )
        with tempfile.TemporaryDirectory() as tmp:
            revision_payload = run_revision_risk_experiments(
                report_dir=Path(tmp),
                category="atomic",
                latency_methods=["oracle_program"],
                latency_repeats=1,
            )
        report = build_claim_evidence_report(
            baseline_payload=baseline_payload,
            ablation_payload=ablation_payload,
            revision_payload=revision_payload,
        )
        statuses = {item["claim_id"]: item["status"] for item in report["claims"]}

        self.assertEqual(statuses["H6_edit_locality_pending"], "partial")
        self.assertEqual(statuses["H7_cost_latency_pending"], "partial")
        self.assertEqual(statuses["H8_public_text_generator_baseline_pending"], "supported")

    def test_markdown_and_json_are_writable(self) -> None:
        baseline_payload = run_methods(methods=["oracle_program", "ours_full", "rule_parser"], category="atomic")
        ablation_payload = run_ablations(
            ablations=["full_oracle", "no_registry_grounding", "repair_challenge_no_repair", "repair_challenge_with_repair"],
            category="atomic",
        )
        report = build_claim_evidence_report(baseline_payload=baseline_payload, ablation_payload=ablation_payload)
        markdown = render_markdown(report)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "claim_evidence.json"
            md_path = root / "claim_evidence.md"
            json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            md_path.write_text(markdown, encoding="utf-8")

            self.assertIn("Idea Evaluator Verdict", md_path.read_text(encoding="utf-8"))
            self.assertTrue(json.loads(json_path.read_text(encoding="utf-8"))["claims"])


if __name__ == "__main__":
    unittest.main()
