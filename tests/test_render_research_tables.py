from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.render_research_tables import render_tables
from evals.run_ablation_study import run_ablations
from evals.run_baselines import run_methods


class ResearchTableRendererTests(unittest.TestCase):
    def test_renderer_builds_main_tables_from_runner_payloads(self) -> None:
        baseline_payload = run_methods(methods=["oracle_program", "ours_full", "rule_parser"], category="atomic")
        ablation_payload = run_ablations(ablations=["full_oracle"], category="atomic")

        markdown = render_tables(baseline_payload=baseline_payload, ablation_payload=ablation_payload)

        self.assertIn("Main Table 1", markdown)
        self.assertIn("Main Table 2", markdown)
        self.assertIn("Main Table 3", markdown)
        self.assertIn("ours_full", markdown)
        self.assertNotIn("oracle_program", markdown)
        self.assertIn("full_oracle", markdown)

    def test_rendered_markdown_can_be_written_after_json_roundtrip(self) -> None:
        baseline_payload = run_methods(methods=["oracle_program"], category="atomic")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload_path = root / "baseline.json"
            table_path = root / "tables.md"
            payload_path.write_text(json.dumps(baseline_payload, indent=2, allow_nan=False), encoding="utf-8")
            loaded = json.loads(payload_path.read_text(encoding="utf-8"))
            table_path.write_text(render_tables(baseline_payload=loaded), encoding="utf-8")

            self.assertTrue(table_path.read_text(encoding="utf-8").startswith("# FaithTS Research Tables"))

    def test_renderer_adds_tsfragment_section_when_t2s_payload_is_provided(self) -> None:
        t2s_payload = {
            "overall": [
                {
                    "method": "T2S native",
                    "cases": 12,
                    "successful_cases": 12,
                    "success_rate": 1.0,
                    "mean_wape": 0.58,
                    "median_wape": 0.52,
                    "mean_corr": 0.08,
                    "median_corr": 0.01,
                    "mean_mrr": 0.10,
                    "median_mrr": 0.10,
                },
                {
                    "method": "Ours",
                    "cases": 120,
                    "successful_cases": 116,
                    "success_rate": 0.96,
                    "mean_wape": 0.41,
                    "median_wape": 0.05,
                    "mean_corr": 0.60,
                    "median_corr": 0.64,
                    "mean_mrr": 0.46,
                    "median_mrr": 0.05,
                },
                {
                    "method": "llm_direct_array",
                    "cases": 120,
                    "successful_cases": 87,
                    "success_rate": 0.725,
                    "mean_wape": 0.72,
                    "median_wape": 0.70,
                    "mean_corr": 0.10,
                    "median_corr": 0.08,
                    "mean_mrr": 0.74,
                    "median_mrr": 0.71,
                    "failure_summary": {
                        "failure_buckets": {"length_mismatch": 33},
                        "length_mismatch_delta_histogram": {"-1": 11, "-4": 4},
                        "analysis_notes": ["Length mismatches dominate."],
                    },
                },
            ]
        }

        markdown = render_tables(t2s_payload=t2s_payload)

        self.assertIn("TSFragment Real-Caption Benchmark", markdown)
        self.assertIn("t2s_native", markdown)
        self.assertIn("FaithTS", markdown)
        self.assertIn("llm_direct_array", markdown)
        self.assertIn("87/120", markdown)
        self.assertIn("length mismatch distribution", markdown)


if __name__ == "__main__":
    unittest.main()
