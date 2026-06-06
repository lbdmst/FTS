from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.baselines.llm_direct import parse_direct_code_output
from evals.run_llm_baseline_outputs import evaluate_t2s_llm_outputs, generate_many_outputs, generate_outputs, merge_t2s_llm_results


class RunLLMBaselineOutputsTests(unittest.TestCase):
    def test_missing_api_key_writes_skip_manifest(self) -> None:
        case = {
            "case_id": "llm_skip_case",
            "category": "atomic",
            "description": "The series remains flat at 2.0 across all 4 timesteps.",
            "length": 4,
            "expected_plan_outline": {"steps": [], "unresolved_semantics": [], "needs_library_extension": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.json"
            output_dir = root / "outputs"
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "", "PROTOTYPE_CODEGEN_API_KEY": ""}, clear=False):
                summary = generate_outputs(
                    baseline="llm_direct_array",
                    output_dir=output_dir,
                    case_paths=[cases_path],
                )

            manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "skipped")
        self.assertEqual(manifest["skip_reason"], "missing_api_key")

    def test_t2s_llm_outputs_are_evaluated_and_mergeable(self) -> None:
        case = {
            "case_id": "t2s_tiny",
            "category": "t2s_real_smoke",
            "source_dataset": "tiny",
            "source_length": 4,
            "description": "A short increasing sequence.",
            "length": 4,
            "reference_series": [1.0, 2.0, 3.0, 4.0],
            "numeric_range": {"min": 0.0, "max": 5.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "t2s_cases.json"
            output_root = root / "outputs"
            comparison_path = root / "comparison.json"
            cases_path.write_text(json.dumps([case]), encoding="utf-8")
            method_dir = output_root / "llm_direct_array"
            method_dir.mkdir(parents=True)
            (method_dir / "t2s_tiny.txt").write_text('{"series": [1, 2, 3, 4]}', encoding="utf-8")
            comparison_path.write_text(
                json.dumps({"version": "1.0", "overall": [{"method": "Ours", "cases": 1}], "artifacts": {}}),
                encoding="utf-8",
            )

            payload = evaluate_t2s_llm_outputs(
                output_root=output_root,
                case_path=cases_path,
                methods=["llm_direct_array"],
            )
            merged = merge_t2s_llm_results(comparison_path=comparison_path, llm_payload=payload)

        result = payload["results"][0]
        self.assertEqual(result["successful_cases"], 1)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(result["mean_wape"], 0.0)
        self.assertIn("llm_direct_array", [row["method"] for row in merged["overall"]])

    def test_generate_many_outputs_uses_t2s_default_case_file_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "outputs"
            fake_case_file = root / "t2s_cases.json"
            fake_case_file.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "t2s_fake",
                            "category": "t2s_real_smoke",
                            "description": "A short increasing sequence.",
                            "length": 4,
                            "reference_series": [1.0, 2.0, 3.0, 4.0],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch("evals.run_llm_baseline_outputs.DEFAULT_T2S_CASE_FILES", [fake_case_file]):
                with patch.dict(
                    os.environ,
                    {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "", "PROTOTYPE_CODEGEN_API_KEY": ""},
                    clear=False,
                ):
                    summary = generate_many_outputs(
                        baselines=["llm_direct_array"],
                        output_root=output_root,
                        use_t2s_main_benchmark=True,
                    )

        self.assertEqual(summary["case_paths"], [str(fake_case_file)])
        self.assertEqual(summary["summaries"][0]["total_cases"], 1)

    def test_parse_direct_code_output_accepts_helper_function_and_print(self) -> None:
        series = parse_direct_code_output(
            """```python
import numpy as np

def generate_series():
    np.random.seed(42)
    series = np.zeros(4)
    series[:] = np.linspace(1.0, 4.0, 4)
    return series

series = generate_series()
print(series)
```""",
            4,
        )

        self.assertEqual(series.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_parse_direct_code_output_seeds_unseeded_numpy_random(self) -> None:
        code = """```python
import numpy as np

rng = np.random.default_rng()
series = rng.normal(size=4)
```"""
        first = parse_direct_code_output(code, 4)
        second = parse_direct_code_output(code, 4)

        self.assertEqual(first.tolist(), second.tolist())


if __name__ == "__main__":
    unittest.main()
