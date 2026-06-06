from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.run_llm_baseline_outputs import DIRECT_LLM_T2S_BASELINES, evaluate_t2s_llm_outputs


DEFAULT_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_outputs_2500"
DEFAULT_JSON_OUTPUT = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_eval_2500.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_eval_2500.md"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Direct LLM TSFragment-600K Evaluation",
        "",
        f"- case_file: `{report['case_file']}`",
        f"- output_root: `{report['output_root']}`",
        f"- total_cases: `{report['total_cases']}`",
        "",
        "| Method | Successful | Success rate | Mean WAPE | Median WAPE | Mean Corr | Median Corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    result["method"],
                    str(result["successful_cases"]),
                    _fmt(result["success_rate"]),
                    _fmt(result["mean_wape"]),
                    _fmt(result["median_wape"]),
                    _fmt(result["mean_corr"]),
                    _fmt(result["median_corr"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failure Summary", ""])
    for result in report["results"]:
        lines.append(f"### {result['method']}")
        failure_summary = result.get("failure_summary") or {}
        if not failure_summary:
            lines.append("")
            continue
        for key, value in failure_summary.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved direct LLM outputs on the shared TSFragment case file.")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", default=DIRECT_LLM_T2S_BASELINES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    report = evaluate_t2s_llm_outputs(
        output_root=args.output_root,
        case_path=args.case_file,
        methods=list(args.methods),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {_display(args.markdown_output)}")
    print(f"Wrote {_display(args.json_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
