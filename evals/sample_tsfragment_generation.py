from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.tsfragment_eval_workflow import (
    build_runtime,
    run_generation_case,
)


CASES_PATH = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
OUTPUT_JSON = REPO_ROOT / "evals" / "tsfragment_eval_sample_outputs.json"
OUTPUT_MD = REPO_ROOT / "evals" / "tsfragment_eval_sample_outputs.md"


def run_sample(seed: int, sample_count: int, provider: str) -> dict[str, Any]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(cases)), sample_count))
    runtime = build_runtime(provider)

    records: list[dict[str, Any]] = []
    for index in indices:
        case = cases[index]
        caption = str(case["description"])
        truth = np.asarray(case["reference_series"], dtype=np.float64)
        record = run_generation_case(
            runtime=runtime,
            index=index,
            description=caption,
            truth=truth,
            target_path=str(case.get("case_id", f"<sample_{index}>")),
        )
        records.append(record)

    return {
        "seed": seed,
        "sample_count": sample_count,
        "provider": provider,
        "cases_path": str(CASES_PATH.relative_to(REPO_ROOT)),
        "registry_validation": runtime["registry_validation"],
        "sample_indices": indices,
        "sample_case_ids": [str(cases[index].get("case_id", index)) for index in indices],
        "samples": records,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TSFragment-Eval Sample Generation",
        "",
        f"- seed: `{report['seed']}`",
        f"- sample_count: `{report['sample_count']}`",
        f"- provider: `{report['provider']}`",
        f"- sample_indices: `{report['sample_indices']}`",
        "",
    ]
    for sample in report["samples"]:
        lines.extend(
            [
                f"## Index {sample['index']}",
                "",
                f"- 状态: `{sample['status']}`",
                f"- 描述: `{sample['description']}`",
                f"- 检索候选 primitive: `{sample.get('candidate_primitives', [])}`",
            ]
        )
        if sample["status"] != "success":
            lines.append(f"- 错误: `{sample.get('error', 'unknown error')}`")
            lines.append("")
            continue

        lines.extend(
            [
                f"- 模型输出 primitive: `{sample['model_output_primitives']}`",
                f"- retrieval_status: `{sample['retrieval_status']}`",
                f"- comparison_metrics: `{sample['comparison_metrics']}`",
                f"- true_series_summary: `{sample['expected_true_series_summary']}`",
                f"- generated_series_summary: `{sample['generated_series_summary']}`",
                "",
                "### Step Outputs",
                "",
            ]
        )
        for step in sample["step_outputs"]:
            lines.append(
                f"- {step['id']} | {step['primitive']} | {step['effect_type']} | summary={step['summary']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample TSFragment-Eval description/series pairs and record end-to-end generation."
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--json-output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_sample(seed=args.seed, sample_count=args.sample_count, provider=args.provider)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
