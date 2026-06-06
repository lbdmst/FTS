from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.series_evaluation import compute_error_metrics


DEFAULT_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_cases_2500_reprocessed_v2"
DEFAULT_JSON_OUTPUT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_eval_2500_reprocessed_v2.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_eval_2500_reprocessed_v2.md"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=float))) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=float))) if values else None


def _summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in [
        "mse",
        "rmse",
        "mae",
        "mrr",
        "wape",
        "range_normalized_mse",
        "range_normalized_rmse",
        "std_normalized_rmse",
        "max_abs_error",
    ]:
        values = [float(item[metric_name]) for item in metrics if item.get(metric_name) is not None]
        summary[f"mean_{metric_name}"] = _mean(values)
        summary[f"median_{metric_name}"] = _median(values)
    correlations = [float(item["correlation"]) for item in metrics if item.get("correlation") is not None]
    summary["mean_correlation"] = _mean(correlations)
    summary["median_correlation"] = _median(correlations)
    return summary


def summarize_artifacts(case_file: Path, artifact_root: Path) -> dict[str, Any]:
    cases = _load_json(case_file)
    records: list[dict[str, Any]] = []
    successful_metrics: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    fallback_cases: list[str] = []
    missing_artifacts: list[str] = []

    for case in cases:
        case_id = str(case["case_id"])
        record_path = artifact_root / case_id / "record.json"
        if not record_path.exists():
            status_counts["missing_artifact"] += 1
            missing_artifacts.append(case_id)
            records.append(
                {
                    "case_id": case_id,
                    "status": "missing_artifact",
                    "source_dataset": case.get("source_dataset"),
                    "source_length": case.get("source_length", case.get("length")),
                    "record_path": _display(record_path),
                    "error": f"missing record file: {_display(record_path)}",
                }
            )
            continue

        raw_record = _load_json(record_path)
        status = str(raw_record.get("status", "unknown"))
        status_counts[status] += 1
        if raw_record.get("fallback_plan_used"):
            fallback_cases.append(case_id)

        output_record: dict[str, Any] = {
            "case_id": case_id,
            "status": status,
            "source_dataset": case.get("source_dataset"),
            "source_length": case.get("source_length", case.get("length")),
            "record_path": _display(record_path),
            "fallback_plan_used": bool(raw_record.get("fallback_plan_used", False)),
        }
        if status in {"success", "partial_success"} and raw_record.get("generated_series") is not None:
            generated = np.asarray(raw_record["generated_series"], dtype=np.float64)
            reference = np.asarray(case["reference_series"], dtype=np.float64)
            if generated.shape == reference.shape:
                metrics = compute_error_metrics(generated, reference)
                output_record["metrics"] = metrics
                successful_metrics.append(metrics)
            else:
                output_record["metrics_error"] = f"generated shape {generated.shape} != reference shape {reference.shape}"
                error_counts["shape_mismatch"] += 1
        else:
            error = str(raw_record.get("error", "unknown error"))
            output_record["error"] = error
            error_counts[error.splitlines()[0][:160]] += 1
        records.append(output_record)

    best_effort = status_counts["success"] + status_counts["partial_success"]
    total_cases = len(cases)
    summary: dict[str, Any] = {
        "method": "FaithTS",
        "cases": total_cases,
        "successful_cases": status_counts["success"],
        "partial_success_cases": status_counts["partial_success"],
        "best_effort_executable_cases": best_effort,
        "failed_cases": total_cases - best_effort,
        "missing_artifact_cases": status_counts["missing_artifact"],
        "success_rate": float(status_counts["success"] / total_cases) if total_cases else 0.0,
        "best_effort_executable_rate": float(best_effort / total_cases) if total_cases else 0.0,
        "fallback_plan_used_cases": len(fallback_cases),
        "status_counts": dict(sorted(status_counts.items())),
        "top_error_counts": dict(error_counts.most_common(12)),
    }
    summary.update(_summarize_metrics(successful_metrics))
    return {
        "version": "1.0",
        "benchmark": "T2S TSFragment-600K main benchmark FaithTS artifact summary",
        "case_file": _display(case_file),
        "artifact_root": _display(artifact_root),
        "summary": summary,
        "fallback_cases": fallback_cases,
        "missing_artifacts": missing_artifacts,
        "records": records,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# FaithTS TSFragment-600K Artifact Evaluation",
        "",
        f"- case_file: `{report['case_file']}`",
        f"- artifact_root: `{report['artifact_root']}`",
        f"- cases: `{summary['cases']}`",
        f"- successful_cases: `{summary['successful_cases']}`",
        f"- partial_success_cases: `{summary['partial_success_cases']}`",
        f"- best_effort_executable_cases: `{summary['best_effort_executable_cases']}`",
        f"- failed_cases: `{summary['failed_cases']}`",
        f"- fallback_plan_used_cases: `{summary['fallback_plan_used_cases']}`",
        f"- mean_wape: `{summary.get('mean_wape')}`",
        f"- median_wape: `{summary.get('median_wape')}`",
        f"- mean_correlation: `{summary.get('mean_correlation')}`",
        f"- median_correlation: `{summary.get('median_correlation')}`",
        f"- top_error_counts: `{summary.get('top_error_counts', {})}`",
        "",
        "## Case Status",
        "",
    ]
    for record in report["records"]:
        line = (
            f"- `{record['case_id']}`: `{record['status']}` "
            f"({record.get('source_dataset')}, length={record.get('source_length')})"
        )
        if record.get("error"):
            line += f" error=`{record['error']}`"
        if record.get("metrics_error"):
            line += f" metrics_error=`{record['metrics_error']}`"
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize existing FaithTS per-case artifacts without regenerating outputs.")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    report = summarize_artifacts(args.case_file, args.artifact_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {_display(args.markdown_output)}")
    print(f"Wrote {_display(args.json_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
