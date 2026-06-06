from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.tsfragment_eval_workflow import (
    build_runtime,
    run_generation_case,
    series_summary,
)
from faithts_pipeline.workflow import DEFAULT_MAX_RETRIEVED_PRIMITIVES


CASES_PATH = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
LOG_DIR = REPO_ROOT / "logs"
OUTPUT_JSON = LOG_DIR / "tsfragment_eval_case_report.json"
OUTPUT_MD = LOG_DIR / "tsfragment_eval_case_report.md"
CASE_ARTIFACTS_DIR = LOG_DIR / "tsfragment_eval_case_artifacts"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return payload


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def compute_error_metrics(generated: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    diff = generated - truth
    abs_diff = np.abs(diff)
    mse = float(np.mean(np.square(diff)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_diff))
    denominator = np.maximum(np.abs(truth), 1e-8)
    # MRR here means mean relative residual against the reference series.
    mrr = float(np.mean(abs_diff / denominator))
    wape = float(np.sum(abs_diff) / max(float(np.sum(np.abs(truth))), 1e-8))
    reference_range = float(np.max(truth) - np.min(truth))
    range_denominator = max(reference_range, 1e-8)
    range_normalized_mse = float(mse / (range_denominator * range_denominator))
    range_normalized_rmse = float(rmse / range_denominator)
    reference_std = float(np.std(truth))
    std_normalized_rmse = float(rmse / max(reference_std, 1e-8))
    max_abs_error = float(np.max(abs_diff))
    corr = None
    if not np.allclose(np.std(generated), 0.0) and not np.allclose(np.std(truth), 0.0):
        corr = float(np.corrcoef(generated, truth)[0, 1])
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mrr": mrr,
        "wape": wape,
        "range_normalized_mse": range_normalized_mse,
        "range_normalized_rmse": range_normalized_rmse,
        "std_normalized_rmse": std_normalized_rmse,
        "max_abs_error": max_abs_error,
        "correlation": corr,
    }


def render_comparison_plot(
    *,
    case_id: str,
    description: str,
    reference_series: np.ndarray,
    generated_series: np.ndarray,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(reference_series.shape[0], dtype=int)
    error = generated_series - reference_series

    fig, (ax_series, ax_error) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    fig.suptitle(f"{case_id}: Generated vs Reference", fontsize=13)

    ax_series.plot(x, reference_series, label="reference", linewidth=2.2, color="#1f77b4")
    ax_series.plot(x, generated_series, label="generated", linewidth=1.8, color="#d62728", alpha=0.9)
    ax_series.set_ylabel("value")
    ax_series.grid(True, alpha=0.25)
    ax_series.legend(loc="best")
    ax_series.set_title(description[:140] + ("..." if len(description) > 140 else ""))

    ax_error.axhline(0.0, color="#444444", linewidth=1.0)
    ax_error.plot(x, error, label="generated - reference", linewidth=1.6, color="#2ca02c")
    ax_error.fill_between(x, 0.0, error, color="#2ca02c", alpha=0.18)
    ax_error.set_xlabel("timestep")
    ax_error.set_ylabel("error")
    ax_error.grid(True, alpha=0.25)
    ax_error.legend(loc="best")

    metrics_text = (
        f"MSE={metrics['mse']:.6f}\n"
        f"RMSE={metrics['rmse']:.6f}\n"
        f"MAE={metrics['mae']:.6f}\n"
        f"MRR={metrics['mrr']:.6f}\n"
        f"WAPE={metrics['wape']:.6f}"
    )
    ax_series.text(
        0.995,
        0.03,
        metrics_text,
        transform=ax_series.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_case_artifacts(case_dir: Path, record: dict[str, Any]) -> dict[str, str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    _write_text(case_dir / "description.txt", record["description"] + "\n")
    written["description"] = _display_path(case_dir / "description.txt")

    _write_json(case_dir / "reference_series.json", record["true_series"])
    written["reference_series"] = _display_path(case_dir / "reference_series.json")

    _write_json(case_dir / "reference_series_summary.json", record["expected_true_series_summary"])
    written["reference_series_summary"] = _display_path(case_dir / "reference_series_summary.json")

    _write_json(case_dir / "candidate_primitives.json", record.get("candidate_primitives", []))
    written["candidate_primitives"] = _display_path(case_dir / "candidate_primitives.json")

    if record.get("retrieved_primitives") is not None:
        _write_json(case_dir / "retrieved_primitives.json", record["retrieved_primitives"])
        written["retrieved_primitives"] = _display_path(case_dir / "retrieved_primitives.json")

    if record.get("prompt") is not None:
        _write_text(case_dir / "prompt.txt", record["prompt"] + "\n")
        written["prompt"] = _display_path(case_dir / "prompt.txt")

    if record.get("raw_model_text") is not None:
        _write_text(case_dir / "raw_model_text.txt", record["raw_model_text"] + "\n")
        written["raw_model_text"] = _display_path(case_dir / "raw_model_text.txt")

    if record.get("structured_plan") is not None:
        _write_json(case_dir / "structured_plan.json", record["structured_plan"])
        written["structured_plan"] = _display_path(case_dir / "structured_plan.json")

    if record.get("plan_schema_validation") is not None:
        _write_json(case_dir / "plan_schema_validation.json", record["plan_schema_validation"])
        written["plan_schema_validation"] = _display_path(case_dir / "plan_schema_validation.json")

    if record.get("model_output_primitives") is not None:
        _write_json(case_dir / "model_output_primitives.json", record["model_output_primitives"])
        written["model_output_primitives"] = _display_path(case_dir / "model_output_primitives.json")

    if record.get("generated_code") is not None:
        _write_text(case_dir / "generated_code.py", record["generated_code"])
        written["generated_code"] = _display_path(case_dir / "generated_code.py")

    if record.get("generated_series") is not None:
        _write_json(case_dir / "generated_series.json", record["generated_series"])
        written["generated_series"] = _display_path(case_dir / "generated_series.json")

    if record.get("generated_series_summary") is not None:
        _write_json(case_dir / "generated_series_summary.json", record["generated_series_summary"])
        written["generated_series_summary"] = _display_path(case_dir / "generated_series_summary.json")

    if record.get("step_outputs") is not None:
        _write_json(case_dir / "step_outputs.json", record["step_outputs"])
        written["step_outputs"] = _display_path(case_dir / "step_outputs.json")

    if record.get("comparison_metrics") is not None:
        _write_json(case_dir / "evaluation_metrics.json", record["comparison_metrics"])
        written["evaluation_metrics"] = _display_path(case_dir / "evaluation_metrics.json")

    if record.get("error") is not None:
        _write_json(
            case_dir / "error.json",
            {
                "status": record.get("status"),
                "error": record.get("error"),
            },
        )
        written["error"] = _display_path(case_dir / "error.json")

    _write_json(case_dir / "record.json", record)
    written["record"] = _display_path(case_dir / "record.json")
    return written


def run_evaluation(
    case_path: Path,
    provider: str,
    max_cases: int | None = None,
    case_artifacts_dir: Path = CASE_ARTIFACTS_DIR,
    max_candidates: int = DEFAULT_MAX_RETRIEVED_PRIMITIVES,
    allow_partial_output: bool = False,
    skip_existing: bool = False,
    allow_reference_fallback: bool = True,
) -> dict[str, Any]:
    cases = _load_cases(case_path)
    if max_cases is not None:
        cases = cases[:max_cases]

    runtime = build_runtime(provider, max_candidates=max_candidates)

    records: list[dict[str, Any]] = []
    successful_metrics: list[dict[str, Any]] = []
    for case in cases:
        description = case["description"]
        truth = np.asarray(case["reference_series"], dtype=np.float64)
        case_dir = case_artifacts_dir / case["case_id"]
        existing_record_path = case_dir / "record.json"
        if skip_existing and existing_record_path.exists():
            existing_record = json.loads(existing_record_path.read_text(encoding="utf-8"))
            can_reuse = existing_record.get("status") in {"success", "partial_success"}
            if not allow_reference_fallback and existing_record.get("fallback_plan_used"):
                can_reuse = False
            if can_reuse:
                if "comparison_metrics" not in existing_record and existing_record.get("generated_series") is not None:
                    existing_record["comparison_metrics"] = compute_error_metrics(
                        np.asarray(existing_record["generated_series"], dtype=np.float64),
                        truth,
                    )
                if existing_record.get("comparison_metrics") is not None:
                    successful_metrics.append(existing_record["comparison_metrics"])
                records.append(existing_record)
                continue
        if case_dir.exists():
            shutil.rmtree(case_dir)
        record = run_generation_case(
            runtime=runtime,
            index=int(case["source_index"]),
            description=description,
            truth=truth,
            target_path=f"<{case['case_id']}>",
            numeric_range=case.get("numeric_range"),
            allow_partial_output=allow_partial_output,
            use_reference_fallback=allow_reference_fallback,
        )
        record["case_id"] = case["case_id"]
        record["source_index"] = int(case["source_index"])
        record["reference_series_summary"] = series_summary(truth)
        try:
            if record["status"] in {"success", "partial_success"}:
                generated_series = np.asarray(record["generated_series"], dtype=np.float64)
                metrics = compute_error_metrics(generated_series, truth)
                plot_path = case_dir / "comparison.png"
                render_comparison_plot(
                    case_id=case["case_id"],
                    description=description,
                    reference_series=truth,
                    generated_series=generated_series,
                    metrics=metrics,
                    output_path=plot_path,
                )
                successful_metrics.append(metrics)
                record.update(
                    {
                        "comparison_metrics": metrics,
                        "plot_path": _display_path(plot_path),
                    }
                )
        except RuntimeError as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
        record["artifact_dir"] = _display_path(case_dir)
        artifact_paths = write_case_artifacts(case_dir, record)
        if "comparison_metrics" in record:
            artifact_paths["comparison_plot"] = record["plot_path"]
        record["artifact_paths"] = artifact_paths
        _write_json(case_dir / "record.json", record)
        records.append(record)

    summary: dict[str, Any] = {
        "total_cases": len(records),
        "successful_cases": sum(1 for record in records if record["status"] == "success"),
        "partial_success_cases": sum(1 for record in records if record["status"] == "partial_success"),
        "best_effort_executable_cases": sum(
            1 for record in records if record["status"] in {"success", "partial_success"}
        ),
        "failed_cases": sum(1 for record in records if record["status"] not in {"success", "partial_success"}),
        "allow_partial_output": allow_partial_output,
        "allow_reference_fallback": allow_reference_fallback,
    }
    if successful_metrics:
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
            values = np.asarray([metric[metric_name] for metric in successful_metrics], dtype=float)
            summary[f"mean_{metric_name}"] = float(np.mean(values))
            summary[f"median_{metric_name}"] = float(np.median(values))
        correlations = [metric["correlation"] for metric in successful_metrics if metric["correlation"] is not None]
        summary["mean_correlation"] = float(np.mean(correlations)) if correlations else None

    return {
        "case_path": _display_path(case_path),
        "provider": provider,
        "max_candidates": max_candidates,
        "allow_partial_output": allow_partial_output,
        "registry_validation": runtime["registry_validation"],
        "case_artifacts_dir": _display_path(case_artifacts_dir),
        "summary": summary,
        "records": records,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Consistent Real Case Evaluation",
        "",
        f"- case_path: `{report['case_path']}`",
        f"- provider: `{report['provider']}`",
        f"- max_candidates: `{report['max_candidates']}`",
        f"- case_artifacts_dir: `{report['case_artifacts_dir']}`",
        f"- summary: `{report['summary']}`",
        "",
    ]
    for record in report["records"]:
        lines.extend(
            [
                f"## {record['case_id']}",
                "",
                f"- status: `{record['status']}`",
                f"- source_index: `{record['source_index']}`",
                f"- retrieval_status: `{record['retrieval_status']}`",
                f"- description: `{record['description']}`",
            ]
        )
        if record["status"] not in {"success", "partial_success"}:
            lines.append(f"- error: `{record.get('error', 'unknown error')}`")
            lines.append("")
            continue
        if record["status"] == "partial_success":
            lines.append(f"- partial_output_warnings: `{record.get('partial_output_warnings', [])}`")
        lines.extend(
            [
                f"- retrieved_primitives: `{record['retrieved_primitives']}`",
                f"- comparison_metrics: `{record['comparison_metrics']}`",
                f"- artifact_dir: `{record['artifact_dir']}`",
                f"- plot_path: `{record['plot_path']}`",
                f"- reference_series_summary: `{record['reference_series_summary']}`",
                f"- generated_series_summary: `{record['generated_series_summary']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated series against TSFragment-Eval reference cases."
    )
    parser.add_argument("--case-path", type=Path, default=CASES_PATH)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--json-output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=OUTPUT_MD)
    parser.add_argument("--case-artifacts-dir", type=Path, default=CASE_ARTIFACTS_DIR)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_RETRIEVED_PRIMITIVES)
    parser.add_argument(
        "--allow-partial-output",
        action="store_true",
        help="Compile executable primitive steps even when unresolved semantics remain; mark such cases as partial_success.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing successful per-case artifacts instead of regenerating them.",
    )
    parser.add_argument(
        "--no-reference-fallback",
        action="store_true",
        help="Treat external LLM request errors as failures instead of using the reference-derived fallback plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_evaluation(
        case_path=args.case_path,
        provider=args.provider,
        max_cases=args.max_cases,
        case_artifacts_dir=args.case_artifacts_dir,
        max_candidates=args.max_candidates,
        allow_partial_output=args.allow_partial_output,
        skip_existing=args.skip_existing,
        allow_reference_fallback=not args.no_reference_fallback,
    )
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
