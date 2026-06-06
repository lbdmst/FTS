from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.series_evaluation import compute_error_metrics, write_case_artifacts
from evals.tsfragment_eval_workflow import (
    compare_series,
    execute_plan_with_steps,
    extract_real_series_plan_payload,
    normalize_plan_for_real_series,
    numeric_range_from_series,
    remove_generic_real_seasonality,
    resolve_representable_real_unresolved,
    series_summary,
)
from faithts_pipeline.hybrid_pipeline import PlanExecutionError, PlanValidationError, compile_plan_to_code, parse_plan
from schema_validator import validate_payload
DEFAULT_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
DEFAULT_SOURCE_ARTIFACT_ROOT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_cases_2500"
DEFAULT_OUTPUT_ARTIFACT_ROOT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_cases_2500_reprocessed"
DEFAULT_JSON_OUTPUT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_eval_2500_reprocessed.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_eval_2500_reprocessed.md"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
        summary[f"mean_{metric_name}"] = float(np.mean(values)) if values else None
        summary[f"median_{metric_name}"] = float(np.median(values)) if values else None
    correlations = [float(item["correlation"]) for item in metrics if item.get("correlation") is not None]
    summary["mean_correlation"] = float(np.mean(correlations)) if correlations else None
    summary["median_correlation"] = float(np.median(correlations)) if correlations else None
    return summary


def _add_candidate_repairs(plan_payload: dict[str, Any], retrieved_names: list[str]) -> list[str]:
    repairs: list[str] = []
    for step in plan_payload.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "primitive":
            continue
        primitive_name = str(step.get("primitive") or "")
        step_id = str(step.get("id") or "")
        if step_id.startswith("real_") and primitive_name in {"add_flat", "add_ramp"} and primitive_name not in retrieved_names:
            retrieved_names.append(primitive_name)
            repairs.append(f"added {primitive_name} to candidate set for deterministic real-series scaffold")
        if primitive_name == "add_outlier" and primitive_name not in retrieved_names:
            retrieved_names.append(primitive_name)
            repairs.append("added add_outlier to candidate set for exact caption point-value guard")
    return repairs


def _reprocess_case(
    *,
    case: dict[str, Any],
    source_artifact_root: Path,
    output_artifact_root: Path,
    allow_partial_output: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    case_id = str(case["case_id"])
    source_record_path = source_artifact_root / case_id / "record.json"
    output_case_dir = output_artifact_root / case_id
    truth = np.asarray(case["reference_series"], dtype=np.float64)
    description = str(case["description"])
    record: dict[str, Any] = {
        "case_id": case_id,
        "index": int(case.get("source_index", case.get("source_row", 0))),
        "source_index": int(case.get("source_index", case.get("source_row", 0))),
        "description": description,
        "expected_true_series_summary": series_summary(truth),
        "reference_series_summary": series_summary(truth),
        "true_series": [float(x) for x in truth.tolist()],
        "status": "started",
        "source_artifact_record": _display(source_record_path),
    }
    metric_record: dict[str, Any] | None = None

    try:
        if not source_record_path.exists():
            raise FileNotFoundError(f"missing source record: {_display(source_record_path)}")
        source_record = _read_json(source_record_path)
        raw_model_text = source_record.get("raw_model_text")
        if not raw_model_text and not isinstance(source_record.get("structured_plan"), dict):
            raise ValueError("source artifact has neither raw_model_text nor structured_plan")

        retrieved_names = list(source_record.get("retrieved_primitives") or source_record.get("candidate_primitives") or [])
        candidate_primitives = list(source_record.get("candidate_primitives") or retrieved_names)
        numeric_range = dict(source_record.get("numeric_range") or case.get("numeric_range") or numeric_range_from_series(truth))
        record.update(
            {
                "candidate_primitives": candidate_primitives,
                "retrieved_primitives": list(retrieved_names),
                "numeric_range": numeric_range,
                "prompt": source_record.get("prompt"),
                "raw_model_text": raw_model_text,
                "model_provider": source_record.get("model_provider"),
                "model_name": source_record.get("model_name"),
                "response_id": source_record.get("response_id"),
                "fallback_plan_used": bool(source_record.get("fallback_plan_used", False)),
            }
        )

        if raw_model_text:
            plan_payload = extract_real_series_plan_payload(str(raw_model_text))
        else:
            plan_payload = dict(source_record["structured_plan"])

        plan_context_repairs: list[str] = []
        plan_context = plan_payload.get("global_context")
        if isinstance(plan_context, dict):
            requested_range = {"min": float(numeric_range["min"]), "max": float(numeric_range["max"])}
            raw_range = plan_context.get("numeric_range")
            range_matches = False
            if isinstance(raw_range, dict) and "min" in raw_range and "max" in raw_range:
                try:
                    range_matches = bool(
                        np.isclose(float(raw_range["min"]), requested_range["min"], atol=1e-9, rtol=0.0)
                        and np.isclose(float(raw_range["max"]), requested_range["max"], atol=1e-9, rtol=0.0)
                    )
                except (TypeError, ValueError):
                    range_matches = False
            if not range_matches:
                plan_context["numeric_range"] = requested_range
                plan_context_repairs.append("normalized global_context.numeric_range to requested evaluation range")
        if plan_context_repairs:
            record["plan_context_repairs"] = plan_context_repairs

        parameter_repairs = normalize_plan_for_real_series(
            plan_payload,
            reference=truth,
            available_primitives=set(retrieved_names),
        )
        if parameter_repairs:
            record["plan_parameter_repairs"] = parameter_repairs
        unresolved_repairs = resolve_representable_real_unresolved(
            plan_payload,
            reference=truth,
            available_primitives=set(retrieved_names),
        )
        if unresolved_repairs:
            record["plan_unresolved_repairs"] = unresolved_repairs
        primitive_repairs = remove_generic_real_seasonality(plan_payload)
        if primitive_repairs:
            record["plan_primitive_repairs"] = primitive_repairs
        candidate_repairs = _add_candidate_repairs(plan_payload, retrieved_names)
        if candidate_repairs:
            record["plan_candidate_repairs"] = candidate_repairs
            record["retrieved_primitives"] = list(retrieved_names)

        remaining_unresolved = [
            {
                "id": str(item.get("id") or f"unresolved_{idx}"),
                "description": str(item.get("description") or item.get("source_text") or "unresolved semantic"),
                "source_text": str(item.get("source_text") or item.get("description") or "unresolved semantic"),
                "reason": str(item.get("reason") or "Model marked this semantic as unresolved."),
            }
            for idx, item in enumerate(plan_payload.get("unresolved_semantics", []), start=1)
            if isinstance(item, dict)
        ]
        if remaining_unresolved and allow_partial_output:
            record["partial_output"] = True
            record["partial_output_warnings"] = remaining_unresolved

        record["structured_plan"] = plan_payload
        record["model_output_primitives"] = [
            step["primitive"] for step in plan_payload.get("steps", []) if isinstance(step, dict) and step.get("kind") == "primitive"
        ]
        schema_validation = validate_payload("plan", plan_payload, target_path=f"<{case_id}>")
        record["plan_schema_validation"] = schema_validation
        plan = parse_plan(
            plan_payload,
            retrieved_names,
            len(truth),
            numeric_range=numeric_range,
            allow_partial_output=allow_partial_output,
        )
        generated_code = compile_plan_to_code(plan, len(truth))
        record["generated_code"] = generated_code
        generated_series, step_outputs = execute_plan_with_steps(plan, description, len(truth))
        record["step_outputs"] = step_outputs
        status = "partial_success" if plan.unresolved_semantics or plan.needs_library_extension else "success"
        comparison_metrics = compute_error_metrics(generated_series, truth)
        record.update(
            {
                "status": status,
                "strict_success": status == "success",
                "best_effort_executable": True,
                "generated_series_summary": series_summary(generated_series),
                "generated_series": [float(x) for x in generated_series.tolist()],
                "comparison_metrics": comparison_metrics,
                "legacy_comparison_metrics": compare_series(generated_series, truth),
            }
        )
        metric_record = comparison_metrics
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        PlanValidationError,
        PlanExecutionError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:
        record["status"] = "failed"
        record["error"] = str(exc)

    record["artifact_dir"] = _display(output_case_dir)
    record["artifact_paths"] = write_case_artifacts(output_case_dir, record)
    return record, metric_record


def reprocess_artifacts(
    *,
    case_file: Path,
    source_artifact_root: Path,
    output_artifact_root: Path,
    allow_partial_output: bool = False,
    max_cases: int | None = None,
) -> dict[str, Any]:
    cases = _read_json(case_file)
    if max_cases is not None:
        cases = cases[:max_cases]
    if output_artifact_root.exists() and any(output_artifact_root.iterdir()):
        raise FileExistsError(
            f"output artifact root is not empty: {_display(output_artifact_root)}; choose a new path"
        )
    output_artifact_root.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    for index, case in enumerate(cases, start=1):
        record, metric_record = _reprocess_case(
            case=case,
            source_artifact_root=source_artifact_root,
            output_artifact_root=output_artifact_root,
            allow_partial_output=allow_partial_output,
        )
        records.append(record)
        status_counts[str(record["status"])] += 1
        if metric_record is not None:
            metrics.append(metric_record)
        if record["status"] not in {"success", "partial_success"}:
            error_counts[str(record.get("error", "unknown error")).splitlines()[0][:160]] += 1
        if index % 100 == 0:
            print(f"Reprocessed {index}/{len(cases)} cases")

    best_effort = status_counts["success"] + status_counts["partial_success"]
    elapsed_seconds = time.time() - start_time
    summary: dict[str, Any] = {
        "method": "FaithTS",
        "cases": len(cases),
        "successful_cases": status_counts["success"],
        "partial_success_cases": status_counts["partial_success"],
        "best_effort_executable_cases": best_effort,
        "failed_cases": len(cases) - best_effort,
        "success_rate": float(status_counts["success"] / len(cases)) if cases else 0.0,
        "best_effort_executable_rate": float(best_effort / len(cases)) if cases else 0.0,
        "allow_partial_output": allow_partial_output,
        "status_counts": dict(sorted(status_counts.items())),
        "top_error_counts": dict(error_counts.most_common(12)),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_case": float(elapsed_seconds / len(cases)) if cases else None,
    }
    summary.update(_summarize_metrics(metrics))
    return {
        "version": "1.0",
        "benchmark": "T2S TSFragment-600K FaithTS deterministic reprocess",
        "case_file": _display(case_file),
        "source_artifact_root": _display(source_artifact_root),
        "artifact_root": _display(output_artifact_root),
        "summary": summary,
        "records": [
            {
                "case_id": record["case_id"],
                "status": record["status"],
                "source_dataset": case.get("source_dataset"),
                "source_length": case.get("source_length", case.get("length")),
                "record_path": _display(output_artifact_root / record["case_id"] / "record.json"),
                "fallback_plan_used": bool(record.get("fallback_plan_used", False)),
                **({"metrics": record["comparison_metrics"]} if record.get("comparison_metrics") is not None else {}),
                **({"error": record["error"]} if record.get("error") is not None else {}),
            }
            for record, case in zip(records, cases)
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# FaithTS TSFragment-600K Deterministic Reprocess",
        "",
        f"- case_file: `{report['case_file']}`",
        f"- source_artifact_root: `{report['source_artifact_root']}`",
        f"- artifact_root: `{report['artifact_root']}`",
        f"- cases: `{summary['cases']}`",
        f"- successful_cases: `{summary['successful_cases']}`",
        f"- partial_success_cases: `{summary['partial_success_cases']}`",
        f"- best_effort_executable_cases: `{summary['best_effort_executable_cases']}`",
        f"- failed_cases: `{summary['failed_cases']}`",
        f"- elapsed_seconds: `{summary['elapsed_seconds']:.2f}`",
        f"- seconds_per_case: `{summary['seconds_per_case']:.4f}`",
        f"- mean_wape: `{summary.get('mean_wape')}`",
        f"- median_wape: `{summary.get('median_wape')}`",
        f"- mean_correlation: `{summary.get('mean_correlation')}`",
        f"- median_correlation: `{summary.get('median_correlation')}`",
        f"- top_error_counts: `{summary.get('top_error_counts', {})}`",
        "",
        "## Failed Cases",
        "",
    ]
    failures = [record for record in report["records"] if record["status"] not in {"success", "partial_success"}]
    if not failures:
        lines.append("- none")
    else:
        for record in failures:
            lines.append(
                f"- `{record['case_id']}`: `{record['status']}` "
                f"({record.get('source_dataset')}, length={record.get('source_length')}) "
                f"error=`{record.get('error', 'unknown error')}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministically reprocess saved FaithTS T2S artifacts without LLM calls.")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--source-artifact-root", type=Path, default=DEFAULT_SOURCE_ARTIFACT_ROOT)
    parser.add_argument("--output-artifact-root", type=Path, default=DEFAULT_OUTPUT_ARTIFACT_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--allow-partial-output", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    report = reprocess_artifacts(
        case_file=args.case_file,
        source_artifact_root=args.source_artifact_root,
        output_artifact_root=args.output_artifact_root,
        allow_partial_output=args.allow_partial_output,
        max_cases=args.max_cases,
    )
    _write_json(args.json_output, report)
    _write_text(args.markdown_output, render_markdown(report))
    print(f"Wrote {_display(args.markdown_output)}")
    print(f"Wrote {_display(args.json_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
