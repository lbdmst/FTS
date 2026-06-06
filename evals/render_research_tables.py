from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _method_name(result: dict[str, Any]) -> str:
    return str(result.get("method") or result.get("baseline") or result.get("ablation") or "unknown")


def _case_reports(result: dict[str, Any]) -> list[dict[str, Any]]:
    reports = []
    for case in result.get("cases", []):
        report = case.get("eval_report")
        if isinstance(report, dict):
            reports.append(report)
    return reports


def _is_skipped(result: dict[str, Any]) -> bool:
    return result.get("status") == "skipped"


def _mean(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _reliability_summary(result: dict[str, Any]) -> dict[str, float | None]:
    if _is_skipped(result):
        return {
            "schema_valid": None,
            "registry_valid": None,
            "executable": None,
            "safe_code": None,
            "failure_report_complete": None,
        }
    reports = _case_reports(result)
    failure_cases = [case for case in result.get("cases", []) if not case.get("passed")]
    schema_valid = _mean([report.get("plan_validation_status", {}).get("status") == "valid" for report in reports])
    registry_valid = _mean([report.get("registry_validation", {}).get("status") == "valid" for report in reports])
    generation_reports = [
        report
        for report in reports
        if not report.get("unresolved_semantics")
        and report.get("execution_status", {}).get("status") != "not_run"
    ]
    executable = _mean([report.get("execution_status", {}).get("status") == "success" for report in generation_reports])
    safe_code = _mean(
        [
            not any(reason.get("code") == "F_safety" for reason in case.get("failure_reasons", []))
            for case in result.get("cases", [])
        ]
    )
    failure_report_complete = _mean(
        [
            bool(case.get("failure_reasons"))
            or bool(case.get("eval_report", {}).get("unresolved_semantics"))
            or bool(case.get("metrics", {}).get("correct_rejection"))
            for case in failure_cases
        ]
    )
    return {
        "schema_valid": schema_valid,
        "registry_valid": registry_valid,
        "executable": executable,
        "safe_code": safe_code,
        "failure_report_complete": 1.0 if failure_report_complete is None else failure_report_complete,
    }


def _success_count(result: dict[str, Any]) -> tuple[int | None, int | None]:
    cases = result.get("cases")
    if isinstance(cases, list):
        total = len(cases)
        passed = sum(1 for case in cases if case.get("passed"))
        return passed, total
    aggregate = result.get("aggregate", {})
    total = aggregate.get("cases")
    pass_rate = aggregate.get("pass_rate")
    if total is None:
        return None, None
    try:
        total_int = int(total)
    except Exception:
        return None, None
    if pass_rate is None:
        return None, total_int
    try:
        passed = int(round(float(pass_rate) * total_int))
    except Exception:
        return None, total_int
    return passed, total_int


def _success_text(result: dict[str, Any]) -> str:
    passed, total = _success_count(result)
    if passed is None or total is None:
        return "n/a"
    return f"{passed}/{total}"


def _semantic_rows(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Main Table 1: Semantic Constraint Satisfaction",
        "",
        "| Method | Success | Atomic CSR | Compositional CSR | Adversarial CSR | Primitive F1 | Window IoU | ParamErr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload.get("results", []):
        if _method_name(result) == "oracle_program":
            continue
        if _is_skipped(result):
            rows.append(f"| {_method_name(result)} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        aggregate = result.get("aggregate", {})
        by_category = result.get("by_category", {})
        rows.append(
            "| "
            + " | ".join(
                [
                    _method_name(result),
                    _success_text(result),
                    _fmt(by_category.get("atomic", {}).get("csr")),
                    _fmt(by_category.get("compositional", {}).get("csr")),
                    _fmt(by_category.get("adversarial", {}).get("csr")),
                    _fmt(aggregate.get("primitive_f1")),
                    _fmt(aggregate.get("window_iou")),
                    _fmt(aggregate.get("parameter_error")),
                ]
            )
            + " |"
        )
    return rows


def _reliability_rows(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Main Table 2: Reliability and Rejection",
        "",
        "| Method | Success | Registry Valid | Schema Valid | Executable | Safe Code | CRR | FRR | HUS | Failure Report Complete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload.get("results", []):
        if _method_name(result) == "oracle_program":
            continue
        if _is_skipped(result):
            rows.append(f"| {_method_name(result)} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        aggregate = result.get("aggregate", {})
        reliability = _reliability_summary(result)
        rows.append(
            "| "
            + " | ".join(
                [
                    _method_name(result),
                    _success_text(result),
                    _fmt(reliability["registry_valid"]),
                    _fmt(reliability["schema_valid"]),
                    _fmt(reliability["executable"]),
                    _fmt(reliability["safe_code"]),
                    _fmt(aggregate.get("crr")),
                    _fmt(aggregate.get("frr")),
                    _fmt(aggregate.get("hus")),
                    _fmt(reliability["failure_report_complete"]),
                ]
            )
            + " |"
        )
    return rows


def _ablation_rows(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Main Table 3: Ablation and Reliability Proxy",
        "",
        "| Variant | CSR | Primitive F1 | ParamErr | HUS | Repair Iter. | Failure Case Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload.get("results", []):
        aggregate = result.get("aggregate", {})
        repair = result.get("repair_summary", {})
        taxonomy = result.get("failure_taxonomy", {})
        rows.append(
            "| "
            + " | ".join(
                [
                    _method_name(result),
                    _fmt(aggregate.get("csr")),
                    _fmt(aggregate.get("primitive_f1")),
                    _fmt(aggregate.get("parameter_error")),
                    _fmt(aggregate.get("hus")),
                    str(int(repair.get("total_iterations", 0))),
                    _fmt(taxonomy.get("failure_case_rate")),
                ]
            )
            + " |"
        )
    return rows


def _t2s_main_rows(payload: dict[str, Any]) -> list[str]:
    overall_rows = payload.get("overall") or payload.get("reference_similarity_overall", [])
    overall_by_method = {}
    if isinstance(overall_rows, list):
        for row in overall_rows:
            if isinstance(row, dict) and row.get("method"):
                overall_by_method[str(row["method"]).strip().lower()] = row
    rows = [
        "## Main Table 2B: TSFragment Real-Caption Benchmark",
        "",
        "| Method | Cases/Settings | Success Count | Success Rate | Mean WAPE ↓ | Median WAPE ↓ | Mean Corr ↑ | Median Corr ↑ | Mean MRR ↓ | Median MRR ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    method_rows = [
        (["t2s native"], "t2s_native"),
        (["verbalts tsfragment"], "verbalts_tsfragment"),
        (["faithts", "ours"], "FaithTS"),
        (["llm_direct_array"], "llm_direct_array"),
        (["llm_direct_code"], "llm_direct_code"),
        (["llm_strong_prompt"], "llm_strong_prompt"),
    ]
    def _t2s_success_text(metrics: dict[str, Any]) -> str:
        successful = metrics.get("successful_cases")
        cases = metrics.get("cases")
        if successful is not None and cases is not None:
            return f"{successful}/{cases}"
        if cases is not None and metrics.get("success_rate") is not None:
            try:
                return f"{int(round(float(metrics['success_rate']) * int(cases)))}/{int(cases)}"
            except Exception:
                return "n/a"
        return "n/a"
    for aliases, label in method_rows:
        metrics = next((overall_by_method[alias] for alias in aliases if alias in overall_by_method), None)
        if not isinstance(metrics, dict):
            continue
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(metrics.get("cases", metrics.get("cases_or_settings", "n/a"))),
                    _t2s_success_text(metrics),
                    _fmt(metrics.get("success_rate")),
                    _fmt(metrics.get("mean_wape")),
                    _fmt(metrics.get("median_wape")),
                    _fmt(metrics.get("mean_corr", metrics.get("mean_correlation"))),
                    _fmt(metrics.get("median_corr", metrics.get("median_correlation"))),
                    _fmt(metrics.get("mean_mrr")),
                    _fmt(metrics.get("median_mrr")),
                ]
            )
            + " |"
        )
    missing_direct = [
        label
        for aliases, label in method_rows
        if aliases[0].startswith("llm_direct") and aliases[0] not in overall_by_method
    ]
    rows.extend(
        [
            "",
            "Real-caption notes:",
            "- This split uses `data/tsfragment_eval/tsfragment_eval_2500_cases.json` and is reported as a separate main-experiment split rather than folded into synthetic CSR columns.",
        ]
    )
    supplement = payload.get("public_generator_native_supplement")
    if isinstance(supplement, dict) and isinstance(supplement.get("verbalts_native"), dict):
        adapter = supplement.get("verbalts_shared_evaluator_adapter")
        adapter_status = adapter.get("status") if isinstance(adapter, dict) else "not_recorded"
        rows.append(
            "- VerbalTS native Weather metrics are retained only as a reproduction/background artifact; shared-case VerbalTS evidence uses the `verbalts` adapter status `"
            + str(adapter_status)
            + "`."
        )
    if missing_direct:
        rows.append(
            "- Pending full-case generation: "
            + ", ".join(f"`{method}`" for method in missing_direct)
            + "."
        )
    else:
        rows.append("- Direct LLM baselines are included from full-case TSFragment raw outputs.")
    for method_key in ["llm_direct_array", "llm_direct_code", "llm_strong_prompt"]:
        metrics = overall_by_method.get(method_key)
        if not isinstance(metrics, dict):
            continue
        failure_summary = metrics.get("failure_summary")
        if not isinstance(failure_summary, dict):
            continue
        bucket_text = ", ".join(
            f"{bucket}={count}" for bucket, count in failure_summary.get("failure_buckets", {}).items()
        )
        if bucket_text:
            rows.append(f"- `{method_key}` failure types: {bucket_text}.")
        length_hist = failure_summary.get("length_mismatch_delta_histogram")
        if isinstance(length_hist, dict) and length_hist:
            sorted_hist = sorted(length_hist.items(), key=lambda item: (-item[1], item[0]))
            rows.append(
                "- `"
                + method_key
                + "` length mismatch distribution (actual-expected): "
                + ", ".join(f"{delta}:{count}" for delta, count in sorted_hist[:5])
                + "."
            )
        notes = failure_summary.get("analysis_notes")
        if isinstance(notes, list):
            for note in notes[:2]:
                rows.append(f"- `{method_key}` analysis: {note}")
    return rows


def render_tables(
    *,
    baseline_payload: dict[str, Any] | None = None,
    ablation_payload: dict[str, Any] | None = None,
    t2s_payload: dict[str, Any] | None = None,
) -> str:
    sections = ["# FaithTS Research Tables", ""]
    if baseline_payload is not None:
        sections.extend(_semantic_rows(baseline_payload))
        sections.append("")
        sections.extend(_reliability_rows(baseline_payload))
        sections.append("")
    if t2s_payload is not None:
        sections.extend(_t2s_main_rows(t2s_payload))
        sections.append("")
    if ablation_payload is not None:
        sections.extend(_ablation_rows(ablation_payload))
        sections.append("")
    return "\n".join(sections)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Render paper-style markdown tables from baseline and ablation JSON outputs.")
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--ablation-json", type=Path, default=None)
    parser.add_argument("--t2s-json", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    markdown = render_tables(
        baseline_payload=_load_json(args.baseline_json),
        ablation_payload=_load_json(args.ablation_json),
        t2s_payload=_load_json(args.t2s_json),
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
