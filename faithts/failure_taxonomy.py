from __future__ import annotations

from collections import Counter
from typing import Any


TAXONOMY_LABELS = [
    "schema_invalid",
    "registry_invalid",
    "unsafe_code",
    "execution_failed",
    "parse_failure",
    "length_mismatch",
    "no_signal",
    "wrong_primitive",
    "wrong_window",
    "wrong_parameter",
    "semantic_constraint_failed",
    "false_rejection",
    "unsupported_hallucination",
]


def _metric(item: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(item.get("metrics", {}).get(name, default))
    except (TypeError, ValueError):
        return default


def _execution_errors(item: dict[str, Any]) -> list[str]:
    report = item.get("eval_report", {})
    status = report.get("execution_status", {})
    errors = status.get("errors", [])
    return [str(error) for error in errors if error]


def classify_failure(item: dict[str, Any]) -> list[str]:
    """Assign stable, paper-facing failure labels to one evaluated case."""

    labels: set[str] = set()
    if item.get("passed"):
        return []

    report = item.get("eval_report", {})
    registry_status = report.get("registry_validation", {}).get("status")
    plan_status = report.get("plan_validation_status", {}).get("status")
    plan_applicable = plan_status not in {None, "not_run"}
    if registry_status not in {None, "valid", "not_run"}:
        labels.add("registry_invalid")
    if plan_status not in {None, "valid", "not_run"}:
        labels.add("schema_invalid")

    for reason in item.get("failure_reasons", []):
        code = reason.get("code")
        stage = reason.get("stage")
        message = str(reason.get("message", "")).lower()
        if code == "F_schema":
            if stage == "registry_validation":
                labels.add("registry_invalid")
            elif plan_applicable:
                labels.add("schema_invalid")
        elif code == "F_safety":
            labels.add("unsafe_code")
        elif code == "F_execution":
            labels.add("execution_failed")
        elif code == "F_semantic":
            labels.add("semantic_constraint_failed")
        elif code == "F_rejection":
            if "incorrectly rejected" in message:
                labels.add("false_rejection")
            else:
                labels.add("unsupported_hallucination")
        elif code == "F_grounding":
            labels.add("wrong_primitive")

    if _metric(item, "hallucinated_unsupported") >= 1.0:
        labels.add("unsupported_hallucination")
    if _metric(item, "false_rejection") >= 1.0:
        labels.add("false_rejection")
    is_rejection_error = "false_rejection" in labels or "unsupported_hallucination" in labels

    for error in _execution_errors(item):
        lowered = error.lower()
        if "no json object" in lowered or "json" in lowered or "parse" in lowered:
            labels.add("parse_failure")
        if "length" in lowered and "does not match" in lowered:
            labels.add("length_mismatch")
        missing_signal = "no executed signal" in lowered or "execution was not provided" in lowered
        if missing_signal and not is_rejection_error:
            labels.add("no_signal")
        if "generated code must include" in lowered or "disallowed" in lowered or "unsafe" in lowered:
            labels.add("unsafe_code")
        if not missing_signal and ("execution" in lowered or "failed" in lowered):
            labels.add("execution_failed")

    if plan_applicable and _metric(item, "primitive_f1", 1.0) < 0.999:
        labels.add("wrong_primitive")
    if plan_applicable and _metric(item, "window_iou", 1.0) < 0.999:
        labels.add("wrong_window")
    if plan_applicable and _metric(item, "parameter_error", 0.0) > 1e-9:
        labels.add("wrong_parameter")
    if _metric(item, "csr", 1.0) < 0.999:
        labels.add("semantic_constraint_failed")

    return [label for label in TAXONOMY_LABELS if label in labels]


def aggregate_failure_taxonomy(cases: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item for item in cases if not item.get("passed")]
    counts: Counter[str] = Counter()
    case_labels: dict[str, list[str]] = {}

    for item in failed:
        labels = classify_failure(item)
        case_labels[str(item.get("case_id", "<unknown>"))] = labels
        counts.update(labels or ["uncategorized"])

    total = len(cases)
    failed_total = len(failed)
    ordered_counts = {label: int(counts[label]) for label in TAXONOMY_LABELS if counts[label]}
    if counts.get("uncategorized"):
        ordered_counts["uncategorized"] = int(counts["uncategorized"])
    return {
        "total_cases": total,
        "failed_cases": failed_total,
        "failure_case_rate": float(failed_total / total) if total else 0.0,
        "counts": ordered_counts,
        "rates_among_all_cases": {
            label: float(count / total) if total else 0.0 for label, count in ordered_counts.items()
        },
        "rates_among_failed_cases": {
            label: float(count / failed_total) if failed_total else 0.0 for label, count in ordered_counts.items()
        },
        "case_labels": case_labels,
    }
