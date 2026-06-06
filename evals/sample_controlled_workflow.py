from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import atomic_timeseries as ats
from faithts_pipeline.hybrid_pipeline import PlanExecutionError, PlanValidationError, execute_plan, parse_plan
from faithts_pipeline.workflow import (
    GeneratedCodeExecutionError,
    RetrievalExample,
    UnsafeGeneratedCodeError,
    build_llm_client,
    load_primitive_specs,
    run_workflow_for_example,
    retrieve_primitives_for_caption,
)
from schema_validator import validate_instance, validate_payload


CONTROLLED_CASES_DIR = REPO_ROOT / "data" / "controlled_gold_cases"
ATOMIC_CASES_PATH = CONTROLLED_CASES_DIR / "single_claim_cases.json"
COMPOSITIONAL_CASES_PATH = CONTROLLED_CASES_DIR / "multi_claim_cases.json"
ADVERSARIAL_CASES_PATH = CONTROLLED_CASES_DIR / "robustness_cases.json"
DEFAULT_JSON_OUTPUT = CONTROLLED_CASES_DIR / "sample_controlled_workflow_report.json"
DEFAULT_MARKDOWN_OUTPUT = CONTROLLED_CASES_DIR / "sample_controlled_workflow_report.md"
DEFAULT_CASE_OUTPUT_DIR = REPO_ROOT / "evals" / "sample_controlled_workflow_cases"
DEFAULT_LOG_OUTPUT_DIR = REPO_ROOT / "logs" / "sample_controlled_workflow_cases"


def _load_cases(path: Path, category: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list payload in {path}")
    return [case for case in payload if case.get("category") == category]


def _sample_cases(cases: list[dict[str, Any]], seed: int, sample_count: int | None) -> list[dict[str, Any]]:
    if sample_count is None:
        return list(cases)
    if len(cases) < sample_count:
        raise ValueError(f"Requested {sample_count} cases but only found {len(cases)} cases.")
    rng = random.Random(seed)
    return rng.sample(cases, sample_count)


def _failure_reason(stage: str, code: str, message: str, related_step_ids: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"stage": stage, "code": code, "message": message}
    if related_step_ids:
        payload["related_step_ids"] = related_step_ids
    return payload


def _expected_step_primitives(case: dict[str, Any]) -> list[str]:
    return [step["primitive"] for step in case["expected_plan_outline"].get("steps", [])]


def _outline_reference_series(case: dict[str, Any]) -> np.ndarray | None:
    outline = case["expected_plan_outline"]
    if outline.get("needs_library_extension") or outline.get("unresolved_semantics"):
        return None
    series = np.zeros(int(case["length"]), dtype=float)
    for step in outline.get("steps", []):
        series = getattr(ats, step["primitive"])(series, **step["parameters"])
    return series


def _numeric_range_for_case(case: dict[str, Any], expected_series: np.ndarray | None) -> tuple[float, float]:
    explicit = case.get("numeric_range")
    if isinstance(explicit, dict) and "min" in explicit and "max" in explicit:
        return float(explicit["min"]), float(explicit["max"])

    if expected_series is not None and expected_series.size:
        finite = expected_series[np.isfinite(expected_series)]
        if finite.size:
            low = float(np.min(finite))
            high = float(np.max(finite))
            span = high - low
            pad = max(1.0, 0.10 * span)
            return low - pad, high + pad

    return -10.0, 10.0


def _fallback_failure_plan(
    *,
    case: dict[str, Any],
    registry_validation: dict[str, Any],
    numeric_range: tuple[float, float],
    reason: str,
) -> dict[str, Any]:
    plan_registry_validation = {
        "status": str(registry_validation.get("status", "not_run")),
        "schema_path": str(registry_validation.get("schema_path", "schemas/primitive.schema.json")),
        "target_path": str(registry_validation.get("target_path", "atomic_timeseries/registry.json")),
        "errors": list(registry_validation.get("errors", [])),
    }
    unresolved = {
        "id": "workflow_failure",
        "description": "Workflow failed before a schema-valid executable plan was available.",
        "source_text": str(case["description"]),
        "reason": reason,
    }
    return {
        "version": "1.0",
        "input_description": str(case["description"]),
        "registry_validation": plan_registry_validation,
        "global_context": {
            "length": int(case["length"]),
            "numeric_range": {"min": float(numeric_range[0]), "max": float(numeric_range[1])},
            "dt": None,
            "seed": None,
            "notes": "Fallback failure plan emitted by the workflow runner to preserve schema-valid artifacts.",
        },
        "steps": [
            {
                "id": "workflow_failure_step",
                "kind": "unresolved",
                "description": unresolved["description"],
                "semantic_role": "workflow failure",
                "confidence": 1.0,
                "source_text": unresolved["source_text"],
                "reason": unresolved["reason"],
                "needs_library_extension": True,
            }
        ],
        "unresolved_semantics": [unresolved],
        "needs_library_extension": True,
        "notes": "No primitive plan was accepted for this case.",
    }


def _eval_unresolved_semantics(plan: dict[str, Any]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in plan.get("unresolved_semantics", []):
        if not isinstance(item, dict):
            continue
        sanitized.append(
            {
                "id": str(item.get("id", "unresolved")),
                "description": str(item.get("description", "Unresolved semantic item.")),
                "reason": str(item.get("reason", "No reason recorded.")),
            }
        )
    return sanitized


def _compare_series(generated: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    diff = generated - truth
    mse = float(np.mean(np.square(diff)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    denom = np.maximum(np.abs(truth), 1e-6)
    mrr = float(np.mean(np.abs(diff) / denom))
    max_abs_error = float(np.max(np.abs(diff)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mrr": mrr,
        "max_abs_error": max_abs_error,
    }


def _primitive_match_score(expected: list[str], generated: list[str]) -> dict[str, Any]:
    expected_counter = Counter(expected)
    generated_counter = Counter(generated)
    overlap = sum(min(expected_counter[name], generated_counter[name]) for name in set(expected_counter) | set(generated_counter))
    precision = overlap / max(1, len(generated))
    recall = overlap / max(1, len(expected))
    f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_match": list(expected) == list(generated),
    }


def _signal_match_score(signal_checks: list[dict[str, Any]], gold_comparison: dict[str, float] | None) -> dict[str, Any]:
    passed = sum(1 for item in signal_checks if item["passed"])
    total = len(signal_checks)
    payload: dict[str, Any] = {
        "check_pass_rate": float(1.0 if total == 0 else passed / total),
        "passed_checks": int(passed),
        "total_checks": int(total),
    }
    if gold_comparison is not None:
        payload["rmse"] = float(gold_comparison["rmse"])
        payload["mae"] = float(gold_comparison["mae"])
        payload["max_abs_error"] = float(gold_comparison["max_abs_error"])
    return payload


def _segment(series: np.ndarray, start: int, end: int) -> np.ndarray:
    return series[start : end + 1]


def _is_relative_level_shift_case(case: dict[str, Any]) -> bool:
    outline_steps = case.get("expected_plan_outline", {}).get("steps", [])
    if len(outline_steps) != 1:
        return False
    step = outline_steps[0]
    if step.get("primitive") != "add_level_shift":
        return False
    point_values = case.get("expected_signal_checks", {}).get("point_values", [])
    return len(point_values) >= 3


def _run_expected_signal_checks(case: dict[str, Any], series: np.ndarray) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    signal_checks = case.get("expected_signal_checks", {})

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record("signal_length", len(series) == case["length"], f"actual={len(series)}")

    point_values = signal_checks.get("point_values", [])
    if _is_relative_level_shift_case(case):
        before_item = point_values[0]
        anchor_item = point_values[1]
        final_item = point_values[-1]
        expected_shift = float(anchor_item["value"]) - float(before_item["value"])
        actual_before = float(series[before_item["timestep"]])
        actual_anchor = float(series[anchor_item["timestep"]])
        actual_final = float(series[final_item["timestep"]])
        atol = max(
            float(before_item.get("tolerance", 1e-6)),
            float(anchor_item.get("tolerance", 1e-6)),
            float(final_item.get("tolerance", 1e-6)),
        )
        record(
            f"relative_shift:{anchor_item['timestep']}",
            np.isclose(actual_anchor - actual_before, expected_shift, atol=atol, rtol=0.0),
            f"actual_delta={actual_anchor - actual_before}",
        )
        record(
            f"relative_shift:{final_item['timestep']}",
            np.isclose(actual_final - actual_before, expected_shift, atol=atol, rtol=0.0),
            f"actual_delta={actual_final - actual_before}",
        )
        record(
            "shift_plateau",
            np.isclose(actual_final, actual_anchor, atol=atol, rtol=0.0),
            f"anchor={actual_anchor}, final={actual_final}",
        )
    else:
        for item in point_values:
            actual = float(series[item["timestep"]])
            passed = np.isclose(actual, item["value"], atol=item.get("tolerance", 1e-6), rtol=0.0)
            record(f"point_value:{item['timestep']}", passed, f"actual={actual}")

    for item in signal_checks.get("constant_segments", []):
        seg = _segment(series, item["start"], item["end"])
        passed = np.allclose(seg, item["value"], atol=item.get("tolerance", 1e-6), rtol=0.0)
        record(f"constant_segment:{item['start']}-{item['end']}", passed, f"value={item['value']}")

    for item in signal_checks.get("unchanged_windows", []):
        seg = _segment(series, item["start"], item["end"])
        passed = np.allclose(seg, item["value"], atol=item.get("tolerance", 1e-6), rtol=0.0)
        record(f"unchanged_window:{item['start']}-{item['end']}", passed, f"value={item['value']}")

    for item in signal_checks.get("monotone_windows", []):
        seg = _segment(series, item["start"], item["end"])
        diffs = np.diff(seg)
        if item["direction"] == "nondecreasing":
            passed = bool(np.all(diffs >= -1e-8))
        elif item["direction"] == "nonincreasing":
            passed = bool(np.all(diffs <= 1e-8))
        else:
            raise ValueError(f"Unsupported monotone direction: {item['direction']}")
        record(f"monotone_window:{item['start']}-{item['end']}", passed, f"direction={item['direction']}")

    bounds = signal_checks.get("value_bounds")
    if bounds is not None:
        actual_min = float(np.nanmin(series))
        actual_max = float(np.nanmax(series))
        passed = actual_min >= bounds["min"] and actual_max <= bounds["max"]
        record("value_bounds", passed, f"min={actual_min}, max={actual_max}")

    argmax_in = signal_checks.get("argmax_in")
    if argmax_in is not None:
        actual = int(np.nanargmax(series))
        passed = argmax_in[0] <= actual <= argmax_in[1]
        record("argmax_in", passed, f"actual={actual}")

    argmin_in = signal_checks.get("argmin_in")
    if argmin_in is not None:
        actual = int(np.nanargmin(series))
        passed = argmin_in[0] <= actual <= argmin_in[1]
        record("argmin_in", passed, f"actual={actual}")

    variance_comparison = signal_checks.get("variance_comparison")
    if variance_comparison is not None:
        left = _segment(series, *variance_comparison["reference_window"])
        right = _segment(series, *variance_comparison["target_window"])
        left_var = float(np.var(left))
        right_var = float(np.var(right))
        if variance_comparison["expect"] == "target_greater":
            passed = right_var > left_var
        else:
            raise ValueError(f"Unsupported variance comparison: {variance_comparison['expect']}")
        record("variance_comparison", passed, f"reference={left_var}, target={right_var}")

    for item in signal_checks.get("nonconstant_windows", []):
        seg = _segment(series, item["start"], item["end"])
        passed = not np.allclose(seg, seg[0], atol=1e-8, rtol=0.0)
        record(f"nonconstant_window:{item['start']}-{item['end']}", passed, "segment should vary")

    return checks


def _make_example(case: dict[str, Any], index: int) -> RetrievalExample:
    workflow_candidates = case.get("workflow_candidate_primitives", case.get("expected_candidate_primitives", []))
    return RetrievalExample(
        index=index,
        caption=case["description"],
        candidate_primitives=list(workflow_candidates),
        retrieval_status="gold-case",
        orchestration_level_descriptors=[],
        raw_record=case,
    )


def _case_summary(series: np.ndarray | None) -> dict[str, Any] | None:
    if series is None:
        return None
    return {
        "length": int(series.shape[0]),
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
    }


def _write_comparison_plot(
    *,
    title: str,
    reference_series: np.ndarray,
    generated_series: np.ndarray,
    output_path: Path,
) -> None:
    x = np.arange(reference_series.shape[0], dtype=int)
    error = generated_series - reference_series

    fig, (ax_series, ax_error) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax_series.plot(x, reference_series, label="reference", linewidth=2.0, color="#1f77b4")
    ax_series.plot(x, generated_series, label="generated", linewidth=1.8, color="#d62728", alpha=0.9)
    ax_series.set_title(title)
    ax_series.set_ylabel("value")
    ax_series.legend(loc="best")
    ax_series.grid(True, alpha=0.25)

    ax_error.plot(x, error, label="generated - reference", linewidth=1.5, color="#2ca02c")
    ax_error.axhline(0.0, color="#555555", linewidth=1.0, alpha=0.8)
    ax_error.set_xlabel("timestep")
    ax_error.set_ylabel("error")
    ax_error.legend(loc="best")
    ax_error.grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_case_artifacts(
    *,
    record: dict[str, Any],
    case: dict[str, Any],
    expected_series: np.ndarray | None,
    output_dir: Path,
    log_dir: Path,
) -> None:
    case_dir = output_dir / record["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = log_dir / record["case_id"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "description.txt").write_text(case["description"] + "\n", encoding="utf-8")
    (case_dir / "true_primitives.json").write_text(
        json.dumps(record["expected_step_primitives"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (case_dir / "true_steps.json").write_text(
        json.dumps(case["expected_plan_outline"].get("steps", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (case_dir / "reference_series.json").write_text(
        json.dumps(
            [float(value) for value in expected_series.tolist()] if expected_series is not None else [],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "plan.json").write_text(
        json.dumps(record.get("structured_plan", {}), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (case_dir / "plan_validation.json").write_text(
        json.dumps(record["plan_validation_status"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (case_dir / "generated_code.py").write_text(record.get("generated_code", "") + ("\n" if record.get("generated_code") else ""), encoding="utf-8")
    (case_dir / "eval_report.json").write_text(
        json.dumps(record["eval_report"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (case_dir / "generated_series.json").write_text(
        json.dumps(record.get("generated_series", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "case_id": record["case_id"],
        "seed": None,
        "series_length": record["length"],
        "numeric_range": record["numeric_range"],
        "summary": _case_summary(expected_series),
        "workflow_status": {
            "passed": record["passed"],
            "plan_validation_status": record["plan_validation_status"]["status"],
            "execution_status": record["execution_status"]["status"],
            "semantic_alignment_overall": record["eval_report"]["semantic_alignment_score"]["overall"],
            "failure_codes": [item["code"] for item in record["failure_reasons"]],
        },
        "primitive_match_score": record["primitive_match_score"],
        "signal_match_score": record["signal_match_score"],
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "evaluation_metrics.json").write_text(
        json.dumps(record.get("gold_comparison") or {}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generated_series_payload = record.get("generated_series")
    if expected_series is not None and isinstance(generated_series_payload, list) and generated_series_payload:
        generated_array = np.asarray(generated_series_payload, dtype=float)
        if generated_array.shape == expected_series.shape:
            _write_comparison_plot(
                title=record["case_id"],
                reference_series=expected_series,
                generated_series=generated_array,
                output_path=case_dir / "comparison.png",
            )


def _run_case(
    *,
    case: dict[str, Any],
    index: int,
    specs: dict[str, Any],
    llm_client: Any,
    registry_validation: dict[str, Any],
    output_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    example = _make_example(case, index)
    expected_series = _outline_reference_series(case)
    numeric_range = _numeric_range_for_case(case, expected_series)
    expected_primitives = _expected_step_primitives(case)

    record: dict[str, Any] = {
        "case_id": case["case_id"],
        "description": case["description"],
        "length": int(case["length"]),
        "numeric_range": {"min": numeric_range[0], "max": numeric_range[1]},
        "registry_validation": registry_validation,
        "retrieved_primitives": [],
        "retrieval_matches_expected_subset": False,
        "expected_candidate_primitives": list(case.get("expected_candidate_primitives", [])),
        "expected_step_primitives": expected_primitives,
        "generated_step_primitives": [],
        "plan_validation_status": {
            "status": "not_run",
            "schema_path": "schemas/plan.schema.json",
            "target_path": f"<{case['case_id']}>",
            "errors": [],
        },
        "code_generation_status": {
            "status": "not_run",
            "draft_generated": False,
            "final_generated": False,
            "artifacts": [],
            "errors": [],
        },
        "execution_status": {
            "status": "not_run",
            "errors": [],
        },
        "semantic_alignment_score": {
            "overall": 0.0,
            "description_to_plan": 0.0,
            "plan_to_code": 0.0,
            "code_to_signal": 0.0,
            "notes": "Not evaluated yet.",
        },
        "unresolved_semantics": [],
        "failure_reasons": [],
        "gold_comparison": None,
        "signal_checks": [],
        "primitive_match_score": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
        },
        "signal_match_score": {
            "check_pass_rate": 0.0,
            "passed_checks": 0,
            "total_checks": 0,
        },
        "plan_matches_expected_primitives": False,
        "eval_report_schema_validation": {
            "status": "not_run",
            "schema_path": "schemas/eval_report.schema.json",
            "target_path": f"<eval_report:{case['case_id']}>",
            "errors": [],
        },
    }

    failure_reasons: list[dict[str, Any]] = []
    evaluation = None
    generated_series = None

    try:
        retrieved_specs = retrieve_primitives_for_caption(
            example.caption,
            specs,
            retrieval_hints=example.candidate_primitives,
        )
        retrieved_names = [spec.function_name for spec in retrieved_specs]
        record["retrieved_primitives"] = retrieved_names
        record["retrieval_matches_expected_subset"] = set(case.get("expected_candidate_primitives", [])).issubset(retrieved_names)
        record["workflow_candidate_primitives"] = list(
            case.get("workflow_candidate_primitives", case.get("expected_candidate_primitives", []))
        )
        workflow_result = run_workflow_for_example(
            example=example,
            specs=specs,
            llm_client=llm_client,
            series_length=int(case["length"]),
            numeric_range=numeric_range,
        )
        record["model_provider"] = workflow_result.llm_provider
        record["model_name"] = workflow_result.llm_model
        record["response_id"] = workflow_result.llm_response_id
        record["structured_plan"] = workflow_result.structured_plan
        record["generated_step_primitives"] = [
            step["primitive"]
            for step in workflow_result.structured_plan.get("steps", [])
            if isinstance(step, dict) and step.get("kind") == "primitive"
        ]
        record["primitive_match_score"] = _primitive_match_score(
            expected_primitives,
            record["generated_step_primitives"],
        )
        record["plan_matches_expected_primitives"] = record["generated_step_primitives"] == expected_primitives
        record["unresolved_semantics"] = list(workflow_result.structured_plan.get("unresolved_semantics", []))
        record["plan_validation_status"] = validate_payload(
            "plan",
            workflow_result.structured_plan,
            target_path=f"<{case['case_id']}>",
        )
        record["generated_code"] = workflow_result.generated_code
        record["code_generation_status"] = {
            "status": "success",
            "draft_generated": True,
            "final_generated": True,
            "artifacts": [],
            "errors": [],
        }
        plan = parse_plan(
            workflow_result.structured_plan,
            workflow_result.retrieved_primitives,
            int(case["length"]),
            numeric_range=numeric_range,
        )
        generated_series = execute_plan(plan, example.caption, int(case["length"]))
        record["generated_series"] = [float(value) for value in generated_series.tolist()]
        record["execution_status"] = {
            "status": "success",
            "signal_summary": workflow_result.series_summary,
            "errors": [],
        }
        evaluation = None
        record["semantic_alignment_score"] = workflow_result.eval_report["semantic_alignment_score"]
        if workflow_result.failure_categories:
            failure_reasons.extend(
                _failure_reason("evaluation", category, category.replace("_", " "))
                for category in workflow_result.failure_categories
            )

        if expected_series is not None:
            record["gold_comparison"] = _compare_series(generated_series, expected_series)
            record["signal_checks"] = _run_expected_signal_checks(case, generated_series)
            record["signal_match_score"] = _signal_match_score(record["signal_checks"], record["gold_comparison"])
            if not all(item["passed"] for item in record["signal_checks"]):
                failure_reasons.append(
                    _failure_reason(
                        "evaluation",
                        "signal_checks_failed",
                        "Generated series did not satisfy the case expected signal checks.",
                    )
                )

    except PlanValidationError as exc:
        if "structured_plan" not in record:
            record["structured_plan"] = _fallback_failure_plan(
                case=case,
                registry_validation=registry_validation,
                numeric_range=numeric_range,
                reason=str(exc),
            )
            record["unresolved_semantics"] = _eval_unresolved_semantics(record["structured_plan"])
            record["plan_validation_status"] = validate_payload(
                "plan",
                record["structured_plan"],
                target_path=f"<{case['case_id']}>",
            )
        record["code_generation_status"]["errors"].append(str(exc))
        failure_reasons.append(_failure_reason("planning", exc.category, str(exc)))
    except UnsafeGeneratedCodeError as exc:
        if "structured_plan" not in record:
            record["structured_plan"] = _fallback_failure_plan(
                case=case,
                registry_validation=registry_validation,
                numeric_range=numeric_range,
                reason=str(exc),
            )
            record["unresolved_semantics"] = _eval_unresolved_semantics(record["structured_plan"])
            record["plan_validation_status"] = validate_payload(
                "plan",
                record["structured_plan"],
                target_path=f"<{case['case_id']}>",
            )
        if record["code_generation_status"]["status"] == "not_run":
            record["code_generation_status"]["status"] = "failed"
        record["code_generation_status"]["errors"].append(str(exc))
        failure_reasons.append(_failure_reason("code_generation", exc.category, str(exc)))
    except (PlanExecutionError, GeneratedCodeExecutionError) as exc:
        if "structured_plan" not in record:
            record["structured_plan"] = _fallback_failure_plan(
                case=case,
                registry_validation=registry_validation,
                numeric_range=numeric_range,
                reason=str(exc),
            )
            record["unresolved_semantics"] = _eval_unresolved_semantics(record["structured_plan"])
            record["plan_validation_status"] = validate_payload(
                "plan",
                record["structured_plan"],
                target_path=f"<{case['case_id']}>",
            )
        if record["execution_status"]["status"] == "not_run":
            record["execution_status"]["status"] = "failed"
        record["execution_status"]["errors"].append(str(exc))
        failure_reasons.append(_failure_reason("execution", getattr(exc, "category", "execution_failed"), str(exc)))
    except Exception as exc:
        if "structured_plan" not in record:
            record["structured_plan"] = _fallback_failure_plan(
                case=case,
                registry_validation=registry_validation,
                numeric_range=numeric_range,
                reason=str(exc),
            )
            record["unresolved_semantics"] = _eval_unresolved_semantics(record["structured_plan"])
            record["plan_validation_status"] = validate_payload(
                "plan",
                record["structured_plan"],
                target_path=f"<{case['case_id']}>",
            )
        if record["code_generation_status"]["status"] == "not_run":
            record["code_generation_status"]["status"] = "failed"
        record["code_generation_status"]["errors"].append(str(exc))
        failure_reasons.append(_failure_reason("planning", "unexpected_error", str(exc)))

    record["failure_reasons"] = failure_reasons

    if evaluation is None:
        from faithts_pipeline.workflow import CodeEvaluation

        evaluation = CodeEvaluation(
            contract_valid=False,
            executed=False,
            numerically_plausible=False,
            semantically_plausible=False,
            usable_success=False,
            failure_categories=[item["code"] for item in failure_reasons] or ["not_run"],
            series=generated_series,
        )

    if "workflow_result" in locals():
        eval_report = dict(workflow_result.eval_report)
        eval_report["failure_reasons"] = failure_reasons
        eval_report["notes"] = f"atomic_case_id={case['case_id']}"
    else:
        from faithts_pipeline.workflow import build_eval_report

        eval_report = build_eval_report(
            caption=example.caption,
            registry_validation=registry_validation,
            plan_validation_status=record["plan_validation_status"],
            code_generation_status=record["code_generation_status"],
            execution_status=record["execution_status"],
            evaluation=evaluation,
            unresolved_semantics=record["unresolved_semantics"],
            failure_reasons=failure_reasons,
            notes=f"atomic_case_id={case['case_id']}",
        )
    record["eval_report"] = eval_report
    record["eval_report_schema_validation"] = validate_payload(
        "eval_report",
        eval_report,
        target_path=f"<eval_report:{case['case_id']}>",
    )
    record["passed"] = (
        registry_validation["status"] == "valid"
        and record["plan_validation_status"]["status"] == "valid"
        and record["code_generation_status"]["status"] == "success"
        and record["execution_status"]["status"] == "success"
        and not failure_reasons
    )
    _write_case_artifacts(
        record=record,
        case=case,
        expected_series=expected_series,
        output_dir=output_dir,
        log_dir=log_dir,
    )
    return record


def run_sample(seed: int, sample_count: int, provider: str, output_dir: Path, log_dir: Path) -> dict[str, Any]:
    registry_validation = validate_instance("primitive", REPO_ROOT / "atomic_timeseries" / "registry.json")
    specs = load_primitive_specs()
    llm_client = build_llm_client(provider)
    sampled_cases = _sample_cases(_load_cases(ATOMIC_CASES_PATH, "atomic"), seed, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    records = [
        _run_case(
            case=case,
            index=index,
            specs=specs,
            llm_client=llm_client,
            registry_validation=registry_validation,
            output_dir=output_dir,
            log_dir=log_dir,
        )
        for index, case in enumerate(sampled_cases, start=1)
    ]

    passed = sum(1 for record in records if record["passed"])
    return {
        "seed": seed,
        "sample_count": sample_count,
        "provider": provider,
        "registry_validation": registry_validation,
        "passed_cases": passed,
        "failed_cases": len(records) - passed,
        "cases": records,
    }


def run_cases_file(
    *,
    cases_path: Path,
    category: str,
    seed: int,
    sample_count: int | None,
    provider: str,
    output_dir: Path,
    log_dir: Path,
    case_id_prefix: str | None = None,
) -> dict[str, Any]:
    registry_validation = validate_instance("primitive", REPO_ROOT / "atomic_timeseries" / "registry.json")
    specs = load_primitive_specs()
    llm_client = build_llm_client(provider)
    cases = _load_cases(cases_path, category)
    if case_id_prefix is not None:
        cases = [case for case in cases if str(case.get("case_id", "")).startswith(case_id_prefix)]
    sampled_cases = _sample_cases(cases, seed, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    records = [
        _run_case(
            case=case,
            index=index,
            specs=specs,
            llm_client=llm_client,
            registry_validation=registry_validation,
            output_dir=output_dir,
            log_dir=log_dir,
        )
        for index, case in enumerate(sampled_cases, start=1)
    ]
    passed = sum(1 for record in records if record["passed"])
    return {
        "seed": seed,
        "sample_count": len(sampled_cases),
        "provider": provider,
        "category": category,
        "cases_path": str(cases_path),
        "registry_validation": registry_validation,
        "passed_cases": passed,
        "failed_cases": len(records) - passed,
        "cases": records,
    }


def render_markdown(report: dict[str, Any]) -> str:
    report_category = report.get("category", "atomic")
    lines = [
        f"# {report_category.title()} Workflow Sample Report",
        "",
        f"- seed: `{report['seed']}`",
        f"- sample_count: `{report['sample_count']}`",
        f"- provider: `{report['provider']}`",
        f"- registry_validation: `{report['registry_validation']['status']}`",
        f"- passed_cases: `{report['passed_cases']}`",
        f"- failed_cases: `{report['failed_cases']}`",
        "",
    ]

    for case in report["cases"]:
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- passed: `{case['passed']}`",
                f"- retrieved_primitives: `{case['retrieved_primitives']}`",
                f"- generated_step_primitives: `{case['generated_step_primitives']}`",
                f"- expected_step_primitives: `{case['expected_step_primitives']}`",
                f"- plan_validation: `{case['plan_validation_status']['status']}`",
                f"- execution_status: `{case['execution_status']['status']}`",
                f"- primitive_match_score: `f1={case['primitive_match_score']['f1']:.3f}, precision={case['primitive_match_score']['precision']:.3f}, recall={case['primitive_match_score']['recall']:.3f}, exact={case['primitive_match_score']['exact_match']}`",
                f"- signal_match_score: `check_pass_rate={case['signal_match_score']['check_pass_rate']:.3f}, passed_checks={case['signal_match_score']['passed_checks']}/{case['signal_match_score']['total_checks']}`",
                f"- semantic_alignment_overall: `{case['eval_report']['semantic_alignment_score']['overall']}`",
            ]
        )
        if case["gold_comparison"] is not None:
            lines.append(
                f"- gold_comparison: `rmse={case['gold_comparison']['rmse']:.6f}, mae={case['gold_comparison']['mae']:.6f}, max_abs_error={case['gold_comparison']['max_abs_error']:.6f}`"
            )
        if case["failure_reasons"]:
            lines.append(f"- failure_reasons: `{[item['code'] for item in case['failure_reasons']]}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample atomic gold cases and run the full workflow.")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--case-output-dir", type=Path, default=DEFAULT_CASE_OUTPUT_DIR)
    parser.add_argument("--log-output-dir", type=Path, default=DEFAULT_LOG_OUTPUT_DIR)
    parser.add_argument("--cases-path", type=Path, default=ATOMIC_CASES_PATH)
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial"], default="atomic")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--case-id-prefix", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_count = None if args.all_cases else args.sample_count
    report = run_cases_file(
        cases_path=args.cases_path,
        category=args.category,
        seed=args.seed,
        sample_count=sample_count,
        provider=args.provider,
        output_dir=args.case_output_dir,
        log_dir=args.log_output_dir,
        case_id_prefix=args.case_id_prefix,
    )
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote case artifacts under {args.case_output_dir}")
    print(f"Wrote metrics under {args.log_output_dir}")
    print(f"Passed {report['passed_cases']} / {report['sample_count']} sampled atomic cases.")
    return 0 if report["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
