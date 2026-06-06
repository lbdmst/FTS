from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import atomic_timeseries as ats
from schema_validator import validate_instance, validate_payload

from .checkers import CheckResult, run_feature_checker
from .semantic_spec import SemanticFeature, SemanticSpec, load_semantic_spec


FAILURE_CODES = {
    "schema": "F_schema",
    "grounding": "F_grounding",
    "execution": "F_execution",
    "semantic": "F_semantic",
    "rejection": "F_rejection",
    "safety": "F_safety",
}


@dataclass(frozen=True)
class CoreEvaluationResult:
    case_id: str
    passed: bool
    requires_rejection: bool
    rejected: bool
    registry_valid: bool
    plan_valid: bool
    execution_success: bool
    semantic_alignment_score: float
    constraint_satisfaction_rate: float
    primitive_f1: float
    window_iou: float
    parameter_error: float
    correct_rejection: bool
    false_rejection: bool
    hallucinated_unsupported: bool
    unresolved_semantics: list[dict[str, Any]]
    failure_reasons: list[dict[str, Any]]
    feature_results: list[CheckResult]
    plan_validation_applicable: bool = True


def _failure(stage: str, code: str, message: str, related_step_ids: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"stage": stage, "code": code, "message": message}
    if related_step_ids:
        payload["related_step_ids"] = related_step_ids
    return payload


def _primitive_steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in plan.get("steps", []) if isinstance(step, dict) and step.get("kind", "primitive") == "primitive"]


def _unresolved_steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in plan.get("steps", []) if isinstance(step, dict) and step.get("kind") == "unresolved"]


def _plan_rejected(plan: dict[str, Any]) -> bool:
    return bool(plan.get("needs_library_extension")) or bool(plan.get("unresolved_semantics")) or bool(_unresolved_steps(plan))


def _candidate_primitives(plan: dict[str, Any]) -> list[str]:
    primitives: list[str] = []
    for step in _primitive_steps(plan):
        primitive = step.get("primitive")
        if primitive and primitive not in primitives:
            primitives.append(str(primitive))
    return primitives


def _plan_execution_trace(plan: dict[str, Any], length: int) -> tuple[dict[str, np.ndarray], np.ndarray | None, str | None]:
    """Replay the plan and keep the series after each step for compositional checks."""

    try:
        from faithts_pipeline.hybrid_pipeline import _ordered_steps, parse_plan

        parsed = parse_plan(plan, candidate_primitives=_candidate_primitives(plan), series_length=length)
        ordered_steps = _ordered_steps(parsed)
        series = np.zeros(length, dtype=float)
        trace: dict[str, np.ndarray] = {}
        for step in ordered_steps:
            series = getattr(ats, step.primitive)(series, **step.parameters)
            trace[step.id] = np.asarray(series, dtype=float).copy()
        return trace, series, None
    except Exception as exc:
        return {}, None, str(exc)


def _window_from_step(step: dict[str, Any], length: int) -> tuple[int, int] | None:
    params = step.get("parameters", {})
    if "start_timestep" in params and "end_timestep" in params:
        return int(params["start_timestep"]), int(params["end_timestep"])
    if "anchor_timestep" in params:
        return int(params["anchor_timestep"]), int(params.get("end_timestep", length - 1))
    if "timestep" in params:
        timestep = int(params["timestep"])
        width = int(params.get("half_width", params.get("width", 0)))
        return max(0, timestep - width), min(length - 1, timestep + width)
    if "start_timestep" in params or "end_timestep" in params:
        return int(params.get("start_timestep", 0)), int(params.get("end_timestep", length - 1))
    return None


def _iou(left: tuple[int, int] | None, right: tuple[int, int] | None) -> float:
    if left is None or right is None:
        return 0.0
    a, b = left
    c, d = right
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(b, d) - min(a, c) + 1
    return float(intersection / union) if union > 0 else 0.0


def _best_step_for_feature(feature: SemanticFeature, steps: list[dict[str, Any]], length: int) -> dict[str, Any] | None:
    candidates = [step for step in steps if step.get("primitive") in set(feature.expected_primitives)]
    if not candidates:
        return None
    if feature.window is None:
        return candidates[0]
    return max(candidates, key=lambda step: _iou(feature.window, _window_from_step(step, length)))


def _multiset_f1(expected: list[str], predicted: list[str]) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    remaining = list(predicted)
    matched = 0
    for item in expected:
        if item in remaining:
            remaining.remove(item)
            matched += 1
    precision = matched / len(predicted)
    recall = matched / len(expected)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _multiset_f1_with_alternatives(expected: list[tuple[str, ...]], predicted: list[str]) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    remaining = list(predicted)
    matched = 0
    for options in expected:
        for index, item in enumerate(remaining):
            if item in options:
                matched += 1
                remaining.pop(index)
                break
    precision = matched / len(predicted)
    recall = matched / len(expected)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _parameter_error(feature: SemanticFeature, step: dict[str, Any] | None, length: int) -> float:
    if step is None:
        return 1.0
    predicted = step.get("parameters", {})
    errors: list[float] = []
    for key, expected_value in feature.parameters.items():
        if key in {"forbidden_primitives", "used_primitives"}:
            continue
        if key not in predicted:
            errors.append(1.0)
            continue
        try:
            expected = float(expected_value)
            actual = float(predicted[key])
        except (TypeError, ValueError):
            errors.append(0.0 if predicted[key] == expected_value else 1.0)
            continue
        if "timestep" in key or key in {"period"}:
            denom = max(1.0, float(length))
        else:
            denom = max(1.0, abs(expected))
        errors.append(min(1.0, abs(actual - expected) / denom))
    return float(np.mean(errors)) if errors else 0.0


def _with_used_primitives(feature: SemanticFeature, used_primitives: list[str]) -> SemanticFeature:
    if feature.checker != "check_negation":
        return feature
    params = dict(feature.parameters)
    params["used_primitives"] = used_primitives
    return SemanticFeature(
        id=feature.id,
        semantic=feature.semantic,
        expected_primitives=feature.expected_primitives,
        window=feature.window,
        parameters=params,
        effect_type=feature.effect_type,
        checker=feature.checker,
        required=feature.required,
        weight=feature.weight,
        tolerance=feature.tolerance,
    )


def _signal_check_result(checker: str, passed: bool, error: float, reason: str) -> CheckResult:
    return CheckResult(
        feature_id=checker,
        checker=checker,
        passed=bool(passed),
        score=1.0 if passed else 0.0,
        error=float(error),
        reason=reason,
    )


def _bounded_segment(array: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end < start or end >= len(array):
        return np.asarray([], dtype=float)
    return array[start : end + 1]


def _run_final_signal_checks(series: np.ndarray, checks: dict[str, Any]) -> list[CheckResult]:
    """Evaluate gold-case final-signal checks against a direct output series."""

    array = np.asarray(series, dtype=float)
    results: list[CheckResult] = []
    if "length" in checks:
        expected = int(checks["length"])
        error = abs(len(array) - expected)
        results.append(_signal_check_result("length", error == 0, float(error), f"length={len(array)}, expected={expected}"))

    for index, item in enumerate(checks.get("point_values", []), start=1):
        timestep = int(item["timestep"])
        expected = float(item["value"])
        tolerance = float(item.get("tolerance", 1e-6))
        if timestep < 0 or timestep >= len(array):
            results.append(_signal_check_result(f"point_value_{index}", False, np.inf, "timestep out of bounds"))
            continue
        error = abs(float(array[timestep]) - expected)
        results.append(_signal_check_result(f"point_value_{index}", error <= tolerance, error, f"t={timestep}, error={error:.6g}"))

    for key in ("constant_segments", "unchanged_windows"):
        for index, item in enumerate(checks.get(key, []), start=1):
            start, end = int(item["start"]), int(item["end"])
            expected = float(item["value"])
            tolerance = float(item.get("tolerance", 1e-6))
            seg = _bounded_segment(array, start, end)
            if seg.size == 0:
                results.append(_signal_check_result(f"{key}_{index}", False, np.inf, "window out of bounds"))
                continue
            value_error = float(np.nanmax(np.abs(seg - expected)))
            std_error = float(np.nanstd(seg))
            error = max(value_error, std_error)
            results.append(
                _signal_check_result(
                    f"{key}_{index}",
                    error <= tolerance,
                    error,
                    f"window=[{start},{end}], max_abs={value_error:.6g}, std={std_error:.6g}",
                )
            )

    for index, item in enumerate(checks.get("monotone_windows", []), start=1):
        start, end = int(item["start"]), int(item["end"])
        direction = str(item["direction"])
        seg = _bounded_segment(array, start, end)
        if seg.size < 2:
            results.append(_signal_check_result(f"monotone_window_{index}", False, np.inf, "window too short"))
            continue
        diffs = np.diff(seg)
        if direction == "nondecreasing":
            violation = float(max(0.0, -np.nanmin(diffs)))
            passed = bool(np.all(diffs >= -1e-8))
        elif direction == "nonincreasing":
            violation = float(max(0.0, np.nanmax(diffs)))
            passed = bool(np.all(diffs <= 1e-8))
        else:
            violation = np.inf
            passed = False
        results.append(
            _signal_check_result(
                f"monotone_window_{index}",
                passed,
                violation,
                f"window=[{start},{end}], direction={direction}, violation={violation:.6g}",
            )
        )

    if "argmax_in" in checks:
        start, end = [int(value) for value in checks["argmax_in"]]
        if array.size == 0 or not np.isfinite(array).any():
            results.append(_signal_check_result("argmax_in", False, np.inf, "empty or non-finite signal"))
        else:
            argmax = int(np.nanargmax(array))
            error = 0.0 if start <= argmax <= end else 1.0
            results.append(_signal_check_result("argmax_in", error == 0.0, error, f"argmax={argmax}, expected=[{start},{end}]"))

    if "argmin_in" in checks:
        start, end = [int(value) for value in checks["argmin_in"]]
        if array.size == 0 or not np.isfinite(array).any():
            results.append(_signal_check_result("argmin_in", False, np.inf, "empty or non-finite signal"))
        else:
            argmin = int(np.nanargmin(array))
            error = 0.0 if start <= argmin <= end else 1.0
            results.append(_signal_check_result("argmin_in", error == 0.0, error, f"argmin={argmin}, expected=[{start},{end}]"))

    if "value_bounds" in checks:
        bounds = checks["value_bounds"]
        lower = float(bounds.get("min", -np.inf))
        upper = float(bounds.get("max", np.inf))
        if array.size == 0:
            results.append(_signal_check_result("value_bounds", False, np.inf, "empty signal"))
        else:
            low_violation = max(0.0, lower - float(np.nanmin(array)))
            high_violation = max(0.0, float(np.nanmax(array)) - upper)
            error = max(low_violation, high_violation)
            results.append(_signal_check_result("value_bounds", error <= 1e-8, error, f"bounds=[{lower},{upper}], error={error:.6g}"))

    for index, item in enumerate(checks.get("nonconstant_windows", []), start=1):
        start, end = int(item["start"]), int(item["end"])
        seg = _bounded_segment(array, start, end)
        if seg.size < 2:
            results.append(_signal_check_result(f"nonconstant_window_{index}", False, np.inf, "window too short"))
            continue
        std = float(np.nanstd(seg))
        results.append(_signal_check_result(f"nonconstant_window_{index}", std > 1e-12, 0.0 if std > 1e-12 else 1.0, f"std={std:.6g}"))

    if "variance_comparison" in checks:
        item = checks["variance_comparison"]
        ref_start, ref_end = [int(value) for value in item["reference_window"]]
        target_start, target_end = [int(value) for value in item["target_window"]]
        ref_seg = _bounded_segment(array, ref_start, ref_end)
        target_seg = _bounded_segment(array, target_start, target_end)
        if ref_seg.size < 2 or target_seg.size < 2:
            results.append(_signal_check_result("variance_comparison", False, np.inf, "comparison window too short"))
        else:
            ref_std = float(np.nanstd(ref_seg))
            target_std = float(np.nanstd(target_seg))
            expect = str(item.get("expect", "target_greater"))
            if expect == "target_greater":
                passed = target_std > ref_std
                error = 0.0 if passed else ref_std - target_std
            else:
                passed = target_std < ref_std
                error = 0.0 if passed else target_std - ref_std
            results.append(
                _signal_check_result(
                    "variance_comparison",
                    passed,
                    float(max(0.0, error)),
                    f"reference_std={ref_std:.6g}, target_std={target_std:.6g}, expect={expect}",
                )
            )

    if "periodicity_hint" in checks:
        # The current gold signal contract records this as a hint, not a strict
        # pass/fail constraint. Keep the check visible without failing the case.
        results.append(_signal_check_result("periodicity_hint", True, 0.0, f"hint={checks['periodicity_hint']}"))

    if "deterministic_seed" in checks:
        # Seed provenance is not directly observable from a final array. Code
        # safety/reproducibility checks should handle it when raw code is in scope.
        results.append(_signal_check_result("deterministic_seed", True, 0.0, f"seed={checks['deterministic_seed']}"))

    return results


def _series_for_feature(
    feature: SemanticFeature,
    matched_step: dict[str, Any] | None,
    trace: dict[str, np.ndarray],
    final_series: np.ndarray,
) -> np.ndarray:
    if feature.checker == "check_negation":
        return final_series
    if matched_step is not None:
        step_id = str(matched_step.get("id", ""))
        if step_id in trace:
            return trace[step_id]
    return final_series


def evaluate_case(
    *,
    semantic_spec: dict[str, Any] | SemanticSpec,
    plan: dict[str, Any],
    series: np.ndarray | None = None,
    registry_path: str = "atomic_timeseries/registry.json",
    execution_success: bool | None = None,
    code_safe: bool | None = None,
) -> CoreEvaluationResult:
    spec = load_semantic_spec(semantic_spec)
    registry_validation = validate_instance("registry", registry_path)
    plan_validation = validate_payload("plan", plan, target_path=f"{spec.case_id}:plan")
    registry_valid = registry_validation["status"] == "valid"
    plan_valid = plan_validation["status"] == "valid"
    rejected = _plan_rejected(plan)
    exec_ok = bool(series is not None) if execution_success is None else bool(execution_success)

    failure_reasons: list[dict[str, Any]] = []
    if not registry_valid:
        failure_reasons.append(_failure("registry_validation", FAILURE_CODES["schema"], "registry schema validation failed"))
    if not plan_valid:
        failure_reasons.append(_failure("plan_validation", FAILURE_CODES["schema"], "plan schema validation failed"))
    if code_safe is False:
        failure_reasons.append(_failure("code_generation", FAILURE_CODES["safety"], "generated code failed safety validation"))

    primitive_steps = _primitive_steps(plan)
    used_primitives = [str(step.get("primitive")) for step in primitive_steps if step.get("primitive")]
    supported_features = [feature for feature in spec.features if feature.required]
    expected_primitive_options = [
        tuple(feature.expected_primitives)
        for feature in supported_features
        if feature.expected_primitives and feature.checker != "check_negation"
    ]
    primitive_f1 = _multiset_f1_with_alternatives(expected_primitive_options, used_primitives)

    matched_steps = [_best_step_for_feature(feature, primitive_steps, spec.length) for feature in supported_features]
    window_features = [
        (feature, step)
        for feature, step in zip(supported_features, matched_steps)
        if feature.checker != "check_negation"
    ]
    window_scores = [
        _iou(feature.window, _window_from_step(step, spec.length))
        for feature, step in window_features
        if feature.window is not None and step is not None
    ]
    window_iou = float(np.mean(window_scores)) if window_scores else (1.0 if not window_features else 0.0)
    param_errors = [
        _parameter_error(feature, step, spec.length)
        for feature, step in zip(supported_features, matched_steps)
        if feature.checker != "check_negation"
    ]
    parameter_error = float(np.mean(param_errors)) if param_errors else 0.0

    correct_rejection = bool(spec.requires_rejection and rejected)
    false_rejection = bool((not spec.requires_rejection) and rejected)
    hallucinated_unsupported = bool(spec.requires_rejection and not rejected)

    if spec.requires_rejection:
        if hallucinated_unsupported:
            failure_reasons.append(
                _failure("planning", FAILURE_CODES["rejection"], "unsupported semantics were generated instead of rejected")
            )
        feature_results: list[CheckResult] = []
        csr = 1.0 if correct_rejection else 0.0
        semantic_score = csr
        passed = bool(registry_valid and plan_valid and correct_rejection)
        return CoreEvaluationResult(
            case_id=spec.case_id,
            passed=passed,
            requires_rejection=True,
            rejected=rejected,
            registry_valid=registry_valid,
            plan_valid=plan_valid,
            execution_success=exec_ok,
            semantic_alignment_score=semantic_score,
            constraint_satisfaction_rate=csr,
            primitive_f1=primitive_f1,
            window_iou=window_iou,
            parameter_error=parameter_error,
            correct_rejection=correct_rejection,
            false_rejection=false_rejection,
            hallucinated_unsupported=hallucinated_unsupported,
            unresolved_semantics=list(plan.get("unresolved_semantics", [])),
            failure_reasons=failure_reasons,
            feature_results=feature_results,
        )

    if false_rejection:
        failure_reasons.append(_failure("planning", FAILURE_CODES["rejection"], "supported case was incorrectly rejected"))
    if series is None and not rejected:
        failure_reasons.append(_failure("execution", FAILURE_CODES["execution"], "no executed signal was provided"))

    signal_matches_plan_execution = True
    trace: dict[str, np.ndarray] = {}
    if plan_valid and series is not None and not rejected:
        trace, traced_final, trace_error = _plan_execution_trace(plan, spec.length)
        if trace_error is not None:
            signal_matches_plan_execution = False
            failure_reasons.append(
                _failure("execution", FAILURE_CODES["execution"], f"plan trace execution failed: {trace_error}")
            )
        elif traced_final is not None and not np.allclose(
            np.asarray(series, dtype=float),
            traced_final,
            atol=1e-8,
            rtol=1e-8,
            equal_nan=True,
        ):
            signal_matches_plan_execution = False
            failure_reasons.append(
                _failure("execution", FAILURE_CODES["execution"], "provided signal does not match replayed plan execution")
            )

    feature_results = []
    if series is not None and not rejected:
        array = np.asarray(series, dtype=float)
        for feature, matched_step in zip(supported_features, matched_steps):
            feature_series = _series_for_feature(feature, matched_step, trace, array)
            feature_results.append(run_feature_checker(feature_series, _with_used_primitives(feature, used_primitives)))

    if supported_features and feature_results:
        required_results = [item for item in feature_results if any(feature.id == item.feature_id for feature in supported_features)]
        csr = float(np.mean([1.0 if item.passed else 0.0 for item in required_results])) if required_results else 0.0
    elif supported_features:
        csr = 0.0
    else:
        csr = 1.0

    for item in feature_results:
        if not item.passed:
            failure_reasons.append(_failure("evaluation", FAILURE_CODES["semantic"], item.reason))

    semantic_score = float(np.mean([csr, primitive_f1, max(0.0, 1.0 - parameter_error)]))
    passed = bool(
        registry_valid
        and plan_valid
        and exec_ok
        and signal_matches_plan_execution
        and not false_rejection
        and csr >= 0.9
    )
    return CoreEvaluationResult(
        case_id=spec.case_id,
        passed=passed,
        requires_rejection=False,
        rejected=rejected,
        registry_valid=registry_valid,
        plan_valid=plan_valid,
        execution_success=exec_ok,
        semantic_alignment_score=semantic_score,
        constraint_satisfaction_rate=csr,
        primitive_f1=primitive_f1,
        window_iou=window_iou,
        parameter_error=parameter_error,
        correct_rejection=False,
        false_rejection=false_rejection,
        hallucinated_unsupported=False,
        unresolved_semantics=list(plan.get("unresolved_semantics", [])),
        failure_reasons=failure_reasons,
        feature_results=feature_results,
    )


def evaluate_direct_signal_case(
    *,
    semantic_spec: dict[str, Any] | SemanticSpec,
    series: np.ndarray | None,
    signal_checks: dict[str, Any] | None = None,
    registry_path: str = "atomic_timeseries/registry.json",
    execution_success: bool | None = None,
) -> CoreEvaluationResult:
    """Evaluate a direct array/code baseline by its final signal, not by plan schema."""

    spec = load_semantic_spec(semantic_spec)
    registry_validation = validate_instance("registry", registry_path)
    registry_valid = registry_validation["status"] == "valid"
    exec_ok = bool(series is not None) if execution_success is None else bool(execution_success)
    failure_reasons: list[dict[str, Any]] = []
    if not registry_valid:
        failure_reasons.append(_failure("registry_validation", FAILURE_CODES["schema"], "registry schema validation failed"))

    if spec.requires_rejection:
        if series is not None:
            failure_reasons.append(
                _failure(
                    "evaluation",
                    FAILURE_CODES["rejection"],
                    "unsupported semantics cannot be verified from a direct final signal",
                )
            )
        csr = 0.0
        return CoreEvaluationResult(
            case_id=spec.case_id,
            passed=False,
            requires_rejection=True,
            rejected=False,
            registry_valid=registry_valid,
            plan_valid=False,
            execution_success=exec_ok,
            semantic_alignment_score=csr,
            constraint_satisfaction_rate=csr,
            primitive_f1=0.0,
            window_iou=0.0,
            parameter_error=1.0,
            correct_rejection=False,
            false_rejection=False,
            hallucinated_unsupported=bool(series is not None),
            unresolved_semantics=[],
            failure_reasons=failure_reasons,
            feature_results=[],
            plan_validation_applicable=False,
        )

    if series is None:
        failure_reasons.append(_failure("execution", FAILURE_CODES["execution"], "no executed signal was provided"))
        feature_results: list[CheckResult] = []
        csr = 0.0
    else:
        array = np.asarray(series, dtype=float)
        if signal_checks:
            feature_results = _run_final_signal_checks(array, signal_checks)
        else:
            feature_results = [
                run_feature_checker(array, _with_used_primitives(feature, []))
                for feature in spec.features
                if feature.required
            ]
        csr = float(np.mean([1.0 if item.passed else 0.0 for item in feature_results])) if feature_results else 1.0
        for item in feature_results:
            if not item.passed:
                failure_reasons.append(_failure("evaluation", FAILURE_CODES["semantic"], item.reason))

    passed = bool(registry_valid and exec_ok and csr >= 0.9)
    return CoreEvaluationResult(
        case_id=spec.case_id,
        passed=passed,
        requires_rejection=False,
        rejected=False,
        registry_valid=registry_valid,
        plan_valid=False,
        execution_success=exec_ok,
        semantic_alignment_score=csr,
        constraint_satisfaction_rate=csr,
        primitive_f1=0.0,
        window_iou=0.0,
        parameter_error=0.0 if passed else 1.0,
        correct_rejection=False,
        false_rejection=False,
        hallucinated_unsupported=False,
        unresolved_semantics=[],
        failure_reasons=failure_reasons,
        feature_results=feature_results,
        plan_validation_applicable=False,
    )


def _mean_or_none(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def aggregate_results(results: list[CoreEvaluationResult]) -> dict[str, float | None]:
    if not results:
        return {
            "cases": 0.0,
            "pass_rate": 0.0,
            "csr": 0.0,
            "primitive_f1": 0.0,
            "window_iou": 0.0,
            "parameter_error": 0.0,
            "crr": 0.0,
            "frr": 0.0,
            "hus": 0.0,
        }
    unsupported = [item for item in results if item.requires_rejection]
    supported = [item for item in results if not item.requires_rejection]
    return {
        "cases": float(len(results)),
        "pass_rate": float(np.mean([item.passed for item in results])),
        "csr": float(np.mean([item.constraint_satisfaction_rate for item in results])),
        "primitive_f1": float(np.mean([item.primitive_f1 for item in results])),
        "window_iou": float(np.mean([item.window_iou for item in results])),
        "parameter_error": float(np.mean([item.parameter_error for item in results])),
        "crr": _mean_or_none([item.correct_rejection for item in unsupported]),
        "frr": _mean_or_none([item.false_rejection for item in supported]),
        "hus": _mean_or_none([item.hallucinated_unsupported for item in unsupported]),
    }


def _validation_status(valid: bool, schema_path: str, target_path: str, *, applicable: bool = True) -> dict[str, Any]:
    status = "valid" if valid else "invalid"
    if not applicable:
        status = "not_run"
    return {"status": status, "schema_path": schema_path, "target_path": target_path, "errors": []}


def _eval_unresolved_semantics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        payload = {
            "id": str(item.get("id") or f"unresolved_{index}"),
            "description": str(item.get("description") or item.get("semantic") or item.get("source_text") or "unresolved semantic"),
            "reason": str(item.get("reason") or "Requested semantic is outside the current primitive library."),
        }
        related = item.get("related_step_ids")
        if related:
            payload["related_step_ids"] = [str(value) for value in related]
        normalized.append(payload)
    return normalized


def build_eval_report_from_result(result: CoreEvaluationResult, description: str) -> dict[str, Any]:
    status = "success" if result.execution_success else "not_run"
    report = {
        "version": "1.0",
        "input_description": description,
        "registry_validation": _validation_status(
            result.registry_valid, "schemas/primitive.schema.json", "atomic_timeseries/registry.json"
        ),
        "plan_validation_status": _validation_status(
            result.plan_valid,
            "schemas/plan.schema.json",
            f"{result.case_id}:plan",
            applicable=result.plan_validation_applicable,
        ),
        "code_generation_status": {
            "status": "not_run",
            "draft_generated": False,
            "final_generated": False,
            "errors": [],
        },
        "execution_status": {
            "status": status,
            "errors": [] if result.execution_success else ["execution was not provided to the core evaluator"],
        },
        "semantic_alignment_score": {
            "overall": result.semantic_alignment_score,
            "description_to_plan": result.primitive_f1 if result.plan_validation_applicable else result.constraint_satisfaction_rate,
            "plan_to_code": 1.0 if result.plan_valid else (1.0 if not result.plan_validation_applicable else 0.0),
            "code_to_signal": result.constraint_satisfaction_rate if result.execution_success else 0.0,
        },
        "unresolved_semantics": _eval_unresolved_semantics(result.unresolved_semantics),
        "failure_reasons": result.failure_reasons,
        "notes": "Generated by faithts core evaluator.",
    }
    validation = validate_payload("eval_report", report, target_path=f"{result.case_id}:eval_report")
    if validation["status"] != "valid":
        raise ValueError("; ".join(validation["errors"]))
    return report
