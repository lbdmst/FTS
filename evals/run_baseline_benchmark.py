from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import atomic_timeseries as ats
from evals.baselines.llm_direct import (
    build_prompt,
    invalid_direct_output_plan,
    load_llm_output,
    parse_direct_array_output,
    parse_direct_code_output,
)
from evals.baselines.rule_parser import plan_for_case as rule_parser_plan_for_case
from faithts_pipeline.hybrid_pipeline import PlanExecutionError, PlanValidationError, execute_plan, parse_plan
from faithts import (
    aggregate_failure_taxonomy,
    aggregate_results,
    build_eval_report_from_result,
    evaluate_case,
    evaluate_direct_signal_case,
)
from faithts.gold_adapter import load_gold_cases, oracle_plan_from_gold_case, semantic_spec_from_gold_case
from faithts.repair import repair_plan


CONTROLLED_CASES_DIR = REPO_ROOT / "data" / "controlled_gold_cases"
DEFAULT_CASE_FILES = [
    CONTROLLED_CASES_DIR / "single_claim_cases.json",
    CONTROLLED_CASES_DIR / "multi_claim_cases.json",
    CONTROLLED_CASES_DIR / "robustness_cases.json",
]


def _candidate_primitives(plan: dict[str, Any]) -> list[str]:
    primitives = []
    for step in plan.get("steps", []):
        primitive = step.get("primitive")
        if primitive and primitive not in primitives:
            primitives.append(str(primitive))
    return primitives


def _is_rejection_plan(plan: dict[str, Any]) -> bool:
    return bool(plan.get("needs_library_extension")) or bool(plan.get("unresolved_semantics"))


def _execute_plan_payload(plan: dict[str, Any], length: int) -> tuple[np.ndarray | None, bool, list[str]]:
    if _is_rejection_plan(plan):
        return None, False, []
    try:
        parsed = parse_plan(plan, candidate_primitives=_candidate_primitives(plan), series_length=length)
        return execute_plan(parsed, plan["input_description"], length), True, []
    except (PlanValidationError, PlanExecutionError) as exc:
        try:
            series = np.zeros(length, dtype=float)
            for step in plan.get("steps", []):
                if step.get("kind") != "primitive":
                    continue
                series = getattr(ats, step["primitive"])(series, **step.get("parameters", {}))
            return series, True, [f"hybrid executor fallback used: {exc}"]
        except Exception as fallback_exc:
            return None, False, [str(exc), f"fallback execution failed: {fallback_exc}"]


def oracle_no_rejection_plan_for_case(case: dict[str, Any]) -> dict[str, Any]:
    plan = oracle_plan_from_gold_case(case)
    if not _is_rejection_plan(plan):
        return plan
    description = case["description"]
    return {
        "version": "1.0",
        "input_description": description,
        "registry_validation": {
            "status": "valid",
            "schema_path": "schemas/primitive.schema.json",
            "target_path": "atomic_timeseries/registry.json",
            "errors": [],
        },
        "global_context": {
            "length": int(case["length"]),
            "numeric_range": case.get("numeric_range") or {"min": -1000.0, "max": 1000.0},
            "dt": None,
            "seed": 7,
        },
        "steps": [
            {
                "id": "step_1",
                "kind": "primitive",
                "primitive": "add_flat",
                "parameters": {"start_timestep": 0, "end_timestep": int(case["length"]) - 1, "target_value": 0.0},
                "semantic_role": "forced fallback generation",
                "effect_type": "overwrite",
                "confidence": 0.2,
                "source_text": description,
                "needs_library_extension": False,
            }
        ],
        "unresolved_semantics": [],
        "needs_library_extension": False,
        "assumptions": ["Unsupported semantics were forced into a flat fallback instead of rejected."],
    }


PLAN_BASELINES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "oracle": oracle_plan_from_gold_case,
    "oracle_program": oracle_plan_from_gold_case,
    "oracle_no_rejection": oracle_no_rejection_plan_for_case,
    "rule_parser": rule_parser_plan_for_case,
}

LLM_OUTPUT_BASELINES = {"llm_direct_array", "llm_direct_code", "llm_strong_prompt", "llm_strong_prompt_code"}
SAVED_PLAN_OUTPUT_BASELINES = {"ours_full"}
EXTERNAL_SERIES_BASELINES = {"verbalts"}
BASELINES = sorted(set(PLAN_BASELINES) | LLM_OUTPUT_BASELINES | SAVED_PLAN_OUTPUT_BASELINES | EXTERNAL_SERIES_BASELINES)


def _evaluate_llm_output(
    baseline: str,
    case: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray | None, bool, list[str]]:
    errors: list[str] = []
    series: np.ndarray | None = None
    try:
        text = load_llm_output(output_dir, str(case["case_id"]))
        if baseline == "llm_direct_array":
            series = parse_direct_array_output(text, int(case["length"]))
        elif baseline in {"llm_direct_code", "llm_strong_prompt", "llm_strong_prompt_code"}:
            series = parse_direct_code_output(text, int(case["length"]))
        else:
            raise ValueError(f"Unsupported LLM output baseline: {baseline}")
    except Exception as exc:
        errors.append(str(exc))
    return invalid_direct_output_plan(case, baseline, errors), series, series is not None, errors


def _load_external_series_output(output_dir: Path, case_id: str, length: int) -> np.ndarray:
    json_path = output_dir / f"{case_id}.json"
    txt_path = output_dir / f"{case_id}.txt"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        raw_series = payload.get("generated_series", payload.get("series", payload.get("values"))) if isinstance(payload, dict) else payload
    elif txt_path.exists():
        raw_series = parse_direct_array_output(txt_path.read_text(encoding="utf-8"), length)
    else:
        raise FileNotFoundError(f"Missing external baseline output file: {json_path} or {txt_path}")
    if raw_series is None:
        raise ValueError("External baseline output did not contain generated_series, series, values, or a parseable array.")
    series = np.asarray(raw_series, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"External generated series must be one-dimensional, got shape {series.shape}.")
    if len(series) != length:
        raise ValueError(f"External generated series length {len(series)} does not match expected length {length}.")
    return series


def _evaluate_external_series_output(
    baseline: str,
    case: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray | None, bool, list[str]]:
    errors: list[str] = []
    series: np.ndarray | None = None
    try:
        series = _load_external_series_output(output_dir, str(case["case_id"]), int(case["length"]))
    except Exception as exc:
        errors.append(str(exc))
    return invalid_direct_output_plan(case, baseline, errors), series, series is not None, errors


def _load_saved_plan_output(output_dir: Path, case_id: str) -> dict[str, Any]:
    json_path = output_dir / f"{case_id}.json"
    txt_path = output_dir / f"{case_id}.txt"
    if json_path.exists():
        text = json_path.read_text(encoding="utf-8")
    elif txt_path.exists():
        text = txt_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"Missing saved plan output file: {json_path} or {txt_path}")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Saved plan output must be a JSON object.")
    return payload


def _evaluate_saved_plan_output(
    baseline: str,
    case: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray | None, bool, list[str]]:
    errors: list[str] = []
    series: np.ndarray | None = None
    execution_success = False
    try:
        payload = _load_saved_plan_output(output_dir, str(case["case_id"]))
        plan = payload.get("plan", payload.get("structured_plan", payload))
        if not isinstance(plan, dict):
            raise ValueError("Saved output must contain a plan object.")
        repaired = repair_plan(plan, series_length=int(case["length"]))
        if repaired.changes:
            plan = repaired.plan
            errors.append(
                "static repair applied: "
                + "; ".join(repaired.changes)
                + f" (iterations={repaired.iterations}, schema_status={repaired.schema_status})"
            )
        if _is_rejection_plan(plan):
            return plan, None, False, []
        raw_series = payload.get("series", payload.get("generated_series"))
        if raw_series is not None and not repaired.changes:
            series = np.asarray(raw_series, dtype=float)
            if series.ndim != 1:
                raise ValueError("Saved generated series must be one-dimensional.")
            if len(series) != int(case["length"]):
                raise ValueError(
                    f"Saved generated series length {len(series)} does not match expected length {int(case['length'])}."
                )
            execution_success = True
        else:
            series, execution_success, execution_errors = _execute_plan_payload(plan, int(case["length"]))
            errors.extend(execution_errors)
        return plan, series, execution_success, errors
    except Exception as exc:
        errors.append(str(exc))
        return invalid_direct_output_plan(case, baseline, errors), None, False, errors


def run_baseline(
    baseline: str,
    paths: list[Path] | None = None,
    category: str | None = None,
    output_dir: Path | None = None,
    repair: bool = False,
) -> dict[str, Any]:
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline `{baseline}`. Choose from {BASELINES}.")
    if baseline in (LLM_OUTPUT_BASELINES | SAVED_PLAN_OUTPUT_BASELINES | EXTERNAL_SERIES_BASELINES) and output_dir is None:
        raise ValueError(f"Baseline `{baseline}` requires --output-dir with one <case_id>.json or <case_id>.txt file per case.")
    plan_for_case = PLAN_BASELINES.get(baseline)
    cases = load_gold_cases(paths or DEFAULT_CASE_FILES)
    if category is not None:
        cases = [case for case in cases if case.get("category") == category]

    outputs: list[dict[str, Any]] = []
    results = []
    repair_records: list[dict[str, Any]] = []
    for case in cases:
        spec = semantic_spec_from_gold_case(case)
        if baseline in LLM_OUTPUT_BASELINES:
            assert output_dir is not None
            plan, series, execution_success, execution_errors = _evaluate_llm_output(baseline, case, output_dir)
        elif baseline in SAVED_PLAN_OUTPUT_BASELINES:
            assert output_dir is not None
            plan, series, execution_success, execution_errors = _evaluate_saved_plan_output(baseline, case, output_dir)
        elif baseline in EXTERNAL_SERIES_BASELINES:
            assert output_dir is not None
            plan, series, execution_success, execution_errors = _evaluate_external_series_output(baseline, case, output_dir)
        else:
            assert plan_for_case is not None
            plan = plan_for_case(case)
            if repair:
                repaired = repair_plan(plan, series_length=int(spec["length"]))
                plan = repaired.plan
                repair_records.append(
                    {
                        "case_id": case["case_id"],
                        "iterations": repaired.iterations,
                        "changes": repaired.changes,
                        "schema_status": repaired.schema_status,
                    }
                )
            series, execution_success, execution_errors = _execute_plan_payload(plan, int(spec["length"]))
        if baseline in (LLM_OUTPUT_BASELINES | EXTERNAL_SERIES_BASELINES):
            result = evaluate_direct_signal_case(
                semantic_spec=spec,
                series=series,
                execution_success=execution_success,
                signal_checks=case.get("expected_signal_checks"),
            )
        else:
            result = evaluate_case(
                semantic_spec=spec,
                plan=plan,
                series=series,
                execution_success=execution_success,
            )
        report = build_eval_report_from_result(result, spec["description"])
        if execution_errors:
            report["execution_status"]["errors"] = execution_errors
        results.append(result)
        outputs.append(
            {
                "case_id": result.case_id,
                "category": case.get("category", "unknown"),
                "passed": result.passed,
                "metrics": {
                    "csr": result.constraint_satisfaction_rate,
                    "primitive_f1": result.primitive_f1,
                    "window_iou": result.window_iou,
                    "parameter_error": result.parameter_error,
                    "correct_rejection": result.correct_rejection,
                    "false_rejection": result.false_rejection,
                    "hallucinated_unsupported": result.hallucinated_unsupported,
                },
                "failure_reasons": result.failure_reasons,
                "eval_report": report,
            }
        )

    by_category: dict[str, Any] = {}
    for item in sorted({output["category"] for output in outputs}):
        subset = [result for result, output in zip(results, outputs) if output["category"] == item]
        by_category[item] = aggregate_results(subset)
    return {
        "baseline": baseline,
        "repair_enabled": bool(repair),
        "repair_summary": {
            "cases": len(repair_records),
            "changed_cases": sum(1 for item in repair_records if item["changes"]),
            "total_iterations": sum(int(item["iterations"]) for item in repair_records),
            "total_changes": sum(len(item["changes"]) for item in repair_records),
        },
        "aggregate": aggregate_results(results),
        "by_category": by_category,
        "failure_taxonomy": aggregate_failure_taxonomy(outputs),
        "cases": outputs,
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _print_summary(payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    passed = int(sum(1 for item in payload["cases"] if item["passed"]))
    total = len(payload["cases"])
    print(
        f"{payload['baseline']}: "
        f"{passed}/{total} passed, "
        f"CSR={aggregate['csr']:.3f}, "
        f"PrimitiveF1={aggregate['primitive_f1']:.3f}, "
        f"WindowIoU={aggregate['window_iou']:.3f}, "
        f"ParamErr={aggregate['parameter_error']:.3f}, "
        f"CRR={_fmt(aggregate['crr'])}, "
        f"HUS={_fmt(aggregate['hus'])}"
    )
    if payload.get("repair_enabled"):
        summary = payload["repair_summary"]
        print(
            f"  repair: changed_cases={summary['changed_cases']}, "
            f"iterations={summary['total_iterations']}, changes={summary['total_changes']}"
        )
    for category, metrics in payload["by_category"].items():
        print(
            f"  {category}: cases={int(metrics['cases'])}, pass={metrics['pass_rate']:.3f}, "
            f"CSR={metrics['csr']:.3f}, F1={metrics['primitive_f1']:.3f}, HUS={_fmt(metrics['hus'])}"
        )
    taxonomy = payload.get("failure_taxonomy", {})
    counts = taxonomy.get("counts", {})
    if counts:
        top = ", ".join(f"{label}={count}" for label, count in list(counts.items())[:6])
        print(f"  failures: {top}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run non-LLM FaithTS baselines through the shared evaluator.")
    parser.add_argument("--baseline", choices=BASELINES, default="rule_parser")
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial"], default=None)
    parser.add_argument("--cases", nargs="*", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory containing <case_id>.txt LLM outputs.")
    parser.add_argument("--repair-plan", action="store_true", help="Run static verifier-guided plan repair before execution.")
    parser.add_argument("--write-prompts", type=Path, default=None, help="Write prompts for the selected baseline and cases, then exit.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.write_prompts is not None:
        cases = load_gold_cases(args.cases or DEFAULT_CASE_FILES)
        if args.category is not None:
            cases = [case for case in cases if case.get("category") == args.category]
        if args.baseline not in LLM_OUTPUT_BASELINES:
            raise SystemExit("--write-prompts is only supported for LLM output baselines.")
        args.write_prompts.mkdir(parents=True, exist_ok=True)
        for case in cases:
            (args.write_prompts / f"{case['case_id']}.txt").write_text(build_prompt(case, args.baseline), encoding="utf-8")
        print(f"Wrote {len(cases)} prompts to {args.write_prompts}")
        return 0

    payload = run_baseline(
        args.baseline,
        paths=args.cases,
        category=args.category,
        output_dir=args.output_dir,
        repair=args.repair_plan,
    )
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
