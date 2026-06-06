from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import atomic_timeseries as ats
from faithts_pipeline.hybrid_pipeline import PlanExecutionError, PlanValidationError, execute_plan, parse_plan
from faithts import aggregate_failure_taxonomy, aggregate_results, build_eval_report_from_result, evaluate_case
from faithts.gold_adapter import (
    is_core_evaluable,
    load_gold_cases,
    oracle_plan_from_gold_case,
    semantic_spec_from_gold_case,
)


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


def _execute_oracle_plan(plan: dict[str, Any], length: int) -> tuple[np.ndarray | None, bool, list[str]]:
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


def run_benchmark(paths: list[Path] | None = None, category: str | None = None) -> dict[str, Any]:
    paths = paths or DEFAULT_CASE_FILES
    cases = load_gold_cases(paths)
    if category is not None:
        cases = [case for case in cases if case.get("category") == category]

    outputs: list[dict[str, Any]] = []
    results = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        if not is_core_evaluable(case):
            skipped.append({"case_id": case["case_id"], "reason": "no oracle steps and no unresolved semantics"})
            continue
        spec = semantic_spec_from_gold_case(case)
        plan = oracle_plan_from_gold_case(case)
        series, execution_success, execution_errors = _execute_oracle_plan(plan, int(spec["length"]))
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
        "aggregate": aggregate_results(results),
        "by_category": by_category,
        "failure_taxonomy": aggregate_failure_taxonomy(outputs),
        "cases": outputs,
        "skipped": skipped,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    passed = int(sum(1 for item in payload["cases"] if item["passed"]))
    total = len(payload["cases"])

    def _fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    print(
        "Core benchmark oracle: "
        f"{passed}/{total} passed, "
        f"CSR={aggregate['csr']:.3f}, "
        f"PrimitiveF1={aggregate['primitive_f1']:.3f}, "
        f"CRR={_fmt(aggregate['crr'])}, "
        f"HUS={_fmt(aggregate['hus'])}, "
        f"skipped={len(payload['skipped'])}"
    )
    for category, metrics in payload["by_category"].items():
        print(
            f"  {category}: cases={int(metrics['cases'])}, pass={metrics['pass_rate']:.3f}, "
            f"CSR={metrics['csr']:.3f}, HUS={_fmt(metrics['hus'])}"
        )
    failures = [item for item in payload["cases"] if not item["passed"]]
    for item in failures[:20]:
        print(f"FAIL {item['case_id']} [{item['category']}]")
        for failure in item["failure_reasons"]:
            print(f"  - {failure['code']}: {failure['message']}")
    if len(failures) > 20:
        print(f"... {len(failures) - 20} more failures omitted")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the core FaithTS benchmark over converted gold cases.")
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial"], default=None)
    parser.add_argument("--cases", nargs="*", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_benchmark(paths=args.cases, category=args.category)
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        _print_summary(payload)
    return 0 if all(item["passed"] for item in payload["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
