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

from faithts_pipeline.hybrid_pipeline import PlanExecutionError, PlanValidationError, execute_plan, parse_plan
from faithts import aggregate_results, build_eval_report_from_result, evaluate_case


DEFAULT_CASE_FILE = REPO_ROOT / "data" / "controlled_gold_cases" / "core_mvp_sample_cases.json"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of cases in {path}")
    return payload


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
        parsed = parse_plan(
            plan,
            candidate_primitives=_candidate_primitives(plan),
            series_length=length,
        )
        return execute_plan(parsed, plan["input_description"], length), True, []
    except (PlanValidationError, PlanExecutionError) as exc:
        return None, False, [str(exc)]


def run_cases(path: Path = DEFAULT_CASE_FILE) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    results = []
    for case in _load_cases(path):
        semantic_spec = case["semantic_spec"]
        plan = case["plan"]
        series, execution_success, execution_errors = _execute_plan_payload(plan, int(semantic_spec["length"]))
        result = evaluate_case(
            semantic_spec=semantic_spec,
            plan=plan,
            series=series,
            execution_success=execution_success,
        )
        report = build_eval_report_from_result(result, semantic_spec["description"])
        if execution_errors:
            report["execution_status"]["errors"] = execution_errors
        results.append(result)
        outputs.append(
            {
                "case_id": result.case_id,
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
    return {"aggregate": aggregate_results(results), "cases": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the scope-locked FaithTS core MVP evaluator.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--json", action="store_true", help="Emit the full JSON payload.")
    args = parser.parse_args()

    payload = run_cases(args.cases)
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        aggregate = payload["aggregate"]
        crr = "n/a" if aggregate["crr"] is None else f"{float(aggregate['crr']):.3f}"
        hus = "n/a" if aggregate["hus"] is None else f"{float(aggregate['hus']):.3f}"
        print(
            "Core MVP: "
            f"{int(sum(1 for item in payload['cases'] if item['passed']))}/{len(payload['cases'])} passed, "
            f"CSR={aggregate['csr']:.3f}, "
            f"PrimitiveF1={aggregate['primitive_f1']:.3f}, "
            f"CRR={crr}, "
            f"HUS={hus}"
        )
        for item in payload["cases"]:
            status = "PASS" if item["passed"] else "FAIL"
            print(f"{status} {item['case_id']}")
            for failure in item["failure_reasons"]:
                print(f"  - {failure['code']}: {failure['message']}")
    return 0 if all(item["passed"] for item in payload["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
