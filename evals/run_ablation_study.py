from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import atomic_timeseries as ats
from evals.run_baseline_benchmark import _execute_plan_payload, oracle_no_rejection_plan_for_case
from evals.baselines.rule_parser import plan_for_case as rule_parser_plan_for_case
from faithts import aggregate_failure_taxonomy, aggregate_results, build_eval_report_from_result, evaluate_case
from faithts.gold_adapter import load_gold_cases, oracle_plan_from_gold_case, semantic_spec_from_gold_case
from faithts.repair import repair_plan


CONTROLLED_CASES_DIR = REPO_ROOT / "data" / "controlled_gold_cases"
DEFAULT_CASE_FILES = [
    CONTROLLED_CASES_DIR / "single_claim_cases.json",
    CONTROLLED_CASES_DIR / "multi_claim_cases.json",
    CONTROLLED_CASES_DIR / "robustness_cases.json",
]

ABLATIONS = [
    "full_oracle",
    "no_rejection",
    "rule_parser_no_repair",
    "rule_parser_with_repair",
    "repair_challenge_no_repair",
    "repair_challenge_with_repair",
    "no_schema_validation",
    "no_registry_grounding",
    "no_effect_ordering",
    "no_evidence_span",
]


def _primitive_steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in plan.get("steps", []) if isinstance(step, dict) and step.get("kind") == "primitive"]


def _execute_in_payload_order(plan: dict[str, Any], length: int) -> tuple[np.ndarray | None, bool, list[str]]:
    if plan.get("needs_library_extension") or plan.get("unresolved_semantics"):
        return None, False, []
    try:
        series = np.zeros(length, dtype=float)
        for step in plan.get("steps", []):
            if step.get("kind") != "primitive":
                continue
            series = getattr(ats, step["primitive"])(series, **step.get("parameters", {}))
        return series, True, []
    except Exception as exc:
        return None, False, [str(exc)]


def _oracle_series(case: dict[str, Any], plan: dict[str, Any]) -> tuple[np.ndarray | None, bool, list[str]]:
    return _execute_plan_payload(plan, int(case["length"]))


def _drop_schema_field(plan: dict[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(plan)
    mutated.pop("version", None)
    mutated.setdefault("notes", "Ablation: schema validation was bypassed before evaluation.")
    return mutated


def _hallucinate_first_primitive(plan: dict[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(plan)
    for step in mutated.get("steps", []):
        if isinstance(step, dict) and step.get("kind") == "primitive":
            step["primitive"] = "add_unregistered_pattern"
            step["semantic_role"] = "hallucinated primitive"
            step["notes"] = "Ablation: registry grounding removed."
            break
    return mutated


def _reverse_primitive_steps(plan: dict[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(plan)
    primitive_steps = _primitive_steps(mutated)
    if len(primitive_steps) <= 1:
        return mutated
    reversed_steps = list(reversed(primitive_steps))
    index = 0
    for position, step in enumerate(mutated["steps"]):
        if isinstance(step, dict) and step.get("kind") == "primitive":
            mutated["steps"][position] = reversed_steps[index]
            index += 1
    mutated.setdefault("assumptions", []).append("Ablation: effect/layer ordering disabled and primitive order reversed.")
    return mutated


def _remove_evidence_spans(plan: dict[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(plan)
    for step in mutated.get("steps", []):
        if isinstance(step, dict):
            step.pop("source_text", None)
    mutated.setdefault("notes", "Ablation: evidence/source spans removed.")
    return mutated


def _make_repair_challenge(plan: dict[str, Any]) -> dict[str, Any]:
    """Inject deterministic planner defects that static repair is expected to normalize."""

    mutated = deepcopy(plan)
    mutated.pop("version", None)
    mutated.pop("registry_validation", None)
    context = dict(mutated.get("global_context") or {})
    if "length" in context:
        context["length"] = int(context["length"]) + 1
    mutated["global_context"] = context
    for step in mutated.get("steps", []):
        if not isinstance(step, dict):
            continue
        if step.get("kind") == "primitive":
            step.pop("id", None)
            step.pop("source_text", None)
            step["effect_type"] = "overwrite" if step.get("effect_type") == "additive" else "additive"
            step["semantic_layer"] = "texture"
            params = dict(step.get("parameters") or {})
            if step.get("primitive") == "add_noise":
                params.pop("random_seed", None)
                params.pop("random_generator", None)
            step["parameters"] = params
            break
        if step.get("kind") == "unresolved":
            step.pop("source_text", None)
            step.pop("reason", None)
            break
    mutated.setdefault("notes", "Ablation: deterministic repair challenge with schema, length, effect, layer, and seed defects.")
    return mutated


def _plan_and_series_for_ablation(
    ablation: str,
    case: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray | None, bool, list[str], dict[str, Any] | None]:
    if ablation == "full_oracle":
        plan = oracle_plan_from_gold_case(case)
        series, ok, errors = _oracle_series(case, plan)
        return plan, series, ok, errors, None
    if ablation == "no_rejection":
        plan = oracle_no_rejection_plan_for_case(case)
        series, ok, errors = _oracle_series(case, plan)
        return plan, series, ok, errors, None
    if ablation in {"rule_parser_no_repair", "rule_parser_with_repair"}:
        plan = rule_parser_plan_for_case(case)
        repair_record = None
        if ablation == "rule_parser_with_repair":
            repaired = repair_plan(plan, series_length=int(case["length"]))
            plan = repaired.plan
            repair_record = {
                "iterations": repaired.iterations,
                "changes": repaired.changes,
                "schema_status": repaired.schema_status,
            }
        series, ok, errors = _oracle_series(case, plan)
        return plan, series, ok, errors, repair_record

    oracle_plan = oracle_plan_from_gold_case(case)
    if ablation in {"repair_challenge_no_repair", "repair_challenge_with_repair"}:
        plan = _make_repair_challenge(oracle_plan)
        repair_record = None
        if ablation == "repair_challenge_with_repair":
            repaired = repair_plan(plan, series_length=int(case["length"]))
            plan = repaired.plan
            repair_record = {
                "iterations": repaired.iterations,
                "changes": repaired.changes,
                "schema_status": repaired.schema_status,
            }
        series, ok, errors = _oracle_series(case, plan)
        return plan, series, ok, errors, repair_record
    if ablation == "no_schema_validation":
        plan = _drop_schema_field(oracle_plan)
        series, ok, errors = _oracle_series(case, oracle_plan)
        return plan, series, ok, errors, None
    if ablation == "no_registry_grounding":
        plan = _hallucinate_first_primitive(oracle_plan)
        return plan, None, False, ["Ablation generated an unregistered primitive and did not execute it."], None
    if ablation == "no_effect_ordering":
        plan = _reverse_primitive_steps(oracle_plan)
        series, ok, errors = _execute_in_payload_order(plan, int(case["length"]))
        return plan, series, ok, errors, None
    if ablation == "no_evidence_span":
        plan = _remove_evidence_spans(oracle_plan)
        series, ok, errors = _oracle_series(case, oracle_plan)
        return plan, series, ok, errors, None
    raise ValueError(f"Unknown ablation `{ablation}`.")


def run_ablation(
    ablation: str,
    *,
    paths: list[Path] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation `{ablation}`. Choose from {ABLATIONS}.")
    cases = load_gold_cases(paths or DEFAULT_CASE_FILES)
    if category is not None:
        cases = [case for case in cases if case.get("category") == category]

    outputs: list[dict[str, Any]] = []
    results = []
    repair_records: list[dict[str, Any]] = []
    for case in cases:
        spec = semantic_spec_from_gold_case(case)
        plan, series, execution_success, execution_errors, repair_record = _plan_and_series_for_ablation(ablation, case)
        if repair_record is not None:
            repair_records.append({"case_id": case["case_id"], **repair_record})
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
        "ablation": ablation,
        "aggregate": aggregate_results(results),
        "by_category": by_category,
        "failure_taxonomy": aggregate_failure_taxonomy(outputs),
        "repair_summary": {
            "cases": len(repair_records),
            "changed_cases": sum(1 for item in repair_records if item["changes"]),
            "total_iterations": sum(int(item["iterations"]) for item in repair_records),
            "total_changes": sum(len(item["changes"]) for item in repair_records),
        },
        "cases": outputs,
    }


def run_ablations(
    *,
    ablations: list[str],
    paths: list[Path] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    return {
        "version": "1.0",
        "ablations": ablations,
        "category": category,
        "results": [run_ablation(ablation, paths=paths, category=category) for ablation in ablations],
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Ablation Study",
        "",
        "| Variant | Cases | Pass | CSR | Primitive F1 | Window IoU | ParamErr | CRR | FRR | HUS | Repair Iter. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        aggregate = result["aggregate"]
        repair = result["repair_summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result["ablation"]),
                    str(int(aggregate["cases"])),
                    _fmt(aggregate["pass_rate"]),
                    _fmt(aggregate["csr"]),
                    _fmt(aggregate["primitive_f1"]),
                    _fmt(aggregate["window_iou"]),
                    _fmt(aggregate["parameter_error"]),
                    _fmt(aggregate["crr"]),
                    _fmt(aggregate["frr"]),
                    _fmt(aggregate["hus"]),
                    str(int(repair["total_iterations"])),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failure Taxonomy", ""])
    for result in payload["results"]:
        counts = result.get("failure_taxonomy", {}).get("counts", {})
        summary = ", ".join(f"{key}={value}" for key, value in counts.items()) if counts else "none"
        lines.append(f"- {result['ablation']}: {summary}")
    lines.append("")
    return "\n".join(lines)


def _print_summary(payload: dict[str, Any]) -> None:
    for result in payload["results"]:
        aggregate = result["aggregate"]
        print(
            f"{result['ablation']}: cases={int(aggregate['cases'])}, "
            f"pass={_fmt(aggregate['pass_rate'])}, CSR={_fmt(aggregate['csr'])}, "
            f"HUS={_fmt(aggregate['hus'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline FaithTS ablations through the shared evaluator.")
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=ABLATIONS)
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial"], default=None)
    parser.add_argument("--cases", nargs="*", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_ablations(ablations=args.ablations, paths=args.cases, category=args.category)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
