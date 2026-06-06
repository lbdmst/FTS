from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import atomic_timeseries as ats
import coverage_audit


CONTROLLED_CASES_DIR = REPO_ROOT / "data" / "controlled_gold_cases"
DEFAULT_CASE_FILES = [
    CONTROLLED_CASES_DIR / "single_claim_cases.json",
    CONTROLLED_CASES_DIR / "multi_claim_cases.json",
    CONTROLLED_CASES_DIR / "robustness_cases.json",
]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    passed: bool
    checks: list[dict[str, Any]]


def _load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload in {path}")
        cases.extend(payload)
    return cases


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _run_retrieval_checks(case: dict[str, Any], catalog: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    row = coverage_audit.evaluate_caption(case["description"], catalog)
    expected = case.get("expected_candidate_primitives", [])
    for primitive in expected:
        _record(
            checks,
            f"retrieve:{primitive}",
            primitive in row["candidate_primitives"],
            f"retrieved={row['candidate_primitives']}",
        )

    assertions = case.get("assertions", {})
    for primitive in assertions.get("must_not_retrieve", []):
        _record(
            checks,
            f"must_not_retrieve:{primitive}",
            primitive not in row["candidate_primitives"],
            f"retrieved={row['candidate_primitives']}",
        )

    primary = row["candidate_primitives"][:1]
    for primitive in assertions.get("must_not_prefer_as_primary", []):
        _record(
            checks,
            f"must_not_prefer_as_primary:{primitive}",
            primitive not in primary,
            f"primary={primary}",
        )

    expected_status = assertions.get("expected_retrieval_status")
    if expected_status is not None:
        _record(
            checks,
            "retrieval_status",
            row["retrieval_status"] == expected_status,
            f"actual={row['retrieval_status']}",
        )

    expected_gap = assertions.get("must_set_needs_library_extension")
    if expected_gap is not None:
        actual_gap = bool(case["expected_plan_outline"].get("needs_library_extension"))
        _record(checks, "needs_library_extension_flag", actual_gap == expected_gap, f"actual={actual_gap}")

    expected_unresolved = set(assertions.get("must_include_unresolved_semantics", []))
    if expected_unresolved:
        actual_unresolved = {
            item["semantic"] for item in case["expected_plan_outline"].get("unresolved_semantics", [])
        }
        _record(
            checks,
            "unresolved_semantics",
            expected_unresolved.issubset(actual_unresolved),
            f"actual={sorted(actual_unresolved)}",
        )


def _execute_steps(case: dict[str, Any]) -> np.ndarray | None:
    outline = case["expected_plan_outline"]
    if outline.get("needs_library_extension") or outline.get("unresolved_semantics"):
        return None
    series = np.zeros(case["length"], dtype=float)
    for step in outline.get("steps", []):
        fn = getattr(ats, step["primitive"])
        series = fn(series, **step["parameters"])
    return series


def _segment(series: np.ndarray, start: int, end: int) -> np.ndarray:
    return series[start : end + 1]


def _run_signal_checks(case: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    series = _execute_steps(case)
    if series is None:
        _record(checks, "signal_execution_skipped", True, "case intentionally unresolved")
        return

    signal_checks = case.get("expected_signal_checks", {})
    _record(checks, "signal_length", len(series) == case["length"], f"actual={len(series)}")

    for item in signal_checks.get("point_values", []):
        actual = float(series[item["timestep"]])
        passed = np.isclose(actual, item["value"], atol=item.get("tolerance", 1e-6), rtol=0.0)
        _record(checks, f"point_value:{item['timestep']}", passed, f"actual={actual}")

    for item in signal_checks.get("constant_segments", []):
        seg = _segment(series, item["start"], item["end"])
        passed = np.allclose(seg, item["value"], atol=item.get("tolerance", 1e-6), rtol=0.0)
        _record(checks, f"constant_segment:{item['start']}-{item['end']}", bool(passed), f"value={item['value']}")

    for item in signal_checks.get("unchanged_windows", []):
        seg = _segment(series, item["start"], item["end"])
        passed = np.allclose(seg, item["value"], atol=item.get("tolerance", 1e-6), rtol=0.0)
        _record(checks, f"unchanged_window:{item['start']}-{item['end']}", bool(passed), f"value={item['value']}")

    for item in signal_checks.get("monotone_windows", []):
        seg = _segment(series, item["start"], item["end"])
        diffs = np.diff(seg)
        if item["direction"] == "nondecreasing":
            passed = bool(np.all(diffs >= -1e-8))
        elif item["direction"] == "nonincreasing":
            passed = bool(np.all(diffs <= 1e-8))
        else:
            raise ValueError(f"Unsupported monotone direction: {item['direction']}")
        _record(checks, f"monotone_window:{item['start']}-{item['end']}", passed, f"direction={item['direction']}")

    bounds = signal_checks.get("value_bounds")
    if bounds is not None:
        actual_min = float(np.nanmin(series))
        actual_max = float(np.nanmax(series))
        passed = actual_min >= bounds["min"] and actual_max <= bounds["max"]
        _record(checks, "value_bounds", passed, f"min={actual_min}, max={actual_max}")

    argmax_in = signal_checks.get("argmax_in")
    if argmax_in is not None:
        actual = int(np.nanargmax(series))
        passed = argmax_in[0] <= actual <= argmax_in[1]
        _record(checks, "argmax_in", passed, f"actual={actual}")

    argmin_in = signal_checks.get("argmin_in")
    if argmin_in is not None:
        actual = int(np.nanargmin(series))
        passed = argmin_in[0] <= actual <= argmin_in[1]
        _record(checks, "argmin_in", passed, f"actual={actual}")

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
        _record(checks, "variance_comparison", passed, f"reference={left_var}, target={right_var}")

    deterministic_seed = signal_checks.get("deterministic_seed")
    if deterministic_seed is not None:
        rerun = _execute_steps(case)
        passed = bool(np.allclose(series, rerun, equal_nan=True))
        _record(checks, "deterministic_seed", passed, f"seed={deterministic_seed}")

    for item in signal_checks.get("nonconstant_windows", []):
        seg = _segment(series, item["start"], item["end"])
        passed = not np.allclose(seg, seg[0], atol=1e-8, rtol=0.0)
        _record(checks, f"nonconstant_window:{item['start']}-{item['end']}", bool(passed), "segment should vary")


def run_cases(paths: list[Path], category: str | None = None) -> list[CaseResult]:
    catalog = coverage_audit.build_catalog()
    cases = _load_cases(paths)
    if category is not None:
        cases = [case for case in cases if case["category"] == category]

    results: list[CaseResult] = []
    for case in cases:
        checks: list[dict[str, Any]] = []
        _run_retrieval_checks(case, catalog, checks)
        _run_signal_checks(case, checks)
        results.append(
            CaseResult(
                case_id=case["case_id"],
                category=case["category"],
                passed=bool(all(item["passed"] for item in checks)),
                checks=checks,
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled gold-case retrieval and signal checks.")
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial"], default=None)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    args = parser.parse_args()

    results = run_cases(DEFAULT_CASE_FILES, category=args.category)
    failures = [result for result in results if not result.passed]

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "case_id": result.case_id,
                        "category": result.category,
                        "passed": result.passed,
                        "checks": result.checks,
                    }
                    for result in results
                ],
                indent=2,
            )
        )
    else:
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.case_id} [{result.category}]")
            if not result.passed:
                for item in result.checks:
                    if not item["passed"]:
                        print(f"  - {item['name']}: {item['detail']}")
        print(f"Summary: {len(results) - len(failures)}/{len(results)} passed")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
