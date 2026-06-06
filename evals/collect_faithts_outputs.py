from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.run_baseline_benchmark import _execute_plan_payload
from faithts.gold_adapter import load_gold_cases
from faithts.rejector import apply_static_rejector
from faithts.repair import repair_plan


CONTROLLED_CASES_DIR = REPO_ROOT / "data" / "controlled_gold_cases"
DEFAULT_CASE_FILES = [
    CONTROLLED_CASES_DIR / "single_claim_cases.json",
    CONTROLLED_CASES_DIR / "multi_claim_cases.json",
    CONTROLLED_CASES_DIR / "robustness_cases.json",
]
DEFAULT_WORKFLOW_DIRS = [
    REPO_ROOT / "evals" / "full_atomic_workflow_cases",
    REPO_ROOT / "evals" / "full_compositional_workflow_cases",
    REPO_ROOT / "evals" / "full_adversarial_workflow_cases",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe_series(values: Any) -> list[float | None]:
    safe: list[float | None] = []
    for value in values if isinstance(values, list) else []:
        numeric = float(value)
        safe.append(numeric if math.isfinite(numeric) else None)
    return safe


def _find_case_dir(case_id: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = root / case_id
        if (candidate / "plan.json").exists():
            return candidate
    return None


def collect_outputs(
    *,
    case_paths: list[Path] | None = None,
    workflow_dirs: list[Path] | None = None,
    output_dir: Path,
    apply_rejector: bool = False,
) -> dict[str, Any]:
    cases = load_gold_cases(case_paths or DEFAULT_CASE_FILES)
    roots = workflow_dirs or DEFAULT_WORKFLOW_DIRS
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        case_dir = _find_case_dir(case_id, roots)
        if case_dir is None:
            records.append({"case_id": case_id, "status": "missing", "reason": "no workflow directory"})
            continue
        plan_path = case_dir / "plan.json"
        series_path = case_dir / "generated_series.json"
        try:
            plan = _load_json(plan_path)
            if apply_rejector:
                plan = apply_static_rejector(
                    plan if isinstance(plan, dict) else {},
                    description=str(case["description"]),
                    length=int(case["length"]),
                    numeric_range=case.get("numeric_range"),
                )
            repair_result = repair_plan(plan if isinstance(plan, dict) else {}, series_length=int(case["length"]))
            plan = repair_result.plan
            payload = {"plan": plan}
            if series_path.exists() and not repair_result.changes:
                payload["generated_series"] = _json_safe_series(_load_json(series_path))
            else:
                series, ok, _ = _execute_plan_payload(plan, int(case["length"]))
                if ok and series is not None:
                    payload["generated_series"] = _json_safe_series(series.tolist())
            (output_dir / f"{case_id}.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            records.append(
                {
                    "case_id": case_id,
                    "status": "collected",
                    "source_dir": str(case_dir),
                    "has_generated_series": series_path.exists(),
                    "static_rejector_applied": bool(apply_rejector and payload["plan"].get("needs_library_extension")),
                    "repair_changes": repair_result.changes,
                }
            )
        except Exception as exc:
            records.append({"case_id": case_id, "status": "error", "source_dir": str(case_dir), "reason": str(exc)})

    summary = {
        "total_cases": len(records),
        "collected": sum(1 for item in records if item["status"] == "collected"),
        "missing": sum(1 for item in records if item["status"] == "missing"),
        "errors": sum(1 for item in records if item["status"] == "error"),
        "output_dir": str(output_dir),
        "records": records,
    }
    (output_dir / "collection_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect existing workflow outputs into the saved-plan format used by ours_full.")
    parser.add_argument("--cases", nargs="*", type=Path, default=None)
    parser.add_argument("--workflow-dirs", nargs="*", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply-static-rejector", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = collect_outputs(
        case_paths=args.cases,
        workflow_dirs=args.workflow_dirs,
        output_dir=args.output_dir,
        apply_rejector=args.apply_static_rejector,
    )
    if args.json:
        print(json.dumps(summary, indent=2, allow_nan=False))
    else:
        print(
            f"Collected {summary['collected']}/{summary['total_cases']} ours_full outputs "
            f"to {summary['output_dir']} (missing={summary['missing']}, errors={summary['errors']})"
        )
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
