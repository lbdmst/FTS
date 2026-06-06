from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.run_baseline_benchmark import (
    BASELINES,
    EXTERNAL_SERIES_BASELINES,
    LLM_OUTPUT_BASELINES,
    SAVED_PLAN_OUTPUT_BASELINES,
    run_baseline,
)
from faithts.gold_adapter import load_gold_cases


DEFAULT_METHODS = ["oracle_program", "rule_parser", "llm_direct_array", "llm_direct_code", "llm_strong_prompt"]


def _method_output_dir(output_root: Path | None, method: str) -> Path | None:
    if output_root is None:
        return None
    direct = output_root / method
    return direct if direct.exists() else output_root


def _skipped_payload(method: str, reason: str) -> dict[str, Any]:
    return {
        "method": method,
        "status": "skipped",
        "skip_reason": reason,
        "aggregate": {
            "cases": 0.0,
            "pass_rate": 0.0,
            "csr": 0.0,
            "primitive_f1": 0.0,
            "window_iou": 0.0,
            "parameter_error": 0.0,
            "crr": None,
            "frr": None,
            "hus": None,
        },
        "by_category": {},
        "failure_taxonomy": {
            "total_cases": 0,
            "failed_cases": 0,
            "failure_case_rate": 0.0,
            "counts": {},
            "rates_among_all_cases": {},
            "rates_among_failed_cases": {},
            "case_labels": {},
        },
        "cases": [],
    }


def _has_case_output_files(output_dir: Path, paths: list[Path] | None, category: str | None) -> bool:
    cases = load_gold_cases(paths or [
        Path(__file__).resolve().parents[1] / "data" / "controlled_gold_cases" / "single_claim_cases.json",
        Path(__file__).resolve().parents[1] / "data" / "controlled_gold_cases" / "multi_claim_cases.json",
        Path(__file__).resolve().parents[1] / "data" / "controlled_gold_cases" / "robustness_cases.json",
    ])
    if category is not None:
        cases = [case for case in cases if case.get("category") == category]
    return any(
        (output_dir / f"{case['case_id']}.txt").exists()
        or (output_dir / f"{case['case_id']}.json").exists()
        for case in cases
    )


def run_methods(
    *,
    methods: list[str],
    paths: list[Path] | None = None,
    category: str | None = None,
    output_root: Path | None = None,
    repair_rule_parser: bool = False,
) -> dict[str, Any]:
    method_payloads: list[dict[str, Any]] = []
    for method in methods:
        if method not in BASELINES:
            raise ValueError(f"Unknown method `{method}`. Choose from {BASELINES}.")
        method_specific_dir = output_root / method if output_root is not None else None
        output_dir = _method_output_dir(output_root, method)
        requires_saved_outputs = method in (LLM_OUTPUT_BASELINES | SAVED_PLAN_OUTPUT_BASELINES | EXTERNAL_SERIES_BASELINES)
        if requires_saved_outputs and output_root is not None and len(methods) > 1 and not method_specific_dir.exists():
            method_payloads.append(_skipped_payload(method, "missing_method_output_dir"))
            continue
        if method in LLM_OUTPUT_BASELINES and output_dir is None:
            method_payloads.append(_skipped_payload(method, "missing_output_dir_or_api_key"))
            continue
        if method in LLM_OUTPUT_BASELINES and output_dir is not None and not _has_case_output_files(output_dir, paths, category):
            method_payloads.append(_skipped_payload(method, "missing_method_output_files"))
            continue
        if method in SAVED_PLAN_OUTPUT_BASELINES and output_dir is None:
            method_payloads.append(_skipped_payload(method, "missing_output_dir"))
            continue
        if method in EXTERNAL_SERIES_BASELINES and output_dir is None:
            method_payloads.append(_skipped_payload(method, "missing_output_dir"))
            continue
        if method in EXTERNAL_SERIES_BASELINES and output_dir is not None and not _has_case_output_files(output_dir, paths, category):
            method_payloads.append(_skipped_payload(method, "missing_method_output_files"))
            continue
        try:
            payload = run_baseline(
                method,
                paths=paths,
                category=category,
                output_dir=output_dir,
                repair=bool(repair_rule_parser and method == "rule_parser"),
            )
        except FileNotFoundError as exc:
            if method in LLM_OUTPUT_BASELINES:
                method_payloads.append(_skipped_payload(method, f"missing_output: {exc}"))
                continue
            raise
        payload["method"] = method
        payload["status"] = "completed"
        method_payloads.append(payload)
    return {
        "version": "1.0",
        "methods": methods,
        "category": category,
        "results": method_payloads,
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Baseline Results",
        "",
        "| Method | Status | Cases | Pass | CSR | Primitive F1 | Window IoU | ParamErr | CRR | FRR | HUS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        aggregate = result["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result["method"]),
                    str(result["status"]),
                    str(int(aggregate["cases"])),
                    _fmt(aggregate["pass_rate"]),
                    _fmt(aggregate["csr"]),
                    _fmt(aggregate["primitive_f1"]),
                    _fmt(aggregate["window_iou"]),
                    _fmt(aggregate["parameter_error"]),
                    _fmt(aggregate["crr"]),
                    _fmt(aggregate["frr"]),
                    _fmt(aggregate["hus"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failure Taxonomy", ""])
    for result in payload["results"]:
        counts = result.get("failure_taxonomy", {}).get("counts", {})
        summary = ", ".join(f"{key}={value}" for key, value in counts.items()) if counts else "none"
        reason = f" ({result['skip_reason']})" if result.get("status") == "skipped" else ""
        lines.append(f"- {result['method']}: {summary}{reason}")
    lines.append("")
    return "\n".join(lines)


def _print_summary(payload: dict[str, Any]) -> None:
    for result in payload["results"]:
        aggregate = result["aggregate"]
        reason = f", reason={result['skip_reason']}" if result.get("status") == "skipped" else ""
        print(
            f"{result['method']}: status={result['status']}{reason}, "
            f"cases={int(aggregate['cases'])}, pass={_fmt(aggregate['pass_rate'])}, "
            f"CSR={_fmt(aggregate['csr'])}, HUS={_fmt(aggregate['hus'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple FaithTS baselines through the shared evaluator.")
    parser.add_argument("--methods", nargs="+", choices=BASELINES, default=DEFAULT_METHODS)
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial"], default=None)
    parser.add_argument("--cases", nargs="*", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None, help="Root containing saved LLM outputs, optionally grouped by method.")
    parser.add_argument("--repair-rule-parser", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_methods(
        methods=args.methods,
        paths=args.cases,
        category=args.category,
        output_root=args.output_root,
        repair_rule_parser=args.repair_rule_parser,
    )
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
