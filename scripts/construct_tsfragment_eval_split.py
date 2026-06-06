from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.timeseries_consistency import evaluate_description_series_consistency
from faithts.gold_adapter import load_gold_cases


DEFAULT_INPUT_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_native_case_inputs"
DEFAULT_RECORD_ROOT = REPO_ROOT / "baselines" / "VerbalTS" / "datasets"
DEFAULT_MAIN_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
DEFAULT_CASE_OUTPUT = REPO_ROOT / "data" / "tsfragment_eval" / "random_audit_cases.json"
DEFAULT_REPORT_JSON = REPO_ROOT / "evals" / "reports" / "t2s_random_audit_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "evals" / "reports" / "t2s_random_audit_report.md"

DATASETS = ["ETTh1", "electricity", "exchangerate", "traffic"]
LENGTHS = [24, 48, 96]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _series_from_ot(raw: Any) -> list[float]:
    parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    series = np.asarray(parsed, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"Expected one-dimensional OT field, got shape {series.shape}.")
    return [float(value) for value in series.tolist()]


def _dataset_path(input_root: Path, dataset: str, length: int) -> Path:
    return input_root / f"embedding_cleaned_{dataset}_{length}.csv"


def _main_rows(case_file: Path) -> set[tuple[str, int, int]]:
    if not case_file.exists():
        return set()
    rows: set[tuple[str, int, int]] = set()
    for case in load_gold_cases([case_file]):
        rows.add((str(case["source_dataset"]), int(case["source_length"]), int(case["source_row"])))
    return rows


def _claim_counts(report: dict[str, Any]) -> dict[str, int]:
    total = len(report.get("claim_results", []))
    unverifiable = len(report.get("unverifiable_claims", []))
    verifiable = int(report.get("verifiable_claim_count", max(0, total - unverifiable)))
    return {
        "total_claims": total,
        "verifiable_claims": verifiable,
        "matched_claims": len(report.get("matched_claims", [])),
        "mismatched_claims": len(report.get("mismatched_claims", [])),
        "unverifiable_claims": unverifiable,
    }


def _load_candidate_records(
    *,
    input_root: Path,
    record_root: Path,
    dataset: str,
    length: int,
) -> tuple[list[dict[str, Any]], str]:
    path = _dataset_path(input_root, dataset, length)
    if path.exists():
        frame = pd.read_csv(path, usecols=["Text", "OT"])
        records = [
            {
                "source_row": int(index),
                "description": str(row["Text"]),
                "series": _series_from_ot(row["OT"]),
            }
            for index, row in frame.iterrows()
        ]
        # These CSVs may be only the already-selected public-generator inputs.
        if len(records) >= 50:
            return records, _display_path(path)

    records_by_row: dict[int, dict[str, Any]] = {}
    for split in ["train", "valid"]:
        record_path = record_root / f"TSFragment_{length}" / f"{split}_records.json"
        if not record_path.exists():
            continue
        for record in _read_json(record_path):
            if str(record.get("source_dataset")) != dataset:
                continue
            if int(record.get("source_length", length)) != length:
                continue
            row_number = int(record["source_row"])
            records_by_row.setdefault(
                row_number,
                {
                    "source_row": row_number,
                    "description": str(record["description"]),
                    "series": [float(value) for value in record["series"]],
                },
            )
    records = [records_by_row[key] for key in sorted(records_by_row)]
    if records:
        return records, _display_path(record_root / f"TSFragment_{length}" / "{train,valid}_records.json")
    return [], "missing"


def _case_record(dataset: str, length: int, record: dict[str, Any]) -> dict[str, Any]:
    row_number = int(record["source_row"])
    series = [float(value) for value in record["series"]]
    report = evaluate_description_series_consistency(str(record["description"]), series)
    lower = float(min(series))
    upper = float(max(series))
    span = upper - lower
    pad = max(1e-6, 0.05 * span)
    counts = _claim_counts(report)
    return {
        "case_id": f"t2s_{dataset}_{length}_{row_number:05d}",
        "category": "t2s_random_audit",
        "source_index": row_number,
        "source_dataset": dataset,
        "source_length": length,
        "source_row": row_number,
        "description": str(record["description"]),
        "length": len(series),
        "reference_series": series,
        "numeric_range": {"min": lower - pad, "max": upper + pad},
        "audit_metadata": {
            "selection_policy": "stratified_random_without_high_consistency_filter",
            "selected_before_method_outputs": True,
            "checker_used_for_selection": False,
        },
        "consistency_evaluation": report,
        "claim_counts": counts,
    }


def export_random_audit_split(
    *,
    input_root: Path,
    record_root: Path,
    main_case_file: Path,
    per_dataset_length: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    main_keys = _main_rows(main_case_file)
    cases: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for length in LENGTHS:
            candidate_records, candidate_source = _load_candidate_records(
                input_root=input_root,
                record_root=record_root,
                dataset=dataset,
                length=length,
            )
            eligible = [
                record
                for record in candidate_records
                if (dataset, length, int(record["source_row"])) not in main_keys
            ]
            if len(eligible) < per_dataset_length:
                raise ValueError(
                    f"{dataset}_{length}: only {len(eligible)} eligible rows for "
                    f"{per_dataset_length} requested audit cases."
                )
            selected_records = sorted(
                rng.sample(eligible, per_dataset_length),
                key=lambda record: int(record["source_row"]),
            )
            selected_rows = [int(record["source_row"]) for record in selected_records]
            for record in selected_records:
                cases.append(_case_record(dataset, length, record))
            strata.append(
                {
                    "dataset": dataset,
                    "length": length,
                    "candidate_source": candidate_source,
                    "candidate_rows": int(len(candidate_records)),
                    "excluded_main_rows": int(len(candidate_records) - len(eligible)),
                    "selected_rows": selected_rows,
                }
            )

    report = _build_report(
        cases=cases,
        input_root=input_root,
        record_root=record_root,
        main_case_file=main_case_file,
        per_dataset_length=per_dataset_length,
        seed=seed,
        strata=strata,
    )
    return cases, report


def _build_report(
    *,
    cases: list[dict[str, Any]],
    input_root: Path,
    record_root: Path,
    main_case_file: Path,
    per_dataset_length: int,
    seed: int,
    strata: list[dict[str, Any]],
) -> dict[str, Any]:
    correctness = [
        float(case["consistency_evaluation"].get("caption_correctness_score", case["consistency_evaluation"]["consistency_score"]))
        for case in cases
    ]
    consistency = [float(case["consistency_evaluation"]["consistency_score"]) for case in cases]
    confidence = [float(case["consistency_evaluation"]["confidence_score"]) for case in cases]
    verifiable_cases = [case for case in cases if int(case["claim_counts"]["verifiable_claims"]) > 0]
    consistent_cases = [
        case for case in cases if case["consistency_evaluation"].get("summary_verdict") == "consistent"
    ]
    partial_cases = [
        case for case in cases if case["consistency_evaluation"].get("summary_verdict") == "partially_consistent"
    ]
    total_claims = sum(int(case["claim_counts"]["total_claims"]) for case in cases)
    verifiable_claims = sum(int(case["claim_counts"]["verifiable_claims"]) for case in cases)
    matched_claims = sum(int(case["claim_counts"]["matched_claims"]) for case in cases)
    return {
        "version": "1.0",
        "split": "t2s_random_audit",
        "purpose": (
            "Audit TSFragment selection effects with a method-blind stratified random split. "
            "The caption checker is applied only after sampling and is not used for case selection."
        ),
        "case_count": len(cases),
        "seed": seed,
        "per_dataset_length": per_dataset_length,
        "input_root": _display_path(input_root),
        "record_root": _display_path(record_root),
        "main_case_file": _display_path(main_case_file),
        "case_file": _display_path(DEFAULT_CASE_OUTPUT),
        "selection_policy": {
            "stratification": "dataset x length",
            "datasets": DATASETS,
            "lengths": LENGTHS,
            "excluded_main_split_rows": True,
            "checker_used_for_selection": False,
            "method_outputs_inspected_for_selection": False,
            "method_comparison_note": (
                "This audit split is sampled from local TSFragment candidate records. "
                "It audits selection and checker coverage only; populate fresh method outputs before using it "
                "for performance comparison."
            ),
        },
        "status": {
            "case_split_registered": True,
            "reference_checker_audit_completed": True,
            "method_outputs_populated": False,
            "method_comparison_claim_allowed": False,
        },
        "reference_checker_audit": {
            "mean_caption_correctness": float(np.mean(correctness)) if correctness else None,
            "mean_semantic_consistency": float(np.mean(consistency)) if consistency else None,
            "mean_confidence": float(np.mean(confidence)) if confidence else None,
            "verifiable_case_count": len(verifiable_cases),
            "verifiable_case_rate": float(len(verifiable_cases) / len(cases)) if cases else None,
            "consistent_count": len(consistent_cases),
            "partially_consistent_count": len(partial_cases),
            "inconsistent_count": max(0, len(cases) - len(consistent_cases) - len(partial_cases)),
            "total_claims": total_claims,
            "verifiable_claims": verifiable_claims,
            "matched_claims": matched_claims,
            "matched_claim_rate": float(matched_claims / verifiable_claims) if verifiable_claims else None,
        },
        "strata": strata,
        "case_ids": [str(case["case_id"]) for case in cases],
    }


def render_markdown(report: dict[str, Any]) -> str:
    audit = report["reference_checker_audit"]
    lines = [
        "# T2S Stratified Random Audit Split",
        "",
        "This split audits selection effects for the TSFragment real-caption evaluation. "
        "Cases are sampled by dataset and horizon with a fixed seed, excluding the main 2500-case split. "
        "The caption checker is applied only after sampling and is not used for case selection.",
        "",
        "## Status",
        "",
        f"- Case file: `{report['case_file']}`",
        f"- Input pool: `{report['input_root']}`",
        f"- Record fallback pool: `{report['record_root']}`",
        f"- Seed: `{report['seed']}`",
        f"- Cases: `{report['case_count']}`",
        "- Method outputs populated: `False`",
        "- Method-comparison claim allowed from this report: `False`",
        "",
        "## Reference Checker Audit",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean caption correctness | {audit['mean_caption_correctness']:.3f} |",
        f"| Mean semantic consistency | {audit['mean_semantic_consistency']:.3f} |",
        f"| Mean confidence | {audit['mean_confidence']:.3f} |",
        f"| Verifiable cases | {audit['verifiable_case_count']}/{report['case_count']} |",
        f"| Consistent cases | {audit['consistent_count']}/{report['case_count']} |",
        f"| Partially consistent cases | {audit['partially_consistent_count']}/{report['case_count']} |",
        f"| Matched claim rate | {audit['matched_claim_rate']:.3f} |",
        "",
        "## Strata",
        "",
        "| Dataset | Length | Candidate source | Candidate rows | Excluded main rows | Selected rows |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in report["strata"]:
        selected = ", ".join(str(item) for item in row["selected_rows"])
        lines.append(
            f"| {row['dataset']} | {row['length']} | `{row['candidate_source']}` | {row['candidate_rows']} | "
            f"{row['excluded_main_rows']} | {selected} |"
        )
    lines.extend(
        [
            "",
            "This report should be cited as a selection-bias audit, not as a completed method-comparison result. "
            "Populate method outputs for this case file before making any performance claim on the random audit split.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a method-blind stratified random TSFragment audit split.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--main-case-file", type=Path, default=DEFAULT_MAIN_CASE_FILE)
    parser.add_argument("--per-dataset-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--case-output", type=Path, default=DEFAULT_CASE_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    cases, report = export_random_audit_split(
        input_root=args.input_root,
        record_root=args.record_root,
        main_case_file=args.main_case_file,
        per_dataset_length=args.per_dataset_length,
        seed=args.seed,
    )
    report["case_file"] = _display_path(args.case_output)
    _write_json(args.case_output, cases)
    _write_json(args.report_json, report)
    _write_text(args.report_md, render_markdown(report))
    print(f"Wrote cases to {_display_path(args.case_output)}")
    print(f"Wrote report to {_display_path(args.report_json)}")
    print(f"Wrote report to {_display_path(args.report_md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
