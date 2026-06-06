from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.timeseries_consistency import evaluate_description_series_consistency
from evals.run_baseline_benchmark import _load_external_series_output
from evals.run_llm_baseline_outputs import DIRECT_LLM_T2S_BASELINES, _parse_llm_series
from faithts.gold_adapter import load_gold_cases


DEFAULT_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
T2S_TSFRAGMENT_ROOT = REPO_ROOT / "baselines" / "T2S" / "Data" / "TSFragment-600K"
if not T2S_TSFRAGMENT_ROOT.exists():
    T2S_TSFRAGMENT_ROOT = REPO_ROOT / "baselines" / "T2S" / "Data" / "Three Levels Data" / "TSFragment-600K"
T2S_TSFRAGMENT_CASE_INPUT_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_native_case_inputs"
DEFAULT_T2S_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_native_main_benchmark_outputs_2500"
DEFAULT_OURS_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_cases_2500_reprocessed_v2"
DEFAULT_OURS_EVAL_JSON = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_eval_2500_reprocessed_v2.json"
DEFAULT_VERBALTS_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "verbalts_main_benchmark_outputs_2500"
DEFAULT_DIRECT_LLM_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_outputs_2500"
DEFAULT_JSON_OUTPUT = REPO_ROOT / "evals" / "reports" / "t2s_semantic_consistency_eval_2500_reprocessed_v3_full.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "evals" / "reports" / "t2s_semantic_consistency_eval_2500_reprocessed_v3_full.md"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
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


def _group_cases(cases: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for case in cases:
        key = (str(case["source_dataset"]), int(case["source_length"]))
        grouped.setdefault(key, []).append(case)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda case: int(case["source_row"]))
    return grouped


def _series_matrix_for(dataset: str, length: int) -> np.ndarray:
    source_csv = T2S_TSFRAGMENT_ROOT / f"embedding_cleaned_{dataset}_{length}.csv"
    if not source_csv.exists():
        source_csv = T2S_TSFRAGMENT_CASE_INPUT_ROOT / f"embedding_cleaned_{dataset}_{length}.csv"
    frame = pd.read_csv(source_csv, usecols=["OT"])
    return np.asarray([ast.literal_eval(item) for item in frame["OT"]], dtype=float)


def _inverse_minmax_transform(dataset: str, length: int, generated: np.ndarray) -> np.ndarray:
    source = _series_matrix_for(dataset, length)
    data_min = np.min(source, axis=0)
    data_max = np.max(source, axis=0)
    data_range = data_max - data_min
    # sklearn MinMaxScaler leaves constant features at min after inverse transform.
    return generated * data_range + data_min


def _load_t2s_native_series_by_case(
    *,
    cases: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, np.ndarray]:
    series_by_case: dict[str, np.ndarray] = {}
    grouped = _group_cases(cases)
    for (dataset, length), group_cases in sorted(grouped.items()):
        generated_path = output_root / f"{dataset}_{length}" / "x_t.npy"
        if not generated_path.exists():
            continue
        generated_all = _inverse_minmax_transform(dataset, length, np.load(generated_path).squeeze(-1))
        if len(generated_all) != len(group_cases):
            raise ValueError(
                f"{dataset}_{length}: generated rows {len(generated_all)} do not match cases {len(group_cases)}"
            )
        for index, case in enumerate(group_cases):
            series_by_case[str(case["case_id"])] = np.asarray(generated_all[index], dtype=float)
    return series_by_case


def _load_ours_series_by_case(
    *,
    cases: list[dict[str, Any]],
    output_root: Path,
    eval_json: Path | None = DEFAULT_OURS_EVAL_JSON,
) -> dict[str, np.ndarray]:
    series_by_case: dict[str, np.ndarray] = {}
    if eval_json is not None and eval_json.exists():
        payload = _read_json(eval_json)
        for record in payload.get("records", []):
            if record.get("status") not in {"success", "partial_success"}:
                continue
            if "generated_series" not in record:
                continue
            series_by_case[str(record["case_id"])] = np.asarray(record["generated_series"], dtype=float)
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in series_by_case:
            continue
        path = output_root / case_id / "generated_series.json"
        if not path.exists():
            continue
        series_by_case[case_id] = np.asarray(_read_json(path), dtype=float)
    return series_by_case


def _load_external_series_by_case(
    *,
    cases: list[dict[str, Any]],
    output_root: Path | None,
) -> dict[str, np.ndarray]:
    if output_root is None or not output_root.exists():
        return {}
    series_by_case: dict[str, np.ndarray] = {}
    for case in cases:
        case_id = str(case["case_id"])
        try:
            series_by_case[case_id] = _load_external_series_output(output_root, case_id, int(case["length"]))
        except Exception:
            continue
    return series_by_case


def _load_direct_llm_series_by_case(
    *,
    cases: list[dict[str, Any]],
    output_root: Path | None,
    method: str,
) -> dict[str, np.ndarray]:
    if output_root is None:
        return {}
    method_dir = output_root / method
    if not method_dir.exists():
        return {}
    series_by_case: dict[str, np.ndarray] = {}
    for case in cases:
        case_id = str(case["case_id"])
        output_path = method_dir / f"{case_id}.txt"
        if not output_path.exists():
            continue
        try:
            series_by_case[case_id] = _parse_llm_series(method, output_path.read_text(encoding="utf-8"), int(case["length"]))
        except Exception:
            continue
    return series_by_case


def _claim_counts(report: dict[str, Any]) -> dict[str, int]:
    return {
        "claims": len(report.get("claim_results", [])),
        "matched": len(report.get("matched_claims", [])),
        "mismatched": len(report.get("mismatched_claims", [])),
        "unverifiable": len(report.get("unverifiable_claims", [])),
    }


def _evaluate_method(
    *,
    method: str,
    cases: list[dict[str, Any]],
    series_by_case: dict[str, np.ndarray],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        series = series_by_case.get(case_id)
        if series is None:
            records.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "status": "missing_series",
                    "source_dataset": case.get("source_dataset"),
                    "source_length": case.get("source_length"),
                }
            )
            continue
        try:
            report = evaluate_description_series_consistency(str(case["description"]), series)
        except Exception as exc:
            records.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "status": "evaluation_error",
                    "error": str(exc),
                    "source_dataset": case.get("source_dataset"),
                    "source_length": case.get("source_length"),
                }
            )
            continue
        records.append(
            {
                "case_id": case_id,
                "method": method,
                "status": "success",
                "source_dataset": case.get("source_dataset"),
                "source_length": case.get("source_length"),
                "summary_verdict": report["summary_verdict"],
                "semantic_consistency_score": report["consistency_score"],
                "caption_correctness_score": report.get("caption_correctness_score", report["consistency_score"]),
                "correctness_details": report.get("correctness_details", {}),
                "confidence_score": report["confidence_score"],
                "verifiable_claim_count": report.get("verifiable_claim_count"),
                "contradicted_claim_rate": report.get("contradicted_claim_rate"),
                **_claim_counts(report),
                "extracted_claims": report.get("extracted_claims", []),
                "aspect_scores": report.get("aspect_scores", {}),
            }
        )
    return {
        "method": method,
        "records": records,
        "aggregate": _aggregate_records(method, records, total_cases=len(cases)),
        "by_dataset_length": _aggregate_by_dataset_length(method, records, cases),
    }


def _aggregate_records(method: str, records: list[dict[str, Any]], *, total_cases: int) -> dict[str, Any]:
    success = [record for record in records if record.get("status") == "success"]
    scores = [float(record["semantic_consistency_score"]) for record in success]
    correctness_scores = [float(record["caption_correctness_score"]) for record in success]
    confidences = [float(record["confidence_score"]) for record in success]
    consistent = [record for record in success if record.get("summary_verdict") == "consistent"]
    partial = [record for record in success if record.get("summary_verdict") == "partially_consistent"]
    inconsistent = [record for record in success if record.get("summary_verdict") == "inconsistent"]
    total_claims = sum(int(record.get("claims", 0)) for record in success)
    total_matched = sum(int(record.get("matched", 0)) for record in success)
    total_mismatched = sum(int(record.get("mismatched", 0)) for record in success)
    total_unverifiable = sum(int(record.get("unverifiable", 0)) for record in success)
    verifiable_claims = max(0, total_claims - total_unverifiable)
    return {
        "method": method,
        "cases": total_cases,
        "successful_cases": len(success),
        "success_rate": float(len(success) / total_cases) if total_cases else 0.0,
        "mean_semantic_consistency": float(np.mean(scores)) if scores else None,
        "median_semantic_consistency": float(np.median(scores)) if scores else None,
        "mean_caption_correctness_valid": float(np.mean(correctness_scores)) if correctness_scores else None,
        "median_caption_correctness_valid": float(np.median(correctness_scores)) if correctness_scores else None,
        "mean_caption_correctness": float(sum(correctness_scores) / total_cases) if total_cases else None,
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "consistent_count": len(consistent),
        "consistent_rate": float(len(consistent) / len(success)) if success else None,
        "consistent_rate_all_cases": float(len(consistent) / total_cases) if total_cases else None,
        "partial_count": len(partial),
        "inconsistent_count": len(inconsistent),
        "total_claims": total_claims,
        "matched_claims": total_matched,
        "mismatched_claims": total_mismatched,
        "unverifiable_claims": total_unverifiable,
        "matched_claim_rate": float(total_matched / verifiable_claims) if verifiable_claims else None,
    }


def _aggregate_by_dataset_length(
    method: str,
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted({(str(case["source_dataset"]), int(case["source_length"])) for case in cases})
    rows: list[dict[str, Any]] = []
    for dataset, length in keys:
        subset = [
            record
            for record in records
            if record.get("status") == "success"
            and str(record.get("source_dataset")) == dataset
            and int(record.get("source_length")) == length
        ]
        scores = [float(record["semantic_consistency_score"]) for record in subset]
        correctness_scores = [float(record["caption_correctness_score"]) for record in subset]
        consistent = [record for record in subset if record.get("summary_verdict") == "consistent"]
        total = sum(1 for case in cases if str(case["source_dataset"]) == dataset and int(case["source_length"]) == length)
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "length": length,
                "cases": total,
                "successful_cases": len(subset),
                "mean_semantic_consistency": float(np.mean(scores)) if scores else None,
                "mean_caption_correctness": float(sum(correctness_scores) / total) if total else None,
                "mean_caption_correctness_valid": float(np.mean(correctness_scores)) if correctness_scores else None,
                "consistent_rate": float(len(consistent) / len(subset)) if subset else None,
            }
        )
    return rows


def _attach_reference_normalized_correctness(methods: list[dict[str, Any]]) -> None:
    reference = next((payload for payload in methods if payload["method"] == "Reference"), None)
    if reference is None:
        return
    reference_scores = {
        str(record["case_id"]): float(record["caption_correctness_score"])
        for record in reference["records"]
        if record.get("status") == "success"
    }
    for payload in methods:
        normalized_scores: list[float] = []
        for record in payload["records"]:
            ref_score = reference_scores.get(str(record.get("case_id")))
            if payload["method"] == "Reference" and record.get("status") == "success":
                normalized = 1.0
            elif record.get("status") == "success" and ref_score is not None:
                normalized = min(1.0, float(record["caption_correctness_score"]) / max(ref_score, 0.25))
            else:
                normalized = 0.0
            record["reference_normalized_caption_correctness"] = round(float(normalized), 4)
            normalized_scores.append(float(normalized))
        aggregate = payload["aggregate"]
        aggregate["mean_reference_normalized_caption_correctness"] = (
            float(np.mean(normalized_scores)) if normalized_scores else None
        )
        aggregate["median_reference_normalized_caption_correctness"] = (
            float(np.median(normalized_scores)) if normalized_scores else None
        )


def evaluate_t2s_semantic_consistency(
    *,
    case_file: Path = DEFAULT_CASE_FILE,
    t2s_output_root: Path = DEFAULT_T2S_OUTPUT_ROOT,
    ours_output_root: Path = DEFAULT_OURS_OUTPUT_ROOT,
    ours_eval_json: Path | None = DEFAULT_OURS_EVAL_JSON,
    verbalts_output_root: Path | None = None,
    direct_llm_output_root: Path | None = DEFAULT_DIRECT_LLM_OUTPUT_ROOT,
    include_direct_llm: bool = True,
) -> dict[str, Any]:
    cases = load_gold_cases([case_file])
    reference_series = {
        str(case["case_id"]): np.asarray(case["reference_series"], dtype=float)
        for case in cases
    }
    native_series = _load_t2s_native_series_by_case(cases=cases, output_root=t2s_output_root)
    ours_series = _load_ours_series_by_case(cases=cases, output_root=ours_output_root, eval_json=ours_eval_json)
    verbalts_series = _load_external_series_by_case(cases=cases, output_root=verbalts_output_root)
    methods = [
        _evaluate_method(method="Reference", cases=cases, series_by_case=reference_series),
        _evaluate_method(method="T2S", cases=cases, series_by_case=native_series),
        _evaluate_method(method="FaithTS", cases=cases, series_by_case=ours_series),
    ]
    if verbalts_output_root is not None:
        methods.append(_evaluate_method(method="VerbalTS", cases=cases, series_by_case=verbalts_series))
    if include_direct_llm:
        method_names = {
            "llm_direct_array": "LLM Direct Array",
            "llm_direct_code": "LLM Direct Code",
            "llm_strong_prompt": "LLM Strong Prompt",
        }
        for method in DIRECT_LLM_T2S_BASELINES:
            methods.append(
                _evaluate_method(
                    method=method_names[method],
                    cases=cases,
                    series_by_case=_load_direct_llm_series_by_case(
                        cases=cases,
                        output_root=direct_llm_output_root,
                        method=method,
                    ),
                )
            )
    _attach_reference_normalized_correctness(methods)
    return {
        "version": "1.1",
        "benchmark": "T2S TSFragment-600K caption-level correctness",
        "case_file": _display_path(case_file),
        "metric_policy": {
            "primary": [
                "caption_correctness_score / mean_caption_correctness (higher better; invalid/missing outputs count as 0)",
                "reference_normalized_caption_score (higher better; per-case capped ratio to the reference row)",
                "claim_satisfaction_rate / matched_claim_rate (higher better)",
            ],
            "diagnostic": [
                "valid_output_claim_score / semantic_consistency_score (higher better; valid outputs only)",
                "full_claim_satisfaction_rate / consistent_rate_all_cases (higher better)",
            ],
            "note": (
                "This real-caption evaluator checks measurable claims extracted from each caption against each "
                "generated series. The primary method-level correctness metric is caption correctness score "
                "(mean_caption_correctness): "
                "invalid or missing outputs count as zero, precise claims receive slightly higher weight, and "
                "explicit contradictions are penalized. The valid-output claim score is retained as a diagnostic, "
                "not the end-to-end correctness metric. This is not the gold primitive CSR used for synthetic "
                "semantic cases."
            ),
        },
        "artifacts": {
            "t2s_native_outputs": _display_path(t2s_output_root),
            "ours_outputs": _display_path(ours_output_root),
            "ours_eval_json": _display_path(ours_eval_json) if ours_eval_json is not None else None,
            "verbalts_outputs": _display_path(verbalts_output_root) if verbalts_output_root is not None else None,
            "direct_llm_outputs": _display_path(direct_llm_output_root) if direct_llm_output_root is not None else None,
        },
        "overall": [method_payload["aggregate"] for method_payload in methods],
        "by_dataset_length": {
            method_payload["method"]: method_payload["by_dataset_length"]
            for method_payload in methods
        },
        "records": {
            method_payload["method"]: method_payload["records"]
            for method_payload in methods
        },
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# T2S Caption-Level Correctness",
        "",
        "| Method | Cases | Valid Outputs | Caption Correctness Score | Reference-Normalized Caption Score | Valid-Output Caption Score | Valid-Output Claim Score | Median Claim Score | Fully Satisfied Cases | Full-Claim Satisfaction Rate (Valid) | Full-Claim Satisfaction Rate (All) | Claim Satisfaction Rate | Mean Conf. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["overall"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["cases"]),
                    f"{row['successful_cases']}/{row['cases']}",
                    _fmt(row["mean_caption_correctness"]),
                    _fmt(row["mean_reference_normalized_caption_correctness"]),
                    _fmt(row["mean_caption_correctness_valid"]),
                    _fmt(row["mean_semantic_consistency"]),
                    _fmt(row["median_semantic_consistency"]),
                    f"{row['consistent_count']}/{row['successful_cases']}",
                    _fmt(row["consistent_rate"]),
                    _fmt(row["consistent_rate_all_cases"]),
                    _fmt(row["matched_claim_rate"]),
                    _fmt(row["mean_confidence"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Note: this evaluator extracts measurable caption claims and checks them against each series. "
            "Caption Correctness Score is the primary caption-level correctness score: missing or invalid outputs count as 0. "
            "Reference-Normalized Caption Score divides each output's caption score by the reference-row caption score for the same case "
            "and caps the ratio at 1. Valid-Output Claim Score is reported only over valid outputs. These metrics complement WAPE/CORR/MRR by measuring caption-level semantic consistency "
            "rather than reference similarity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate T2S and VTS outputs with the same caption-level semantic checker.")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--t2s-output-root", type=Path, default=DEFAULT_T2S_OUTPUT_ROOT)
    parser.add_argument("--ours-output-root", type=Path, default=DEFAULT_OURS_OUTPUT_ROOT)
    parser.add_argument("--ours-eval-json", type=Path, default=DEFAULT_OURS_EVAL_JSON)
    parser.add_argument(
        "--verbalts-output-root",
        type=Path,
        default=DEFAULT_VERBALTS_OUTPUT_ROOT,
        help=(
            "One-file-per-case VerbalTS output directory. Files are read as "
            "<case_id>.json or <case_id>.txt using generated_series/series/values."
        ),
    )
    parser.add_argument("--direct-llm-output-root", type=Path, default=DEFAULT_DIRECT_LLM_OUTPUT_ROOT)
    parser.add_argument("--no-direct-llm", action="store_true")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = evaluate_t2s_semantic_consistency(
        case_file=args.case_file,
        t2s_output_root=args.t2s_output_root,
        ours_output_root=args.ours_output_root,
        ours_eval_json=args.ours_eval_json,
        verbalts_output_root=args.verbalts_output_root,
        direct_llm_output_root=args.direct_llm_output_root,
        include_direct_llm=not args.no_direct_llm,
    )
    _write_json(args.json_output, payload)
    _write_text(args.markdown_output, render_markdown(payload))
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        print(f"Wrote JSON to {_display_path(args.json_output)}")
        print(f"Wrote Markdown to {_display_path(args.markdown_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
