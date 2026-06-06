from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.baselines.llm_direct import build_prompt, parse_direct_array_output, parse_direct_code_output
from evals.series_evaluation import compute_error_metrics
from evals.run_baseline_benchmark import DEFAULT_CASE_FILES, LLM_OUTPUT_BASELINES
from faithts_pipeline.workflow import ExternalLLMConfigurationError, ExternalLLMRequestError, RetrievalExample, build_llm_client
from faithts.gold_adapter import load_gold_cases


DEFAULT_T2S_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
DEFAULT_T2S_CASE_FILES = [DEFAULT_T2S_CASE_FILE]
DEFAULT_T2S_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_outputs_2500"
DEFAULT_T2S_COMPARISON_JSON = REPO_ROOT / "evals" / "reports" / "t2s_main_benchmark_comparison_2500.json"
DEFAULT_T2S_EVAL_JSON = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_eval_2500.json"
DIRECT_LLM_T2S_BASELINES = ["llm_direct_array", "llm_direct_code", "llm_strong_prompt"]


def _has_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("PROTOTYPE_CODEGEN_API_KEY"))


def _fake_example(case: dict[str, Any], index: int) -> RetrievalExample:
    return RetrievalExample(
        index=index,
        caption=str(case["description"]),
        candidate_primitives=list(case.get("expected_candidate_primitives", [])),
        retrieval_status="baseline_prompt",
        orchestration_level_descriptors=[],
        raw_record=case,
    )


def resolve_case_paths(case_paths: list[Path] | None, *, use_t2s_main_benchmark: bool = False) -> list[Path]:
    if case_paths:
        return case_paths
    if use_t2s_main_benchmark:
        return list(DEFAULT_T2S_CASE_FILES)
    return list(DEFAULT_CASE_FILES)


def generate_outputs(
    *,
    baseline: str,
    output_dir: Path,
    case_paths: list[Path] | None = None,
    category: str | None = None,
    provider: str = "gemini",
    limit: int | None = None,
    skip_existing: bool = True,
    require_api_key: bool = False,
    use_t2s_main_benchmark: bool = False,
) -> dict[str, Any]:
    if baseline not in LLM_OUTPUT_BASELINES:
        raise ValueError(f"`{baseline}` is not an LLM output baseline. Choose from {sorted(LLM_OUTPUT_BASELINES)}.")
    resolved_case_paths = resolve_case_paths(case_paths, use_t2s_main_benchmark=use_t2s_main_benchmark)
    cases = load_gold_cases(resolved_case_paths)
    if category is not None:
        cases = [case for case in cases if case.get("category") == category]
    if limit is not None:
        cases = cases[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    if not _has_api_key():
        summary = {
            "baseline": baseline,
            "status": "skipped",
            "skip_reason": "missing_api_key",
            "output_dir": str(output_dir),
            "case_paths": [str(path) for path in resolved_case_paths],
            "total_cases": len(cases),
            "written": 0,
            "records": [],
        }
        (output_dir / "generation_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        if require_api_key:
            raise ExternalLLMConfigurationError("Missing API key for LLM baseline generation.")
        return summary

    client = build_llm_client(provider)
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = str(case["case_id"])
        output_path = output_dir / f"{case_id}.txt"
        prompt_path = output_dir / f"{case_id}.prompt.txt"
        if skip_existing and output_path.exists():
            records.append({"case_id": case_id, "status": "skipped_existing", "output_path": str(output_path)})
            continue
        prompt = build_prompt(case, baseline)
        prompt_path.write_text(prompt, encoding="utf-8")
        try:
            generation = client.generate_code(prompt, _fake_example(case, index))
            output_path.write_text(generation.raw_text, encoding="utf-8")
            records.append(
                {
                    "case_id": case_id,
                    "status": "written",
                    "provider": generation.provider,
                    "model": generation.model,
                    "output_path": str(output_path),
                    "prompt_path": str(prompt_path),
                }
            )
        except (ExternalLLMConfigurationError, ExternalLLMRequestError) as exc:
            records.append({"case_id": case_id, "status": "error", "reason": str(exc), "prompt_path": str(prompt_path)})

    summary = {
        "baseline": baseline,
        "status": "completed",
        "output_dir": str(output_dir),
        "case_paths": [str(path) for path in resolved_case_paths],
        "total_cases": len(cases),
        "written": sum(1 for item in records if item["status"] == "written"),
        "skipped_existing": sum(1 for item in records if item["status"] == "skipped_existing"),
        "errors": sum(1 for item in records if item["status"] == "error"),
        "records": records,
    }
    (output_dir / "generation_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def generate_many_outputs(
    *,
    baselines: list[str],
    output_root: Path,
    case_paths: list[Path] | None = None,
    category: str | None = None,
    provider: str = "gemini",
    limit: int | None = None,
    skip_existing: bool = True,
    require_api_key: bool = False,
    use_t2s_main_benchmark: bool = False,
) -> dict[str, Any]:
    resolved_case_paths = resolve_case_paths(case_paths, use_t2s_main_benchmark=use_t2s_main_benchmark)
    summaries = []
    for baseline in baselines:
        summaries.append(
            generate_outputs(
                baseline=baseline,
                output_dir=output_root / baseline,
                case_paths=resolved_case_paths,
                category=category,
                provider=provider,
                limit=limit,
                skip_existing=skip_existing,
                require_api_key=require_api_key,
                use_t2s_main_benchmark=use_t2s_main_benchmark,
            )
        )
    manifest = {
        "version": "1.0",
        "output_root": str(output_root),
        "baselines": baselines,
        "case_paths": [str(path) for path in resolved_case_paths],
        "total_written": sum(int(item.get("written", 0)) for item in summaries),
        "total_errors": sum(int(item.get("errors", 0)) for item in summaries if item.get("status") != "skipped"),
        "summaries": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "generation_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def _parse_llm_series(method: str, text: str, length: int) -> np.ndarray:
    if method == "llm_direct_array":
        return parse_direct_array_output(text, length)
    if method in {"llm_direct_code", "llm_strong_prompt", "llm_strong_prompt_code"}:
        return parse_direct_code_output(text, length)
    raise ValueError(f"Unsupported TSFragment LLM baseline: {method}")


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _failure_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.get("status") == "success":
            continue
        status = str(record.get("status") or "failed")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _normalize_t2s_failure(error: str) -> str:
    if "does not match expected length" in error:
        return "length_mismatch"
    if "__name__" in error:
        return "missing___name__"
    if "could not broadcast input array" in error:
        return "shape_broadcast_mismatch"
    if "Disallowed numpy call" in error or "Only whitelisted `np.*` calls are allowed." in error:
        return "unsupported_numpy_api"
    if "Disallowed function call" in error:
        return "disallowed_function_call"
    if "No JSON object found" in error:
        return "json_parse_failure"
    return "other_execution_failure"


def _length_mismatch_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        error = str(record.get("error") or "")
        match = re.search(r"length (\d+) does not match expected length (\d+)", error)
        if match is None:
            continue
        actual = int(match.group(1))
        expected = int(match.group(2))
        delta = actual - expected
        key = f"{delta:+d}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _failure_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [record for record in records if record.get("status") != "success"]
    bucket_counts: dict[str, int] = {}
    for record in failed:
        bucket = _normalize_t2s_failure(str(record.get("error") or ""))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    summary: dict[str, Any] = {
        "failed_cases": len(failed),
        "failure_buckets": bucket_counts,
    }
    length_hist = _length_mismatch_distribution(failed)
    if length_hist:
        summary["length_mismatch_delta_histogram"] = length_hist
    notes: list[str] = []
    if bucket_counts.get("length_mismatch"):
        sorted_hist = sorted(length_hist.items(), key=lambda item: (-item[1], item[0]))
        top = ", ".join(f"{delta}:{count}" for delta, count in sorted_hist[:5])
        notes.append(f"Length mismatches dominate; delta histogram (actual-expected) is {top}.")
    if bucket_counts.get("missing___name__"):
        notes.append("Many code failures come from `if __name__ == \"__main__\":` style scripts that do not expose `series` in the restricted executor.")
    if bucket_counts.get("shape_broadcast_mismatch"):
        notes.append("Some generated programs fail because slice lengths and generated segment lengths are inconsistent.")
    if notes:
        summary["analysis_notes"] = notes
    return summary


def evaluate_t2s_llm_outputs(
    *,
    output_root: Path = DEFAULT_T2S_OUTPUT_ROOT,
    case_path: Path = DEFAULT_T2S_CASE_FILE,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    methods = methods or DIRECT_LLM_T2S_BASELINES
    cases = load_gold_cases([case_path])
    results: list[dict[str, Any]] = []
    for method in methods:
        method_dir = output_root / method
        records: list[dict[str, Any]] = []
        for case in cases:
            case_id = str(case["case_id"])
            output_path = method_dir / f"{case_id}.txt"
            reference = np.asarray(case["reference_series"], dtype=float)
            try:
                if not output_path.exists():
                    raise FileNotFoundError(f"missing output file: {output_path}")
                generated = _parse_llm_series(method, output_path.read_text(encoding="utf-8"), int(case["length"]))
                if generated.shape != reference.shape:
                    raise ValueError(f"shape {generated.shape} != reference shape {reference.shape}")
                metrics = compute_error_metrics(generated, reference)
                records.append(
                    {
                        "case_id": case_id,
                        "status": "success",
                        "source_dataset": case.get("source_dataset"),
                        "source_length": case.get("source_length", case.get("length")),
                        "output_path": str(output_path),
                        "metrics": metrics,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "case_id": case_id,
                        "status": "failed",
                        "source_dataset": case.get("source_dataset"),
                        "source_length": case.get("source_length", case.get("length")),
                        "output_path": str(output_path),
                        "error": str(exc),
                    }
                )

        success = [record for record in records if record.get("status") == "success"]
        metrics = [record["metrics"] for record in success]
        results.append(
            {
                "method": method,
                "status": "completed" if len(success) == len(records) else "partial",
                "case_file": str(case_path),
                "output_dir": str(method_dir),
                "cases": len(records),
                "successful_cases": len(success),
                "success_rate": float(len(success) / len(records)) if records else 0.0,
                "mean_wape": _mean([item.get("wape") for item in metrics]),
                "median_wape": _median([item.get("wape") for item in metrics]),
                "mean_corr": _mean([item.get("correlation") for item in metrics]),
                "median_corr": _median([item.get("correlation") for item in metrics]),
                "mean_mrr": _mean([item.get("mrr") for item in metrics]),
                "median_mrr": _median([item.get("mrr") for item in metrics]),
                "mean_raw_mse": _mean([item.get("mse") for item in metrics]),
                "median_raw_mse": _median([item.get("mse") for item in metrics]),
                "failure_counts": _failure_counts(records),
                "failure_summary": _failure_summary(records),
                "records": records,
            }
        )
    return {
        "version": "1.0",
        "benchmark": "T2S TSFragment-600K direct LLM baseline outputs",
        "case_file": str(case_path),
        "output_root": str(output_root),
        "methods": methods,
        "total_cases": len(cases),
        "results": results,
    }


def t2s_report_paths(*, report_dir: Path, output_root: Path | None = None) -> dict[str, Path]:
    resolved_output_root = output_root or (report_dir / "t2s_llm_main_benchmark_outputs_2500")
    return {
        "output_root": resolved_output_root,
        "eval_json": report_dir / "t2s_llm_main_benchmark_eval_2500.json",
        "comparison_json": report_dir / "t2s_main_benchmark_comparison_2500.json",
        "comparison_md": report_dir / "t2s_main_benchmark_comparison_2500.md",
    }


def merge_t2s_llm_results(
    *,
    comparison_path: Path = DEFAULT_T2S_COMPARISON_JSON,
    llm_payload: dict[str, Any],
    eval_path: Path = DEFAULT_T2S_EVAL_JSON,
) -> dict[str, Any]:
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    else:
        comparison = {
            "version": "1.0",
            "benchmark": "T2S TSFragment-600K main benchmark grid",
            "overall": [],
            "artifacts": {},
        }
    existing = []
    for row in comparison.get("overall", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("method")) in set(llm_payload.get("methods", [])):
            continue
        existing.append(row)

    llm_rows = []
    for result in llm_payload.get("results", []):
        llm_rows.append(
            {
                "method": result["method"],
                "cases": result["cases"],
                "success_rate": result["success_rate"],
                "mean_wape": result["mean_wape"],
                "median_wape": result["median_wape"],
                "mean_corr": result["mean_corr"],
                "median_corr": result["median_corr"],
                "mean_mrr": result["mean_mrr"],
                "median_mrr": result["median_mrr"],
                "mean_raw_mse": result["mean_raw_mse"],
                "median_raw_mse": result["median_raw_mse"],
                "status": result["status"],
                "successful_cases": result["successful_cases"],
                "failure_summary": result.get("failure_summary"),
            }
        )
    comparison["overall"] = existing + llm_rows
    comparison.setdefault("artifacts", {})["llm_direct_outputs"] = str(llm_payload["output_root"])
    comparison.setdefault("artifacts", {})["llm_direct_eval"] = str(eval_path)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate saved raw outputs for direct LLM baselines.")
    parser.add_argument("--baseline", choices=sorted(LLM_OUTPUT_BASELINES), default=None)
    parser.add_argument("--baselines", nargs="+", choices=sorted(LLM_OUTPUT_BASELINES), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cases", nargs="*", type=Path, default=None)
    parser.add_argument("--t2s-main-benchmark", action="store_true")
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial", "t2s_real_smoke"], default=None)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-api-key", action="store_true")
    parser.add_argument("--evaluate-t2s", action="store_true")
    parser.add_argument("--write-t2s-comparison", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    case_paths = resolve_case_paths(args.cases, use_t2s_main_benchmark=args.t2s_main_benchmark)
    baselines = args.baselines or ([args.baseline] if args.baseline is not None else None)
    if baselines is None:
        baselines = DIRECT_LLM_T2S_BASELINES if args.t2s_main_benchmark else None
    if baselines is None:
        parser.error("Provide --baseline/--baselines, or use --t2s-main-benchmark for the default direct LLM trio.")

    if args.output_root is not None:
        summary = generate_many_outputs(
            baselines=baselines,
            output_root=args.output_root,
            case_paths=case_paths,
            category=args.category,
            provider=args.provider,
            limit=args.limit,
            skip_existing=not args.overwrite,
            require_api_key=args.require_api_key,
            use_t2s_main_benchmark=args.t2s_main_benchmark,
        )
    elif len(baselines) == 1:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = (DEFAULT_T2S_OUTPUT_ROOT / baselines[0]) if args.t2s_main_benchmark else None
        if output_dir is None:
            parser.error("--output-dir is required for a single baseline unless --t2s-main-benchmark is used.")
        summary = generate_outputs(
            baseline=baselines[0],
            output_dir=output_dir,
            case_paths=case_paths,
            category=args.category,
            provider=args.provider,
            limit=args.limit,
            skip_existing=not args.overwrite,
            require_api_key=args.require_api_key,
            use_t2s_main_benchmark=args.t2s_main_benchmark,
        )
    else:
        output_root = args.output_root or (DEFAULT_T2S_OUTPUT_ROOT if args.t2s_main_benchmark else None)
        if output_root is None:
            parser.error("--output-root is required when generating multiple baselines.")
        summary = generate_many_outputs(
            baselines=baselines,
            output_root=output_root,
            case_paths=case_paths,
            category=args.category,
            provider=args.provider,
            limit=args.limit,
            skip_existing=not args.overwrite,
            require_api_key=args.require_api_key,
            use_t2s_main_benchmark=args.t2s_main_benchmark,
        )

    if args.evaluate_t2s or args.write_t2s_comparison:
        output_root = args.output_root or DEFAULT_T2S_OUTPUT_ROOT
        if args.output_dir is not None and len(baselines) == 1:
            output_root = args.output_dir.parent
        t2s_eval = evaluate_t2s_llm_outputs(
            output_root=output_root,
            case_path=case_paths[0],
            methods=baselines,
        )
        DEFAULT_T2S_EVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_T2S_EVAL_JSON.write_text(json.dumps(t2s_eval, indent=2, allow_nan=False), encoding="utf-8")
        summary["t2s_eval_path"] = str(DEFAULT_T2S_EVAL_JSON)
        if args.write_t2s_comparison:
            comparison = merge_t2s_llm_results(llm_payload=t2s_eval, eval_path=DEFAULT_T2S_EVAL_JSON)
            DEFAULT_T2S_COMPARISON_JSON.write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
            summary["t2s_comparison_path"] = str(DEFAULT_T2S_COMPARISON_JSON)

    if args.json:
        print(json.dumps(summary, indent=2, allow_nan=False))
    else:
        if summary.get("status") == "skipped":
            print(f"Skipped {args.baseline}: {summary['skip_reason']}")
        elif "summaries" in summary:
            print(
                f"Generated {summary['total_written']} outputs across {len(summary['baselines'])} baselines "
                f"(errors={summary['total_errors']})"
            )
        else:
            print(
                f"Generated {summary['written']}/{summary['total_cases']} outputs for {args.baseline} "
                f"(skipped_existing={summary['skipped_existing']}, errors={summary['errors']})"
            )
    errors = int(summary.get("errors", summary.get("total_errors", 0)))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
