from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.baselines.llm_direct import invalid_direct_output_plan, parse_direct_array_output
from evals.run_baselines import run_methods
from faithts import aggregate_results, evaluate_case
from faithts.gold_adapter import load_gold_cases, semantic_spec_from_gold_case


DEFAULT_REPORT_DIR = REPO_ROOT / "evals" / "reports"
DEFAULT_REALISM_PATH = REPO_ROOT / "evals" / "real_series_sample_outputs.json"
DEFAULT_T2S_CASE_LEVEL_EVAL = REPO_ROOT / "evals" / "reports" / "t2s_native_main_benchmark_eval_2500.json"
DEFAULT_T2S_CASE_LEVEL_OUTPUTS = REPO_ROOT / "evals" / "reports" / "t2s_native_main_benchmark_outputs_2500"
DEFAULT_VERBALTS_REPORT = REPO_ROOT / "evals" / "reports" / "verbalts_tsfragment_eval_2500.json"
DEFAULT_CASE_FILES = [
    REPO_ROOT / "data" / "controlled_gold_cases" / "single_claim_cases.json",
    REPO_ROOT / "data" / "controlled_gold_cases" / "multi_claim_cases.json",
    REPO_ROOT / "data" / "controlled_gold_cases" / "robustness_cases.json",
]

PUBLIC_TEXT_BASELINES = [
    {
        "baseline_id": "t2s_text_to_series",
        "display_name": "T2S / Text-to-Series",
        "public_artifact": "https://github.com/WinfredGe/T2S",
        "paper": "https://arxiv.org/abs/2505.02417",
        "comparison_role": "public text-conditioned generator",
        "required_adapter": "Convert VTS gold descriptions to the T2S prompt/data interface and save one generated series per case.",
        "setup_requirements": [
            "torch==2.3.1",
            "TSFragment/preprocessed data under ./Data",
            "released LA-VAE and T2S-DiT checkpoints",
        ],
        "run_blocker": "Completed for the shared TSFragment-Eval 2500-case benchmark when evals/reports/t2s_native_main_benchmark_eval_2500.json is present; arbitrary non-TSFragment VTS prompts still require a prompt-to-embedding/data adapter.",
        "prompt_support_note": "The completed comparison uses TSFragment captions and text embeddings exported to evals/reports/t2s_native_case_inputs; arbitrary VTS-gold prompts still need an embedding adapter.",
        "status": "case_level_2500_completed" if DEFAULT_T2S_CASE_LEVEL_EVAL.exists() else "native_run_completed_semantic_adapter_required",
    },
    {
        "baseline_id": "verbalts",
        "display_name": "VerbalTS",
        "public_artifact": "https://github.com/seqml/VerbalTS/tree/main",
        "paper": "https://seqml.github.io/VerbalTS/",
        "comparison_role": "public text-conditioned generator with editing support",
        "required_adapter": "Run base and edited prompts through VerbalTS and save paired outputs for locality scoring.",
        "setup_requirements": [
            "torch==2.2.1",
            "dataset folder from the VerbalTS Google Drive release",
            "LongCLIP weights",
            "VerbalTS checkpoints",
            "GPU by default",
        ],
        "run_blocker": "Completed for the shared TSFragment-Eval 2500-case benchmark when evals/reports/verbalts_tsfragment_eval_2500.json is present.",
        "prompt_support_note": "The reported VerbalTS comparison uses TSFragment-compatible case-level outputs and the same 2500 case identifiers as the other methods.",
        "status": "case_level_2500_completed" if DEFAULT_VERBALTS_REPORT.exists() else "runner_ready_artifacts_required",
    },
    {
        "baseline_id": "unstructured_nl_conditioned_ts",
        "display_name": "Unstructured-NL-conditioned TS generator",
        "public_artifact": "literature_or_dataset_check_required",
        "paper": "https://arxiv.org/abs/2506.22927",
        "comparison_role": "related public text-series data or method, depending on released artifacts",
        "required_adapter": "Use only if code or generated outputs are available; otherwise cite as related work, not a main baseline.",
        "setup_requirements": ["released code or generated outputs"],
        "run_blocker": "Treat as related work until public runnable artifacts are confirmed.",
        "prompt_support_note": "Unknown until runnable artifacts are confirmed.",
        "status": "artifact_check_required",
    },
]


def _t2s_case_level_completed() -> bool:
    if not DEFAULT_T2S_CASE_LEVEL_EVAL.exists():
        return False
    try:
        payload = json.loads(DEFAULT_T2S_CASE_LEVEL_EVAL.read_text(encoding="utf-8"))
    except Exception:
        return False
    aggregate = payload.get("aggregate", {})
    return (
        aggregate.get("status") == "completed"
        and int(aggregate.get("successful_cases", 0)) == 2500
        and int(aggregate.get("cases", 0)) == 2500
    )


def _verbalts_native_completed() -> bool:
    if not DEFAULT_VERBALTS_REPORT.exists():
        return False
    try:
        payload = json.loads(DEFAULT_VERBALTS_REPORT.read_text(encoding="utf-8"))
    except Exception:
        return False
    aggregate = payload.get("aggregate", {})
    return (
        aggregate.get("status") == "completed"
        and int(aggregate.get("successful_cases", 0)) == 2500
        and int(aggregate.get("cases", 0)) == 2500
    )


def _public_text_baselines() -> list[dict[str, Any]]:
    baselines = [dict(item) for item in PUBLIC_TEXT_BASELINES]
    if _t2s_case_level_completed():
        for item in baselines:
            if item.get("baseline_id") == "t2s_text_to_series":
                item["status"] = "case_level_2500_completed"
                item["completed_artifacts"] = {
                    "case_file": "data/tsfragment_eval/tsfragment_eval_2500_cases.json",
                    "eval_json": "evals/reports/t2s_native_main_benchmark_eval_2500.json",
                    "outputs": "evals/reports/t2s_native_main_benchmark_outputs_2500",
                }
    if DEFAULT_VERBALTS_REPORT.exists():
        try:
            report = json.loads(DEFAULT_VERBALTS_REPORT.read_text(encoding="utf-8"))
        except Exception:
            report = {}
        for item in baselines:
            if item.get("baseline_id") == "verbalts":
                item["status"] = "case_level_2500_completed"
                item["completed_artifacts"] = {
                    "eval_json": "evals/reports/verbalts_tsfragment_eval_2500.json",
                    "outputs": "evals/reports/verbalts_main_benchmark_outputs_2500",
                    "reference_metrics": report.get("aggregate", {}),
                    "shared_evaluator_method": "verbalts",
                }
    return baselines


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _series_scale(series: np.ndarray) -> float:
    spread = float(np.nanmax(series) - np.nanmin(series)) if series.size else 0.0
    std = float(np.nanstd(series)) if series.size else 0.0
    return max(spread, std, 1e-6)


def _locality_metrics(base: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if base.shape != edited.shape or base.shape != mask.shape:
        raise ValueError("base, edited, and mask must have matching one-dimensional shapes.")
    outside = ~mask
    scale = _series_scale(base)
    inside_change = float(np.mean(np.abs(edited[mask] - base[mask])) / scale) if np.any(mask) else 0.0
    outside_drift = float(np.mean(np.abs(edited[outside] - base[outside])) / scale) if np.any(outside) else 0.0
    locality_score = inside_change / (1.0 + outside_drift)
    return {
        "inside_change": inside_change,
        "outside_drift": outside_drift,
        "locality_score": locality_score,
    }


def _edit_cases(length: int = 100) -> list[dict[str, Any]]:
    x = np.linspace(0.0, 1.0, length)
    base = 2.0 + 0.25 * x + 0.08 * np.sin(2.0 * np.pi * 3.0 * x)
    cases: list[dict[str, Any]] = []

    def add_case(case_id: str, edit_type: str, start: int, end: int, delta: np.ndarray) -> None:
        mask = np.zeros(length, dtype=bool)
        mask[start : end + 1] = True
        edited = base.copy()
        edited[mask] = edited[mask] + delta[: int(np.sum(mask))]
        cases.append(
            {
                "case_id": case_id,
                "edit_type": edit_type,
                "base": base.tolist(),
                "edited": edited.tolist(),
                "mask": mask.tolist(),
            }
        )

    add_case("edit_level_window", "local_level_shift", 35, 45, np.full(11, 0.45))
    add_case("edit_spike_position", "single_event_insert", 62, 62, np.array([0.9]))
    add_case("edit_noise_window", "local_noise_increase", 70, 82, np.linspace(-0.12, 0.12, 13))
    add_case("edit_trend_segment", "local_slope_change", 15, 35, np.linspace(0.0, 0.5, 21))
    return cases


def run_edit_locality_benchmark() -> dict[str, Any]:
    cases = _edit_cases()
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(17)
    for case in cases:
        base = np.asarray(case["base"], dtype=float)
        ideal = np.asarray(case["edited"], dtype=float)
        mask = np.asarray(case["mask"], dtype=bool)
        global_rerun = ideal + rng.normal(0.0, 0.025, size=ideal.shape) + 0.04 * np.sin(np.linspace(0.0, 8.0, ideal.size))
        for method, edited in [
            ("local_program_edit_oracle", ideal),
            ("global_rerun_control", global_rerun),
        ]:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "edit_type": case["edit_type"],
                    "method": method,
                    **_locality_metrics(base, edited, mask),
                }
            )

    aggregates = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        aggregates.append(
            {
                "method": method,
                "cases": len(method_rows),
                "mean_inside_change": float(np.mean([row["inside_change"] for row in method_rows])),
                "mean_outside_drift": float(np.mean([row["outside_drift"] for row in method_rows])),
                "mean_locality_score": float(np.mean([row["locality_score"] for row in method_rows])),
            }
        )
    return {
        "version": "1.0",
        "status": "internal_oracle_benchmark",
        "gate": "external text-conditioned generator outputs are required before claiming public-baseline edit locality",
        "rows": rows,
        "aggregate": aggregates,
    }


def run_cost_latency_benchmark(
    *,
    methods: list[str],
    case_paths: list[Path] | None = None,
    category: str | None = "atomic",
    output_root: Path | None = None,
    repeats: int = 3,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for method in methods:
        timings = []
        status = "completed"
        error = None
        cases = 0
        for _ in range(repeats):
            start = time.perf_counter()
            try:
                payload = run_methods(
                    methods=[method],
                    paths=case_paths,
                    category=category,
                    output_root=output_root,
                )
                elapsed = time.perf_counter() - start
                result = payload["results"][0]
                if result.get("status") == "skipped":
                    status = "skipped"
                    error = result.get("skip_reason")
                    break
                cases = int(result.get("aggregate", {}).get("cases", 0))
                timings.append(elapsed)
            except Exception as exc:  # pragma: no cover - exercised by CLI users with missing artifacts.
                status = "failed"
                error = str(exc)
                break
        sec_per_case = None
        if timings and cases > 0:
            sec_per_case = float(np.mean(timings) / cases)
        records.append(
            {
                "method": method,
                "status": status,
                "cases": cases,
                "repeats": len(timings),
                "mean_seconds": float(np.mean(timings)) if timings else None,
                "seconds_per_case": sec_per_case,
                "token_cost_usd": None,
                "token_accounting_status": "not_applicable_or_not_logged",
                "error": error,
            }
        )
    return {
        "version": "1.0",
        "status": "latency_measured_token_cost_pending",
        "category": category,
        "records": records,
    }


def _lag1_autocorr(series: np.ndarray) -> float | None:
    if series.size < 2:
        return None
    left = series[:-1]
    right = series[1:]
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _realism_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, float | None]:
    if reference.shape != generated.shape:
        raise ValueError("reference and generated series must have the same shape.")
    scale = _series_scale(reference)
    diff = generated - reference
    ref_acf = _lag1_autocorr(reference)
    gen_acf = _lag1_autocorr(generated)
    return {
        "normalized_rmse": float(np.sqrt(np.mean(diff * diff)) / scale),
        "normalized_mae": float(np.mean(np.abs(diff)) / scale),
        "mean_error": float(abs(np.mean(generated) - np.mean(reference)) / scale),
        "std_error": float(abs(np.std(generated) - np.std(reference)) / scale),
        "range_error": float(abs((np.max(generated) - np.min(generated)) - (np.max(reference) - np.min(reference))) / scale),
        "lag1_autocorr_error": None if ref_acf is None or gen_acf is None else float(abs(gen_acf - ref_acf)),
    }


def run_realism_smoke(source_path: Path = DEFAULT_REALISM_PATH) -> dict[str, Any]:
    if not source_path.exists():
        return {"version": "1.0", "status": "skipped", "reason": f"missing source file: {source_path}", "rows": []}
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for sample in payload.get("samples", []):
        if "true_series" not in sample or "generated_series" not in sample:
            continue
        reference = np.asarray(sample["true_series"], dtype=float)
        generated = np.asarray(sample["generated_series"], dtype=float)
        if reference.ndim != 1 or generated.ndim != 1 or reference.shape != generated.shape:
            continue
        rows.append(
            {
                "case_id": str(sample.get("index")),
                "description": sample.get("description", ""),
                **_realism_metrics(reference, generated),
            }
        )
    numeric_fields = [
        "normalized_rmse",
        "normalized_mae",
        "mean_error",
        "std_error",
        "range_error",
        "lag1_autocorr_error",
    ]
    aggregate = {
        field: (
            float(np.mean([row[field] for row in rows if row[field] is not None]))
            if any(row[field] is not None for row in rows)
            else None
        )
        for field in numeric_fields
    }
    return {
        "version": "1.0",
        "status": "completed" if rows else "skipped",
        "source_path": str(source_path),
        "cases": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "gate": "realism smoke test only; do not claim distributional realism without public generator and TSTR/discriminative tests",
    }


def _extract_plan_signature(plan: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(step.get("primitive")) for step in plan.get("steps", []) if step.get("primitive"))


def run_multimodel_stability(model_dirs: dict[str, Path] | None = None) -> dict[str, Any]:
    if not model_dirs:
        return {
            "version": "1.0",
            "status": "protocol_only",
            "required_artifacts": "one JSON plan per case per model, named <case_id>.json",
            "metrics": [
                "plan_presence_rate",
                "primitive_signature_agreement",
                "unresolved_semantics_rate",
                "needs_library_extension_rate",
            ],
        }

    per_model: dict[str, dict[str, Any]] = {}
    all_case_ids: set[str] = set()
    signatures: dict[str, dict[str, tuple[str, ...]]] = {}
    for model_name, model_dir in model_dirs.items():
        files = sorted(model_dir.glob("*.json"))
        rows = []
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            plan = payload.get("plan", payload.get("structured_plan", payload))
            if not isinstance(plan, dict):
                continue
            case_id = path.stem
            all_case_ids.add(case_id)
            signatures.setdefault(case_id, {})[model_name] = _extract_plan_signature(plan)
            rows.append(
                {
                    "case_id": case_id,
                    "primitive_signature": list(_extract_plan_signature(plan)),
                    "unresolved_semantics": len(plan.get("unresolved_semantics", [])),
                    "needs_library_extension": bool(plan.get("needs_library_extension")),
                }
            )
        per_model[model_name] = {"plan_dir": str(model_dir), "cases": len(rows), "rows": rows}

    agreement_values = []
    for case_id in sorted(all_case_ids):
        case_signatures = list(signatures.get(case_id, {}).values())
        if len(case_signatures) < 2:
            continue
        first = case_signatures[0]
        agreement_values.append(sum(1.0 for item in case_signatures if item == first) / len(case_signatures))

    return {
        "version": "1.0",
        "status": "completed",
        "models": per_model,
        "cases": len(all_case_ids),
        "primitive_signature_agreement": float(np.mean(agreement_values)) if agreement_values else None,
    }


def _load_public_generated_series(path: Path, length: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_series = payload.get("generated_series", payload.get("series", payload.get("values")))
            metadata = {
                key: payload.get(key)
                for key in ["seconds", "seconds_per_case", "token_cost_usd", "model", "seed"]
                if key in payload
            }
        else:
            raw_series = payload
    else:
        raw_series = parse_direct_array_output(path.read_text(encoding="utf-8"), length).tolist()
    if raw_series is None:
        return None, metadata
    series = np.asarray(raw_series, dtype=float)
    if series.ndim != 1 or len(series) != length:
        raise ValueError(f"Expected one-dimensional series of length {length}, got shape {series.shape}.")
    return series, metadata


def run_public_generator_comparison(
    *,
    output_root: Path | None = None,
    case_paths: list[Path] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    baselines = _public_text_baselines()
    if output_root is None:
        if _t2s_case_level_completed() and _verbalts_native_completed():
            status = "t2s_and_verbalts_case_level_2500_completed"
        elif _t2s_case_level_completed():
            status = "t2s_case_level_2500_completed_other_external_runs_pending"
        else:
            status = "comparison_protocol_ready_external_runs_pending"
        return {
            "version": "1.0",
            "status": status,
            "baselines": baselines,
            "output_format": {
                "root": "<output_root>/<baseline_id>/<case_id>.json",
                "json_fields": ["generated_series", "series", "values", "seconds", "token_cost_usd", "model", "seed"],
                "txt_format": "a direct numeric array accepted by evals.baselines.llm_direct.parse_direct_array_output",
            },
            "gate": (
                "T2S and VerbalTS are completed on the shared 2500-case TSFragment-Eval benchmark. Broad "
                "public-generator superiority beyond this shared split still requires additional generators "
                "or settings."
                if _t2s_case_level_completed()
                else "At least one public text-conditioned generator must be run before claiming comparison against public generators."
            ),
            "results": [],
        }

    cases = load_gold_cases(case_paths or DEFAULT_CASE_FILES)
    if category is not None:
        cases = [case for case in cases if case.get("category") == category]

    results: list[dict[str, Any]] = []
    for baseline in baselines:
        baseline_id = str(baseline["baseline_id"])
        baseline_dir = output_root / baseline_id
        if not baseline_dir.exists():
            results.append(
                {
                    "baseline_id": baseline_id,
                    "status": "missing_output_dir",
                    "output_dir": str(baseline_dir),
                    "aggregate": aggregate_results([]),
                    "cases": [],
                }
            )
            continue
        evaluated = []
        core_results = []
        for case in cases:
            case_id = str(case["case_id"])
            candidate_paths = [baseline_dir / f"{case_id}.json", baseline_dir / f"{case_id}.txt"]
            output_path = next((path for path in candidate_paths if path.exists()), None)
            if output_path is None:
                evaluated.append({"case_id": case_id, "status": "missing_output"})
                continue
            try:
                series, metadata = _load_public_generated_series(output_path, int(case["length"]))
                spec = semantic_spec_from_gold_case(case)
                plan = invalid_direct_output_plan(case, baseline_id, [])
                result = evaluate_case(
                    semantic_spec=spec,
                    plan=plan,
                    series=series,
                    execution_success=series is not None,
                )
                core_results.append(result)
                evaluated.append(
                    {
                        "case_id": case_id,
                        "status": "completed",
                        "csr": result.constraint_satisfaction_rate,
                        "semantic_alignment_score": result.semantic_alignment_score,
                        "metadata": metadata,
                    }
                )
            except Exception as exc:
                evaluated.append({"case_id": case_id, "status": "failed", "error": str(exc)})
        completed = [item for item in evaluated if item.get("status") == "completed"]
        results.append(
            {
                "baseline_id": baseline_id,
                "status": "completed" if completed else "no_completed_outputs",
                "output_dir": str(baseline_dir),
                "aggregate": aggregate_results(core_results),
                "cases": evaluated,
            }
        )

    completed_baselines = {item["baseline_id"] for item in results if item.get("status") == "completed"}
    for baseline in baselines:
        if baseline["baseline_id"] in completed_baselines:
            baseline["status"] = "completed"
    return {
        "version": "1.0",
        "status": "completed" if completed_baselines else "external_runs_pending",
        "baselines": baselines,
        "output_root": str(output_root),
        "gate": "Supported public-baseline comparison requires at least one completed public text-conditioned generator run.",
        "results": results,
    }


def export_public_generator_inputs(
    *,
    output_dir: Path,
    case_paths: list[Path] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    cases = load_gold_cases(case_paths or DEFAULT_CASE_FILES)
    if category is not None:
        cases = [case for case in cases if case.get("category") == category]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for baseline in PUBLIC_TEXT_BASELINES:
        baseline_id = str(baseline["baseline_id"])
        prompt_dir = output_dir / baseline_id / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            case_id = str(case["case_id"])
            prompt_path = prompt_dir / f"{case_id}.txt"
            prompt_text = (
                f"Generate a one-dimensional time series with length {int(case['length'])} from this description.\n"
                f"Description: {case['description']}\n"
                "Return only the generated numeric series when possible."
            )
            prompt_path.write_text(prompt_text, encoding="utf-8")
            manifest_records.append(
                {
                    "baseline_id": baseline_id,
                    "case_id": case_id,
                    "category": case.get("category"),
                    "length": int(case["length"]),
                    "prompt_path": str(prompt_path),
                    "expected_output_path": str(output_dir / baseline_id / "outputs" / f"{case_id}.json"),
                }
            )
    manifest = {
        "version": "1.0",
        "cases": len(cases),
        "baselines": [item["baseline_id"] for item in PUBLIC_TEXT_BASELINES],
        "records": manifest_records,
    }
    _write_json(output_dir / "public_generator_input_manifest.json", manifest)
    return {
        "version": "1.0",
        "status": "completed",
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "public_generator_input_manifest.json"),
        "records": len(manifest_records),
    }


def build_public_baseline_matrix(output_root: Path | None = None, case_paths: list[Path] | None = None, category: str | None = None) -> dict[str, Any]:
    if output_root is not None:
        return run_public_generator_comparison(output_root=output_root, case_paths=case_paths, category=category)
    if _t2s_case_level_completed() and _verbalts_native_completed():
        status = "t2s_and_verbalts_case_level_2500_completed"
    elif _t2s_case_level_completed():
        status = "t2s_case_level_2500_completed_other_external_runs_pending"
    else:
        status = "comparison_protocol_ready_external_runs_pending"
    return {
        "version": "1.0",
        "status": status,
        "baselines": _public_text_baselines(),
        "output_format": {
            "root": "<output_root>/<baseline_id>/<case_id>.json",
            "json_fields": ["generated_series", "series", "values", "seconds", "token_cost_usd", "model", "seed"],
            "txt_format": "a direct numeric array accepted by evals.baselines.llm_direct.parse_direct_array_output",
        },
        "gate": (
            "T2S and VerbalTS are completed on the shared 2500-case TSFragment-Eval benchmark. Broad "
            "public-generator superiority beyond this shared split still requires additional generators "
            "or settings."
            if _t2s_case_level_completed()
            else "At least one public text-conditioned generator must be run before claiming comparison against public generators."
        ),
    }


def run_revision_risk_experiments(
    *,
    report_dir: Path = DEFAULT_REPORT_DIR,
    case_paths: list[Path] | None = None,
    category: str | None = "atomic",
    latency_methods: list[str] | None = None,
    output_root: Path | None = None,
    public_output_root: Path | None = None,
    latency_repeats: int = 3,
    skip_latency: bool = False,
) -> dict[str, Any]:
    latency_methods = latency_methods or ["oracle_program", "rule_parser"]
    public_input_export = export_public_generator_inputs(
        output_dir=report_dir / "public_generator_inputs",
        case_paths=case_paths,
        category=category,
    )
    payload = {
        "version": "1.0",
        "public_text_generator_baselines": build_public_baseline_matrix(
            output_root=public_output_root,
            case_paths=case_paths,
            category=category,
        ),
        "public_generator_inputs": public_input_export,
        "edit_locality": run_edit_locality_benchmark(),
        "cost_latency": None
        if skip_latency
        else run_cost_latency_benchmark(
            methods=latency_methods,
            case_paths=case_paths,
            category=category,
            output_root=output_root,
            repeats=latency_repeats,
        ),
        "realism_smoke": run_realism_smoke(),
        "multi_model_stability": run_multimodel_stability(),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "revision_risk_report.json", payload)
    (report_dir / "revision_risk_report.md").write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Revision Risk Report", ""]
    public = payload["public_text_generator_baselines"]
    lines.extend(
        [
            "## Public Text-Conditioned Baselines",
            "",
            f"Status: {public['status']}",
            "",
            "| Baseline | Role | Status | Artifact |",
            "|---|---|---:|---|",
        ]
    )
    for item in public["baselines"]:
        lines.append(
            f"| {item['display_name']} | {item['comparison_role']} | {item['status']} | {item['public_artifact']} |"
        )
    lines.extend(["", "Setup blockers:", ""])
    for item in public["baselines"]:
        lines.append(f"- {item['display_name']}: {item.get('run_blocker', 'none')}")
        lines.append(f"  Prompt support: {item.get('prompt_support_note', 'not checked')}")
    if public.get("output_format"):
        fmt = public["output_format"]
        lines.extend(
            [
                "",
                "Expected output format:",
                f"- Root: `{fmt.get('root')}`",
                f"- JSON fields: `{', '.join(fmt.get('json_fields', []))}`",
                f"- TXT format: {fmt.get('txt_format')}",
            ]
        )
    if public.get("results"):
        lines.extend(["", "| Baseline ID | Status | Cases | CSR |", "|---|---:|---:|---:|"])
        for item in public["results"]:
            aggregate = item.get("aggregate", {})
            lines.append(
                f"| {item['baseline_id']} | {item['status']} | {int(aggregate.get('cases', 0))} | {_fmt(aggregate.get('csr'))} |"
            )
    public_inputs = payload.get("public_generator_inputs")
    if public_inputs:
        lines.extend(
            [
                "",
                "Input export:",
                f"- Status: {public_inputs.get('status')}",
                f"- Manifest: `{public_inputs.get('manifest_path')}`",
                f"- Prompt records: {public_inputs.get('records')}",
            ]
        )

    edit = payload["edit_locality"]
    lines.extend(["", "## Edit Locality", "", f"Status: {edit['status']}", "", "| Method | Cases | Inside Change | Outside Drift | Locality Score |", "|---|---:|---:|---:|---:|"])
    for item in edit["aggregate"]:
        lines.append(
            f"| {item['method']} | {item['cases']} | {_fmt(item['mean_inside_change'])} | {_fmt(item['mean_outside_drift'])} | {_fmt(item['mean_locality_score'])} |"
        )

    cost = payload.get("cost_latency")
    lines.extend(["", "## Cost and Latency", ""])
    if cost is None:
        lines.append("Status: skipped")
    else:
        lines.extend([f"Status: {cost['status']}", "", "| Method | Status | Cases | Seconds/Case | Token Cost |", "|---|---:|---:|---:|---:|"])
        for item in cost["records"]:
            lines.append(
                f"| {item['method']} | {item['status']} | {item['cases']} | {_fmt(item['seconds_per_case'])} | {item['token_cost_usd']} |"
            )

    realism = payload["realism_smoke"]
    lines.extend(["", "## Realism Smoke", "", f"Status: {realism['status']}", f"Cases: {realism.get('cases', 0)}", ""])
    if realism.get("aggregate"):
        aggregate = realism["aggregate"]
        lines.append(
            "Aggregate: "
            + ", ".join(f"{key}={_fmt(value)}" for key, value in aggregate.items())
        )

    stability = payload["multi_model_stability"]
    lines.extend(["", "## Multi-Model Stability", "", f"Status: {stability['status']}"])
    if stability["status"] == "protocol_only":
        lines.append(f"Required artifacts: {stability['required_artifacts']}")
    else:
        lines.append(f"Primitive signature agreement: {_fmt(stability.get('primitive_signature_agreement'))}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run revision-risk experiments for public baselines, edit locality, cost, realism, and stability.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cases", nargs="*", type=Path, default=None)
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial"], default="atomic")
    parser.add_argument("--latency-methods", nargs="+", default=["oracle_program", "rule_parser"])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--public-output-root", type=Path, default=None)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_revision_risk_experiments(
        report_dir=args.report_dir,
        case_paths=args.cases,
        category=args.category,
        latency_methods=args.latency_methods,
        output_root=args.output_root,
        public_output_root=args.public_output_root,
        latency_repeats=args.latency_repeats,
        skip_latency=args.skip_latency,
    )
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        print(f"Wrote revision risk reports to {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
