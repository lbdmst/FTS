from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.series_evaluation import compute_error_metrics
from evals.run_llm_baseline_outputs import DEFAULT_T2S_COMPARISON_JSON
from faithts.gold_adapter import load_gold_cases


T2S_ROOT = REPO_ROOT / "baselines" / "T2S"
T2S_TSFRAGMENT_ROOT = T2S_ROOT / "Data" / "TSFragment-600K"
if not T2S_TSFRAGMENT_ROOT.exists():
    T2S_TSFRAGMENT_ROOT = T2S_ROOT / "Data" / "Three Levels Data" / "TSFragment-600K"
T2S_MINI_DATA_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_native_case_inputs"
DEFAULT_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_native_main_benchmark_outputs_2500"
DEFAULT_EVAL_JSON = REPO_ROOT / "evals" / "reports" / "t2s_native_main_benchmark_eval_2500.json"
DEFAULT_EVAL_MD = REPO_ROOT / "evals" / "reports" / "t2s_native_main_benchmark_eval_2500.md"

T2S_NATIVE_CONFIGS: dict[tuple[str, int], dict[str, Any]] = {
    ("ETTh1", 24): {"cfg_scale": 9.0, "total_step": 10},
    ("ETTh1", 48): {"cfg_scale": 9.0, "total_step": 10},
    ("ETTh1", 96): {"cfg_scale": 9.0, "total_step": 10},
    ("electricity", 24): {"cfg_scale": 5.0, "total_step": 60},
    ("electricity", 48): {"cfg_scale": 5.0, "total_step": 10},
    ("electricity", 96): {"cfg_scale": 13.0, "total_step": 30},
    ("exchangerate", 24): {"cfg_scale": 7.0, "total_step": 100},
    ("exchangerate", 48): {"cfg_scale": 12.0, "total_step": 60},
    ("exchangerate", 96): {"cfg_scale": 5.0, "total_step": 100},
    ("traffic", 24): {"cfg_scale": 5.0, "total_step": 100},
    ("traffic", 48): {"cfg_scale": 5.0, "total_step": 10},
    ("traffic", 96): {"cfg_scale": 5.0, "total_step": 30},
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _group_cases(cases: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for case in cases:
        key = (str(case["source_dataset"]), int(case["source_length"]))
        grouped.setdefault(key, []).append(case)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda case: int(case["source_row"]))
    return grouped


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _select_rows(frame: pd.DataFrame, group_cases: list[dict[str, Any]]) -> pd.DataFrame:
    row_ids = [int(case["source_row"]) for case in group_cases]
    if row_ids and max(row_ids) < len(frame):
        return frame.iloc[row_ids].copy()

    if "Text" not in frame.columns:
        raise IndexError("source_row is out of bounds and source frame does not contain `Text` for fallback matching.")

    by_text = {_normalize_text(value): idx for idx, value in enumerate(frame["Text"].tolist())}
    matched_indices = []
    for case in group_cases:
        key = _normalize_text(case["description"])
        if key not in by_text:
            raise KeyError(f"Could not recover row for case `{case['case_id']}` by caption match.")
        matched_indices.append(by_text[key])
    return frame.iloc[matched_indices].copy()


def export_case_csvs(*, case_file: Path = DEFAULT_CASE_FILE, output_root: Path = T2S_MINI_DATA_ROOT) -> dict[str, Any]:
    cases = load_gold_cases([case_file])
    grouped = _group_cases(cases)
    records = []
    output_root.mkdir(parents=True, exist_ok=True)
    for (dataset, length), group_cases in sorted(grouped.items()):
        source_csv = T2S_TSFRAGMENT_ROOT / f"embedding_cleaned_{dataset}_{length}.csv"
        frame = pd.read_csv(source_csv)
        selected = _select_rows(frame, group_cases)
        output_csv = output_root / f"embedding_cleaned_{dataset}_{length}.csv"
        selected.to_csv(output_csv, index=False)
        records.append(
            {
                "dataset": dataset,
                "length": length,
                "rows": len(group_cases),
                "source_csv": _display_path(source_csv),
                "output_csv": _display_path(output_csv),
                "case_ids": [str(case["case_id"]) for case in group_cases],
            }
        )
    return {
        "status": "completed",
        "case_file": _display_path(case_file),
        "output_root": _display_path(output_root),
        "records": records,
    }


def _scaler_for(dataset: str, length: int) -> MinMaxScaler:
    source_csv = T2S_TSFRAGMENT_ROOT / f"embedding_cleaned_{dataset}_{length}.csv"
    frame = pd.read_csv(source_csv, usecols=["OT"])
    series_matrix = np.asarray([ast.literal_eval(item) for item in frame["OT"]], dtype=float)
    return MinMaxScaler().fit(series_matrix)


@contextmanager
def _in_t2s_root() -> Any:
    previous = Path.cwd()
    try:
        os.chdir(T2S_ROOT)
        if str(T2S_ROOT) not in sys.path:
            sys.path.insert(0, str(T2S_ROOT))
        yield
    finally:
        os.chdir(previous)


def _build_args(dataset: str, length: int, output_dir: Path) -> SimpleNamespace:
    import torch

    config = T2S_NATIVE_CONFIGS[(dataset, length)]
    args = SimpleNamespace(
        batch_size=5,
        save_path="./results/denoiser_results",
        usepretrainedvae=True,
        backbone="flowmatching",
        denoiser="DiT",
        cfg_scale=float(config["cfg_scale"]),
        total_step=int(config["total_step"]),
        checkpoint_id=19999,
        dataset_name=f"{dataset}_{length}",
        run_multi=False,
    )
    args.mix_train = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.checkpoint_path = os.path.join(
        args.save_path,
        "checkpoints",
        f"{args.backbone}_{args.denoiser}_{dataset}",
        f"model_{args.checkpoint_id}.pth",
    )
    args.generation_save_path = os.path.join(
        args.save_path,
        "generation",
        f"{args.backbone}_{args.denoiser}_{args.dataset_name}_{args.cfg_scale}_{args.total_step}",
    )
    args.generation_save_path_result = str(output_dir)
    return args


def _artifact_status(dataset: str, args: SimpleNamespace) -> dict[str, Any]:
    pretrained = T2S_ROOT / "results" / "saved_pretrained_models" / f"dataset{dataset}_epoch2000" / "final_model.pth"
    checkpoint = T2S_ROOT / args.checkpoint_path.replace("./", "", 1)
    return {
        "pretrained_lavae": str(pretrained),
        "pretrained_exists": pretrained.exists(),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
    }


def _custom_loader_provider(dataset_name: str, batch_size: int, data_root: Path):
    from datafactory.dataset import T2SDataset
    from torch.utils.data import DataLoader

    def loader_provider(_args: Any, period: str):  # pragma: no cover - exercised by integration run.
        dataset = T2SDataset(
            name=f"embedding_cleaned_{dataset_name}",
            data_root=str(data_root),
            period="test",
            proportion=0.0,
            seed=123,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return dataset, dataloader

    return loader_provider


def run_native_generation(
    *,
    case_file: Path = DEFAULT_CASE_FILE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    skip_existing: bool = True,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    export_summary = export_case_csvs(case_file=case_file, output_root=T2S_MINI_DATA_ROOT)
    cases = load_gold_cases([case_file])
    grouped = _group_cases(cases)
    run_records = []

    with _in_t2s_root():
        import infer as t2s_infer  # pragma: no cover - integration-only import.

        for (dataset, length), group_cases in sorted(grouped.items()):
            dataset_output_dir = output_root / f"{dataset}_{length}"
            args = _build_args(dataset, length, dataset_output_dir)
            artifact_info = _artifact_status(dataset, args)
            if not artifact_info["pretrained_exists"] or not artifact_info["checkpoint_exists"]:
                run_records.append(
                    {
                        "dataset": dataset,
                        "length": length,
                        "status": "missing_artifact",
                        "artifacts": artifact_info,
                    }
                )
                continue
            if skip_existing and (dataset_output_dir / "x_t.npy").exists() and (dataset_output_dir / "x_1.npy").exists():
                run_records.append(
                    {
                        "dataset": dataset,
                        "length": length,
                        "status": "skipped_existing",
                        "output_dir": _display_path(dataset_output_dir),
                        "artifacts": artifact_info,
                    }
                )
                continue

            original_loader_provider = t2s_infer.loader_provider
            original_torch_load = t2s_infer.torch.load

            def _torch_load_legacy_checkpoint(*load_args: Any, **load_kwargs: Any) -> Any:
                load_kwargs.setdefault("weights_only", False)
                if not t2s_infer.torch.cuda.is_available():
                    load_kwargs.setdefault("map_location", t2s_infer.torch.device("cpu"))
                return original_torch_load(*load_args, **load_kwargs)

            try:
                t2s_infer.loader_provider = _custom_loader_provider(
                    f"{dataset}_{length}",
                    batch_size=args.batch_size,
                    data_root=T2S_MINI_DATA_ROOT,
                )
                t2s_infer.torch.load = _torch_load_legacy_checkpoint
                x_true, x_generated, *_ = t2s_infer.infer(args)
            finally:
                t2s_infer.loader_provider = original_loader_provider
                t2s_infer.torch.load = original_torch_load
            run_records.append(
                {
                    "dataset": dataset,
                    "length": length,
                    "status": "completed",
                    "output_dir": _display_path(dataset_output_dir),
                    "artifacts": artifact_info,
                    "generated_shape": list(x_generated.shape),
                    "reference_shape": list(x_true.shape),
                    "cases": len(group_cases),
                }
            )

    completed = sum(1 for record in run_records if record["status"] in {"completed", "skipped_existing"})
    return {
        "status": "completed" if completed == len(run_records) else "partial",
        "case_file": _display_path(case_file),
        "output_root": _display_path(output_root),
        "export_summary": export_summary,
        "runs": run_records,
    }


def evaluate_native_generation(
    *,
    case_file: Path = DEFAULT_CASE_FILE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    cases = load_gold_cases([case_file])
    grouped = _group_cases(cases)
    records: list[dict[str, Any]] = []
    missing_groups: list[dict[str, Any]] = []

    for (dataset, length), group_cases in sorted(grouped.items()):
        dataset_output_dir = output_root / f"{dataset}_{length}"
        true_path = dataset_output_dir / "x_1.npy"
        generated_path = dataset_output_dir / "x_t.npy"
        if not true_path.exists() or not generated_path.exists():
            missing_groups.append(
                {
                    "dataset": dataset,
                    "length": length,
                    "status": "missing_output",
                    "output_dir": _display_path(dataset_output_dir),
                }
            )
            continue

        scaler = _scaler_for(dataset, length)
        true_all = scaler.inverse_transform(np.load(true_path).squeeze(-1))
        generated_all = scaler.inverse_transform(np.load(generated_path).squeeze(-1))
        if len(true_all) != len(group_cases) or len(generated_all) != len(group_cases):
            missing_groups.append(
                {
                    "dataset": dataset,
                    "length": length,
                    "status": "shape_mismatch",
                    "output_dir": _display_path(dataset_output_dir),
                    "expected_cases": len(group_cases),
                    "reference_rows": int(len(true_all)),
                    "generated_rows": int(len(generated_all)),
                }
            )
            continue

        for index, case in enumerate(group_cases):
            reference = np.asarray(case["reference_series"], dtype=float)
            generated = np.asarray(generated_all[index], dtype=float)
            reference_from_t2s = np.asarray(true_all[index], dtype=float)
            if reference.shape != generated.shape:
                raise ValueError(f"{case['case_id']}: generated shape {generated.shape} != reference shape {reference.shape}")
            align_error = float(np.sqrt(np.mean(np.square(reference_from_t2s - reference))))
            records.append(
                {
                    "case_id": str(case["case_id"]),
                    "source_dataset": dataset,
                    "source_length": length,
                    "source_row": int(case["source_row"]),
                    "status": "success",
                    "output_dir": _display_path(dataset_output_dir),
                    "metrics": compute_error_metrics(generated, reference),
                    "reference_alignment_rmse": align_error,
                }
            )

    success = [record for record in records if record["status"] == "success"]
    metrics = [record["metrics"] for record in success]
    aggregate = {
        "method": "T2S native",
        "cases": len(success),
        "successful_cases": len(success),
        "success_rate": float(len(success) / len(cases)) if cases else 0.0,
        "mean_wape": float(np.mean([item["wape"] for item in metrics])) if metrics else None,
        "median_wape": float(np.median([item["wape"] for item in metrics])) if metrics else None,
        "mean_corr": float(np.mean([item["correlation"] for item in metrics if item["correlation"] is not None]))
        if any(item["correlation"] is not None for item in metrics)
        else None,
        "median_corr": float(np.median([item["correlation"] for item in metrics if item["correlation"] is not None]))
        if any(item["correlation"] is not None for item in metrics)
        else None,
        "mean_mrr": float(np.mean([item["mrr"] for item in metrics])) if metrics else None,
        "median_mrr": float(np.median([item["mrr"] for item in metrics])) if metrics else None,
        "mean_raw_mse": float(np.mean([item["mse"] for item in metrics])) if metrics else None,
        "median_raw_mse": float(np.median([item["mse"] for item in metrics])) if metrics else None,
        "status": "completed" if len(success) == len(cases) else "partial",
        "mean_reference_alignment_rmse": float(np.mean([record["reference_alignment_rmse"] for record in success]))
        if success
        else None,
    }
    by_dataset_length = []
    for (dataset, length), group_cases in sorted(grouped.items()):
        subset = [record for record in success if record["source_dataset"] == dataset and int(record["source_length"]) == length]
        subset_metrics = [record["metrics"] for record in subset]
        by_dataset_length.append(
            {
                "dataset": dataset,
                "length": length,
                "cases": len(group_cases),
                "successful_cases": len(subset),
                "success_rate": float(len(subset) / len(group_cases)) if group_cases else 0.0,
                "mean_wape": float(np.mean([item["wape"] for item in subset_metrics])) if subset_metrics else None,
                "median_wape": float(np.median([item["wape"] for item in subset_metrics])) if subset_metrics else None,
                "mean_corr": float(np.mean([item["correlation"] for item in subset_metrics if item["correlation"] is not None]))
                if any(item["correlation"] is not None for item in subset_metrics)
                else None,
                "median_corr": float(np.median([item["correlation"] for item in subset_metrics if item["correlation"] is not None]))
                if any(item["correlation"] is not None for item in subset_metrics)
                else None,
                "mean_mrr": float(np.mean([item["mrr"] for item in subset_metrics])) if subset_metrics else None,
                "median_mrr": float(np.median([item["mrr"] for item in subset_metrics])) if subset_metrics else None,
            }
        )
    return {
        "version": "1.0",
        "benchmark": "T2S TSFragment-600K main benchmark case-level native outputs",
        "case_file": _display_path(case_file),
        "output_root": _display_path(output_root),
        "aggregate": aggregate,
        "missing_groups": missing_groups,
        "records": records,
        "by_dataset_length": by_dataset_length,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# T2S Native Case-Level Benchmark",
        "",
        "| Method | Cases | Success Count | Success Rate | Mean WAPE | Median WAPE | Mean Corr | Median Corr | Mean MRR | Median MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                str(aggregate["method"]),
                str(aggregate["cases"]),
                f"{aggregate['successful_cases']}/{aggregate['cases']}",
                f"{aggregate['success_rate']:.3f}" if aggregate["success_rate"] is not None else "n/a",
                f"{aggregate['mean_wape']:.3f}" if aggregate["mean_wape"] is not None else "n/a",
                f"{aggregate['median_wape']:.3f}" if aggregate["median_wape"] is not None else "n/a",
                f"{aggregate['mean_corr']:.3f}" if aggregate["mean_corr"] is not None else "n/a",
                f"{aggregate['median_corr']:.3f}" if aggregate["median_corr"] is not None else "n/a",
                f"{aggregate['mean_mrr']:.3f}" if aggregate["mean_mrr"] is not None else "n/a",
                f"{aggregate['median_mrr']:.3f}" if aggregate["median_mrr"] is not None else "n/a",
            ]
        )
        + " |",
        "",
        "## By Dataset-Length",
        "",
        "| Dataset | Length | Cases | Success | Mean WAPE | Median WAPE | Mean Corr | Median Corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("by_dataset_length", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["length"]),
                    str(row["cases"]),
                    f"{row['successful_cases']}/{row['cases']}",
                    f"{row['mean_wape']:.3f}" if row["mean_wape"] is not None else "n/a",
                    f"{row['median_wape']:.3f}" if row["median_wape"] is not None else "n/a",
                    f"{row['mean_corr']:.3f}" if row["mean_corr"] is not None else "n/a",
                    f"{row['median_corr']:.3f}" if row["median_corr"] is not None else "n/a",
                ]
            )
            + " |"
        )
    if payload.get("missing_groups"):
        lines.extend(["", "## Missing Groups", ""])
        for row in payload["missing_groups"]:
            lines.append(f"- {row['dataset']}_{row['length']}: {row['status']} ({row['output_dir']})")
    return "\n".join(lines) + "\n"


def update_comparison_json(
    *,
    native_payload: dict[str, Any],
    comparison_path: Path = DEFAULT_T2S_COMPARISON_JSON,
) -> dict[str, Any]:
    comparison = _read_json(comparison_path) if comparison_path.exists() else {
        "version": "1.0",
        "benchmark": "T2S TSFragment-600K 2500-case shared benchmark",
        "overall": [],
        "artifacts": {},
    }
    comparison["benchmark"] = "T2S TSFragment-600K 2500-case shared benchmark"
    comparison["scope"] = {
        "datasets": ["ETTh1", "electricity", "exchangerate", "traffic"],
        "lengths": [24, 48, 96],
        "split": "stratified 2500-case TSFragment-Eval split",
        "native_case_file": _display_path(DEFAULT_CASE_FILE),
        "native_total_cases": native_payload["aggregate"]["cases"],
        "comparison_unit": "case-level caption-series pairs shared by T2S native, ours, and direct LLM baselines",
    }
    comparison["metric_policy"] = {
        "primary": ["WAPE (lower better)", "CORR/correlation (higher better)", "MRR (lower better)", "success_rate"],
        "not_directly_compared": ["Historical native full-dataset T2S summary metrics"],
        "reason": (
            "Main-table comparison uses the same 2500 TSFragment-Eval caption-series cases for T2S native, "
            "FaithTS, VerbalTS, and direct LLM baselines; older native setting summaries are excluded from "
            "the main comparison."
        ),
    }
    existing = []
    for row in comparison.get("overall", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("method")) == "T2S native":
            continue
        existing.append(row)
    aggregate = native_payload["aggregate"]
    native_row = {
        "method": "T2S native",
        "cases": aggregate["cases"],
        "success_rate": aggregate["success_rate"],
        "mean_wape": aggregate["mean_wape"],
        "median_wape": aggregate["median_wape"],
        "mean_corr": aggregate["mean_corr"],
        "median_corr": aggregate["median_corr"],
        "mean_mrr": aggregate["mean_mrr"],
        "median_mrr": aggregate["median_mrr"],
        "mean_raw_mse": aggregate["mean_raw_mse"],
        "median_raw_mse": aggregate["median_raw_mse"],
        "status": aggregate["status"],
        "successful_cases": aggregate["successful_cases"],
    }
    comparison["overall"] = [native_row] + existing
    comparison["by_dataset_length"] = native_payload.get("by_dataset_length", [])
    comparison.setdefault("artifacts", {}).pop("t2s_summary", None)
    comparison.setdefault("artifacts", {})["t2s_native_case_level_eval"] = _display_path(DEFAULT_EVAL_JSON)
    comparison.setdefault("artifacts", {})["t2s_native_case_level_outputs"] = _display_path(DEFAULT_OUTPUT_ROOT)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T2S native inference on the shared 2500-case TSFragment-Eval benchmark.")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--eval-md", type=Path, default=DEFAULT_EVAL_MD)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--update-comparison", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        payload = export_case_csvs(case_file=args.case_file, output_root=T2S_MINI_DATA_ROOT)
        if args.json:
            print(json.dumps(payload, indent=2, allow_nan=False))
        else:
            print(f"Exported benchmark CSVs to {payload['output_root']}")
        return 0
    if args.evaluate_only:
        eval_payload = evaluate_native_generation(case_file=args.case_file, output_root=args.output_root)
        _write_json(args.eval_json, eval_payload)
        _write_text(args.eval_md, render_markdown(eval_payload))
        if args.update_comparison:
            comparison = update_comparison_json(native_payload=eval_payload)
            _write_json(DEFAULT_T2S_COMPARISON_JSON, comparison)
        if args.json:
            print(json.dumps({"evaluation": eval_payload}, indent=2, allow_nan=False))
        else:
            print(f"Wrote native eval JSON to {_display_path(args.eval_json)}")
            print(f"Wrote native eval Markdown to {_display_path(args.eval_md)}")
            print(f"Status: {eval_payload['aggregate']['status']} ({eval_payload['aggregate']['successful_cases']}/{len(load_gold_cases([args.case_file]))} cases)")
        return 0

    generation_payload = run_native_generation(
        case_file=args.case_file,
        output_root=args.output_root,
        skip_existing=args.skip_existing,
    )
    eval_payload = evaluate_native_generation(case_file=args.case_file, output_root=args.output_root)
    _write_json(args.eval_json, eval_payload)
    _write_text(args.eval_md, render_markdown(eval_payload))
    if args.update_comparison:
        comparison = update_comparison_json(native_payload=eval_payload)
        _write_json(DEFAULT_T2S_COMPARISON_JSON, comparison)
    if args.json:
        print(json.dumps({"generation": generation_payload, "evaluation": eval_payload}, indent=2, allow_nan=False))
    else:
        print(f"Wrote native eval JSON to {_display_path(args.eval_json)}")
        print(f"Wrote native eval Markdown to {_display_path(args.eval_md)}")
        print(f"Status: {eval_payload['aggregate']['status']} ({eval_payload['aggregate']['successful_cases']}/{len(load_gold_cases([args.case_file]))} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
