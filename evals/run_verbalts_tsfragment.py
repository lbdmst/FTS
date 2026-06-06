from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.series_evaluation import compute_error_metrics

VERBALTS_ROOT = REPO_ROOT / "baselines" / "VerbalTS"
TSFRAGMENT_ROOT = REPO_ROOT / "baselines" / "T2S" / "Data" / "TSFragment-600K"
if not TSFRAGMENT_ROOT.exists():
    TSFRAGMENT_ROOT = REPO_ROOT / "baselines" / "T2S" / "Data" / "Three Levels Data" / "TSFragment-600K"
DEFAULT_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
DEFAULT_DATASET_ROOT = VERBALTS_ROOT / "datasets"
DEFAULT_CONFIG_ROOT = VERBALTS_ROOT / "configs"
DEFAULT_SAVE_ROOT = VERBALTS_ROOT / "save"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "verbalts_main_benchmark_outputs_2500"
DEFAULT_PLAN_PATH = REPO_ROOT / "evals" / "reports" / "verbalts_tsfragment_plan_2500.json"
DEFAULT_EVAL_JSON = REPO_ROOT / "evals" / "reports" / "verbalts_tsfragment_eval_2500.json"
DEFAULT_EVAL_MD = REPO_ROOT / "evals" / "reports" / "verbalts_tsfragment_eval_2500.md"
DEFAULT_COMPARISON_JSON = REPO_ROOT / "evals" / "reports" / "t2s_main_benchmark_comparison_2500.json"

SOURCE_DATASETS = ["ETTh1", "electricity", "exchangerate", "traffic"]
LENGTHS = [24, 48, 96]


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _series_from_ot(raw: Any) -> list[float]:
    parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    series = np.asarray(parsed, dtype=np.float32)
    if series.ndim != 1:
        raise ValueError(f"Expected one-dimensional OT field, got shape {series.shape}.")
    return [float(value) for value in series.tolist()]


def _dataset_name(length: int) -> str:
    return f"TSFragment_{length}"


def _dataset_path(dataset: str, length: int, tsfragment_root: Path = TSFRAGMENT_ROOT) -> Path:
    return tsfragment_root / f"embedding_cleaned_{dataset}_{length}.csv"


def _case_rows_by_dataset_length(cases: list[dict[str, Any]]) -> dict[tuple[str, int], set[int]]:
    rows: dict[tuple[str, int], set[int]] = {}
    for case in cases:
        dataset = str(case["source_dataset"])
        length = int(case["source_length"])
        source_row = int(case.get("source_row", case.get("source_index", -1)))
        rows.setdefault((dataset, length), set()).add(source_row)
    return rows


def _sample_rows(
    *,
    dataset: str,
    length: int,
    excluded_rows: set[int],
    train_per_dataset: int,
    valid_per_dataset: int,
    scan_limit: int,
    seed: int,
    tsfragment_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = _dataset_path(dataset, length, tsfragment_root)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["Text", "OT"], nrows=scan_limit)
    candidates: list[dict[str, Any]] = []
    for row_number, row in frame.iterrows():
        if int(row_number) in excluded_rows:
            continue
        try:
            series = _series_from_ot(row["OT"])
        except (SyntaxError, ValueError):
            continue
        if len(series) != length:
            continue
        candidates.append(
            {
                "source_dataset": dataset,
                "source_length": length,
                "source_row": int(row_number),
                "description": str(row["Text"]),
                "series": series,
            }
        )
    rng = random.Random(seed + length + sum(ord(ch) for ch in dataset))
    rng.shuffle(candidates)
    train_count = min(train_per_dataset, len(candidates))
    valid_count = min(valid_per_dataset, max(0, len(candidates) - train_count))
    train = candidates[:train_count]
    valid = candidates[train_count : train_count + valid_count]
    return train, valid


def _records_to_arrays(records: list[dict[str, Any]], length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.asarray([record["series"] for record in records], dtype=np.float32)
    if ts.ndim != 2 or ts.shape[1] != length:
        raise ValueError(f"Expected records to form [N, {length}] array, got {ts.shape}.")
    ts = ts[..., np.newaxis]
    caps = np.asarray([[str(record["description"])] for record in records])
    attrs = np.zeros((len(records), 1), dtype=np.int64)
    return ts, attrs, caps


def _write_verbalts_dataset(
    *,
    output_dir: Path,
    length: int,
    train_records: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in [("train", train_records), ("valid", valid_records), ("test", test_records)]:
        ts, attrs, caps = _records_to_arrays(records, length)
        np.save(output_dir / f"{split}_ts.npy", ts)
        np.save(output_dir / f"{split}_attrs_idx.npy", attrs)
        np.save(output_dir / f"{split}_text_caps.npy", caps)
    meta = {
        "attr_list": ["source"],
        "attr_n_ops": [1],
        "attrs_split": {
            "train": len(train_records),
            "valid": len(valid_records),
            "test": len(test_records),
        },
        "final_split": {
            "train": len(train_records),
            "valid": len(valid_records),
            "test": len(test_records),
        },
        "source": "TSFragment-600K",
        "length": length,
        "n_var": 1,
    }
    _write_json(output_dir / "meta.json", meta)
    _write_json(output_dir / "train_records.json", train_records)
    _write_json(output_dir / "valid_records.json", valid_records)
    _write_json(output_dir / "test_records.json", test_records)
    _write_json(output_dir / "test_case_ids.json", [record["case_id"] for record in test_records])
    return {
        "dataset_dir": _display(output_dir),
        "length": length,
        "train": len(train_records),
        "valid": len(valid_records),
        "test": len(test_records),
        "test_case_ids": [record["case_id"] for record in test_records],
    }


def prepare_datasets(
    *,
    case_file: Path = DEFAULT_CASE_FILE,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    tsfragment_root: Path = TSFRAGMENT_ROOT,
    train_per_dataset: int = 2000,
    valid_per_dataset: int = 200,
    scan_limit: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    cases = _read_json(case_file)
    if not isinstance(cases, list):
        raise ValueError(f"Expected case list in {case_file}.")
    excluded = _case_rows_by_dataset_length(cases)
    summaries: list[dict[str, Any]] = []
    for length in LENGTHS:
        train_records: list[dict[str, Any]] = []
        valid_records: list[dict[str, Any]] = []
        for dataset in SOURCE_DATASETS:
            train, valid = _sample_rows(
                dataset=dataset,
                length=length,
                excluded_rows=excluded.get((dataset, length), set()),
                train_per_dataset=train_per_dataset,
                valid_per_dataset=valid_per_dataset,
                scan_limit=scan_limit,
                seed=seed,
                tsfragment_root=tsfragment_root,
            )
            train_records.extend(train)
            valid_records.extend(valid)
        test_records = [
            {
                "case_id": str(case["case_id"]),
                "source_dataset": str(case["source_dataset"]),
                "source_length": int(case["source_length"]),
                "source_row": int(case.get("source_row", case.get("source_index", -1))),
                "description": str(case["description"]),
                "series": [float(value) for value in case["reference_series"]],
            }
            for case in cases
            if int(case["source_length"]) == length
        ]
        test_records.sort(key=lambda item: (item["source_dataset"], item["source_row"], item["case_id"]))
        summaries.append(
            _write_verbalts_dataset(
                output_dir=dataset_root / _dataset_name(length),
                length=length,
                train_records=train_records,
                valid_records=valid_records,
                test_records=test_records,
            )
        )
    return {
        "version": "1.0",
        "status": "prepared",
        "case_file": _display(case_file),
        "dataset_root": _display(dataset_root),
        "train_per_dataset": train_per_dataset,
        "valid_per_dataset": valid_per_dataset,
        "scan_limit": scan_limit,
        "seed": seed,
        "datasets": summaries,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_configs(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    config_root: Path = DEFAULT_CONFIG_ROOT,
    device: str = "cuda:0",
    epochs: int = 200,
    batch_size: int = 256,
    eval_batch_size: int = 128,
    lr: float = 1e-3,
) -> dict[str, Any]:
    base_dir = DEFAULT_CONFIG_ROOT / "Weather"
    base_diff = _load_yaml(base_dir / "diff" / "model_text2ts_dep.yaml")
    base_cond = _load_yaml(base_dir / "cond" / "text_msmdiffmv.yaml")
    summaries: list[dict[str, Any]] = []
    for length in LENGTHS:
        name = _dataset_name(length)
        dataset_dir = dataset_root / name
        config_dir = config_root / name

        diff = dict(base_diff)
        diff["device"] = device
        diff["diffusion"] = dict(base_diff["diffusion"])
        diff["diffusion"]["n_var"] = 1
        diff["diffusion"]["base_patch"] = 1
        diff["diffusion"]["L_patch_len"] = 2
        diff["diffusion"]["multipatch_num"] = 2
        diff["diffusion"]["side"] = dict(base_diff["diffusion"]["side"])
        diff["diffusion"]["side"]["num_var"] = 1

        cond = dict(base_cond)
        cond["device"] = device
        cond["cond_modal"] = "simple_text"
        cond["generator_pretrain_path"] = ""
        cond["text"] = dict(base_cond["text"])
        cond["text"]["output_type"] = "all"
        cond["text"]["num_stages"] = 3
        cond["text"]["pos_emb"] = "none"

        train = {
            "data": {"name": "custom", "folder": str(dataset_dir)},
            "train": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "itr_per_epoch": 1.0e8,
                "model_path": "",
                "output_folder": "pretrain",
                "val_epoch_interval": 10,
                "display_interval": 1,
            },
        }
        evaluate = {
            "data": {"name": "custom", "folder": str(dataset_dir)},
            "eval": {
                "n_samples": 10,
                "model_path": "",
                "batch_size": eval_batch_size,
                "display_interval": 1,
                "device": device,
                "cache_folder": f"./cache/{name}_text",
            },
        }

        _write_yaml(config_dir / "diff" / "model_text2ts_dep.yaml", diff)
        _write_yaml(config_dir / "cond" / "text_msmdiffmv.yaml", cond)
        _write_yaml(config_dir / "train.yaml", train)
        _write_yaml(config_dir / "evaluate.yaml", evaluate)
        summaries.append(
            {
                "length": length,
                "config_dir": _display(config_dir),
                "dataset_dir": _display(dataset_dir),
                "diff_config": _display(config_dir / "diff" / "model_text2ts_dep.yaml"),
                "cond_config": _display(config_dir / "cond" / "text_msmdiffmv.yaml"),
                "train_config": _display(config_dir / "train.yaml"),
                "evaluate_config": _display(config_dir / "evaluate.yaml"),
            }
        )
    return {
        "version": "1.0",
        "status": "configs_written",
        "config_root": _display(config_root),
        "datasets": summaries,
    }


def train_command(*, length: int, save_root: Path, batch_size: int, epochs: int, n_runs: int, lr: float = 1e-3) -> list[str]:
    name = _dataset_name(length)
    return [
        sys.executable,
        "run.py",
        "--cond_modal",
        "simple_text",
        "--training_stage",
        "finetune",
        "--save_folder",
        str(save_root / name / "text2ts_msmdiffmv"),
        "--model_diff_config_path",
        f"configs/{name}/diff/model_text2ts_dep.yaml",
        "--model_cond_config_path",
        f"configs/{name}/cond/text_msmdiffmv.yaml",
        "--train_config_path",
        f"configs/{name}/train.yaml",
        "--evaluate_config_path",
        f"configs/{name}/evaluate.yaml",
        "--data_folder",
        str(DEFAULT_DATASET_ROOT / name),
        "--multipatch_num",
        "2",
        "--L_patch_len",
        "2",
        "--base_patch",
        "1",
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--n_runs",
        str(n_runs),
    ]


def build_plan(
    *,
    output_path: Path = DEFAULT_PLAN_PATH,
    save_root: Path = DEFAULT_SAVE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    batch_size: int = 256,
    epochs: int = 200,
    n_runs: int = 1,
    lr: float = 1e-3,
) -> dict[str, Any]:
    commands = {
        str(length): {
            "train": train_command(
                length=length,
                save_root=save_root,
                batch_size=batch_size,
                epochs=epochs,
                n_runs=n_runs,
                lr=lr,
            ),
            "export": [
                sys.executable,
                "evals/run_verbalts_tsfragment.py",
                "export",
                "--length",
                str(length),
                "--checkpoint",
                str(save_root / _dataset_name(length) / "text2ts_msmdiffmv" / "0" / "ckpts" / "model_best_loss.pth"),
                "--output-root",
                str(output_root),
            ],
        }
        for length in LENGTHS
    }
    payload = {
        "version": "1.0",
        "status": "planned",
        "training_note": (
            "Train one TSFragment-compatible VerbalTS model per length. The cases in the selected benchmark "
            "file are exported one case per JSON file for the shared evaluator."
        ),
        "commands": commands,
        "evaluate_command": [
            sys.executable,
            "evals/run_baselines.py",
            "--methods",
            "verbalts",
            "--cases",
            "data/tsfragment_eval/tsfragment_eval_2500_cases.json",
            "--output-root",
            str(output_root),
        ],
    }
    _write_json(output_path, payload)
    return payload


def _load_verbalts_modules() -> Any:
    if str(VERBALTS_ROOT) not in sys.path:
        sys.path.insert(0, str(VERBALTS_ROOT))
    from data import GenerationDataset
    from models.conditional_generator import ConditionalGenerator

    return GenerationDataset, ConditionalGenerator


def _resolve_verbalts_relative_paths(cond_configs: dict[str, Any]) -> None:
    text_configs = cond_configs.get("text")
    if not isinstance(text_configs, dict):
        return
    for key in ["pretrain_model_path", "tokenizer_path"]:
        value = text_configs.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            text_configs[key] = str((VERBALTS_ROOT / path).resolve())


def export_predictions(
    *,
    length: int,
    checkpoint: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    config_root: Path = DEFAULT_CONFIG_ROOT,
    batch_size: int = 128,
    n_samples: int = 10,
    sampler: str = "ddim",
    seed: int = 42,
) -> dict[str, Any]:
    import torch

    GenerationDataset, ConditionalGenerator = _load_verbalts_modules()
    name = _dataset_name(length)
    dataset_dir = dataset_root / name
    config_dir = config_root / name
    case_ids = _read_json(dataset_dir / "test_case_ids.json")
    diff_configs = _load_yaml(config_dir / "diff" / "model_text2ts_dep.yaml")
    cond_configs = _load_yaml(config_dir / "cond" / "text_msmdiffmv.yaml")
    eval_configs = _load_yaml(config_dir / "evaluate.yaml")
    diff_configs["generator_pretrain_path"] = diff_configs.get("generator_pretrain_path", "")
    cond_configs["cond_modal"] = cond_configs.get("cond_modal", "simple_text")
    cond_configs.setdefault("text", {})
    cond_configs["text"]["output_type"] = cond_configs["text"].get("output_type", "all")
    cond_configs["text"]["num_stages"] = cond_configs["text"].get("num_stages", 3)
    cond_configs["text"]["pos_emb"] = cond_configs["text"].get("pos_emb", "none")
    _resolve_verbalts_relative_paths(cond_configs)
    eval_configs["data"]["folder"] = str(dataset_dir)
    eval_configs["eval"]["batch_size"] = batch_size

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = GenerationDataset(eval_configs["data"])
    if "attrs" in cond_configs:
        cond_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
    model = ConditionalGenerator(diff_configs, cond_configs)
    model.load_state_dict(torch.load(checkpoint, map_location=model.device))
    model = model.to(model.device)
    model.eval()

    output_root.mkdir(parents=True, exist_ok=True)
    loader = dataset.get_loader(split="test", batch_size=batch_size, shuffle=False, num_workers=0)
    written: list[str] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            multi_preds = model.generate(batch, n_samples, sampler)
            pred = multi_preds.permute(0, 1, 3, 2).median(dim=0).values.cpu().numpy()
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred[:, :, 0]
            for row in pred:
                case_id = str(case_ids[offset])
                path = output_root / f"{case_id}.json"
                _write_json(
                    path,
                    {
                        "generated_series": [float(value) for value in row.tolist()],
                        "method": "VerbalTS TSFragment",
                        "length": length,
                        "checkpoint": _display(checkpoint),
                    },
                )
                written.append(_display(path))
                offset += 1
    if offset != len(case_ids):
        raise RuntimeError(f"Exported {offset} rows but expected {len(case_ids)} case ids.")
    return {
        "version": "1.0",
        "status": "exported",
        "length": length,
        "checkpoint": _display(checkpoint),
        "output_root": _display(output_root),
        "written": len(written),
        "files": written,
    }


def _load_generated_series(output_root: Path, case_id: str) -> list[float]:
    path = output_root / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    raw = payload.get("generated_series", payload.get("series", payload.get("values")))
    if raw is None:
        raise ValueError(f"{path} does not contain generated_series, series, or values.")
    series = np.asarray(raw, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"{case_id}: generated series must be one-dimensional, got shape {series.shape}.")
    return [float(value) for value in series.tolist()]


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def _median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.median(clean)) if clean else None


def _aggregate_records(records: list[dict[str, Any]], *, cases: int) -> dict[str, Any]:
    success = [record for record in records if record.get("status") == "success"]
    metrics = [record["metrics"] for record in success]
    return {
        "method": "VerbalTS TSFragment",
        "cases": cases,
        "successful_cases": len(success),
        "success_rate": float(len(success) / cases) if cases else 0.0,
        "mean_wape": _mean([item.get("wape") for item in metrics]),
        "median_wape": _median([item.get("wape") for item in metrics]),
        "mean_corr": _mean([item.get("correlation") for item in metrics]),
        "median_corr": _median([item.get("correlation") for item in metrics]),
        "mean_mrr": _mean([item.get("mrr") for item in metrics]),
        "median_mrr": _median([item.get("mrr") for item in metrics]),
        "mean_raw_mse": _mean([item.get("mse") for item in metrics]),
        "median_raw_mse": _median([item.get("mse") for item in metrics]),
        "status": "completed" if len(success) == cases else "partial",
    }


def _aggregate_by_dataset_length(records: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(str(case["source_dataset"]), int(case["source_length"])) for case in cases})
    for dataset, length in keys:
        group_cases = [
            case for case in cases
            if str(case["source_dataset"]) == dataset and int(case["source_length"]) == length
        ]
        success = [
            record for record in records
            if record.get("status") == "success"
            and str(record.get("source_dataset")) == dataset
            and int(record.get("source_length")) == length
        ]
        metrics = [record["metrics"] for record in success]
        rows.append(
            {
                "method": "VerbalTS TSFragment",
                "dataset": dataset,
                "length": length,
                "cases": len(group_cases),
                "successful_cases": len(success),
                "success_rate": float(len(success) / len(group_cases)) if group_cases else 0.0,
                "mean_wape": _mean([item.get("wape") for item in metrics]),
                "median_wape": _median([item.get("wape") for item in metrics]),
                "mean_corr": _mean([item.get("correlation") for item in metrics]),
                "median_corr": _median([item.get("correlation") for item in metrics]),
                "mean_mrr": _mean([item.get("mrr") for item in metrics]),
                "median_mrr": _median([item.get("mrr") for item in metrics]),
            }
        )
    return rows


def evaluate_predictions(
    *,
    case_file: Path = DEFAULT_CASE_FILE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    cases = _read_json(case_file)
    if not isinstance(cases, list):
        raise ValueError(f"Expected case list in {case_file}.")
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        try:
            generated = _load_generated_series(output_root, case_id)
            expected_length = int(case["length"])
            if len(generated) != expected_length:
                raise ValueError(f"generated length {len(generated)} does not match expected {expected_length}")
            metrics = compute_error_metrics(
                np.asarray(generated, dtype=float),
                np.asarray(case["reference_series"], dtype=float),
            )
            records.append(
                {
                    "case_id": case_id,
                    "source_dataset": str(case["source_dataset"]),
                    "source_length": int(case["source_length"]),
                    "source_row": int(case.get("source_row", case.get("source_index", -1))),
                    "status": "success",
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "case_id": case_id,
                    "source_dataset": str(case.get("source_dataset", "")),
                    "source_length": int(case.get("source_length", case.get("length", 0))),
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return {
        "version": "1.0",
        "benchmark": f"VerbalTS TSFragment {len(cases)}-case shared benchmark",
        "case_file": _display(case_file),
        "output_root": _display(output_root),
        "training_boundary": (
            "One VerbalTS model is used per length. Evaluation reads exactly the cases in the selected "
            "case file and counts one exported prediction per case."
        ),
        "aggregate": _aggregate_records(records, cases=len(cases)),
        "by_dataset_length": _aggregate_by_dataset_length(records, cases),
        "records": records,
    }


def render_eval_markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# VerbalTS TSFragment Shared Benchmark",
        "",
        payload["training_boundary"],
        "",
        "| Method | Cases | Success Count | Success Rate | Mean WAPE | Median WAPE | Mean Corr | Median Corr | Mean MRR | Median MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                "VerbalTS TSFragment",
                str(aggregate["cases"]),
                f"{aggregate['successful_cases']}/{aggregate['cases']}",
                f"{aggregate['success_rate']:.3f}",
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
        "| Dataset | Length | Cases | Success | Mean WAPE | Median WAPE | Mean Corr | Median Corr | Mean MRR | Median MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    f"{row['mean_mrr']:.3f}" if row["mean_mrr"] is not None else "n/a",
                    f"{row['median_mrr']:.3f}" if row["median_mrr"] is not None else "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def update_comparison(
    *,
    verbalts_payload: dict[str, Any],
    comparison_path: Path = DEFAULT_COMPARISON_JSON,
    eval_json: Path = DEFAULT_EVAL_JSON,
) -> dict[str, Any]:
    comparison = _read_json(comparison_path) if comparison_path.exists() else {
        "version": "1.0",
        "benchmark": "T2S TSFragment-600K real-caption benchmark",
        "overall": [],
        "artifacts": {},
    }
    existing = [
        row for row in comparison.get("overall", [])
        if isinstance(row, dict) and str(row.get("method")) != "VerbalTS TSFragment"
    ]
    aggregate = verbalts_payload["aggregate"]
    verbalts_row = {
        "method": "VerbalTS TSFragment",
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
    comparison["overall"] = existing + [verbalts_row]
    comparison.setdefault("artifacts", {})["verbalts_tsfragment_eval"] = _display(eval_json)
    comparison.setdefault("artifacts", {})["verbalts_tsfragment_outputs"] = verbalts_payload["output_root"]
    comparison.setdefault("scope", {})["verbalts_comparison_unit"] = (
        "VerbalTS TSFragment uses three separately trained length-specific models and one generated output "
        "per shared TSFragment test case."
    )
    comparison_path.write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    return comparison


def run_training(
    *,
    length: int,
    save_root: Path = DEFAULT_SAVE_ROOT,
    batch_size: int = 256,
    epochs: int = 200,
    n_runs: int = 1,
    lr: float = 1e-3,
) -> int:
    command = train_command(length=length, save_root=save_root, batch_size=batch_size, epochs=epochs, n_runs=n_runs, lr=lr)
    completed = subprocess.run(command, cwd=VERBALTS_ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, train, and export TSFragment-compatible VerbalTS baselines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build VerbalTS datasets and configs for TSFragment lengths 24/48/96.")
    prepare_parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    prepare_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    prepare_parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    prepare_parser.add_argument("--train-per-dataset", type=int, default=2000)
    prepare_parser.add_argument("--valid-per-dataset", type=int, default=200)
    prepare_parser.add_argument("--scan-limit", type=int, default=20000)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--device", default="cuda:0")
    prepare_parser.add_argument("--epochs", type=int, default=200)
    prepare_parser.add_argument("--batch-size", type=int, default=256)
    prepare_parser.add_argument("--eval-batch-size", type=int, default=128)
    prepare_parser.add_argument("--lr", type=float, default=1e-3)
    prepare_parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_PATH)

    plan_parser = subparsers.add_parser("plan", help="Write train/export/evaluate commands without preparing data.")
    plan_parser.add_argument("--output-path", type=Path, default=DEFAULT_PLAN_PATH)
    plan_parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    plan_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    plan_parser.add_argument("--batch-size", type=int, default=256)
    plan_parser.add_argument("--epochs", type=int, default=200)
    plan_parser.add_argument("--n-runs", type=int, default=1)
    plan_parser.add_argument("--lr", type=float, default=1e-3)

    train_parser = subparsers.add_parser("train", help="Launch VerbalTS training for one TSFragment length.")
    train_parser.add_argument("--length", type=int, choices=LENGTHS, required=True)
    train_parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--n-runs", type=int, default=1)
    train_parser.add_argument("--lr", type=float, default=1e-3)

    export_parser = subparsers.add_parser("export", help="Export one-file-per-case predictions from a trained length model.")
    export_parser.add_argument("--length", type=int, choices=LENGTHS, required=True)
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    export_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    export_parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    export_parser.add_argument("--batch-size", type=int, default=128)
    export_parser.add_argument("--n-samples", type=int, default=10)
    export_parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    export_parser.add_argument("--seed", type=int, default=42)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate exported VerbalTS TSFragment outputs on the shared benchmark.")
    eval_parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    eval_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    eval_parser.add_argument("--json-output", type=Path, default=DEFAULT_EVAL_JSON)
    eval_parser.add_argument("--markdown-output", type=Path, default=DEFAULT_EVAL_MD)
    eval_parser.add_argument("--update-comparison", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        data_payload = prepare_datasets(
            case_file=args.case_file,
            dataset_root=args.dataset_root,
            train_per_dataset=args.train_per_dataset,
            valid_per_dataset=args.valid_per_dataset,
            scan_limit=args.scan_limit,
            seed=args.seed,
        )
        config_payload = write_configs(
            dataset_root=args.dataset_root,
            config_root=args.config_root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
        )
        plan_payload = build_plan(output_path=args.plan_output, batch_size=args.batch_size, epochs=args.epochs)
        payload = {"data": data_payload, "configs": config_payload, "plan": plan_payload}
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    if args.command == "plan":
        payload = build_plan(
            output_path=args.output_path,
            save_root=args.save_root,
            output_root=args.output_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            n_runs=args.n_runs,
            lr=args.lr,
        )
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    if args.command == "train":
        return run_training(
            length=args.length,
            save_root=args.save_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            n_runs=args.n_runs,
            lr=args.lr,
        )
    if args.command == "export":
        payload = export_predictions(
            length=args.length,
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            dataset_root=args.dataset_root,
            config_root=args.config_root,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            sampler=args.sampler,
            seed=args.seed,
        )
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    if args.command == "evaluate":
        payload = evaluate_predictions(case_file=args.case_file, output_root=args.output_root)
        _write_json(args.json_output, payload)
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_eval_markdown(payload), encoding="utf-8")
        if args.update_comparison:
            update_comparison(verbalts_payload=payload, eval_json=args.json_output)
        print(json.dumps(payload["aggregate"], indent=2, allow_nan=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
