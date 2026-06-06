from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VERBALTS_ROOT = REPO_ROOT / "baselines" / "VerbalTS"
DEFAULT_REPORT_JSON = REPO_ROOT / "evals" / "reports" / "verbalts_native_weather_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "evals" / "reports" / "verbalts_native_weather_report.md"
DEFAULT_T2S_COMPARISON_JSON = REPO_ROOT / "evals" / "reports" / "t2s_main_benchmark_comparison_2500.json"
DEFAULT_SHARED_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "verbalts_main_benchmark_outputs_2500"


DATASET_ARGS = {
    "Weather": {
        "base_patch": 1,
        "config_dir": "Weather",
    },
    "synth-m": {
        "base_patch": 4,
        "config_dir": "synth-m",
    },
}


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _required_paths(*, dataset: str, data_folder: Path, save_folder: Path, clip_folder: Path) -> list[Path]:
    config_dir = DATASET_ARGS[dataset]["config_dir"]
    longclip_folder = VERBALTS_ROOT / "save" / "Longclip"
    paths = [
        VERBALTS_ROOT / "run.py",
        VERBALTS_ROOT / "download_longclip.py",
        VERBALTS_ROOT / "configs" / config_dir / "train.yaml",
        VERBALTS_ROOT / "configs" / config_dir / "evaluate.yaml",
        VERBALTS_ROOT / "configs" / config_dir / "diff" / "model_text2ts_dep.yaml",
        VERBALTS_ROOT / "configs" / config_dir / "cond" / "text_msmdiffmv.yaml",
        longclip_folder / "config.json",
        longclip_folder / "preprocessor_config.json",
        longclip_folder / "tokenizer_config.json",
        data_folder / "meta.json",
        data_folder / "test_ts.npy",
        data_folder / "test_attrs_idx.npy",
        data_folder / "test_text_caps.npy",
        save_folder / "0" / "ckpts" / "model_best_loss.pth",
        clip_folder / "clip_model_best.pth",
        clip_folder / "model_configs.yaml",
    ]
    return paths


def check_artifacts(*, dataset: str, data_folder: Path, save_folder: Path, clip_folder: Path) -> dict[str, Any]:
    required = _required_paths(dataset=dataset, data_folder=data_folder, save_folder=save_folder, clip_folder=clip_folder)
    missing = [path for path in required if not path.exists()]
    return {
        "dataset": dataset,
        "status": "ready" if not missing else "missing_artifacts",
        "required_paths": [_display(path) for path in required],
        "missing_paths": [_display(path) for path in missing],
    }


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_native_metrics(save_folder: Path) -> dict[str, Any]:
    results_csv = save_folder / "results.csv"
    if not results_csv.exists():
        return {
            "status": "missing_results",
            "results_csv": _display(results_csv),
            "rows": [],
            "aggregate": {},
        }
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed_rows.append(
            {
                "run": int(float(row["run"])) if row.get("run") not in {None, ""} else None,
                "mode": row.get("mode"),
                "sampler": row.get("sampler"),
                "n_samples": int(float(row["n_samples"])) if row.get("n_samples") not in {None, ""} else None,
                "steps": int(float(row["steps"])) if row.get("steps") not in {None, ""} else None,
                "cttp": _float_or_none(row.get("cttp")),
                "fid": _float_or_none(row.get("fid")),
                "jftsd": _float_or_none(row.get("jftsd")),
            }
        )
    aggregate: dict[str, Any] = {
        "runs": len(parsed_rows),
        "mode": parsed_rows[0].get("mode") if parsed_rows else None,
        "sampler": parsed_rows[0].get("sampler") if parsed_rows else None,
        "n_samples": parsed_rows[0].get("n_samples") if parsed_rows else None,
    }
    for key in ["cttp", "fid", "jftsd"]:
        values = [row[key] for row in parsed_rows if row.get(key) is not None]
        if values:
            aggregate[f"mean_{key}"] = sum(float(value) for value in values) / len(values)
    return {
        "status": "completed" if parsed_rows else "empty_results",
        "results_csv": _display(results_csv),
        "results_stat_condgen_csv": _display(save_folder / "results_stat_condgen.csv"),
        "rows": parsed_rows,
        "aggregate": aggregate,
        "metric_policy": {
            "cttp": "higher is better; VerbalTS native text-time-series alignment score",
            "fid": "lower is better; VerbalTS native time-series embedding distribution distance",
            "jftsd": "lower is better; VerbalTS native joint text-time-series distribution distance",
            "boundary": (
                "These are VerbalTS native distribution/alignment metrics on the released Weather split. "
                "They are not direct CSR/PrimitiveF1/WindowIoU correctness metrics and should not be used "
                "to claim exact semantic controllability."
            ),
        },
    }


def shared_evaluator_adapter(output_root: Path = DEFAULT_SHARED_OUTPUT_ROOT) -> dict[str, Any]:
    files = sorted(output_root.glob("*.json")) + sorted(output_root.glob("*.txt")) if output_root.exists() else []
    return {
        "method_id": "verbalts",
        "status": "case_outputs_available" if files else "adapter_ready_missing_case_outputs",
        "output_root": _display(output_root),
        "available_output_files": len(files),
        "expected_case_file": "data/tsfragment_eval/tsfragment_eval_2500_cases.json",
        "expected_file_pattern": "<case_id>.json or <case_id>.txt",
        "expected_json_fields": ["generated_series", "series", "values"],
        "shared_evaluator_command": (
            "python evals/run_baselines.py --methods verbalts "
            "--cases data/tsfragment_eval/tsfragment_eval_2500_cases.json "
            f"--output-root {_display(output_root)}"
        ),
        "correctness_boundary": (
            "Shared-case correctness claims require one generated one-dimensional series per TSFragment case. "
            "Native VerbalTS CTTP/FID/JFTSD remain supplementary similarity/fidelity metrics until those "
            "case-level outputs are populated."
        ),
    }


def _native_command(*, dataset: str, data_folder: Path, save_folder: Path, clip_folder: Path, batch_size: int, n_runs: int) -> list[str]:
    config_dir = DATASET_ARGS[dataset]["config_dir"]
    return [
        sys.executable,
        "run.py",
        "--cond_modal",
        "simple_text",
        "--training_stage",
        "finetune",
        "--save_folder",
        str(save_folder),
        "--model_diff_config_path",
        f"configs/{config_dir}/diff/model_text2ts_dep.yaml",
        "--model_cond_config_path",
        f"configs/{config_dir}/cond/text_msmdiffmv.yaml",
        "--train_config_path",
        f"configs/{config_dir}/train.yaml",
        "--evaluate_config_path",
        f"configs/{config_dir}/evaluate.yaml",
        "--data_folder",
        str(data_folder),
        "--clip_folder",
        str(clip_folder),
        "--multipatch_num",
        "3",
        "--L_patch_len",
        "3",
        "--base_patch",
        str(DATASET_ARGS[dataset]["base_patch"]),
        "--epochs",
        "700",
        "--batch_size",
        str(batch_size),
        "--clip_cache_path",
        str(save_folder / "cache"),
        "--only_evaluate",
        "true",
        "--n_runs",
        str(n_runs),
    ]


def run_native_verbalts(
    *,
    dataset: str,
    data_folder: Path,
    save_folder: Path,
    clip_folder: Path,
    batch_size: int,
    n_runs: int,
    force: bool,
) -> dict[str, Any]:
    artifact_check = check_artifacts(dataset=dataset, data_folder=data_folder, save_folder=save_folder, clip_folder=clip_folder)
    command = _native_command(
        dataset=dataset,
        data_folder=data_folder,
        save_folder=save_folder,
        clip_folder=clip_folder,
        batch_size=batch_size,
        n_runs=n_runs,
    )
    if artifact_check["status"] != "ready" and not force:
        return {
            "version": "1.0",
            "method": "verbalts",
            "status": "skipped",
            "skip_reason": "missing_artifacts",
            "artifact_check": artifact_check,
            "command": command,
            "native_metrics": load_native_metrics(save_folder),
            "shared_evaluator_adapter": shared_evaluator_adapter(),
            "results_csv": _display(save_folder / "results.csv"),
            "results_stat_condgen_csv": _display(save_folder / "results_stat_condgen.csv"),
        }

    completed = subprocess.run(command, cwd=VERBALTS_ROOT, text=True, capture_output=True, check=False)
    status = "completed" if completed.returncode == 0 else "failed"
    return {
        "version": "1.0",
        "method": "verbalts",
        "status": status,
        "returncode": completed.returncode,
        "artifact_check": artifact_check,
        "command": command,
        "native_metrics": load_native_metrics(save_folder),
        "shared_evaluator_adapter": shared_evaluator_adapter(),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "results_csv": _display(save_folder / "results.csv"),
        "results_stat_condgen_csv": _display(save_folder / "results_stat_condgen.csv"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("native_metrics", {})
    aggregate = metrics.get("aggregate", {}) if isinstance(metrics, dict) else {}
    lines = [
        "# VerbalTS Baseline Report",
        "",
        f"Status: {payload['status']}",
        "",
        "## Native VerbalTS Run",
        "",
        f"- Dataset: `{payload.get('artifact_check', {}).get('dataset')}`",
        f"- Results CSV: `{payload.get('results_csv')}`",
        f"- Results stats: `{payload.get('results_stat_condgen_csv')}`",
        f"- Command: `{' '.join(str(item) for item in payload.get('command', []))}`",
    ]
    if payload.get("skip_reason"):
        lines.append(f"- Skip reason: `{payload['skip_reason']}`")
    missing = payload.get("artifact_check", {}).get("missing_paths", [])
    if missing:
        lines.extend(["", "Missing artifacts:"])
        lines.extend(f"- `{path}`" for path in missing)
    if payload.get("stderr_tail"):
        lines.extend(["", "Stderr tail:", "", "```text", payload["stderr_tail"], "```"])
    if aggregate:
        lines.extend(
            [
                "",
                "## Native Metrics",
                "",
                "| Dataset | Runs | Mode | Sampler | Samples | CTTP ↑ | FID ↓ | JFTSD ↓ |",
                "|---|---:|---|---|---:|---:|---:|---:|",
                "| "
                + " | ".join(
                    [
                        str(payload.get("artifact_check", {}).get("dataset")),
                        str(aggregate.get("runs", "n/a")),
                        str(aggregate.get("mode", "n/a")),
                        str(aggregate.get("sampler", "n/a")),
                        str(aggregate.get("n_samples", "n/a")),
                        f"{aggregate.get('mean_cttp'):.3f}" if aggregate.get("mean_cttp") is not None else "n/a",
                        f"{aggregate.get('mean_fid'):.3f}" if aggregate.get("mean_fid") is not None else "n/a",
                        f"{aggregate.get('mean_jftsd'):.3f}" if aggregate.get("mean_jftsd") is not None else "n/a",
                    ]
                )
                + " |",
                "",
                "Metric boundary: " + metrics.get("metric_policy", {}).get("boundary", ""),
            ]
        )
    adapter = payload.get("shared_evaluator_adapter", {})
    lines.extend(
        [
            "",
            "## Shared Evaluator Adapter",
            "",
            f"- Status: `{adapter.get('status', 'unknown')}`",
            f"- Output root: `{adapter.get('output_root', _display(DEFAULT_SHARED_OUTPUT_ROOT))}`",
            f"- Available output files: `{adapter.get('available_output_files', 0)}`",
            f"- Expected file pattern: `{adapter.get('expected_file_pattern', '<case_id>.json or <case_id>.txt')}`",
            f"- Expected JSON fields: `{', '.join(adapter.get('expected_json_fields', ['generated_series', 'series', 'values']))}`",
            f"- Command: `{adapter.get('shared_evaluator_command', '')}`",
            "",
            "Correctness boundary: " + str(adapter.get("correctness_boundary", "")),
        ]
    )
    lines.extend(
        [
            "",
            "## Reproduction Notes",
            "",
            "Use `python baselines/VerbalTS/download_longclip.py` to populate the VerbalTS `./save/Longclip` path referenced by the text condition config.",
            "",
            "VerbalTS generated case outputs can be evaluated by the main baseline runner as `verbalts` when saved under the adapter output root.",
        ]
    )
    return "\n".join(lines) + "\n"


def update_t2s_comparison(payload: dict[str, Any], comparison_path: Path = DEFAULT_T2S_COMPARISON_JSON) -> dict[str, Any]:
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    else:
        comparison = {"version": "1.0", "overall": [], "artifacts": {}}
    comparison.setdefault("scope", {})
    comparison["scope"]["public_text_generator_baselines"] = sorted(
        set(comparison["scope"].get("public_text_generator_baselines", []))
        | {"t2s_native", "verbalts_native"}
    )
    comparison["scope"]["verbalts_comparison_unit"] = (
        "VerbalTS native Weather metrics are retained only as a reproduction/background artifact. "
        "Main VerbalTS claims use shared TSFragment case-level outputs through the `verbalts` adapter."
    )
    metrics = payload.get("native_metrics", {})
    comparison.setdefault("public_generator_native_supplement", {})
    comparison["public_generator_native_supplement"].update(
        {
            "role": (
                "Native Weather metrics are retained for VerbalTS reproduction/background only. "
                "They are not same-benchmark evidence against FaithTS and should not enter main claim tables."
            ),
            "verbalts_native": {
                "method": "VerbalTS native",
                "status": payload.get("status"),
                "dataset": payload.get("artifact_check", {}).get("dataset"),
                "results_csv": payload.get("results_csv"),
                "results_stat_condgen_csv": payload.get("results_stat_condgen_csv"),
                "aggregate": metrics.get("aggregate") if isinstance(metrics, dict) else None,
                "metric_policy": metrics.get("metric_policy") if isinstance(metrics, dict) else None,
                "metric_boundary": (metrics.get("metric_policy", {}) if isinstance(metrics, dict) else {}).get(
                    "boundary",
                    "VerbalTS native metrics are not direct CSR-style correctness metrics.",
                ),
            },
            "verbalts_shared_evaluator_adapter": payload.get(
                "shared_evaluator_adapter",
                shared_evaluator_adapter(),
            ),
        }
    )
    comparison.setdefault("artifacts", {})["verbalts_native_report"] = _display(DEFAULT_REPORT_JSON)
    comparison_path.write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Check/run the VerbalTS baseline and write a reproducible report.")
    parser.add_argument("--dataset", choices=sorted(DATASET_ARGS), default="Weather")
    parser.add_argument("--data-folder", type=Path, default=VERBALTS_ROOT / "datasets" / "Weather")
    parser.add_argument("--save-folder", type=Path, default=VERBALTS_ROOT / "save" / "Weather_eval" / "text2ts_msmdiffmv")
    parser.add_argument("--clip-folder", type=Path, default=VERBALTS_ROOT / "save" / "Weather_cttp")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Run even when the artifact preflight check reports missing files.")
    parser.add_argument("--report-only", action="store_true", help="Do not run VerbalTS; only refresh the report from existing artifacts/results.")
    parser.add_argument("--update-t2s-comparison", action="store_true", help="Add VerbalTS native metrics to the T2S comparison artifact supplement.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    if args.report_only:
        artifact_check = check_artifacts(
            dataset=args.dataset,
            data_folder=args.data_folder,
            save_folder=args.save_folder,
            clip_folder=args.clip_folder,
        )
        payload = {
            "version": "1.0",
            "method": "verbalts",
            "status": "completed" if load_native_metrics(args.save_folder).get("status") == "completed" else "report_only",
            "artifact_check": artifact_check,
            "command": _native_command(
                dataset=args.dataset,
                data_folder=args.data_folder,
                save_folder=args.save_folder,
                clip_folder=args.clip_folder,
                batch_size=args.batch_size,
                n_runs=args.n_runs,
            ),
            "native_metrics": load_native_metrics(args.save_folder),
            "shared_evaluator_adapter": shared_evaluator_adapter(),
            "results_csv": _display(args.save_folder / "results.csv"),
            "results_stat_condgen_csv": _display(args.save_folder / "results_stat_condgen.csv"),
        }
    else:
        payload = run_native_verbalts(
            dataset=args.dataset,
            data_folder=args.data_folder,
            save_folder=args.save_folder,
            clip_folder=args.clip_folder,
            batch_size=args.batch_size,
            n_runs=args.n_runs,
            force=args.force,
        )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    if args.update_t2s_comparison:
        update_t2s_comparison(payload)
    print(f"{payload['method']}: status={payload['status']}")
    print(f"Wrote JSON to {_display(args.json_output)}")
    print(f"Wrote Markdown to {_display(args.markdown_output)}")
    return 0 if payload["status"] in {"completed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
