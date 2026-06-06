from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.baselines.llm_direct import parse_direct_array_output, parse_direct_code_output
from evals.baselines.rule_parser import plan_for_case as rule_parser_plan_for_case
from evals.run_baseline_benchmark import _execute_plan_payload
from faithts.gold_adapter import load_gold_cases, oracle_plan_from_gold_case


GOLD_CASE_FILES = [
    REPO_ROOT / "data" / "controlled_gold_cases" / "single_claim_cases.json",
    REPO_ROOT / "data" / "controlled_gold_cases" / "multi_claim_cases.json",
    REPO_ROOT / "data" / "controlled_gold_cases" / "robustness_cases.json",
]
T2S_CASE_FILE = REPO_ROOT / "data" / "controlled_gold_cases" / "tsfragment_eval_2500_cases.json"
BASELINE_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "baseline_outputs"
T2S_OURS_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_ours_main_benchmark_cases"
T2S_NATIVE_ROOT = REPO_ROOT / "baselines" / "T2S" / "results" / "denoiser_results" / "generation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evals" / "reports" / "random_baseline_samples"

GOLD_METHODS = [
    "ours_full",
    "rule_parser",
    "llm_direct_array",
    "llm_direct_code",
    "llm_strong_prompt",
]
COLORS = {
    "true": "#111111",
    "ours_full": "#1f77b4",
    "rule_parser": "#2ca02c",
    "llm_direct_array": "#d62728",
    "llm_direct_code": "#9467bd",
    "llm_strong_prompt": "#ff7f0e",
    "t2s_native": "#8c564b",
}
ALL_DISPLAY_METHODS = [
    "true",
    "ours_full",
    "rule_parser",
    "llm_direct_array",
    "llm_direct_code",
    "llm_strong_prompt",
    "t2s_native",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _as_array(values: Any) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"Expected one-dimensional series, got shape {series.shape}.")
    return series


def _wape(reference: np.ndarray, generated: np.ndarray) -> float:
    denom = float(np.sum(np.abs(reference)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(reference - generated)) / denom)


def _corr(reference: np.ndarray, generated: np.ndarray) -> float:
    if reference.size < 2 or np.std(reference) == 0 or np.std(generated) == 0:
        return float("nan")
    return float(np.corrcoef(reference, generated)[0, 1])


def _metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(reference - generated))),
        "rmse": float(np.sqrt(np.mean((reference - generated) ** 2))),
        "wape": _wape(reference, generated),
        "corr": _corr(reference, generated),
    }


def _metric_text(values: dict[str, float] | None) -> str:
    if not values:
        return "parse/execution failed"
    corr = values["corr"]
    corr_text = "nan" if math.isnan(corr) else f"{corr:.3f}"
    wape = values["wape"]
    wape_text = "nan" if math.isnan(wape) else f"{wape:.3f}"
    return f"WAPE={wape_text}, corr={corr_text}"


def _normalize_caption_text(text: str) -> str:
    return " ".join(str(text).split())


def _oracle_series(case: dict[str, Any]) -> np.ndarray | None:
    plan = oracle_plan_from_gold_case(case)
    series, ok, _errors = _execute_plan_payload(plan, int(case["length"]))
    if not ok or series is None:
        return None
    return _as_array(series)


def _ours_gold_series(case: dict[str, Any]) -> np.ndarray:
    payload = _load_json(BASELINE_OUTPUT_ROOT / "ours_full" / f"{case['case_id']}.json")
    if "generated_series" in payload:
        return _as_array(payload["generated_series"])
    plan = payload.get("plan", payload.get("structured_plan", payload))
    series, ok, errors = _execute_plan_payload(plan, int(case["length"]))
    if not ok or series is None:
        raise ValueError("; ".join(errors) or "ours plan did not execute")
    return _as_array(series)


def _rule_parser_series(case: dict[str, Any]) -> np.ndarray:
    plan = rule_parser_plan_for_case(case)
    series, ok, errors = _execute_plan_payload(plan, int(case["length"]))
    if not ok or series is None:
        raise ValueError("; ".join(errors) or "rule parser plan did not execute")
    return _as_array(series)


def _llm_series(case: dict[str, Any], method: str) -> np.ndarray:
    text = (BASELINE_OUTPUT_ROOT / method / f"{case['case_id']}.txt").read_text(encoding="utf-8")
    if method == "llm_direct_array":
        return parse_direct_array_output(text, int(case["length"]))
    return parse_direct_code_output(text, int(case["length"]))


def _gold_method_series(case: dict[str, Any], method: str) -> np.ndarray:
    if method == "ours_full":
        return _ours_gold_series(case)
    if method == "rule_parser":
        return _rule_parser_series(case)
    if method in {"llm_direct_array", "llm_direct_code", "llm_strong_prompt"}:
        return _llm_series(case, method)
    raise ValueError(f"Unknown method: {method}")


def _gold_candidates() -> list[dict[str, Any]]:
    cases = load_gold_cases(GOLD_CASE_FILES)
    candidates = []
    for case in cases:
        if case.get("category") not in {"atomic", "compositional"}:
            continue
        case_id = str(case["case_id"])
        required = [
            BASELINE_OUTPUT_ROOT / "ours_full" / f"{case_id}.json",
            BASELINE_OUTPUT_ROOT / "llm_direct_array" / f"{case_id}.txt",
            BASELINE_OUTPUT_ROOT / "llm_direct_code" / f"{case_id}.txt",
            BASELINE_OUTPUT_ROOT / "llm_strong_prompt" / f"{case_id}.txt",
        ]
        if all(path.exists() for path in required) and _oracle_series(case) is not None:
            candidates.append(case)
    return candidates


def _plot_gold_samples(samples: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    if not samples:
        return []
    fig, axes = plt.subplots(len(samples), 1, figsize=(14, 5.0 * len(samples)), squeeze=False)
    records: list[dict[str, Any]] = []
    for row, case in enumerate(samples):
        ax = axes[row][0]
        reference = _oracle_series(case)
        assert reference is not None
        x = np.arange(len(reference))
        method_records = {}
        for method in GOLD_METHODS:
            try:
                generated = _gold_method_series(case, method)
                if len(generated) != len(reference):
                    raise ValueError(f"length {len(generated)} != {len(reference)}")
                values = _metrics(reference, generated)
                method_records[method] = {"status": "ok", "metrics": values}
                ax.plot(
                    x,
                    generated,
                    linewidth=1.35,
                    alpha=0.88,
                    label=f"{method} ({_metric_text(values)})",
                    color=COLORS[method],
                )
            except Exception as exc:
                method_records[method] = {"status": "failed", "error": str(exc)}
        ax.plot(
            x,
            reference,
            color=COLORS["true"],
            linewidth=3.0,
            linestyle="--",
            label="true/oracle",
            zorder=10,
        )
        failed = [method for method, result in method_records.items() if result["status"] != "ok"]
        if failed:
            ax.text(
                0.99,
                0.04,
                "failed/missing: " + ", ".join(failed),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
            )
        caption = textwrap.fill(str(case["description"]), width=120)
        title = f"{case['case_id']} [{case.get('category', 'unknown')}]"
        ax.set_title(title, fontsize=10, loc="left")
        ax.text(
            0.0,
            1.08,
            f"Caption: {caption}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            clip_on=False,
            bbox={"facecolor": "#fbfbfb", "edgecolor": "#dddddd", "alpha": 0.96},
        )
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncols=2, loc="best")
        records.append(
            {
                "case_id": case["case_id"],
                "category": case.get("category"),
                "description": case["description"],
                "methods": method_records,
            }
        )
    axes[-1][0].set_xlabel("timestep")
    fig.tight_layout()
    fig.savefig(output_dir / "gold_random_samples.png", dpi=180)
    plt.close(fig)
    return records


def _t2s_generated_dir(dataset: str, length: int) -> Path | None:
    matches = sorted(T2S_NATIVE_ROOT.glob(f"flowmatching_DiT_{dataset}_{length}_*"))
    return matches[-1] if matches else None


def _native_t2s_pair(dataset: str, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int] | None:
    root = _t2s_generated_dir(dataset, length)
    if root is None:
        return None
    true_path = root / "x_1.npy"
    gen_path = root / "x_t.npy"
    if not true_path.exists() or not gen_path.exists():
        return None
    true_all = np.load(true_path)
    gen_all = np.load(gen_path)
    n = min(len(true_all), len(gen_all))
    if n == 0:
        return None
    index = int(rng.integers(0, n))
    return _as_array(true_all[index].squeeze()), _as_array(gen_all[index].squeeze()), index


@lru_cache(maxsize=None)
def _t2s_dataset_table(dataset: str, length: int) -> pd.DataFrame:
    root = REPO_ROOT / "baselines" / "T2S" / "Data" / "TSFragment-600K"
    if not root.exists():
        root = REPO_ROOT / "baselines" / "T2S" / "Data" / "Three Levels Data" / "TSFragment-600K"
    path = root / f"embedding_cleaned_{dataset}_{length}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing T2S dataset csv: {path}")
    return pd.read_csv(path, usecols=["Text", "OT"])


@lru_cache(maxsize=None)
def _t2s_scaler(dataset: str, length: int) -> MinMaxScaler:
    frame = _t2s_dataset_table(dataset, length)
    series_matrix = np.asarray([ast.literal_eval(item) for item in frame["OT"]], dtype=float)
    return MinMaxScaler().fit(series_matrix)


@lru_cache(maxsize=None)
def _t2s_native_bundle(dataset: str, length: int) -> dict[str, Any]:
    root = _t2s_generated_dir(dataset, length)
    if root is None:
        raise FileNotFoundError(f"missing T2S native generation directory for {dataset}_{length}")
    scaler = _t2s_scaler(dataset, length)
    true_all = scaler.inverse_transform(np.load(root / "x_1.npy").squeeze(-1))
    gen_all = scaler.inverse_transform(np.load(root / "x_t.npy").squeeze(-1))
    if true_all.shape != gen_all.shape:
        raise ValueError(f"T2S true/gen shape mismatch: {true_all.shape} vs {gen_all.shape}")
    captions_path = root / "y.npy"
    captions: list[str] | None = None
    if captions_path.exists():
        captions = [_normalize_caption_text(item) for item in np.load(captions_path, allow_pickle=True).tolist()]
    return {
        "root": root,
        "true": true_all,
        "generated": gen_all,
        "captions": captions,
    }


def _match_t2s_native_series(case: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    dataset = str(case["source_dataset"])
    length = int(case["source_length"])
    bundle = _t2s_native_bundle(dataset, length)
    reference = _as_array(case["reference_series"])
    native_true = np.asarray(bundle["true"], dtype=float)
    native_generated = np.asarray(bundle["generated"], dtype=float)
    captions = bundle["captions"]
    caption = _normalize_caption_text(case["description"])

    if captions is not None:
        matches = [index for index, value in enumerate(captions) if value == caption]
        if len(matches) == 1:
            index = matches[0]
            return _as_array(native_generated[index]), {
                "match_strategy": "caption_exact",
                "matched_index": index,
                "native_root": _repo_relative(Path(bundle["root"])),
            }

    if native_true.shape[1] != reference.shape[0]:
        raise ValueError(f"T2S native length {native_true.shape[1]} != reference length {reference.shape[0]}")
    errors = native_true - reference
    rmse_by_row = np.sqrt(np.mean(errors ** 2, axis=1))
    best_index = int(np.argmin(rmse_by_row))
    best_rmse = float(rmse_by_row[best_index])
    ref_span = float(np.max(reference) - np.min(reference))
    normalized_rmse = best_rmse / ref_span if ref_span > 1e-9 else best_rmse
    if normalized_rmse > 0.10:
        raise ValueError(
            f"could not reliably align T2S native sample (best normalized RMSE={normalized_rmse:.3f}, index={best_index})"
        )
    return _as_array(native_generated[best_index]), {
        "match_strategy": "nearest_reference",
        "matched_index": best_index,
        "match_rmse": best_rmse,
        "match_normalized_rmse": normalized_rmse,
        "native_root": _repo_relative(Path(bundle["root"])),
    }


def _t2s_candidates(
    require_all_llm: bool = False,
    llm_output_root: Path | None = None,
    require_native_match: bool = False,
) -> list[dict[str, Any]]:
    if not T2S_CASE_FILE.exists():
        return []
    required_methods = ["llm_direct_array", "llm_direct_code", "llm_strong_prompt"]
    cases = _load_json(T2S_CASE_FILE)
    candidates = []
    for case in cases:
        case_dir = T2S_OURS_ROOT / str(case["case_id"])
        if not (case_dir / "generated_series.json").exists() or not (case_dir / "reference_series.json").exists():
            continue
        if require_all_llm:
            if llm_output_root is None:
                raise ValueError("llm_output_root is required when require_all_llm=True")
            missing = [
                method
                for method in required_methods
                if not (llm_output_root / method / f"{case['case_id']}.txt").exists()
            ]
            if missing:
                continue
        candidates.append(case)
    return candidates


def _plot_t2s_samples(samples: list[dict[str, Any]], output_dir: Path, rng: np.random.Generator) -> list[dict[str, Any]]:
    if not samples:
        return []
    llm_output_root = output_dir / "t2s_llm_outputs"
    if not llm_output_root.exists():
        llm_output_root = DEFAULT_OUTPUT_DIR / "t2s_llm_outputs"
    fig, axes = plt.subplots(len(samples), 1, figsize=(14, 5.1 * len(samples)), squeeze=False)
    records: list[dict[str, Any]] = []
    for row, case in enumerate(samples):
        case_id = str(case["case_id"])
        case_dir = T2S_OURS_ROOT / case_id
        reference = _as_array(_load_json(case_dir / "reference_series.json"))
        ours = _as_array(_load_json(case_dir / "generated_series.json"))
        left = axes[row][0]
        method_records: dict[str, Any] = {
            "ours_full": {"status": "ok", "metrics": _metrics(reference, ours)}
        }
        try:
            t2s_native, native_match = _match_t2s_native_series(case)
            if len(t2s_native) != len(reference):
                raise ValueError(f"length {len(t2s_native)} != {len(reference)}")
            values = _metrics(reference, t2s_native)
            method_records["t2s_native"] = {
                "status": "ok",
                "metrics": values,
                "match": native_match,
            }
            left.plot(
                t2s_native,
                color=COLORS["t2s_native"],
                linewidth=1.35,
                alpha=0.92,
                label=f"t2s_native ({_metric_text(values)})",
            )
        except Exception as exc:
            method_records["t2s_native"] = {"status": "failed", "error": str(exc)}
        left.plot(
            ours,
            color=COLORS["ours_full"],
            linewidth=1.6,
            label=f"ours ({_metric_text(method_records['ours_full']['metrics'])})",
        )
        for method in ["llm_direct_array", "llm_direct_code", "llm_strong_prompt"]:
            output_path = llm_output_root / method / f"{case_id}.txt"
            try:
                if not output_path.exists():
                    raise FileNotFoundError(f"missing {output_path.relative_to(REPO_ROOT)}")
                text = output_path.read_text(encoding="utf-8")
                if method == "llm_direct_array":
                    generated = parse_direct_array_output(text, int(case["length"]))
                else:
                    generated = parse_direct_code_output(text, int(case["length"]))
                if len(generated) != len(reference):
                    raise ValueError(f"length {len(generated)} != {len(reference)}")
                values = _metrics(reference, generated)
                method_records[method] = {"status": "ok", "metrics": values}
                left.plot(
                    generated,
                    color=COLORS[method],
                    linewidth=1.25,
                    alpha=0.9,
                    label=f"{method} ({_metric_text(values)})",
                )
            except Exception as exc:
                method_records[method] = {"status": "failed", "error": str(exc)}
        left.plot(
            reference,
            color=COLORS["true"],
            linewidth=3.0,
            linestyle="--",
            label="real/raw",
            zorder=10,
        )
        failed = [method for method, result in method_records.items() if result["status"] != "ok"]
        note = "failed/missing: " + ", ".join(failed) if failed else "all available baselines parsed"
        left.text(
            0.99,
            0.04,
            note,
            transform=left.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )
        caption = textwrap.fill(str(case["description"]), width=120)
        left.set_title(
            f"{case_id}: TSFragment-600K raw sample ({case['source_dataset']}, L={case['source_length']}, row={case['source_row']})",
            fontsize=10,
            loc="left",
        )
        left.text(
            0.0,
            1.08,
            f"Caption: {caption}",
            transform=left.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            clip_on=False,
            bbox={"facecolor": "#fbfbfb", "edgecolor": "#dddddd", "alpha": 0.96},
        )
        left.grid(alpha=0.25)
        left.legend(fontsize=7)

        records.append(
            {
                "case_id": case_id,
                "dataset": case["source_dataset"],
                "length": case["source_length"],
                "source_row": case["source_row"],
                "description": case["description"],
                "ours_metrics": _metrics(reference, ours),
                "methods": method_records,
            }
        )
    axes[-1][0].set_xlabel("timestep")
    fig.tight_layout()
    fig.savefig(output_dir / "t2s_random_samples.png", dpi=180)
    plt.close(fig)
    return records


def _write_markdown(payload: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Random Baseline Samples",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Gold cases sampled: `{len(payload['gold_samples'])}`",
        f"- T2S cases sampled: `{len(payload['t2s_samples'])}`",
        "",
        "Figures:",
        "- `gold_random_samples.png`: true/oracle vs ours/rule-parser/LLM baselines on executable gold cases.",
        "- `t2s_random_samples.png`: TSFragment-600K raw samples with caption, real/raw, T2S native, ours, and available LLM direct baselines on the same subplot.",
        "",
        "Why method counts differ:",
        "- Gold/synthetic cases have saved outputs for ours, rule_parser, and LLM direct baselines; T2S native was not run on these cases.",
        "- T2S real-caption cases are rendered as same-panel multi-method comparisons: real/raw, T2S native when a reliable alignment is recoverable, ours, and any LLM direct outputs found under `t2s_llm_outputs/`.",
        "- If a method output fails parsing or safe execution, it is listed as failed/missing instead of being plotted as a curve.",
        "",
        "## Gold Samples",
        "",
    ]
    for item in payload["gold_samples"]:
        lines.append(f"### {item['case_id']}")
        lines.append(item["description"])
        for method, result in item["methods"].items():
            if result["status"] == "ok":
                lines.append(f"- {method}: {_metric_text(result['metrics'])}")
            else:
                lines.append(f"- {method}: failed ({result['error']})")
        lines.append("")
    lines.append("## T2S Samples")
    lines.append("")
    for item in payload["t2s_samples"]:
        lines.append(f"### {item['case_id']}")
        lines.append(item["description"])
        for method in ["t2s_native", "ours_full", "llm_direct_array", "llm_direct_code", "llm_strong_prompt"]:
            result = item.get("methods", {}).get(method)
            if result is None:
                continue
            if result["status"] == "ok":
                lines.append(f"- {method}: {_metric_text(result['metrics'])}")
            else:
                lines.append(f"- {method}: failed ({result['error']})")
        lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot random sample-level baseline comparisons from saved artifacts.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gold-samples", type=int, default=4)
    parser.add_argument("--t2s-samples", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--t2s-require-all-llm", action="store_true")
    parser.add_argument("--exclude-t2s-case-id", action="append", default=[])
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gold_candidates = _gold_candidates()
    if len(gold_candidates) < args.gold_samples:
        raise SystemExit(f"Only {len(gold_candidates)} gold candidates are available.")
    gold_samples = []
    if args.gold_samples > 0:
        gold_indices = rng.choice(len(gold_candidates), size=args.gold_samples, replace=False)
        gold_samples = [gold_candidates[int(index)] for index in gold_indices]

    llm_output_root = args.output_dir / "t2s_llm_outputs"
    if not llm_output_root.exists():
        llm_output_root = DEFAULT_OUTPUT_DIR / "t2s_llm_outputs"
    t2s_candidates = _t2s_candidates(
        require_all_llm=args.t2s_require_all_llm,
        llm_output_root=llm_output_root,
        require_native_match=True,
    )
    excluded_t2s = {str(case_id) for case_id in args.exclude_t2s_case_id}
    if excluded_t2s:
        t2s_candidates = [case for case in t2s_candidates if str(case["case_id"]) not in excluded_t2s]
    if len(t2s_candidates) < args.t2s_samples:
        raise SystemExit(f"Only {len(t2s_candidates)} T2S candidates are available.")
    t2s_samples = []
    if args.t2s_samples > 0:
        t2s_order = rng.permutation(len(t2s_candidates))
        for index in t2s_order:
            case = t2s_candidates[int(index)]
            try:
                _match_t2s_native_series(case)
            except Exception:
                continue
            t2s_samples.append(case)
            if len(t2s_samples) == args.t2s_samples:
                break
        if len(t2s_samples) < args.t2s_samples:
            raise SystemExit(
                f"Only {len(t2s_samples)} T2S candidates could be reliably aligned with T2S native outputs."
            )

    payload = {
        "seed": args.seed,
        "display_methods": ALL_DISPLAY_METHODS,
        "gold_samples": _plot_gold_samples(gold_samples, args.output_dir),
        "t2s_samples": _plot_t2s_samples(t2s_samples, args.output_dir, rng),
        "outputs": {
            "gold_figure": _repo_relative(args.output_dir / "gold_random_samples.png") if gold_samples else None,
            "t2s_figure": _repo_relative(args.output_dir / "t2s_random_samples.png") if t2s_samples else None,
            "selected_t2s_cases": _repo_relative(args.output_dir / "selected_t2s_cases.json") if t2s_samples else None,
        },
    }
    if t2s_samples:
        selected_cases_path = args.output_dir / "selected_t2s_cases.json"
        selected_cases_path.write_text(
            json.dumps(t2s_samples, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "sample_metrics.json").write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(payload, args.output_dir)
    if gold_samples:
        print(f"Wrote {args.output_dir / 'gold_random_samples.png'}")
    if t2s_samples:
        print(f"Wrote {args.output_dir / 't2s_random_samples.png'}")
    print(f"Wrote {args.output_dir / 'sample_metrics.json'}")
    print(f"Wrote {args.output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
