"""Select and plot representative LLM direct-generation failures.

This script traverses the controlled gold cases and the shared 2500-case
TSFragment-Eval split using existing benchmark artifacts only. It records cases where
FaithTS succeeds while direct array/code generation either fails
structurally or violates the applicable semantic contract. Controlled gold
quality labels use a conservative caption-only policy: oracle/reference-only
mismatch is not counted as bad quality unless the caption states the relevant
baseline, window, or persistence contract explicitly. TSFragment quality labels
use the caption-level semantic-consistency report, not reference similarity.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.baselines.llm_direct import (  # noqa: E402
    _execute_permissive_direct_code,
    parse_direct_array_output,
    parse_direct_code_output,
)
from evals.run_baseline_benchmark import _evaluate_saved_plan_output, _execute_plan_payload  # noqa: E402
from faithts_pipeline.workflow import execute_generated_code  # noqa: E402
from faithts.gold_adapter import oracle_plan_from_gold_case  # noqa: E402


GOLD_CASE_FILES = [
    REPO_ROOT / "data" / "controlled_gold_cases" / "single_claim_cases.json",
    REPO_ROOT / "data" / "controlled_gold_cases" / "multi_claim_cases.json",
    REPO_ROOT / "data" / "controlled_gold_cases" / "robustness_cases.json",
]
T2S_CASE_FILE = REPO_ROOT / "data" / "tsfragment_eval" / "tsfragment_eval_2500_cases.json"
BASELINE_RESULTS_PATH = REPO_ROOT / "evals" / "reports" / "baseline_results.json"
BASELINE_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "baseline_outputs"
T2S_LLM_EVAL_PATH = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_eval_2500.json"
T2S_SEMANTIC_PATH = REPO_ROOT / "evals" / "reports" / "t2s_semantic_consistency_eval_2500_reprocessed_v3_full.json"
T2S_LLM_OUTPUT_ROOT = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_outputs_2500"
T2S_OURS_ROOT = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_cases_2500_reprocessed_v2"
T2S_OURS_EVAL_PATH = REPO_ROOT / "evals" / "reports" / "faithts_main_benchmark_eval_2500_reprocessed_v2.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evals" / "reports" / "representative_failure_comparisons"

METHODS = ["llm_direct_array", "llm_direct_code"]
METHOD_LABELS = {
    "reference": "Reference/Oracle",
    "ours": "Ours",
    "llm_direct_array": "LLM Direct Array",
    "llm_direct_code": "LLM Direct Code",
}
METHOD_COLORS = {
    "reference": "#111111",
    "ours": "#1f77b4",
    "llm_direct_array": "#d62728",
    "llm_direct_code": "#9467bd",
}
ALLOWED_PRIMITIVES = [
    "add_change_point",
    "add_decay",
    "add_dip",
    "add_flat",
    "add_gap",
    "add_growth",
    "add_level_shift",
    "add_noise",
    "add_outlier",
    "add_peak",
    "add_plateau",
    "add_ramp",
    "add_seasonality",
    "add_spike",
    "add_trend",
    "add_trough",
    "add_volatility",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _as_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.ndim != 1:
        raise ValueError(f"expected one-dimensional series, got shape {array.shape}")
    return array


def _wape(reference: np.ndarray, generated: np.ndarray) -> float:
    denom = float(np.sum(np.abs(reference)))
    num = float(np.sum(np.abs(reference - generated)))
    if denom <= 1e-12:
        return 0.0 if num <= 1e-12 else float("inf")
    return num / denom


def _corr(reference: np.ndarray, generated: np.ndarray) -> float:
    if reference.size < 2 or float(np.std(reference)) <= 1e-12 or float(np.std(generated)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(reference, generated)[0, 1])


def _metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    if len(reference) != len(generated):
        raise ValueError(f"length {len(generated)} != reference length {len(reference)}")
    span = float(np.max(reference) - np.min(reference)) if len(reference) else 0.0
    rmse = float(np.sqrt(np.mean((reference - generated) ** 2)))
    mae = float(np.mean(np.abs(reference - generated)))
    return {
        "mae": mae,
        "rmse": rmse,
        "wape": _wape(reference, generated),
        "correlation": _corr(reference, generated),
        "range_normalized_rmse": rmse / span if span > 1e-12 else (0.0 if rmse <= 1e-12 else float("inf")),
        "max_abs_error": float(np.max(np.abs(reference - generated))) if len(reference) else 0.0,
    }


def _metric_brief(metrics: dict[str, float] | None) -> str:
    if not metrics:
        return "no comparable metrics"
    corr = metrics.get("correlation")
    corr_text = "nan" if corr is None or math.isnan(corr) else f"{corr:.3f}"
    wape = metrics.get("wape")
    wape_text = "inf" if wape is not None and math.isinf(wape) else f"{float(wape):.3f}"
    return f"WAPE={wape_text}, corr={corr_text}, RMSE={metrics['rmse']:.3g}"


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    return match.group(1).strip() if match else cleaned


def _parse_raw_array(text: str) -> np.ndarray:
    cleaned = _strip_fence(text)
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            payload = json.loads(cleaned[start : end + 1])
            if isinstance(payload, dict) and isinstance(payload.get("series"), list):
                return _as_array(payload["series"])
    except Exception:
        pass
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match is None:
        raise ValueError("no numeric array found")
    return _as_array(json.loads(match.group(0)))


def _parse_raw_code(text: str) -> np.ndarray:
    try:
        return _as_array(execute_generated_code(text, ALLOWED_PRIMITIVES))
    except Exception:
        return _as_array(_execute_permissive_direct_code(text))


def _parse_method_output(method: str, text: str, length: int) -> dict[str, Any]:
    strict_error: str | None = None
    try:
        if method == "llm_direct_array":
            series = parse_direct_array_output(text, length)
        else:
            series = parse_direct_code_output(text, length)
        return {
            "parse_status": "ok",
            "strict_error": None,
            "series": _as_array(series),
            "raw_length": int(len(series)),
        }
    except Exception as exc:
        strict_error = str(exc)

    raw_series: np.ndarray | None = None
    raw_error: str | None = None
    try:
        raw_series = _parse_raw_array(text) if method == "llm_direct_array" else _parse_raw_code(text)
    except Exception as exc:
        raw_error = str(exc)

    if raw_series is not None and len(raw_series) != length:
        parse_status = "length_mismatch"
    elif raw_series is not None:
        parse_status = "strict_parse_failed"
    else:
        parse_status = "parse_or_execution_failed"
    return {
        "parse_status": parse_status,
        "strict_error": strict_error,
        "raw_error": raw_error,
        "series": raw_series,
        "raw_length": int(len(raw_series)) if raw_series is not None else None,
    }


def _baseline_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    payload = _load_json(BASELINE_RESULTS_PATH)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for method_block in payload["results"]:
        method = method_block["method"]
        for record in method_block["cases"]:
            lookup[(method, record["case_id"])] = record
    return lookup


def _t2s_eval_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    payload = _load_json(T2S_LLM_EVAL_PATH)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for method_block in payload["results"]:
        method = method_block["method"]
        for record in method_block["records"]:
            lookup[(method, record["case_id"])] = record
    return lookup


def _t2s_semantic_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    if not T2S_SEMANTIC_PATH.exists():
        return {}
    payload = _load_json(T2S_SEMANTIC_PATH)
    label_to_method = {
        "Reference": "reference",
        "LLM Direct Array": "llm_direct_array",
        "LLM Direct Code": "llm_direct_code",
        "Ours": "ours",
    }
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for label, records in payload.get("records", {}).items():
        method = label_to_method.get(label)
        if method is None:
            continue
        for record in records:
            lookup[(method, record["case_id"])] = record
    return lookup


def _failure_codes(record: dict[str, Any] | None) -> list[str]:
    if not record:
        return []
    return [str(reason.get("code", "")) for reason in record.get("failure_reasons", [])]


def _failure_messages(record: dict[str, Any] | None) -> list[str]:
    if not record:
        return []
    return [str(reason.get("message", "")) for reason in record.get("failure_reasons", [])]


def _controlled_has_explicit_contract(description: str) -> bool:
    text = " ".join(description.lower().split())
    phrases = [
        "flat at",
        "stays flat",
        "remains flat",
        "across all",
        "from timestep 0",
        "to 99",
        "through the end",
        "plateau",
        "local window",
        "same baseline",
        "baseline",
        "new level",
        "levels off",
        "not a",
        "rather than",
        "after timestep",
    ]
    return any(phrase in text for phrase in phrases)


def _semantic_verdict_status(semantic_record: dict[str, Any] | None) -> str | None:
    if not semantic_record:
        return None
    status = semantic_record.get("summary_verdict")
    if status is None:
        return None
    return str(status)


def _is_caption_semantic_issue(status: str | None) -> bool:
    return status in {"partial", "partially_consistent", "inconsistent"}


def _caption_semantic_reason(semantic_record: dict[str, Any] | None) -> str:
    if not semantic_record:
        return "caption checker unavailable"
    status = _semantic_verdict_status(semantic_record)
    score = semantic_record.get("semantic_consistency_score")
    claims = semantic_record.get("extracted_claims") or []
    claim_text = ", ".join(str(claim) for claim in claims[:4])
    if len(claims) > 4:
        claim_text += ", ..."
    score_text = "NA" if score is None else f"{float(score):.3f}"
    return f"caption checker verdict={status}, score={score_text}, claims=[{claim_text}]"


def _is_poor_quality(dataset: str, metrics: dict[str, float] | None, semantic_failed: bool) -> bool:
    if semantic_failed:
        return True
    if dataset == "controlled_gold":
        return False
    return False


def _issue_reason(
    *,
    dataset: str,
    method: str,
    parse_status: str,
    raw_length: int | None,
    length: int,
    strict_error: str | None,
    metrics: dict[str, float] | None,
    baseline_record: dict[str, Any] | None = None,
    semantic_record: dict[str, Any] | None = None,
    controlled_explicit_contract: bool = True,
) -> tuple[str, str, float]:
    raw_semantic_failed = "F_semantic" in _failure_codes(baseline_record)
    semantic_failed = raw_semantic_failed
    if dataset == "controlled_gold" and raw_semantic_failed and not controlled_explicit_contract:
        semantic_failed = False
    status = _semantic_verdict_status(semantic_record)
    semantic_inconsistent = _is_caption_semantic_issue(status)
    if parse_status == "missing":
        return "missing", "output file missing", 10.0
    if parse_status == "length_mismatch":
        return (
            "hard_fail",
            f"length mismatch: generated {raw_length}, expected {length}",
            9.0 + (abs((raw_length or 0) - length) / max(length, 1)),
        )
    if parse_status in {"parse_or_execution_failed", "strict_parse_failed"}:
        message = strict_error or "parse/execution failed"
        return "hard_fail", message[:180], 8.5
    if _is_poor_quality(dataset, metrics, semantic_failed or semantic_inconsistent):
        evidence = _metric_brief(metrics)
        if semantic_failed:
            messages = "; ".join(_failure_messages(baseline_record))
            return "poor_quality", f"semantic checker failed ({messages}); {evidence}", 6.5
        if semantic_inconsistent:
            return "poor_quality", _caption_semantic_reason(semantic_record), 6.0
    if raw_semantic_failed and not semantic_failed:
        messages = "; ".join(_failure_messages(baseline_record))
        return (
            "ok",
            f"ok under caption-only policy; ignored closed-world gold check ({messages}); {_metric_brief(metrics)}",
            0.0,
        )
    if dataset == "tsfragment" and semantic_record:
        return "ok", f"caption ok: {_caption_semantic_reason(semantic_record)}", 0.0
    return "ok", f"ok: {_metric_brief(metrics)}", 0.0


def _oracle_series(case: dict[str, Any]) -> np.ndarray:
    plan = oracle_plan_from_gold_case(case)
    series, ok, errors = _execute_plan_payload(plan, int(case["length"]))
    if not ok or series is None:
        raise ValueError("; ".join(errors) or "oracle plan did not execute")
    return _as_array(series)


def _controlled_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_file in GOLD_CASE_FILES:
        data = _load_json(case_file)
        for index, case in enumerate(data):
            item = dict(case)
            item["_case_file"] = _repo_relative(case_file)
            item["_case_file_index"] = index
            cases.append(item)
    return cases


def _evaluate_direct_method(
    *,
    dataset: str,
    method: str,
    case_id: str,
    length: int,
    reference: np.ndarray,
    output_root: Path,
    baseline_record: dict[str, Any] | None = None,
    semantic_record: dict[str, Any] | None = None,
    case_description: str = "",
) -> dict[str, Any]:
    path = output_root / method / f"{case_id}.txt"
    if not path.exists():
        parse = {"parse_status": "missing", "strict_error": None, "series": None, "raw_length": None}
    else:
        parse = _parse_method_output(method, path.read_text(encoding="utf-8"), length)

    series = parse.get("series")
    metrics = None
    if isinstance(series, np.ndarray) and len(series) == len(reference):
        metrics = _metrics(reference, series)

    issue_type, reason, severity = _issue_reason(
        dataset=dataset,
        method=method,
        parse_status=str(parse["parse_status"]),
        raw_length=parse.get("raw_length"),
        length=length,
        strict_error=parse.get("strict_error"),
        metrics=metrics,
        baseline_record=baseline_record,
        semantic_record=semantic_record,
        controlled_explicit_contract=_controlled_has_explicit_contract(case_description),
    )
    raw_semantic_failed = "F_semantic" in _failure_codes(baseline_record)
    controlled_explicit_contract = _controlled_has_explicit_contract(case_description)
    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "issue_type": issue_type,
        "reason": reason,
        "severity": severity,
        "parse_status": parse["parse_status"],
        "strict_error": parse.get("strict_error"),
        "raw_error": parse.get("raw_error"),
        "raw_length": parse.get("raw_length"),
        "metrics": metrics,
        "series": series,
        "output_path": _repo_relative(path) if path.exists() else None,
        "semantic_verdict": semantic_record.get("summary_verdict") if semantic_record else None,
        "semantic_score": semantic_record.get("semantic_consistency_score") if semantic_record else None,
        "raw_semantic_failed": raw_semantic_failed,
        "controlled_explicit_contract": controlled_explicit_contract if dataset == "controlled_gold" else None,
        "ignored_closed_world_semantic_failure": (
            dataset == "controlled_gold"
            and raw_semantic_failed
            and not controlled_explicit_contract
            and issue_type == "ok"
        ),
    }


def _screen_controlled() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = _baseline_lookup()
    candidates: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for case in _controlled_cases():
        case_id = str(case["case_id"])
        length = int(case["length"])
        ours_path = BASELINE_OUTPUT_ROOT / "ours_full" / f"{case_id}.json"
        if not ours_path.exists():
            continue
        try:
            reference = _oracle_series(case)
        except Exception:
            # Correct-rejection adversarial cases do not have a comparable
            # oracle signal for plotting, so they are not useful here.
            continue
        _, repaired_ours, ours_ok, _ = _evaluate_saved_plan_output(
            "ours_full",
            case,
            BASELINE_OUTPUT_ROOT / "ours_full",
        )
        if not ours_ok or repaired_ours is None:
            continue
        ours_series = _as_array(repaired_ours)
        if len(ours_series) != length:
            continue
        ours_metrics = _metrics(reference, ours_series)
        methods = {}
        for method in METHODS:
            methods[method] = _evaluate_direct_method(
                dataset="controlled_gold",
                method=method,
                case_id=case_id,
                length=length,
                reference=reference,
                output_root=BASELINE_OUTPUT_ROOT,
                baseline_record=baseline.get((method, case_id)),
                case_description=str(case["description"]),
            )
        row = _case_row(
            dataset="controlled_gold",
            case=case,
            reference=reference,
            ours_series=ours_series,
            ours_metrics=ours_metrics,
            methods=methods,
            index_fields={
                "case_file": case["_case_file"],
                "case_file_index": case["_case_file_index"],
                "case_file_index_1based": case["_case_file_index"] + 1,
            },
        )
        all_rows.append(row)
        if row["issue_count"] > 0:
            candidates.append(row)
    return candidates, all_rows


def _screen_t2s() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    llm_eval = _t2s_eval_lookup()
    semantic = _t2s_semantic_lookup()
    ours_eval = {record["case_id"]: record for record in _load_json(T2S_OURS_EVAL_PATH)["records"]}
    cases = _load_json(T2S_CASE_FILE)
    candidates: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        case_id = str(case["case_id"])
        length = int(case["length"])
        ours_record = ours_eval.get(case_id, {})
        case_dir = T2S_OURS_ROOT / case_id
        ours_path = case_dir / "generated_series.json"
        if ours_record.get("status") != "success" or not ours_path.exists():
            continue
        reference = _as_array(case["reference_series"])
        ours_series = _as_array(_load_json(ours_path))
        if len(ours_series) != length:
            continue
        ours_semantic = semantic.get(("ours", case_id))
        if _semantic_verdict_status(ours_semantic) != "consistent":
            continue
        reference_semantic = semantic.get(("reference", case_id))
        if _semantic_verdict_status(reference_semantic) != "consistent":
            continue
        ours_metrics = _metrics(reference, ours_series)
        methods = {}
        for method in METHODS:
            methods[method] = _evaluate_direct_method(
                dataset="tsfragment",
                method=method,
                case_id=case_id,
                length=length,
                reference=reference,
                output_root=T2S_LLM_OUTPUT_ROOT,
                baseline_record=llm_eval.get((method, case_id)),
                semantic_record=semantic.get((method, case_id)),
                case_description=str(case["description"]),
            )
        row = _case_row(
            dataset="tsfragment",
            case=case,
            reference=reference,
            ours_series=ours_series,
            ours_metrics=ours_metrics,
            methods=methods,
            index_fields={
                "case_file": _repo_relative(T2S_CASE_FILE),
                "case_file_index": case_index,
                "case_file_index_1based": case_index + 1,
                "source_index": case.get("source_index"),
                "source_row": case.get("source_row"),
                "source_dataset": case.get("source_dataset"),
                "source_length": case.get("source_length"),
            },
        )
        all_rows.append(row)
        if row["issue_count"] > 0:
            candidates.append(row)
    return candidates, all_rows


def _case_row(
    *,
    dataset: str,
    case: dict[str, Any],
    reference: np.ndarray,
    ours_series: np.ndarray,
    ours_metrics: dict[str, float],
    methods: dict[str, dict[str, Any]],
    index_fields: dict[str, Any],
) -> dict[str, Any]:
    issue_count = sum(1 for method in methods.values() if method["issue_type"] != "ok")
    hard_count = sum(1 for method in methods.values() if method["issue_type"] in {"hard_fail", "missing"})
    poor_count = sum(1 for method in methods.values() if method["issue_type"] == "poor_quality")
    score = sum(float(method["severity"]) for method in methods.values())
    if issue_count == len(METHODS):
        score += 3.0
    if hard_count and poor_count:
        score += 2.0
    return {
        "dataset": dataset,
        "case_id": case["case_id"],
        "category": case.get("category"),
        "description": case["description"],
        "length": int(case["length"]),
        "index": index_fields,
        "issue_count": issue_count,
        "hard_count": hard_count,
        "poor_count": poor_count,
        "selection_score": score,
        "ours_status": "ok",
        "ours_metrics": ours_metrics,
        "methods": methods,
        "series": {
            "reference": reference,
            "ours": ours_series,
            "llm_direct_array": methods["llm_direct_array"]["series"],
            "llm_direct_code": methods["llm_direct_code"]["series"],
        },
    }


def _select_representatives(candidates: list[dict[str, Any]], count: int = 2) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_best(pool: list[dict[str, Any]]) -> None:
        for item in sorted(pool, key=lambda row: (row["selection_score"], row["issue_count"]), reverse=True):
            if item["case_id"] not in selected_ids:
                selected.append(item)
                selected_ids.add(item["case_id"])
                return

    # Prefer diversity over raw severity: one structural failure and one
    # length-correct quality/semantic failure makes the qualitative evidence
    # more informative than two near-duplicate length-mismatch examples.
    add_best([row for row in candidates if row["hard_count"] > 0 and row["issue_count"] >= 2])
    if len(selected) < count:
        add_best([row for row in candidates if row["hard_count"] == 0 and row["poor_count"] > 0 and row["issue_count"] >= 2])
    if len(selected) < count:
        add_best([row for row in candidates if row["hard_count"] > 0])
    if len(selected) < count:
        add_best(candidates)
    while len(selected) < count:
        remaining = [row for row in candidates if row["case_id"] not in selected_ids]
        if not remaining:
            break
        add_best(remaining)
    return selected[:count]


def _select_by_case_ids(candidates: list[dict[str, Any]], case_ids: list[str], dataset_name: str) -> list[dict[str, Any]]:
    by_id = {str(row["case_id"]): row for row in candidates}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise SystemExit(f"Missing {dataset_name} candidate case ids: {', '.join(missing)}")
    return [by_id[case_id] for case_id in case_ids]


def _selection_reason(row: dict[str, Any]) -> str:
    parts = []
    for method in METHODS:
        method_row = row["methods"][method]
        if method_row["issue_type"] != "ok":
            parts.append(f"{METHOD_LABELS[method]}: {method_row['reason']}")
    return " | ".join(parts) if parts else "No direct-baseline issue."


def _plot_rows(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    if not rows:
        return
    fig = plt.figure(figsize=(18, 5.2 * len(rows)), constrained_layout=True)
    spec = fig.add_gridspec(len(rows), 2, width_ratios=[2.25, 1.35])
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for row_index, row in enumerate(rows):
        ax = fig.add_subplot(spec[row_index, 0])
        text_ax = fig.add_subplot(spec[row_index, 1])
        text_ax.axis("off")

        for key in ["reference", "ours", "llm_direct_array", "llm_direct_code"]:
            series = row["series"].get(key)
            if series is None:
                continue
            array = _as_array(series)
            label = METHOD_LABELS[key]
            if key in row["methods"]:
                label += f" ({row['methods'][key]['issue_type']}, n={len(array)})"
            elif key == "ours":
                label += f" ({_metric_brief(row['ours_metrics'])})"
            else:
                label += f" (n={len(array)})"
            ax.plot(
                np.arange(len(array)),
                array,
                color=METHOD_COLORS[key],
                linewidth=2.8 if key == "reference" else (2.0 if key == "ours" else 1.35),
                linestyle="--" if key == "reference" else "-",
                alpha=0.95 if key in {"reference", "ours"} else 0.88,
                label=label,
            )

        ax.axvline(row["length"] - 1, color="#777777", linewidth=0.8, alpha=0.45)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.85)
        ax.set_xlabel("timestep")
        ax.set_ylabel("value")
        header = f"{row_index + 1}. {row['dataset']} | {row['case_id']} | L={row['length']}"
        ax.set_title(header, loc="left", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="best")

        index_lines = [f"{key}: {value}" for key, value in row["index"].items() if value is not None]
        method_lines = []
        for method in METHODS:
            method_row = row["methods"][method]
            method_lines.append(
                f"{METHOD_LABELS[method]} [{method_row['issue_type']}]: "
                f"{method_row['reason']}"
            )
        caption = textwrap.fill(str(row["description"]), width=62)
        reason = textwrap.fill(_selection_reason(row), width=62)
        metrics = textwrap.fill(
            "Ours ok: " + _metric_brief(row["ours_metrics"]),
            width=62,
        )
        text = "\n\n".join(
            [
                "Index:\n" + "\n".join(index_lines),
                "Caption:\n" + caption,
                "Selection reason:\n" + reason,
                "Method details:\n" + "\n".join(textwrap.fill(line, width=62) for line in method_lines),
                metrics,
            ]
        )
        text_ax.text(
            0.0,
            1.0,
            text,
            va="top",
            ha="left",
            fontsize=8.1,
            family="DejaVu Sans",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8f8f8", "edgecolor": "#cccccc"},
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_readme(
    *,
    output_dir: Path,
    controlled_candidates: list[dict[str, Any]],
    t2s_candidates: list[dict[str, Any]],
    selected: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Representative Failure Comparisons",
        "",
        "Generated from existing benchmark artifacts; no model calls were made.",
        "",
        "Screening policy:",
        "- Ours must have an existing successful generated series with the expected length.",
        "- TSFragment selected examples additionally require both the reference series and Ours to pass the caption-level semantic checker.",
        "- Direct LLM hard failures include missing output, parse/execution failure, and length mismatch.",
        "- Controlled poor quality is flagged only for explicit caption-contract semantic failures.",
        "- Controlled oracle/reference-only mismatch is not labeled bad quality under this plotting policy.",
        "- TSFragment poor quality is flagged only by caption-level semantic-consistency outcomes.",
        "- Reference-similarity metrics are retained for context but do not decide TSFragment failure labels.",
        "",
        "Outputs:",
        "- `candidate_indices.json`: all screened candidate cases with index provenance.",
        "- `selected_examples.json`: two selected controlled cases and two selected TSFragment cases.",
        "- `controlled_gold_examples.png`: selected controlled gold plots.",
        "- `tsfragment_examples.png`: selected TSFragment plots.",
        "- `combined_examples.png`: all four selected examples in one figure.",
        "",
        f"Controlled candidates: `{len(controlled_candidates)}`",
        f"TSFragment candidates: `{len(t2s_candidates)}`",
        "",
        "## Selected Controlled Gold Cases",
        "",
    ]
    for row in selected["controlled_gold"]:
        lines.append(f"- `{row['case_id']}` index `{row['index'].get('case_file_index')}` in `{row['index'].get('case_file')}`: {_selection_reason(row)}")
    lines.extend(["", "## Selected TSFragment Cases", ""])
    for row in selected["tsfragment"]:
        lines.append(
            f"- `{row['case_id']}` index `{row['index'].get('case_file_index')}` "
            f"(source row `{row['index'].get('source_row')}`): {_selection_reason(row)}"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _drop_series(row: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in row.items() if key != "series"}
    compact["method_preview"] = {}
    for method in METHODS:
        series = row["series"].get(method)
        compact["method_preview"][method] = None if series is None else _as_array(series)[:8].tolist()
        compact["methods"][method] = {key: value for key, value in compact["methods"][method].items() if key != "series"}
    compact["series_preview"] = {
        key: None if value is None else _as_array(value)[:8].tolist()
        for key, value in row["series"].items()
    }
    return compact


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Render representative LLM direct-generation failure comparisons.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-dataset", type=int, default=2)
    parser.add_argument("--controlled-case-id", action="append", default=[])
    parser.add_argument("--t2s-case-id", action="append", default=[])
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    controlled_candidates, controlled_all = _screen_controlled()
    t2s_candidates, t2s_all = _screen_t2s()
    selected = {
        "controlled_gold": (
            _select_by_case_ids(controlled_candidates, args.controlled_case_id, "controlled_gold")
            if args.controlled_case_id
            else _select_representatives(controlled_candidates, args.per_dataset)
        ),
        "tsfragment": (
            _select_by_case_ids(t2s_candidates, args.t2s_case_id, "tsfragment")
            if args.t2s_case_id
            else _select_representatives(t2s_candidates, args.per_dataset)
        ),
    }

    _write_json(
        output_dir / "candidate_indices.json",
        {
            "controlled_gold": [_drop_series(row) for row in sorted(controlled_candidates, key=lambda item: item["selection_score"], reverse=True)],
            "tsfragment": [_drop_series(row) for row in sorted(t2s_candidates, key=lambda item: item["selection_score"], reverse=True)],
            "screened_totals": {
                "controlled_gold_ours_ok": len(controlled_all),
                "tsfragment_ours_ok": len(t2s_all),
            },
        },
    )
    _write_json(
        output_dir / "selected_examples.json",
        {key: [_drop_series(row) for row in rows] for key, rows in selected.items()},
    )

    _plot_rows(selected["controlled_gold"], output_dir / "controlled_gold_examples.png", "Controlled Gold: Ours succeeds while direct LLM outputs fail or degrade")
    _plot_rows(selected["tsfragment"], output_dir / "tsfragment_examples.png", "TSFragment: Ours succeeds while direct LLM outputs fail or degrade")
    _plot_rows(
        selected["controlled_gold"] + selected["tsfragment"],
        output_dir / "combined_examples.png",
        "Representative direct LLM failures across controlled gold and TSFragment cases",
    )
    _render_readme(
        output_dir=output_dir,
        controlled_candidates=controlled_candidates,
        t2s_candidates=t2s_candidates,
        selected=selected,
    )

    print(f"Wrote {_repo_relative(output_dir / 'candidate_indices.json')}")
    print(f"Wrote {_repo_relative(output_dir / 'selected_examples.json')}")
    print(f"Wrote {_repo_relative(output_dir / 'controlled_gold_examples.png')}")
    print(f"Wrote {_repo_relative(output_dir / 'tsfragment_examples.png')}")
    print(f"Wrote {_repo_relative(output_dir / 'combined_examples.png')}")
    print(f"Wrote {_repo_relative(output_dir / 'README.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
