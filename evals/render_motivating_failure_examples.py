"""Render motivating failure examples from existing benchmark artifacts.

The script does not call any model. It selects six existing cases, records
their indices/provenance, and draws a comparison figure with the caption and
the selected baseline failure reason.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "evals" / "reports" / "motivating_failure_examples"


GOLD_SELECTIONS = [
    {
        "case_id": "adv_spike_not_peak",
        "case_file": "data/controlled_gold_cases/robustness_cases.json",
        "workflow_dir": "evals/full_adversarial_workflow_cases/adv_spike_not_peak",
        "failed_method": "llm_direct_code",
        "failed_output_path": "evals/reports/baseline_outputs/llm_direct_code/adv_spike_not_peak.txt",
        "failure_reason": (
            "Direct-to-code emits a raw NumPy script and sets the impulse to 10 "
            "instead of the checked one-step add_spike target near 4."
        ),
        "selection_reason": "single-timestep local event precision",
    },
    {
        "case_id": "adv_shift_not_plateau",
        "case_file": "data/controlled_gold_cases/robustness_cases.json",
        "workflow_dir": "evals/full_adversarial_workflow_cases/adv_shift_not_plateau",
        "failed_method": "llm_direct_array",
        "failed_output_path": "evals/reports/baseline_outputs/llm_direct_array/adv_shift_not_plateau.txt",
        "failure_reason": (
            "Direct-to-array uses a sustained level of 1.0, but the gold "
            "semantic program checks a level shift of 1.5 from timestep 50 onward."
        ),
        "selection_reason": "regime-shift amplitude and persistence",
    },
    {
        "case_id": "combo_noise_then_volatility_scale",
        "case_file": "data/controlled_gold_cases/multi_claim_cases.json",
        "workflow_dir": "evals/full_compositional_workflow_cases/combo_noise_then_volatility_scale",
        "failed_method": "llm_direct_array",
        "failed_output_path": "evals/reports/baseline_outputs/llm_direct_array/combo_noise_then_volatility_scale.txt",
        "failure_reason": (
            "Direct-to-array returns the wrong length and has no auditable seed or "
            "primitive boundary for the noise-plus-volatility composition."
        ),
        "selection_reason": "composition, length, stochastic reproducibility",
    },
]


TSFRAGMENT_SELECTIONS = [
    {
        "case_id": "t2s_ETTh1_96_00154",
        "failed_method": "llm_direct_array",
        "failed_output_path": "evals/reports/t2s_llm_main_benchmark_outputs/llm_direct_array/t2s_ETTh1_96_00154.txt",
        "failure_reason": (
            "Direct-to-array length check fails: 101 generated values for a "
            "required length of 96; direct-to-code also produced only 88 values."
        ),
        "selection_reason": "real-caption length control with good VTS match",
    },
    {
        "case_id": "t2s_exchangerate_96_00043",
        "failed_method": "llm_direct_array",
        "failed_output_path": "evals/reports/t2s_llm_main_benchmark_outputs/llm_direct_array/t2s_exchangerate_96_00043.txt",
        "failure_reason": (
            "Direct-to-array length check fails: 90 generated values for a "
            "required length of 96, despite an explicit length requirement."
        ),
        "selection_reason": "real-caption truncation under numeric decoding",
    },
    {
        "case_id": "t2s_electricity_96_00002",
        "failed_method": "llm_direct_array",
        "failed_output_path": "evals/reports/t2s_llm_main_benchmark_outputs/llm_direct_array/t2s_electricity_96_00002.txt",
        "failure_reason": (
            "Direct-to-array length check fails: 102 generated values for a "
            "required length of 96, while VTS emits a schema-valid length-96 program."
        ),
        "selection_reason": "real-caption over-generation under numeric decoding",
    },
]


def rel(path: str | Path) -> Path:
    return REPO_ROOT / path


def read_json(path: str | Path) -> Any:
    with rel(path).open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def strip_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def read_array_output(path: str | Path) -> list[float]:
    raw = strip_fence(rel(path).read_text())
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "series" in parsed:
        parsed = parsed["series"]
    if not isinstance(parsed, list):
        raise TypeError(f"Expected an array in {path}")
    return [float(v) for v in parsed]


def read_code_series(path: str | Path) -> list[float]:
    """Execute a reviewed direct-code snippet in a narrow namespace."""

    code = strip_fence(rel(path).read_text())
    code = "\n".join(
        line for line in code.splitlines() if not line.strip().startswith("import ")
    )
    namespace: dict[str, Any] = {
        "np": np,
        "range": range,
        "len": len,
        "float": float,
        "int": int,
        "max": max,
        "min": min,
        "abs": abs,
        "__builtins__": {},
    }
    exec(code, namespace, namespace)
    series = namespace.get("series")
    if series is None:
        raise RuntimeError(f"No `series` produced by {path}")
    return np.asarray(series, dtype=float).ravel().tolist()


def read_failed_series(method: str, path: str | Path) -> list[float]:
    if method.endswith("code"):
        return read_code_series(path)
    return read_array_output(path)


def find_case_index(case_file: str, case_id: str) -> tuple[int, dict[str, Any]]:
    cases = read_json(case_file)
    for index, case in enumerate(cases):
        if case.get("case_id") == case_id:
            return index, case
    raise KeyError(f"{case_id} not found in {case_file}")


def baseline_result_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    result = read_json("evals/reports/baseline_results.json")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for method_block in result["results"]:
        method = method_block["method"]
        for case in method_block["cases"]:
            lookup[(method, case["case_id"])] = case
    return lookup


def t2s_eval_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    result = read_json("evals/reports/t2s_llm_main_benchmark_eval.json")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for method_block in result["results"]:
        method = method_block["method"]
        for record in method_block["records"]:
            lookup[(method, record["case_id"])] = record
    return lookup


def t2s_ours_lookup() -> dict[str, dict[str, Any]]:
    result = read_json("evals/reports/t2s_ours_main_benchmark_after_fix_reprocessed_eval.json")
    return {record["case_id"]: record for record in result["records"]}


def plan_primitives(plan: dict[str, Any]) -> list[str]:
    return [
        step["primitive"]
        for step in plan.get("steps", [])
        if step.get("kind") == "primitive" and "primitive" in step
    ]


def build_gold_examples() -> list[dict[str, Any]]:
    baseline_lookup = baseline_result_lookup()
    examples = []
    for selected in GOLD_SELECTIONS:
        case_index, case = find_case_index(selected["case_file"], selected["case_id"])
        workflow_dir = rel(selected["workflow_dir"])
        ours = read_json(f"evals/reports/baseline_outputs/ours_full/{selected['case_id']}.json")
        failed_series = read_failed_series(
            selected["failed_method"], selected["failed_output_path"]
        )
        reference_series = read_json(workflow_dir / "reference_series.json")
        vts_series = ours["generated_series"]
        baseline_result = baseline_lookup.get(
            (selected["failed_method"], selected["case_id"]), {}
        )
        ours_result = baseline_lookup.get(("ours_full", selected["case_id"]), {})
        examples.append(
            {
                "source": "gold_cases",
                "case_id": selected["case_id"],
                "case_file": selected["case_file"],
                "case_file_index": case_index,
                "category": case.get("category"),
                "caption": case["description"],
                "required_length": case["length"],
                "failed_method": selected["failed_method"],
                "failed_output_path": selected["failed_output_path"],
                "failed_length": len(failed_series),
                "reference_length": len(reference_series),
                "vts_length": len(vts_series),
                "failure_reason": selected["failure_reason"],
                "selection_reason": selected["selection_reason"],
                "vts_primitives": plan_primitives(ours["plan"]),
                "vts_plan_valid": bool(ours_result.get("passed")),
                "failed_baseline_metrics": baseline_result.get("metrics", {}),
                "failed_baseline_failure_reasons": baseline_result.get("failure_reasons", []),
                "series": {
                    "reference": reference_series,
                    "vts": vts_series,
                    "failed": failed_series,
                },
            }
        )
    return examples


def build_tsfragment_examples() -> list[dict[str, Any]]:
    t2s_cases = {case["case_id"]: (i, case) for i, case in enumerate(read_json("data/tsfragment_eval/tsfragment_eval_2500_cases.json"))}
    t2s_fail_lookup = t2s_eval_lookup()
    t2s_ours = t2s_ours_lookup()
    examples = []
    for selected in TSFRAGMENT_SELECTIONS:
        case_index, case = t2s_cases[selected["case_id"]]
        artifact_dir = Path("evals/reports/faithts_main_benchmark_cases_2500_reprocessed_v2") / selected["case_id"]
        reference_series = read_json(artifact_dir / "reference_series.json")
        vts_series = read_json(artifact_dir / "generated_series.json")
        failed_series = read_failed_series(
            selected["failed_method"], selected["failed_output_path"]
        )
        fail_record = t2s_fail_lookup.get((selected["failed_method"], selected["case_id"]), {})
        ours_record = t2s_ours[selected["case_id"]]
        structured_plan = read_json(artifact_dir / "structured_plan.json")
        examples.append(
            {
                "source": "TSFragment_real_caption",
                "case_id": selected["case_id"],
                "case_file": "data/tsfragment_eval/tsfragment_eval_2500_cases.json",
                "case_file_index": case_index,
                "source_index": case.get("source_index"),
                "source_dataset": case.get("source_dataset"),
                "source_length": case.get("source_length"),
                "caption": case["description"],
                "required_length": case["length"],
                "failed_method": selected["failed_method"],
                "failed_output_path": selected["failed_output_path"],
                "failed_length": len(failed_series),
                "reference_length": len(reference_series),
                "vts_length": len(vts_series),
                "failure_reason": selected["failure_reason"],
                "selection_reason": selected["selection_reason"],
                "failure_record_error": fail_record.get("error"),
                "vts_primitives": plan_primitives(structured_plan),
                "vts_plan_validation_path": str(artifact_dir / "plan_schema_validation.json"),
                "vts_metrics": ours_record.get("comparison_metrics", {}),
                "series": {
                    "reference": reference_series,
                    "vts": vts_series,
                    "failed": failed_series,
                },
            }
        )
    return examples


def compact_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for example in examples:
        item = {k: v for k, v in example.items() if k != "series"}
        item["series_preview"] = {
            name: values[:8] for name, values in example["series"].items()
        }
        compact.append(item)
    return compact


def render_markdown(examples: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Motivating Failure Example Candidates",
        "",
        "These examples are selected from existing repository artifacts. No new model generation was run.",
        "",
    ]
    for example in examples:
        lines.extend(
            [
                f"## {example['source']} / {example['case_id']}",
                "",
                f"- Case index: `{example['case_file_index']}` in `{example['case_file']}`",
                f"- Source index: `{example.get('source_index', 'n/a')}`",
                f"- Failed method: `{example['failed_method']}`",
                f"- Lengths: failed `{example['failed_length']}`, required `{example['required_length']}`, VTS `{example['vts_length']}`",
                f"- VTS primitives: `{', '.join(example['vts_primitives'])}`",
                f"- Caption: {example['caption']}",
                f"- Failure reason: {example['failure_reason']}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def plot_examples(examples: list[dict[str, Any]], path: Path) -> None:
    fig = plt.figure(figsize=(18, 26), constrained_layout=True)
    spec = fig.add_gridspec(len(examples), 2, width_ratios=[2.2, 1.25])
    fig.suptitle(
        "Motivating Failure Examples: direct outputs vs checked VTS programs",
        fontsize=18,
        fontweight="bold",
    )

    colors = {
        "reference": "#202020",
        "vts": "#1f77b4",
        "failed": "#d62728",
    }

    for row, example in enumerate(examples):
        ax = fig.add_subplot(spec[row, 0])
        text_ax = fig.add_subplot(spec[row, 1])
        text_ax.axis("off")

        for label, series in [
            ("Reference", example["series"]["reference"]),
            ("VTS checked", example["series"]["vts"]),
            (example["failed_method"], example["series"]["failed"]),
        ]:
            x = np.arange(len(series))
            key = "failed" if label == example["failed_method"] else ("vts" if label == "VTS checked" else "reference")
            ax.plot(
                x,
                series,
                label=f"{label} (n={len(series)})",
                color=colors[key],
                linewidth=1.9 if key != "failed" else 1.4,
                linestyle="--" if key == "failed" else "-",
                alpha=0.95,
            )

        ax.axvline(example["required_length"] - 1, color="#777777", linewidth=0.8, alpha=0.5)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
        ax.set_title(
            f"{row + 1}. {example['source']} | {example['case_id']}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("timestep")
        ax.set_ylabel("value")
        ax.legend(loc="best", fontsize=8, frameon=True)

        index_bits = [f"case index: {example['case_file_index']}"]
        if example.get("source_index") is not None:
            index_bits.append(f"source index: {example['source_index']}")
        if example.get("source_dataset"):
            index_bits.append(str(example["source_dataset"]))

        text = "\n\n".join(
            [
                "\n".join(index_bits),
                "Caption:\n" + textwrap.fill(example["caption"], 58),
                "Fail reason:\n" + textwrap.fill(example["failure_reason"], 58),
                "VTS primitives:\n" + textwrap.fill(", ".join(example["vts_primitives"]), 58),
            ]
        )
        text_ax.text(
            0.0,
            1.0,
            text,
            va="top",
            ha="left",
            fontsize=9,
            family="DejaVu Sans",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f7f7f7", "edgecolor": "#cfcfcf"},
        )

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = build_gold_examples() + build_tsfragment_examples()
    selected_json = OUT_DIR / "selected_indices.json"
    selected_md = OUT_DIR / "selected_examples.md"
    figure_path = OUT_DIR / "comparison.png"
    write_json(selected_json, compact_examples(examples))
    render_markdown(examples, selected_md)
    plot_examples(examples, figure_path)

    print("Relevant source files:")
    print("- data/controlled_gold_cases/robustness_cases.json")
    print("- data/controlled_gold_cases/multi_claim_cases.json")
    print("- data/tsfragment_eval/tsfragment_eval_2500_cases.json")
    print("- evals/reports/baseline_results.json")
    print("- evals/reports/t2s_llm_main_benchmark_eval.json")
    print("- evals/reports/t2s_ours_main_benchmark_after_fix_reprocessed_eval.json")
    print("Data provenance: existing repository benchmark artifacts; no new model generation")
    print(f"Selected indices JSON: {selected_json.relative_to(REPO_ROOT)}")
    print(f"Summary Markdown: {selected_md.relative_to(REPO_ROOT)}")
    print(f"Comparison figure: {figure_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
