"""Render generated series for the motivating TSFragment example.

This script uses existing saved artifacts only. It does not call any model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "t2s_ETTh1_96_00049"
TARGET_LENGTH = 96
TARGET_LAST_INDEX = TARGET_LENGTH - 1
OLD_EXAMPLE_DIR = (
    REPO_ROOT / "evals" / "reports" / "direct_llm_rerun_example_t2s_ETTh1_96_00049"
)

REPORT_DIR = REPO_ROOT / "evals" / "reports" / "motivating_failure_examples"
FIG_DIR = REPO_ROOT / "Manuscript" / "Figs"


def rel(path: str | Path) -> Path:
    return REPO_ROOT / path


def strip_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def read_json_series(path: str | Path) -> list[float]:
    raw = strip_fence(rel(path).read_text())
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        for key in ("series", "generated_series", "values", "prediction"):
            if key in parsed:
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return [float(v) for v in parsed]


def read_code_series(path: str | Path) -> list[float]:
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


def read_series(kind: str, path: str | Path) -> list[float]:
    if kind == "code":
        return read_code_series(path)
    return read_json_series(path)


def common_limits(series_items: list[dict[str, Any]]) -> tuple[tuple[int, int], tuple[float, float]]:
    max_len = max(len(item["series"]) for item in series_items)
    values = np.concatenate([np.asarray(item["series"], dtype=float) for item in series_items])
    y_min, y_max = float(np.nanmin(values)), float(np.nanmax(values))
    padding = max((y_max - y_min) * 0.08, 0.5)
    return (0, max(max_len - 1, TARGET_LAST_INDEX) + 2), (y_min - padding, y_max + padding)


def render_panel_figure(series_items: list[dict[str, Any]], output_stem: str) -> None:
    xlim, ylim = common_limits(series_items)
    reference = np.asarray(read_series("json", OLD_EXAMPLE_DIR / "reference_series.json"), dtype=float)
    reference_x = np.arange(len(reference))
    fig, axes = plt.subplots(
        1,
        len(series_items),
        figsize=(14.4, 3.1),
        sharey=True,
        constrained_layout=True,
    )
    if len(series_items) == 1:
        axes = [axes]

    for ax, item in zip(axes, series_items):
        series = np.asarray(item["series"], dtype=float)
        x = np.arange(len(series))
        ax.plot(
            reference_x,
            reference,
            color="#222222",
            linewidth=1.7,
            linestyle=(0, (5, 3)),
            alpha=0.8,
            label="Reference",
        )
        ax.plot(
            x,
            series,
            color=item["color"],
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            markeredgewidth=0.0,
            label=item["label"],
        )
        ax.axvline(
            TARGET_LAST_INDEX,
            color="#6c6c6c",
            linestyle=(0, (3, 3)),
            linewidth=1.1,
            alpha=0.9,
        )
        ax.scatter(
            [len(series) - 1],
            [series[-1]],
            s=58,
            facecolor="white",
            edgecolor=item["color"],
            linewidth=1.8,
            zorder=5,
        )
        ax.set_title(item["label"], fontsize=11, fontweight="bold", pad=7)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Timestep", fontsize=9)
        ax.grid(True, color="#d9d9d9", linewidth=0.65, alpha=0.8)
        ax.legend(loc="lower right", fontsize=7.7, frameon=True)
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            length=4.2,
            width=0.9,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.95)
            spine.set_color("#222222")

    axes[0].set_ylabel("Value", fontsize=9)
    fig.suptitle(
        "Generated series for the TSFragment ETTh1 motivating example",
        fontsize=12.5,
        fontweight="bold",
    )

    for base in (REPORT_DIR, FIG_DIR):
        base.mkdir(parents=True, exist_ok=True)
        fig.savefig(base / f"{output_stem}.png", dpi=240)
        fig.savefig(base / f"{output_stem}.pdf")
    plt.close(fig)


def main() -> None:
    direct_array = {
        "label": "LLM Direct Array",
        "color": "#D55E00",
        "series": read_series(
            "json",
            OLD_EXAMPLE_DIR / "llm_direct_array.parsed_series.json",
        ),
    }
    direct_code = {
        "label": "LLM Direct Code",
        "color": "#0072B2",
        "series": read_series(
            "json",
            OLD_EXAMPLE_DIR / "llm_direct_code.generated_series.json",
        ),
    }
    faithts = {
        "label": "FaithTS",
        "color": "#009E73",
        "series": read_series(
            "json",
            OLD_EXAMPLE_DIR / "faithts.generated_series.json",
        ),
    }

    render_panel_figure(
        [direct_array, direct_code, faithts],
        "motivating_example_generation_paths",
    )
    summary = {
        "case_id": CASE_ID,
        "source_artifact_dir": str(OLD_EXAMPLE_DIR.relative_to(REPO_ROOT)),
        "target_length": TARGET_LENGTH,
        "outputs": [
            {"label": item["label"], "length": len(item["series"])}
            for item in [direct_array, direct_code, faithts]
        ],
        "figure_paths": {
            "generation_paths_png": str(
                (FIG_DIR / "motivating_example_generation_paths.png").relative_to(REPO_ROOT)
            ),
            "generation_paths_pdf": str(
                (FIG_DIR / "motivating_example_generation_paths.pdf").relative_to(REPO_ROOT)
            ),
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "motivating_example_series_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
