"""Render a focused TSFragment example after the caption-index prompt update."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "t2s_ETTh1_48_00049"
TARGET_LENGTH = 48
ANCHOR_INDEX = 24
ANCHOR_VALUE = 16.109

REPORT_DIR = REPO_ROOT / "evals" / "reports" / "tmp_rerun_t2s_ETTh1_48_00049_new_prompt"
CASE_DIR = REPORT_DIR / "cases" / CASE_ID
MANUSCRIPT_FIG_DIR = REPO_ROOT / "Manuscript" / "Figs"


def strip_fence(text: str) -> str:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    return match.group(1).strip() if match else cleaned


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_series(path: Path) -> np.ndarray:
    payload = json.loads(strip_fence(path.read_text(encoding="utf-8")))
    if isinstance(payload, dict):
        for key in ["series", "generated_series", "values", "prediction"]:
            if key in payload:
                payload = payload[key]
                break
    return np.asarray(payload, dtype=float).ravel()


def read_code_series(path: Path) -> np.ndarray:
    code = strip_fence(path.read_text(encoding="utf-8"))
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
    if "series" not in namespace:
        raise RuntimeError(f"No `series` produced by {path}")
    return np.asarray(namespace["series"], dtype=float).ravel()


def load_series() -> dict[str, np.ndarray]:
    outputs = REPO_ROOT / "evals" / "reports" / "t2s_llm_main_benchmark_outputs"
    return {
        "Reference": read_json_series(CASE_DIR / "reference_series.json"),
        "LLM Direct Array": read_json_series(outputs / "llm_direct_array" / f"{CASE_ID}.txt"),
        "LLM Direct Code": read_code_series(outputs / "llm_direct_code" / f"{CASE_ID}.txt"),
        "FaithTS old": read_json_series(
            REPO_ROOT
            / "evals"
            / "reports"
            / "t2s_ours_main_benchmark_after_fix_cases"
            / CASE_ID
            / "generated_series.json"
        ),
        "FaithTS new": read_json_series(CASE_DIR / "generated_series.json"),
    }


def annotate_anchor(ax: plt.Axes, y_offset: float = 0.45) -> None:
    ax.axvline(ANCHOR_INDEX, color="#7A7A7A", linestyle=(0, (3, 3)), linewidth=1.0)
    ax.axhline(ANCHOR_VALUE, color="#7A7A7A", linestyle=(0, (3, 3)), linewidth=0.9)
    ax.scatter([ANCHOR_INDEX], [ANCHOR_VALUE], s=58, color="#009E73", edgecolor="white", linewidth=0.8, zorder=6)
    ax.annotate(
        "point 25 -> index 24\nvalue 16.109",
        xy=(ANCHOR_INDEX, ANCHOR_VALUE),
        xytext=(ANCHOR_INDEX + 2.2, ANCHOR_VALUE + y_offset),
        fontsize=8,
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#404040"},
    )


def render() -> dict[str, str]:
    series = load_series()
    description = read_json(REPORT_DIR / "case.json")[0]["description"]

    colors = {
        "Reference": "#222222",
        "LLM Direct Array": "#D55E00",
        "LLM Direct Code": "#0072B2",
        "FaithTS old": "#8C8C8C",
        "FaithTS new": "#009E73",
    }
    styles = {
        "Reference": {"linestyle": (0, (5, 3)), "linewidth": 1.7, "alpha": 0.85},
        "LLM Direct Array": {"linestyle": "-", "linewidth": 1.45, "alpha": 0.75},
        "LLM Direct Code": {"linestyle": "-", "linewidth": 1.45, "alpha": 0.75},
        "FaithTS old": {"linestyle": (0, (2, 2)), "linewidth": 1.4, "alpha": 0.9},
        "FaithTS new": {"linestyle": "-", "linewidth": 2.1, "alpha": 0.95},
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 3.45),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
        constrained_layout=True,
    )
    ax_full, ax_zoom = axes

    for name in ["Reference", "LLM Direct Array", "LLM Direct Code", "FaithTS new"]:
        values = series[name]
        x = np.arange(len(values))
        label = f"{name} (n={len(values)})" if name.startswith("LLM") else name
        ax_full.plot(x, values, color=colors[name], label=label, **styles[name])
    ax_full.axvline(TARGET_LENGTH - 1, color="#999999", linestyle=(0, (3, 3)), linewidth=0.9)
    annotate_anchor(ax_full, y_offset=0.6)
    ax_full.set_title("Full horizon comparison", fontsize=10.5, fontweight="bold")
    ax_full.set_xlabel("Index")
    ax_full.set_ylabel("Value")
    ax_full.set_xlim(-1, max(len(values) for values in series.values()))
    ax_full.grid(True, color="#DCDCDC", linewidth=0.6, alpha=0.8)
    ax_full.legend(loc="upper left", fontsize=7.3, frameon=True)

    for name in ["Reference", "FaithTS old", "FaithTS new"]:
        values = series[name]
        x = np.arange(len(values))
        ax_zoom.plot(x, values, color=colors[name], label=name, **styles[name])
    annotate_anchor(ax_zoom, y_offset=0.28)
    ax_zoom.set_title("Anchor zoom", fontsize=10.5, fontweight="bold")
    ax_zoom.set_xlabel("Index")
    ax_zoom.set_xlim(15, 27)
    ax_zoom.set_ylim(15.7, 19.8)
    ax_zoom.grid(True, color="#DCDCDC", linewidth=0.6, alpha=0.8)
    ax_zoom.legend(loc="upper right", fontsize=7.3, frameon=True)

    wrapped = textwrap.fill(description, width=132)
    fig.suptitle("TSFragment ETTh1 example: caption-index anchor after prompt update", fontsize=11.5, fontweight="bold")
    fig.text(0.01, -0.02, wrapped, ha="left", va="top", fontsize=7.4)

    outputs = {
        "report_png": REPORT_DIR / "method_comparison_t2s_ETTh1_48_00049_preview.png",
        "report_pdf": REPORT_DIR / "method_comparison_t2s_ETTh1_48_00049_preview.pdf",
        "manuscript_png": MANUSCRIPT_FIG_DIR / "method_comparison_t2s_ETTh1_48_00049_preview.png",
        "manuscript_pdf": MANUSCRIPT_FIG_DIR / "method_comparison_t2s_ETTh1_48_00049_preview.pdf",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["report_png"], dpi=260, bbox_inches="tight")
    fig.savefig(outputs["report_pdf"], bbox_inches="tight")
    fig.savefig(outputs["manuscript_png"], dpi=260, bbox_inches="tight")
    fig.savefig(outputs["manuscript_pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path.relative_to(REPO_ROOT)) for key, path in outputs.items()}


def main() -> None:
    print(json.dumps(render(), indent=2))


if __name__ == "__main__":
    main()
