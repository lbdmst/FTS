from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faithts_pipeline.hybrid_pipeline import PlanValidationError, extract_json_payload, parse_plan
from faithts_pipeline.workflow import (
    RetrievalExample,
    build_parameter_inference_prompt,
    build_llm_client,
    build_prompt,
    extract_parameter_payload,
    load_primitive_specs,
    load_registry_primitives,
    merge_inferred_parameters,
    retrieve_primitives_for_caption,
)
from schema_validator import validate_instance, validate_payload


CONTROLLED_CASES_DIR = REPO_ROOT / "data" / "controlled_gold_cases"
DEFAULT_MARKDOWN_OUTPUT = CONTROLLED_CASES_DIR / "sample_planner_outputs.md"
DEFAULT_JSON_OUTPUT = CONTROLLED_CASES_DIR / "sample_planner_outputs.json"
CATEGORY_FILES = {
    "atomic": CONTROLLED_CASES_DIR / "single_claim_cases.json",
    "compositional": CONTROLLED_CASES_DIR / "multi_claim_cases.json",
    "adversarial": CONTROLLED_CASES_DIR / "robustness_cases.json",
}


def _case_numeric_range(case: dict[str, Any]) -> tuple[float, float]:
    raw_numeric_range = case.get("numeric_range")
    if isinstance(raw_numeric_range, dict) and {"min", "max"}.issubset(raw_numeric_range):
        return (float(raw_numeric_range["min"]), float(raw_numeric_range["max"]))
    expected_checks = case.get("expected_signal_checks", {})
    values: list[float] = []
    for segment in expected_checks.get("constant_segments", []):
        value = segment.get("value")
        if isinstance(value, (int, float)):
            values.append(float(value))
    plan_outline = case.get("expected_plan_outline", {})
    for step in plan_outline.get("steps", []):
        parameters = step.get("parameters", {})
        if not isinstance(parameters, dict):
            continue
        for key in ["target_value", "start_value", "end_value", "amplitude", "delta", "shift", "level_change", "slope_change"]:
            value = parameters.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return (-1000.0, 1000.0)
    lower = min(values)
    upper = max(values)
    span = max(1.0, upper - lower)
    return (lower - 0.25 * span, upper + 0.25 * span)


def _load_cases(path: Path, category: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return [case for case in payload if case.get("category") == category]


def _sample_cases(seed: int, per_category: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for category, path in CATEGORY_FILES.items():
        cases = _load_cases(path, category)
        if len(cases) < per_category:
            raise ValueError(f"Not enough cases in {path} to sample {per_category}")
        selected = rng.sample(cases, per_category)
        for case in selected:
            sampled.append({"category": category, "case": case})
    return sampled


def _step_primitives(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        return []
    return [
        step["primitive"]
        for step in steps
        if isinstance(step, dict) and step.get("kind") == "primitive" and "primitive" in step
    ]


def _json_code_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _text_code_block(value: str) -> str:
    return "```text\n" + value.strip() + "\n```"


def run_samples(seed: int, per_category: int, provider: str) -> dict[str, Any]:
    registry_validation = validate_instance("primitive", REPO_ROOT / "atomic_timeseries" / "registry.json")
    if registry_validation["status"] != "valid":
        raise RuntimeError(f"Registry validation failed: {registry_validation['errors']}")

    specs = load_primitive_specs()
    registry_primitives = load_registry_primitives()
    llm_client = build_llm_client(provider)
    sampled = _sample_cases(seed=seed, per_category=per_category)

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(sampled, start=1):
        category = item["category"]
        case = item["case"]
        example = RetrievalExample(
            index=idx,
            caption=case["description"],
            candidate_primitives=list(case.get("expected_candidate_primitives", [])),
            retrieval_status="gold-case",
            orchestration_level_descriptors=[],
            raw_record=case,
        )
        retrieved_specs = retrieve_primitives_for_caption(
            example.caption,
            specs,
            retrieval_hints=example.candidate_primitives,
        )
        retrieved_primitives = [spec.function_name for spec in retrieved_specs]
        prompt = build_prompt(
            example=example,
            retrieved_specs=retrieved_specs,
            series_length=int(case["length"]),
            numeric_range=_case_numeric_range(case),
        )
        selection_generation = llm_client.generate_code(prompt, example)

        plan_payload: dict[str, Any] | None = None
        selection_plan: dict[str, Any] | None = None
        extract_error: str | None = None
        schema_validation: dict[str, Any] | None = None
        parse_status = {"status": "not_run", "error": None}
        try:
            selection_plan = extract_json_payload(selection_generation.raw_text)
            parameter_prompt = build_parameter_inference_prompt(
                example=example,
                selection_plan=selection_plan,
                registry_primitives=registry_primitives,
                series_length=int(case["length"]),
                numeric_range=_case_numeric_range(case),
            )
            parameter_generation = llm_client.generate_code(parameter_prompt, example)
            parameter_payload = extract_parameter_payload(parameter_generation.raw_text)
            plan_payload = merge_inferred_parameters(selection_plan, parameter_payload)
            schema_validation = validate_payload("plan", plan_payload, target_path=f"<{case['case_id']}>")
            try:
                parse_plan(
                    plan_payload,
                    retrieved_primitives,
                    int(case["length"]),
                    numeric_range=_case_numeric_range(case),
                )
                parse_status = {"status": "valid", "error": None}
            except PlanValidationError as exc:
                parse_status = {"status": exc.category, "error": str(exc)}
        except PlanValidationError as exc:
            extract_error = str(exc)

        records.append(
            {
                "category": category,
                "case_id": case["case_id"],
                "input_description": case["description"],
                "length": case["length"],
                "expected_candidate_primitives": list(case.get("expected_candidate_primitives", [])),
                "retrieved_primitives": retrieved_primitives,
                "model_provider": selection_generation.provider,
                "model_name": selection_generation.model,
                "response_id": selection_generation.response_id,
                "raw_text": selection_generation.raw_text,
                "selection_plan": selection_plan,
                "extracted_plan": plan_payload,
                "extract_error": extract_error,
                "schema_validation": schema_validation,
                "parse_status": parse_status,
                "step_primitives": _step_primitives(plan_payload),
            }
        )

    return {
        "seed": seed,
        "per_category": per_category,
        "provider": provider,
        "registry_validation": registry_validation,
        "samples": records,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sample Gold Case Primitive Comparison",
        "",
        f"- seed: `{report['seed']}`",
        f"- samples_per_category: `{report['per_category']}`",
        f"- provider: `{report['provider']}`",
        f"- registry_validation: `{report['registry_validation']['status']}`",
        "",
    ]

    for sample in report["samples"]:
        status_parts = [
            f"schema={sample['schema_validation']['status'] if sample['schema_validation'] else 'not_run'}",
            f"parse={sample['parse_status']['status']}",
        ]
        lines.extend(
            [
                f"## {sample['category']} | {sample['case_id']}",
                "",
                f"- 描述: `{sample['input_description']}`",
                f"- 预期 primitive: `{sample['expected_candidate_primitives']}`",
                f"- 模型输出 primitive: `{sample['step_primitives']}`",
                f"- 喂给模型的候选 primitive: `{sample['retrieved_primitives']}`",
                f"- 状态: `{' | '.join(status_parts)}`",
            ]
        )
        if sample["schema_validation"] and sample["schema_validation"]["errors"]:
            lines.append(f"- schema_errors: `{sample['schema_validation']['errors']}`")
        if sample["parse_status"]["error"]:
            lines.append(f"- parse_error: `{sample['parse_status']['error']}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample gold cases and record planner JSON outputs.")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_samples(seed=args.seed, per_category=args.per_category, provider=args.provider)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
