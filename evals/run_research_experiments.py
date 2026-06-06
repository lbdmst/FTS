from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.baselines.llm_direct import build_prompt
from evals.collect_faithts_outputs import DEFAULT_WORKFLOW_DIRS, collect_outputs
from evals.render_claim_evidence import build_claim_evidence_report, render_markdown as render_claim_evidence_markdown
from evals.render_research_tables import render_tables
from evals.run_ablation_study import ABLATIONS, render_markdown as render_ablation_markdown, run_ablations
from evals.run_baseline_benchmark import LLM_OUTPUT_BASELINES
from evals.run_baselines import render_markdown as render_baseline_markdown, run_methods
from evals.run_llm_baseline_outputs import (
    DEFAULT_T2S_CASE_FILE,
    DEFAULT_T2S_OUTPUT_ROOT,
    DIRECT_LLM_T2S_BASELINES,
    evaluate_t2s_llm_outputs,
    generate_many_outputs,
    merge_t2s_llm_results,
    t2s_report_paths,
)
from evals.run_revision_risk_experiments import run_revision_risk_experiments
from faithts.gold_adapter import load_gold_cases


DEFAULT_METHODS = ["oracle_program", "ours_full", "rule_parser", "llm_direct_array", "llm_direct_code", "llm_strong_prompt"]
DEFAULT_REPORT_DIR = REPO_ROOT / "evals" / "reports"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _write_llm_prompts(
    *,
    methods: list[str],
    case_paths: list[Path] | None,
    category: str | None,
    output_root: Path,
) -> dict[str, Any]:
    cases = load_gold_cases(case_paths or None) if case_paths else load_gold_cases(
        [
            REPO_ROOT / "data" / "controlled_gold_cases" / "single_claim_cases.json",
            REPO_ROOT / "data" / "controlled_gold_cases" / "multi_claim_cases.json",
            REPO_ROOT / "data" / "controlled_gold_cases" / "robustness_cases.json",
        ]
    )
    if category is not None:
        cases = [case for case in cases if case.get("category") == category]

    records: list[dict[str, Any]] = []
    for method in methods:
        if method not in LLM_OUTPUT_BASELINES:
            continue
        prompt_dir = output_root / "llm_prompts" / method
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            path = prompt_dir / f"{case['case_id']}.txt"
            path.write_text(build_prompt(case, method), encoding="utf-8")
        records.append({"method": method, "prompt_dir": str(prompt_dir), "prompts": len(cases)})
    return {"records": records, "total_prompts": sum(item["prompts"] for item in records)}


def run_research_experiments(
    *,
    report_dir: Path = DEFAULT_REPORT_DIR,
    case_paths: list[Path] | None = None,
    workflow_dirs: list[Path] | None = None,
    methods: list[str] | None = None,
    ablations: list[str] | None = None,
    category: str | None = None,
    apply_static_rejector: bool = True,
    write_llm_prompts: bool = True,
    run_t2s_llm_baselines: bool = False,
    t2s_case_path: Path = DEFAULT_T2S_CASE_FILE,
    t2s_llm_output_root: Path = DEFAULT_T2S_OUTPUT_ROOT,
    t2s_llm_methods: list[str] | None = None,
    skip_ablations: bool = False,
    skip_revision_risks: bool = False,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    methods = methods or DEFAULT_METHODS
    ablations = ablations or ABLATIONS
    baseline_output_root = report_dir / "baseline_outputs"
    t2s_paths = t2s_report_paths(report_dir=report_dir, output_root=t2s_llm_output_root)

    steps: dict[str, Any] = {}
    if "ours_full" in methods:
        steps["collect_ours_full"] = collect_outputs(
            case_paths=case_paths,
            workflow_dirs=workflow_dirs or DEFAULT_WORKFLOW_DIRS,
            output_dir=baseline_output_root / "ours_full",
            apply_rejector=apply_static_rejector,
        )

    if write_llm_prompts:
        steps["llm_prompts"] = _write_llm_prompts(
            methods=methods,
            case_paths=case_paths,
            category=category,
            output_root=baseline_output_root,
        )

    baseline_payload = run_methods(
        methods=methods,
        paths=case_paths,
        category=category,
        output_root=baseline_output_root,
    )
    baseline_json = report_dir / "baseline_results.json"
    baseline_md = report_dir / "baseline_results.md"
    _write_json(baseline_json, baseline_payload)
    baseline_md.write_text(render_baseline_markdown(baseline_payload), encoding="utf-8")

    t2s_generation_payload = None
    t2s_llm_eval_payload = None
    if run_t2s_llm_baselines:
        selected_t2s_methods = t2s_llm_methods or [
            method for method in DIRECT_LLM_T2S_BASELINES if method in methods
        ] or DIRECT_LLM_T2S_BASELINES
        t2s_generation_payload = generate_many_outputs(
            baselines=selected_t2s_methods,
            output_root=t2s_paths["output_root"],
            case_paths=[t2s_case_path],
            provider="gemini",
            skip_existing=True,
            require_api_key=True,
            use_t2s_main_benchmark=True,
        )
        t2s_llm_eval_payload = evaluate_t2s_llm_outputs(
            output_root=t2s_paths["output_root"],
            case_path=t2s_case_path,
            methods=selected_t2s_methods,
        )
        _write_json(t2s_paths["eval_json"], t2s_llm_eval_payload)
        merged_t2s = merge_t2s_llm_results(
            comparison_path=t2s_paths["comparison_json"],
            llm_payload=t2s_llm_eval_payload,
            eval_path=t2s_paths["eval_json"],
        )
        _write_json(t2s_paths["comparison_json"], merged_t2s)
        t2s_paths["comparison_md"].parent.mkdir(parents=True, exist_ok=True)
        t2s_paths["comparison_md"].write_text(render_tables(t2s_payload=merged_t2s), encoding="utf-8")

    t2s_payload = None
    if t2s_paths["comparison_json"].exists():
        t2s_payload = json.loads(t2s_paths["comparison_json"].read_text(encoding="utf-8"))

    ablation_payload = None
    ablation_json = None
    ablation_md = None
    if not skip_ablations:
        ablation_payload = run_ablations(ablations=ablations, paths=case_paths, category=category)
        ablation_json = report_dir / "ablation_results.json"
        ablation_md = report_dir / "ablation_results.md"
        _write_json(ablation_json, ablation_payload)
        ablation_md.write_text(render_ablation_markdown(ablation_payload), encoding="utf-8")

    tables_md = report_dir / "research_tables.md"
    tables_md.write_text(
        render_tables(
            baseline_payload=baseline_payload,
            ablation_payload=ablation_payload,
            t2s_payload=t2s_payload,
        ),
        encoding="utf-8",
    )

    claim_evidence_json = None
    claim_evidence_md = None
    revision_json = None
    revision_md = None
    if ablation_payload is not None:
        revision_payload = None
        if not skip_revision_risks:
            revision_payload = run_revision_risk_experiments(
                report_dir=report_dir,
                case_paths=case_paths,
                category=category or "atomic",
                output_root=baseline_output_root,
                latency_repeats=1,
            )
            revision_json = report_dir / "revision_risk_report.json"
            revision_md = report_dir / "revision_risk_report.md"
        claim_payload = build_claim_evidence_report(
            baseline_payload=baseline_payload,
            ablation_payload=ablation_payload,
            revision_payload=revision_payload,
        )
        claim_evidence_json = report_dir / "claim_evidence.json"
        claim_evidence_md = report_dir / "claim_evidence.md"
        _write_json(claim_evidence_json, claim_payload)
        claim_evidence_md.write_text(render_claim_evidence_markdown(claim_payload), encoding="utf-8")

    manifest = {
        "version": "1.0",
        "report_dir": str(report_dir),
        "category": category,
        "methods": methods,
        "ablations": [] if skip_ablations else ablations,
        "apply_static_rejector": apply_static_rejector,
        "paths": {
            "baseline_results_json": str(baseline_json),
            "baseline_results_md": str(baseline_md),
            "ablation_results_json": str(ablation_json) if ablation_json is not None else None,
            "ablation_results_md": str(ablation_md) if ablation_md is not None else None,
            "research_tables_md": str(tables_md),
            "t2s_main_benchmark_json": str(t2s_paths["comparison_json"]) if t2s_payload is not None else None,
            "t2s_main_benchmark_md": str(t2s_paths["comparison_md"]) if t2s_payload is not None else None,
            "verbalts_tsfragment_json": str(report_dir / "verbalts_tsfragment_eval_2500.json")
            if (report_dir / "verbalts_tsfragment_eval_2500.json").exists()
            else None,
            "verbalts_tsfragment_md": str(report_dir / "verbalts_tsfragment_eval_2500.md")
            if (report_dir / "verbalts_tsfragment_eval_2500.md").exists()
            else None,
            "t2s_llm_eval_json": str(t2s_paths["eval_json"]) if t2s_llm_eval_payload is not None else None,
            "t2s_llm_output_root": str(t2s_paths["output_root"]) if t2s_generation_payload is not None else None,
            "claim_evidence_json": str(claim_evidence_json) if claim_evidence_json is not None else None,
            "claim_evidence_md": str(claim_evidence_md) if claim_evidence_md is not None else None,
            "revision_risk_json": str(revision_json) if revision_json is not None else None,
            "revision_risk_md": str(revision_md) if revision_md is not None else None,
            "baseline_output_root": str(baseline_output_root),
        },
        "steps": {
            **steps,
            "t2s_llm_generation": t2s_generation_payload,
            "t2s_llm_eval": t2s_llm_eval_payload,
        },
    }
    _write_json(report_dir / "experiment_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full offline FaithTS research experiment pipeline.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cases", nargs="*", type=Path, default=None)
    parser.add_argument("--workflow-dirs", nargs="*", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=ABLATIONS)
    parser.add_argument("--category", choices=["atomic", "compositional", "adversarial"], default=None)
    parser.add_argument("--no-static-rejector", action="store_true")
    parser.add_argument("--no-llm-prompts", action="store_true")
    parser.add_argument("--run-t2s-llm-baselines", action="store_true")
    parser.add_argument("--t2s-case-path", type=Path, default=DEFAULT_T2S_CASE_FILE)
    parser.add_argument("--t2s-llm-output-root", type=Path, default=DEFAULT_T2S_OUTPUT_ROOT)
    parser.add_argument("--t2s-llm-methods", nargs="+", choices=DIRECT_LLM_T2S_BASELINES, default=None)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-revision-risks", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest = run_research_experiments(
        report_dir=args.report_dir,
        case_paths=args.cases,
        workflow_dirs=args.workflow_dirs,
        methods=args.methods,
        ablations=args.ablations,
        category=args.category,
        apply_static_rejector=not args.no_static_rejector,
        write_llm_prompts=not args.no_llm_prompts,
        run_t2s_llm_baselines=args.run_t2s_llm_baselines,
        t2s_case_path=args.t2s_case_path,
        t2s_llm_output_root=args.t2s_llm_output_root,
        t2s_llm_methods=args.t2s_llm_methods,
        skip_ablations=args.skip_ablations,
        skip_revision_risks=args.skip_revision_risks,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, allow_nan=False))
    else:
        print(f"Wrote research experiment reports to {manifest['report_dir']}")
        print(f"Main tables: {manifest['paths']['research_tables_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
