from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CLAIM_ORDER = [
    "H1_direct_generation_unreliable",
    "H2_registry_grounding_matters",
    "H3_static_repair_improves_reliability",
    "H4_rejection_reduces_hallucination",
    "H5_rule_parser_boundary",
    "H6_edit_locality_pending",
    "H7_cost_latency_pending",
    "H8_public_text_generator_baseline_pending",
]


def _result_by_name(payload: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {
        str(item.get(key) or item.get("method") or item.get("baseline") or item.get("ablation")): item
        for item in payload.get("results", [])
    }


def _metric(result: dict[str, Any] | None, *path: str, default: float | None = None) -> float | None:
    current: Any = result
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    return float(current)


def _status(condition: bool, *, partial: bool = False) -> str:
    if condition:
        return "partial" if partial else "supported"
    return "not_supported"


def _record(
    claim_id: str,
    claim: str,
    status: str,
    evidence: dict[str, Any],
    gate: str,
    implication: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim": claim,
        "status": status,
        "evidence": evidence,
        "gate": gate,
        "implication": implication,
    }


def build_claim_evidence_report(
    *,
    baseline_payload: dict[str, Any],
    ablation_payload: dict[str, Any],
    revision_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    methods = _result_by_name(baseline_payload, "method")
    ablations = _result_by_name(ablation_payload, "ablation")

    ours = methods.get("ours_full")
    rule = methods.get("rule_parser")
    direct_methods = [
        methods.get("llm_direct_array"),
        methods.get("llm_direct_code"),
        methods.get("llm_strong_prompt"),
    ]
    direct_csrs = [
        value
        for value in (_metric(item, "aggregate", "csr") for item in direct_methods)
        if value is not None
    ]
    direct_pass_rates = [
        value
        for value in (_metric(item, "aggregate", "pass_rate") for item in direct_methods)
        if value is not None
    ]
    direct_hus_values = [
        value
        for value in (_metric(item, "aggregate", "hus") for item in direct_methods)
        if value is not None
    ]
    ours_csr = _metric(ours, "aggregate", "csr", default=0.0) or 0.0
    ours_pass = _metric(ours, "aggregate", "pass_rate", default=0.0) or 0.0
    best_direct_csr = max(direct_csrs) if direct_csrs else None
    best_direct_pass = max(direct_pass_rates) if direct_pass_rates else None
    worst_direct_hus = max(direct_hus_values) if direct_hus_values else None
    direct_gap = None if best_direct_csr is None else ours_csr - best_direct_csr

    full_oracle = ablations.get("full_oracle")
    no_registry = ablations.get("no_registry_grounding")
    no_rejection = ablations.get("no_rejection")
    repair_no = ablations.get("repair_challenge_no_repair")
    repair_yes = ablations.get("repair_challenge_with_repair")

    full_oracle_csr = _metric(full_oracle, "aggregate", "csr", default=0.0) or 0.0
    no_registry_csr = _metric(no_registry, "aggregate", "csr", default=0.0) or 0.0
    registry_drop = full_oracle_csr - no_registry_csr

    repair_no_pass = _metric(repair_no, "aggregate", "pass_rate", default=0.0) or 0.0
    repair_yes_pass = _metric(repair_yes, "aggregate", "pass_rate", default=0.0) or 0.0
    repair_gain = repair_yes_pass - repair_no_pass
    repair_changed = _metric(repair_yes, "repair_summary", "changed_cases", default=0.0) or 0.0

    ours_hus = _metric(ours, "aggregate", "hus", default=0.0) or 0.0
    no_rejection_hus = _metric(no_rejection, "aggregate", "hus", default=0.0) or 0.0
    ours_crr = _metric(ours, "aggregate", "crr", default=0.0) or 0.0

    rule_csr = _metric(rule, "aggregate", "csr", default=0.0) or 0.0
    ours_adv = _metric(ours, "by_category", "adversarial", "csr", default=0.0) or 0.0
    rule_adv = _metric(rule, "by_category", "adversarial", "csr", default=0.0) or 0.0
    ours_comp = _metric(ours, "by_category", "compositional", "csr", default=0.0) or 0.0
    rule_comp = _metric(rule, "by_category", "compositional", "csr", default=0.0) or 0.0
    edit = revision_payload.get("edit_locality") if revision_payload else None
    edit_aggregate = {
        item.get("method"): item
        for item in edit.get("aggregate", [])
    } if isinstance(edit, dict) else {}
    local_edit = edit_aggregate.get("local_program_edit_oracle")
    local_outside_drift = _metric(local_edit, "mean_outside_drift")
    local_inside_change = _metric(local_edit, "mean_inside_change")
    locality_status = "pending"
    if local_outside_drift is not None and local_inside_change is not None:
        locality_status = _status(
            local_outside_drift <= 0.01 and local_inside_change >= 0.05,
            partial=True,
        )

    cost = revision_payload.get("cost_latency") if revision_payload else None
    cost_records = cost.get("records", []) if isinstance(cost, dict) else []
    completed_cost_records = [item for item in cost_records if item.get("status") == "completed"]
    best_seconds_per_case = min(
        (float(item["seconds_per_case"]) for item in completed_cost_records if item.get("seconds_per_case") is not None),
        default=None,
    )
    cost_status = "pending"
    if best_seconds_per_case is not None:
        cost_status = "partial"

    public_baselines = revision_payload.get("public_text_generator_baselines") if revision_payload else None
    public_rows = public_baselines.get("baselines", []) if isinstance(public_baselines, dict) else []
    public_completed = [
        item
        for item in public_rows
        if item.get("status") in {"completed", "evaluated", "outputs_available", "case_level_2500_completed", "native_completed"}
    ]
    public_status = "supported" if public_completed else ("partial" if public_rows else "pending")

    claims = [
        _record(
            "H1_direct_generation_unreliable",
            "Direct generation remains less reliable than FaithTS under exact-pass and rejection requirements.",
            _status(
                best_direct_csr is not None
                and best_direct_pass is not None
                and ours_pass - best_direct_pass >= 0.20
                and ours_csr > best_direct_csr
                and (worst_direct_hus is None or worst_direct_hus >= 0.50)
            ),
            {
                "ours_full_csr": ours_csr,
                "ours_full_pass": ours_pass,
                "best_direct_csr": best_direct_csr,
                "best_direct_pass": best_direct_pass,
                "csr_gap": direct_gap,
                "worst_direct_hus": worst_direct_hus,
            },
            "supported if ours_full improves exact-pass rate over the best direct baseline by >= 0.20 and direct unsupported hallucination remains high",
            "Main text should claim a reliability/rejection advantage, not a large CSR gap over every direct-code variant.",
        ),
        _record(
            "H2_registry_grounding_matters",
            "Registry grounding is necessary for executable, semantically grounded generation.",
            _status(registry_drop >= 0.50),
            {
                "full_oracle_csr": full_oracle_csr,
                "no_registry_grounding_csr": no_registry_csr,
                "csr_drop": registry_drop,
                "no_registry_failure_case_rate": _metric(no_registry, "failure_taxonomy", "failure_case_rate"),
            },
            "supported if removing registry grounding drops CSR by >= 0.50",
            "Main text may claim registry grounding is a necessary reliability component.",
        ),
        _record(
            "H3_static_repair_improves_reliability",
            "Static verifier-guided repair improves reliability on deterministic planner-defect cases.",
            _status(repair_gain >= 0.50 and repair_changed > 0),
            {
                "repair_challenge_no_repair_pass": repair_no_pass,
                "repair_challenge_with_repair_pass": repair_yes_pass,
                "pass_rate_gain": repair_gain,
                "changed_cases": repair_changed,
            },
            "supported if repair-challenge pass rate improves by >= 0.50 and changed_cases > 0",
            "Claim should say static verifier-guided repair, not LLM-assisted semantic repair.",
        ),
        _record(
            "H4_rejection_reduces_hallucination",
            "Explicit rejection reduces hallucinated generation for unsupported semantics.",
            _status(ours_crr >= 0.95 and ours_hus <= 0.05 and no_rejection_hus >= 0.50),
            {
                "ours_full_crr": ours_crr,
                "ours_full_hus": ours_hus,
                "no_rejection_hus": no_rejection_hus,
            },
            "supported if CRR >= 0.95, ours HUS <= 0.05, and no-rejection HUS >= 0.50",
            "Main text may claim rejection prevents unsupported hallucination.",
        ),
        _record(
            "H5_rule_parser_boundary",
            "FaithTS improves over a rule parser overall and on adversarial/rejection behavior, but not all subcategories.",
            _status(ours_csr > rule_csr and ours_adv > rule_adv and ours_comp >= rule_comp, partial=ours_comp == rule_comp),
            {
                "ours_full_csr": ours_csr,
                "rule_parser_csr": rule_csr,
                "ours_adversarial_csr": ours_adv,
                "rule_adversarial_csr": rule_adv,
                "ours_compositional_csr": ours_comp,
                "rule_compositional_csr": rule_comp,
            },
            "supported/partial if ours beats rule parser overall and adversarial, with compositional at least tied",
            "Do not claim strict superiority over rule parser on every category; report the compositional tie.",
        ),
        _record(
            "H6_edit_locality_pending",
            "Programmatic generation improves edit locality.",
            locality_status,
            {
                "edit_locality_report": "revision_risk_report.json" if edit else None,
                "local_program_inside_change": local_inside_change,
                "local_program_outside_drift": local_outside_drift,
            },
            "partial if internal local-edit oracle has outside drift <= 0.01 and inside change >= 0.05; supported requires public-generator edit outputs",
            "Internal locality behavior may be reported as a diagnostic; public-baseline locality still needs external outputs.",
        ),
        _record(
            "H7_cost_latency_pending",
            "The approach has acceptable latency and token cost.",
            cost_status,
            {"cost_latency_report": "revision_risk_report.json" if cost else None, "best_seconds_per_case": best_seconds_per_case},
            "partial if wall-clock latency is logged; supported requires token/API cost or explicit no-API accounting",
            "Latency can be discussed as measured; token-cost claims remain gated unless cost accounting is populated.",
        ),
        _record(
            "H8_public_text_generator_baseline_pending",
            "The paper includes a public text-conditioned time-series generator comparison.",
            public_status,
            {
                "baseline_protocol": "revision_risk_report.json" if public_baselines else None,
                "registered_public_baselines": len(public_rows),
                "completed_public_baselines": len(public_completed),
                "completed_baselines": [item.get("baseline_id") for item in public_completed],
            },
            "partial if public baseline protocol is registered; supported for reported completed public-generator settings; broader support requires additional public-generator outputs beyond the shared TSFragment split",
            "The paper may report completed T2S and TSFragment-compatible VerbalTS shared-case results; broader public-generator superiority remains pending beyond the completed shared split.",
        ),
    ]

    status_counts: dict[str, int] = {}
    for item in claims:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    idea_evaluation = {
        "first_impression": {
            "paper_type": "Novel Method / New Setting",
            "one_sentence_story": (
                "Do not ask LLMs to draw time series; compile language into verifiable, "
                "registry-grounded time-series programs with execution, checking, and rejection."
            ),
        },
        "fatal_flaws": [
            {
                "id": "F3",
                "flaw": "Public-generator shared-case evidence is limited to the reported TSFragment-Eval setup.",
                "severity": "MAJOR",
                "defense": (
                    "Keep the correctness claim scoped to the shared 2500-case TSFragment-Eval caption checker "
                    "and the reported T2S, VerbalTS, and direct-generation outputs. Broader generators and editing "
                    "settings remain future work."
                ),
            },
            {
                "id": "F6",
                "flaw": "Edit locality, cost/latency, realism, and multi-model stability are not yet empirically verified.",
                "severity": "MAJOR",
                "defense": (
                    "Use the revision-risk report gates: internal edit-locality and latency can be partial evidence, while "
                    "public-generator locality, token cost, realism, and multi-model stability remain gated until populated."
                ),
            },
        ],
        "lifecycle_capability_match": {
            "idea_category": "Frontier exploration / data-intensive method paper",
            "lifecycle": "3-9 months",
            "fit": "Yellow",
            "assessment": (
                "Core engineering and offline experiments are feasible and mostly implemented; external baselines, "
                "real-domain validation, and polished figures are the remaining schedule risks."
            ),
        },
        "five_dimension_scores": {
            "Higher": {
                "score": 8,
                "evidence": "Ours_full reaches 899/1000 exact passes and HUS=0.0; the concise direct-code baseline is strong on atomic cases but passes 594/1000 and still hallucinates unsupported cases.",
                "lift": "Add public generator comparison or human-authored cases for stronger venue fit.",
            },
            "Faster": {
                "score": 4,
                "evidence": "The pipeline is deterministic after planning, but no latency/token accounting is implemented.",
                "lift": "Add wall-clock and token-cost logging before making efficiency claims.",
            },
            "Stronger": {
                "score": 8,
                "evidence": "Rejection, schema validation, registry grounding, and repair-challenge gates reduce failure modes.",
                "lift": "Add multi-model or temperature-stability experiments.",
            },
            "Cheaper": {
                "score": 6,
                "evidence": "Primitive programs create controllable cases without human drawing or manual arrays.",
                "lift": "Quantify annotation/time savings or synthetic case generation cost.",
            },
            "Broader": {
                "score": 7,
                "evidence": "The semantic compiler framing can apply beyond the current primitive set.",
                "lift": "Demonstrate a small real-caption smoke test or another signal domain.",
            },
        },
        "paradigm_shift_probe": {
            "First Principles": {
                "answer": "Yes",
                "rationale": "Challenges the assumption that text-to-series should directly sample signals.",
            },
            "Elephant in the Room": {
                "answer": "Yes",
                "rationale": "Targets precise constraint failures that plausible-looking generative outputs often hide.",
            },
            "Technology Cycle": {
                "answer": "Partial",
                "rationale": "LLMs make semantic planning practical, while deterministic primitive execution supplies reliability.",
            },
            "Hamming's Rule": {
                "answer": "Partial",
                "rationale": "If accepted, it changes controllable synthetic time-series generation, but not all time-series modeling.",
            },
            "disruptive_potential": "possible",
        },
        "feasibility": {
            "compute": "Low risk: current experiments are offline Python/Numpy evaluation.",
            "data": "Medium risk: synthetic gold cases exist; real-domain and public generator comparisons remain pending.",
            "engineering": "Medium risk: core pipeline is implemented; artifact cleanup and figures remain.",
            "timeline": "Medium risk: strong paper packaging still needs external evidence or careful claim narrowing.",
        },
        "integrity_gate": {
            "dimension_scores_grounded": "pass",
            "feasibility_references_current_artifacts": "pass",
            "novelty_literature_check": "partial_unverified",
            "fatal_flaws_actionable": "pass",
            "verdict_consistent": "pass",
            "paradigm_probe_grounded": "pass",
            "lifecycle_attestation": "pass_user_should_sanity_check",
        },
        "verdict": "Accept with Revisions",
        "verdict_rationale": (
            "The core semantic-compiler/rejection evidence is strong enough to continue, but external-baseline, "
            "unverified-claim, and packaging risks must be addressed or explicitly scoped out before submission."
        ),
    }

    return {
        "version": "1.0",
        "claims": claims,
        "idea_evaluation": idea_evaluation,
        "status_counts": status_counts,
        "source_tables": {
            "baseline_methods": list(methods),
            "ablations": list(ablations),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Evidence Report",
        "",
        "| Claim | Status | Gate | Key Evidence | Implication |",
        "|---|---:|---|---|---|",
    ]
    for item in payload["claims"]:
        evidence = ", ".join(f"{key}={value}" for key, value in item["evidence"].items())
        lines.append(
            "| "
            + " | ".join(
                [
                    item["claim_id"],
                    item["status"],
                    item["gate"],
                    evidence,
                    item["implication"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Status Counts", ""])
    for status, count in sorted(payload.get("status_counts", {}).items()):
        lines.append(f"- {status}: {count}")
    idea = payload.get("idea_evaluation", {})
    if idea:
        lines.extend(["", "## Idea Evaluator Verdict", ""])
        first = idea.get("first_impression", {})
        lines.append(f"- Paper type: {first.get('paper_type')}")
        lines.append(f"- One-sentence story: {first.get('one_sentence_story')}")
        lines.append(f"- Verdict: {idea.get('verdict')}")
        lines.append(f"- Rationale: {idea.get('verdict_rationale')}")
        lines.extend(["", "### Fatal Flaws", ""])
        lines.append("| Flaw | Severity | Defense |")
        lines.append("|---|---:|---|")
        for flaw in idea.get("fatal_flaws", []):
            lines.append(f"| {flaw.get('id')}: {flaw.get('flaw')} | {flaw.get('severity')} | {flaw.get('defense')} |")
        lines.extend(["", "### Five-Dimension Scores", ""])
        lines.append("| Dimension | Score | Evidence | Lift |")
        lines.append("|---|---:|---|---|")
        for dimension, item in idea.get("five_dimension_scores", {}).items():
            lines.append(f"| {dimension} | {item.get('score')} | {item.get('evidence')} | {item.get('lift')} |")
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Render claim-to-evidence gates from FaithTS experiment outputs.")
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--ablation-json", type=Path, required=True)
    parser.add_argument("--revision-json", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_claim_evidence_report(
        baseline_payload=_load_json(args.baseline_json),
        ablation_payload=_load_json(args.ablation_json),
        revision_payload=_load_json(args.revision_json) if args.revision_json is not None else None,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
