"""Core research evaluation utilities for FaithTS.

This package contains the scope-locked research layer: gold semantic
specifications, executable semantic checkers, plan/signal evaluation, rejection
metrics, and failure taxonomy.
"""

from .evaluator import (
    CoreEvaluationResult,
    aggregate_results,
    build_eval_report_from_result,
    evaluate_case,
    evaluate_direct_signal_case,
)
from .failure_taxonomy import aggregate_failure_taxonomy, classify_failure
from .gold_adapter import oracle_plan_from_gold_case, semantic_spec_from_gold_case
from .repair import RepairResult, repair_plan
from .rejector import RejectionDecision, apply_static_rejector, detect_unsupported_semantics, rejection_plan
from .semantic_spec import SemanticFeature, SemanticSpec, load_semantic_spec

__all__ = [
    "CoreEvaluationResult",
    "RepairResult",
    "RejectionDecision",
    "SemanticFeature",
    "SemanticSpec",
    "aggregate_results",
    "aggregate_failure_taxonomy",
    "apply_static_rejector",
    "build_eval_report_from_result",
    "classify_failure",
    "detect_unsupported_semantics",
    "evaluate_case",
    "evaluate_direct_signal_case",
    "load_semantic_spec",
    "oracle_plan_from_gold_case",
    "rejection_plan",
    "repair_plan",
    "semantic_spec_from_gold_case",
]
