from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RejectionDecision:
    should_reject: bool
    semantic: str | None = None
    reason: str | None = None
    source_text: str | None = None


def detect_unsupported_semantics(description: str) -> RejectionDecision:
    text = description.lower()
    if "mean-reverting" in text or "mean reversion" in text or "long-run mean" in text:
        return RejectionDecision(
            should_reject=True,
            semantic="mean_reversion",
            reason="No registered primitive explicitly models pullback toward a long-run mean.",
            source_text=description,
        )
    denies_period = (
        "does not indicate any repeating period" in text
        or "no repeating period" in text
        or "not indicate any repeating period" in text
    )
    mentions_oscillation = "oscillat" in text
    denies_seasonality = "seasonal cycle" in text or "seasonality" in text
    if mentions_oscillation and denies_period and denies_seasonality:
        return RejectionDecision(
            should_reject=True,
            semantic="mild_nonperiodic_oscillation",
            reason=(
                "The description requests non-periodic oscillation while denying seasonality, "
                "but does not provide registered stochastic parameters for noise or volatility."
            ),
            source_text=description,
        )
    return RejectionDecision(should_reject=False)


def rejection_plan(
    *,
    description: str,
    length: int,
    numeric_range: dict[str, Any] | None = None,
    decision: RejectionDecision | None = None,
) -> dict[str, Any]:
    decision = decision or detect_unsupported_semantics(description)
    if not decision.should_reject:
        raise ValueError("Cannot build a rejection plan when no unsupported semantic was detected.")
    semantic = str(decision.semantic or "unsupported_semantic")
    source_text = str(decision.source_text or description)
    reason = str(decision.reason or "Requested semantic is outside the current primitive library.")
    unresolved = {
        "id": "u1",
        "description": semantic,
        "source_text": source_text,
        "reason": reason,
    }
    return {
        "version": "1.0",
        "input_description": description,
        "registry_validation": {
            "status": "valid",
            "schema_path": "schemas/primitive.schema.json",
            "target_path": "atomic_timeseries/registry.json",
            "errors": [],
        },
        "global_context": {
            "length": int(length),
            "numeric_range": numeric_range or {"min": -1000.0, "max": 1000.0},
            "dt": None,
            "seed": None,
        },
        "steps": [
            {
                "id": "u1",
                "kind": "unresolved",
                "description": semantic,
                "semantic_role": semantic,
                "confidence": 0.9,
                "source_text": source_text,
                "reason": reason,
                "needs_library_extension": True,
            }
        ],
        "unresolved_semantics": [unresolved],
        "needs_library_extension": True,
        "assumptions": ["Static rejector converted an unsupported description into an unresolved plan."],
    }


def apply_static_rejector(plan: dict[str, Any], *, description: str, length: int, numeric_range: dict[str, Any] | None = None) -> dict[str, Any]:
    if plan.get("needs_library_extension") or plan.get("unresolved_semantics"):
        return deepcopy(plan)
    decision = detect_unsupported_semantics(description)
    if not decision.should_reject:
        return deepcopy(plan)
    return rejection_plan(description=description, length=length, numeric_range=numeric_range, decision=decision)
