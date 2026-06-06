from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticFeature:
    """Gold semantic unit used by the constraint-centered evaluator."""

    id: str
    semantic: str
    expected_primitives: tuple[str, ...]
    window: tuple[int, int] | None
    parameters: dict[str, Any]
    effect_type: str | None
    checker: str
    required: bool = True
    weight: float = 1.0
    tolerance: float = 1e-6

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SemanticFeature":
        if not isinstance(payload, dict):
            raise TypeError("Semantic feature must be a mapping.")
        raw_window = payload.get("window")
        window: tuple[int, int] | None
        if raw_window is None:
            window = None
        else:
            if not isinstance(raw_window, (list, tuple)) or len(raw_window) != 2:
                raise ValueError(f"Feature {payload.get('id', '<unknown>')} has invalid window.")
            window = (int(raw_window[0]), int(raw_window[1]))
        return cls(
            id=str(payload["id"]),
            semantic=str(payload["semantic"]),
            expected_primitives=tuple(str(item) for item in payload.get("expected_primitives", [])),
            window=window,
            parameters=dict(payload.get("parameters", {})),
            effect_type=str(payload["effect_type"]) if payload.get("effect_type") is not None else None,
            checker=str(payload["checker"]),
            required=bool(payload.get("required", True)),
            weight=float(payload.get("weight", 1.0)),
            tolerance=float(payload.get("tolerance", 1e-6)),
        )


@dataclass(frozen=True)
class SemanticSpec:
    """Gold semantic specification ``S*`` for one evaluation case."""

    case_id: str
    description: str
    length: int
    features: tuple[SemanticFeature, ...]
    unsupported_semantics: tuple[dict[str, Any], ...] = ()
    protected_constraints: tuple[dict[str, Any], ...] = ()
    numeric_range: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def requires_rejection(self) -> bool:
        return bool(self.unsupported_semantics)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SemanticSpec":
        if not isinstance(payload, dict):
            raise TypeError("Semantic spec must be a mapping.")
        return cls(
            case_id=str(payload.get("case_id") or payload.get("id") or "case"),
            description=str(payload["description"]),
            length=int(payload["length"]),
            features=tuple(SemanticFeature.from_mapping(item) for item in payload.get("features", [])),
            unsupported_semantics=tuple(dict(item) for item in payload.get("unsupported_semantics", [])),
            protected_constraints=tuple(dict(item) for item in payload.get("protected_constraints", [])),
            numeric_range=dict(payload["numeric_range"]) if payload.get("numeric_range") is not None else None,
            metadata=dict(payload.get("metadata", {})),
        )


def load_semantic_spec(payload: dict[str, Any] | SemanticSpec) -> SemanticSpec:
    if isinstance(payload, SemanticSpec):
        return payload
    return SemanticSpec.from_mapping(payload)
