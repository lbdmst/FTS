from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?"
POSITION_UNIT_PATTERN = r"(?:time\s*step|timestep|data\s*points?|observations?|points?)"
UNIT_BEFORE_POSITION_PATTERN = re.compile(
    rf"\b(?P<unit>{POSITION_UNIT_PATTERN})\s+(?P<position>{NUMBER_PATTERN})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimResult:
    name: str
    score: float
    confidence: float
    evidence: str
    supported: bool
    contradicted: bool
    unverifiable: bool


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _series_scale(series: np.ndarray) -> float:
    scale = max(float(np.max(series) - np.min(series)), float(abs(np.mean(series))), 1.0)
    return scale


def _position_to_index(raw_position: str, unit: str | None, series_length: int) -> int | None:
    try:
        position = int(round(float(raw_position)))
    except (TypeError, ValueError):
        return None
    normalized_unit = (unit or "").lower().replace(" ", "")
    if "datapoint" in normalized_unit or "observation" in normalized_unit or normalized_unit in {"point", "points"}:
        position -= 1
    return max(0, min(series_length - 1, position))


def _normalize_unit_before_position(text: str) -> str:
    return UNIT_BEFORE_POSITION_PATTERN.sub(
        lambda match: f"{match.group('position')} {match.group('unit')}",
        text,
    )


def _value_alignment_score(observed: float, expected: float, series: np.ndarray) -> float:
    tolerance = max(0.05 * _series_scale(series), 0.25)
    return max(0.0, 1.0 - abs(observed - expected) / tolerance)


def _linear_slope(series: np.ndarray) -> float:
    x = np.arange(series.shape[0], dtype=float)
    slope, _ = np.polyfit(x, series, 1)
    return float(slope)


def _relative_volatility(series: np.ndarray) -> float:
    return float(np.std(series) / _series_scale(series))


def _change_point_strength(series: np.ndarray, window: int) -> tuple[int, float]:
    best_index = 0
    best_strength = 0.0
    for idx in range(window, len(series) - window):
        left = series[idx - window : idx]
        right = series[idx : idx + window]
        strength = abs(float(np.mean(right) - np.mean(left)))
        if strength > best_strength:
            best_index = idx
            best_strength = strength
    return best_index, best_strength


def _make_claim(
    *,
    name: str,
    score: float,
    confidence: float,
    evidence: str,
    support_threshold: float = 0.7,
    contradict_threshold: float = 0.35,
) -> ClaimResult:
    bounded_score = _clip01(score)
    bounded_confidence = _clip01(confidence)
    return ClaimResult(
        name=name,
        score=bounded_score,
        confidence=bounded_confidence,
        evidence=evidence,
        supported=bounded_score >= support_threshold,
        contradicted=bounded_score <= contradict_threshold,
        unverifiable=False,
    )


def _make_unverifiable(name: str, evidence: str) -> ClaimResult:
    return ClaimResult(
        name=name,
        score=0.5,
        confidence=0.25,
        evidence=evidence,
        supported=False,
        contradicted=False,
        unverifiable=True,
    )


def _claim_correctness_weight(claim: ClaimResult) -> float:
    """Weight precise, anchored claims slightly more than broad visual claims."""
    if claim.unverifiable:
        return 0.0
    multiplier = 1.0
    if claim.name.startswith(("point_value", "range_bounds", "peak_alignment", "trough_alignment")):
        multiplier = 1.25
    elif claim.name.startswith(("change_point", "segment_")):
        multiplier = 1.15
    return max(0.05, claim.confidence * multiplier)


def _caption_correctness_score(claim_results: list[ClaimResult]) -> tuple[float, dict[str, float]]:
    usable = [claim for claim in claim_results if not claim.unverifiable]
    if not usable:
        return 0.5, {
            "weighted_mean": 0.5,
            "contradicted_weight_rate": 0.0,
            "severe_mismatch_weight_rate": 0.0,
        }

    weights = [_claim_correctness_weight(claim) for claim in usable]
    total_weight = float(sum(weights))
    weighted_mean = float(sum(weight * claim.score for weight, claim in zip(weights, usable)) / total_weight)
    contradicted_weight = float(
        sum(weight for weight, claim in zip(weights, usable) if claim.contradicted)
    )
    severe_mismatch_weight = float(
        sum(weight for weight, claim in zip(weights, usable) if claim.score <= 0.2)
    )
    contradicted_rate = contradicted_weight / total_weight
    severe_mismatch_rate = severe_mismatch_weight / total_weight
    score = _clip01(weighted_mean - 0.20 * contradicted_rate - 0.10 * severe_mismatch_rate)
    return score, {
        "weighted_mean": weighted_mean,
        "contradicted_weight_rate": contradicted_rate,
        "severe_mismatch_weight_rate": severe_mismatch_rate,
    }


def _summary_verdict_from_claims(claim_results: list[ClaimResult], caption_correctness_score: float) -> str:
    usable = [claim for claim in claim_results if not claim.unverifiable]
    supported = [claim for claim in usable if claim.supported]
    contradicted = [claim for claim in usable if claim.contradicted]
    if usable and not contradicted and len(supported) == len(usable):
        return "consistent"
    if contradicted and caption_correctness_score < 0.6:
        return "inconsistent"
    if supported or caption_correctness_score >= 0.6:
        return "partially_consistent"
    return "inconsistent"


def _extract_peak_claims(description: str, series_length: int) -> list[tuple[int | None, float | None]]:
    description = _normalize_unit_before_position(description)
    patterns = [
        re.compile(
            rf"\b(?:peak(?:s|ed|ing)?|maximum|local maximum|reaches?\s+a\s+peak|reaching\s+a\s+peak)"
            rf"[^.,;]*?\b(?:of|at|to|near|around|value\s*(?:of)?|with\s+value)\s+"
            rf"(?P<value>{NUMBER_PATTERN})[^.,;]*?\b(?:at|near|around|by)\s+(?:the\s+)?"
            rf"(?P<position>{NUMBER_PATTERN})(?:st|nd|rd|th)?(?:\s*(?P<unit>{POSITION_UNIT_PATTERN}))?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:peak|maximum|local maximum)(?:\s+is\s+observed|\s+observed|\s+occurs)?"
            rf"[^.]*?\b(?:at|near|around)\s+timestep\s+(?P<position>{NUMBER_PATTERN})"
            rf"(?:[^.]*?\b(?:value\s*(?:of)?|with\s+value)\s*(?P<value>{NUMBER_PATTERN}))?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:peak(?:s|ed|ing)?|maximum|local maximum|reaches?\s+a\s+peak|reaching\s+a\s+peak)"
            rf"[^.]*?\b(?:at|near|around|by)\s+(?:the\s+)?(?P<position>{NUMBER_PATTERN})"
            rf"(?:st|nd|rd|th)?\s*(?P<unit>{POSITION_UNIT_PATTERN})",
            re.IGNORECASE,
        ),
    ]
    claims: list[tuple[int | None, float | None]] = []
    seen: set[tuple[int | None, float | None]] = set()
    for pattern in patterns:
        for match in pattern.finditer(description):
            groups = match.groupdict()
            raw_unit = groups.get("unit")
            if raw_unit is None and groups.get("value") and "timestep" not in match.group(0).lower():
                prefix_before_position = match.group(0).lower().rsplit(groups["position"].lower(), 1)[0]
                if not re.search(r"\b(?:at|by)\s+(?:the\s+)?$", prefix_before_position):
                    continue
            unit = raw_unit or ("point" if groups.get("value") and "timestep" not in match.group(0).lower() else "timestep")
            timestep = _position_to_index(groups.get("position", ""), unit, series_length)
            value = float(groups["value"]) if groups.get("value") else None
            key = (timestep, value)
            if key in seen:
                continue
            seen.add(key)
            claims.append((timestep, value))
    return claims


def _extract_trough_claims(description: str, series_length: int) -> list[tuple[int | None, float | None]]:
    description = _normalize_unit_before_position(description)
    patterns = [
        re.compile(
            rf"\b(?:trough(?:s|ed|ing)?|minimum|local minimum|low)"
            rf"[^.,;]*?\b(?:of|at|to|near|around|value\s*(?:of)?|with\s+value)\s+"
            rf"(?P<value>{NUMBER_PATTERN})[^.,;]*?\b(?:at|near|around|by)\s+(?:the\s+)?"
            rf"(?P<position>{NUMBER_PATTERN})(?:st|nd|rd|th)?(?:\s*(?P<unit>{POSITION_UNIT_PATTERN}))?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:trough|minimum|local minimum|low)(?:\s+is\s+observed|\s+observed|\s+occurs)?"
            rf"[^.]*?\b(?:at|near|around)\s+timestep\s+(?P<position>{NUMBER_PATTERN})"
            rf"(?:[^.]*?\b(?:value\s*(?:of)?|with\s+value)\s*(?P<value>{NUMBER_PATTERN}))?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:trough(?:s|ed|ing)?|minimum|local minimum|low)"
            rf"[^.]*?\b(?:at|near|around|by)\s+(?:the\s+)?(?P<position>{NUMBER_PATTERN})"
            rf"(?:st|nd|rd|th)?\s*(?P<unit>{POSITION_UNIT_PATTERN})",
            re.IGNORECASE,
        ),
    ]
    claims: list[tuple[int | None, float | None]] = []
    seen: set[tuple[int | None, float | None]] = set()
    for pattern in patterns:
        for match in pattern.finditer(description):
            groups = match.groupdict()
            raw_unit = groups.get("unit")
            if raw_unit is None and groups.get("value") and "timestep" not in match.group(0).lower():
                prefix_before_position = match.group(0).lower().rsplit(groups["position"].lower(), 1)[0]
                if not re.search(r"\b(?:at|by)\s+(?:the\s+)?$", prefix_before_position):
                    continue
            unit = raw_unit or ("point" if groups.get("value") and "timestep" not in match.group(0).lower() else "timestep")
            timestep = _position_to_index(groups.get("position", ""), unit, series_length)
            value = float(groups["value"]) if groups.get("value") else None
            key = (timestep, value)
            if key in seen:
                continue
            seen.add(key)
            claims.append((timestep, value))
    return claims


def _extract_change_point_claims(description: str) -> list[int]:
    pattern = re.compile(
        r"(?:after|at|from)\s+timestep\s+(\d+)[^.]{0,60}\b(?:shift|jump|change|break|plateau|stabiliz|climb|declin|drop|rise)",
        re.IGNORECASE,
    )
    return [int(match.group(1)) for match in pattern.finditer(description)]


def _trend_claim(description: str, series: np.ndarray) -> ClaimResult | None:
    lowered = description.lower()
    slope = _linear_slope(series)
    endpoint_delta = float(series[-1] - series[0])
    scale = _series_scale(series)
    strength = abs(endpoint_delta) / scale
    evidence = f"slope={slope:.6f}, endpoint_delta={endpoint_delta:.6f}, normalized_strength={strength:.3f}"

    upward = any(token in lowered for token in ["upward trend", "upward", "increase", "increasing", "climb", "rises"])
    downward = any(token in lowered for token in ["downward trend", "downward", "decline", "decrease", "fall", "drops"])
    flat = bool(
        re.search(
            r"\b(?:overall|generally|mostly|relatively)?\s*(?:flat|constant|stable trajectory|stable profile)\b",
            lowered,
        )
        or re.search(r"\bremains?\s+(?:flat|constant|stable|fixed)\b", lowered)
        or "no discernible trend" in lowered
    )
    weak_trend_language = bool(
        re.search(r"\b(?:general|overall|gradual|slight|minor|modest|steady|consistent)\b", lowered)
    )
    strong_trend_language = bool(
        re.search(r"\b(?:strong|significant|pronounced|prominent|sharp|clear|robust)\b", lowered)
    )
    trend_support_threshold = 0.7 if strong_trend_language and not weak_trend_language else 0.6

    if upward and not downward:
        score = 0.15
        if slope > 0 or endpoint_delta > 0:
            score = 0.6 + 0.4 * min(1.0, strength / 0.2)
        return _make_claim(
            name="overall_trend_up",
            score=score,
            confidence=0.85,
            evidence=evidence,
            support_threshold=trend_support_threshold,
        )

    if downward and not upward:
        score = 0.15
        if slope < 0 or endpoint_delta < 0:
            score = 0.6 + 0.4 * min(1.0, strength / 0.2)
        return _make_claim(
            name="overall_trend_down",
            score=score,
            confidence=0.85,
            evidence=evidence,
            support_threshold=trend_support_threshold,
        )

    if flat:
        score = 1.0 - min(1.0, strength / 0.08)
        return _make_claim(name="overall_trend_flat_or_stable", score=score, confidence=0.75, evidence=evidence)

    return None


def _volatility_claim(description: str, series: np.ndarray) -> ClaimResult | None:
    lowered = description.lower()
    rel_vol = _relative_volatility(series)
    mean_abs_diff = float(np.mean(np.abs(np.diff(series)))) / _series_scale(series)
    evidence = f"relative_std={rel_vol:.6f}, relative_mean_abs_diff={mean_abs_diff:.6f}"

    minor_fluctuation = bool(
        re.search(r"\b(?:minor|small|slight|mild|modest|limited)\s+fluctuat", lowered)
        or re.search(r"\bfluctuat\w*\s+(?:slightly|mildly|modestly)\b", lowered)
    )
    moderate_fluctuation = bool(re.search(r"\bmoderate\w*\s+fluctuat", lowered))
    high_fluctuation = bool(
        re.search(r"\b(?:high|large|wide|strong|substantial|significant|noticeable)\s+fluctuat", lowered)
    )
    generic_fluctuation = bool(re.search(r"\bfluctuat\w*", lowered))
    low = any(
        token in lowered
        for token in ["low volatility", "very low volatility", "minimal deviation", "smooth", "stable trajectory", "non-existent"]
    )
    moderate = "moderate volatility" in lowered or "moderately volatile" in lowered or moderate_fluctuation
    high = any(token in lowered for token in ["high volatility", "noticeable volatility", "volatile", "noisy"]) or high_fluctuation

    if minor_fluctuation and not high:
        low_variation_fit = 1.0 - min(1.0, rel_vol / 0.08)
        fluctuation_presence = min(1.0, mean_abs_diff / 0.006)
        score = 0.8 * low_variation_fit + 0.2 * fluctuation_presence
        return _make_claim(
            name="fluctuation_minor",
            score=score,
            confidence=0.6,
            evidence=evidence,
            support_threshold=0.6,
            contradict_threshold=0.25,
        )
    if low and not high:
        score = 1.0 - min(1.0, rel_vol / 0.03)
        return _make_claim(name="volatility_low", score=score, confidence=0.7, evidence=evidence)
    if moderate and not high:
        center = 0.04
        distance = abs(rel_vol - center)
        score = 1.0 - min(1.0, distance / 0.04)
        return _make_claim(name="volatility_moderate", score=score, confidence=0.6, evidence=evidence)
    if high:
        score = min(1.0, rel_vol / 0.05)
        return _make_claim(name="volatility_high", score=score, confidence=0.7, evidence=evidence)
    if generic_fluctuation:
        score = min(1.0, max(rel_vol / 0.015, mean_abs_diff / 0.006))
        return _make_claim(
            name="fluctuation_present",
            score=score,
            confidence=0.55,
            evidence=evidence,
            support_threshold=0.55,
            contradict_threshold=0.2,
        )
    return None


def _peak_and_trough_claims(description: str, series: np.ndarray) -> list[ClaimResult]:
    claims: list[ClaimResult] = []
    scale = _series_scale(series)
    argmax = int(np.argmax(series))
    argmin = int(np.argmin(series))
    max_value = float(series[argmax])
    min_value = float(series[argmin])

    peak_claims = _extract_peak_claims(description, len(series))
    if any(timestep is not None or value is not None for timestep, value in peak_claims):
        peak_claims = [(timestep, value) for timestep, value in peak_claims if timestep is not None or value is not None]
    for timestep, value in peak_claims:
        if timestep is None and value is None:
            claims.append(_make_unverifiable("peak", "peak mentioned without measurable anchor"))
            continue
        time_score = 1.0
        value_score = 1.0
        evidence_parts = [f"observed_argmax={argmax}", f"observed_peak_value={max_value:.6f}"]
        if timestep is not None:
            time_score = max(0.0, 1.0 - abs(argmax - timestep) / 12.0)
            evidence_parts.append(f"claimed_timestep={timestep}")
        if value is not None:
            value_score = max(0.0, 1.0 - abs(max_value - value) / max(0.05 * scale, 0.5))
            evidence_parts.append(f"claimed_value={value:.6f}")
        claims.append(
            _make_claim(
                name="peak_alignment",
                score=0.55 * time_score + 0.45 * value_score,
                confidence=0.9 if timestep is not None or value is not None else 0.4,
                evidence=", ".join(evidence_parts),
            )
        )

    trough_claims = _extract_trough_claims(description, len(series))
    if any(timestep is not None or value is not None for timestep, value in trough_claims):
        trough_claims = [(timestep, value) for timestep, value in trough_claims if timestep is not None or value is not None]
    for timestep, value in trough_claims:
        if timestep is None and value is None:
            claims.append(_make_unverifiable("trough", "trough mentioned without measurable anchor"))
            continue
        time_score = 1.0
        value_score = 1.0
        evidence_parts = [f"observed_argmin={argmin}", f"observed_trough_value={min_value:.6f}"]
        if timestep is not None:
            time_score = max(0.0, 1.0 - abs(argmin - timestep) / 12.0)
            evidence_parts.append(f"claimed_timestep={timestep}")
        if value is not None:
            value_score = max(0.0, 1.0 - abs(min_value - value) / max(0.05 * scale, 0.5))
            evidence_parts.append(f"claimed_value={value:.6f}")
        claims.append(
            _make_claim(
                name="trough_alignment",
                score=0.55 * time_score + 0.45 * value_score,
                confidence=0.9 if timestep is not None or value is not None else 0.4,
                evidence=", ".join(evidence_parts),
            )
        )

    return claims


def _change_point_claim(description: str, series: np.ndarray) -> ClaimResult | None:
    candidate_points = _extract_change_point_claims(description)
    if not candidate_points:
        return None
    window = max(5, len(series) // 10)
    observed_index, observed_strength = _change_point_strength(series, window)
    scale = _series_scale(series)
    observed_norm = observed_strength / scale
    point_scores: list[float] = []
    point_evidence: list[str] = []
    for point in candidate_points:
        if point < window or point >= len(series) - window:
            continue
        left = series[point - window : point]
        right = series[point : point + window]
        local_shift = abs(float(np.mean(right) - np.mean(left)))
        local_shift_norm = local_shift / scale
        left_slope = _linear_slope(left)
        right_slope = _linear_slope(right)
        slope_delta_norm = abs(right_slope - left_slope) * len(series) / scale
        proximity = max(0.0, 1.0 - abs(observed_index - point) / float(2 * window))
        local_score = 0.45 * min(1.0, local_shift_norm / 0.08) + 0.35 * min(1.0, slope_delta_norm / 0.4) + 0.20 * proximity
        point_scores.append(local_score)
        point_evidence.append(
            f"point={point}: local_shift={local_shift:.6f}, local_shift_norm={local_shift_norm:.6f}, "
            f"slope_delta_norm={slope_delta_norm:.6f}, proximity_to_global={proximity:.3f}"
        )
    score = max(point_scores) if point_scores else min(1.0, observed_norm / 0.1)
    evidence = (
        f"observed_change_point={observed_index}, claimed_points={candidate_points}, "
        f"window={window}, global_mean_shift={observed_strength:.6f}, normalized_global_shift={observed_norm:.6f}; "
        + " | ".join(point_evidence)
    )
    return _make_claim(name="change_point_alignment", score=score, confidence=0.8, evidence=evidence)


def _explicit_point_value_claims(description: str, series: np.ndarray) -> list[ClaimResult]:
    description = _normalize_unit_before_position(description)
    claims: list[ClaimResult] = []
    seen: set[tuple[str, int, float]] = set()
    patterns = [
        (
            "peak",
            re.compile(
                rf"\b(?:peak(?:s|ed|ing)?|maximum|local maximum|reaches?\s+a\s+peak|reaching\s+a\s+peak)"
                rf"[^.,;]*?\b(?:of|at|to|near|around|value\s*(?:of)?|with\s+value)\s+"
                rf"(?P<value>{NUMBER_PATTERN})[^.,;]*?\b(?:at|near|around|by)\s+(?:the\s+)?"
                rf"(?P<position>{NUMBER_PATTERN})(?:st|nd|rd|th)?(?:\s*(?P<unit>{POSITION_UNIT_PATTERN}))?",
                re.IGNORECASE,
            ),
        ),
        (
            "trough",
            re.compile(
                rf"\b(?:trough(?:s|ed|ing)?|minimum|local minimum|low)"
                rf"[^.,;]*?\b(?:of|at|to|near|around|value\s*(?:of)?|with\s+value)\s+"
                rf"(?P<value>{NUMBER_PATTERN})[^.,;]*?\b(?:at|near|around|by)\s+(?:the\s+)?"
                rf"(?P<position>{NUMBER_PATTERN})(?:st|nd|rd|th)?(?:\s*(?P<unit>{POSITION_UNIT_PATTERN}))?",
                re.IGNORECASE,
            ),
        ),
        (
            "movement_endpoint",
            re.compile(
                rf"\b(?:declin\w*|decreas\w*|drop\w*|fall\w*|rise\w*|increas\w*|recover\w*|climb\w*)"
                rf"[^.]{0,100}?\bto\s+(?P<value>{NUMBER_PATTERN})\s+by\s+(?:the\s+)?"
                rf"(?P<position>{NUMBER_PATTERN})(?:st|nd|rd|th)?\s*(?P<unit>{POSITION_UNIT_PATTERN})",
                re.IGNORECASE,
            ),
        ),
        (
            "point_value",
            re.compile(
                rf"\b(?:value|level|reading|starts?\s+(?:at|from)|begins?\s+(?:at|from)|high\s+point\s+of|low\s+of)\s+"
                rf"(?P<value>{NUMBER_PATTERN})[^.]{0,80}?\b(?:at|near|around|by)\s+(?:the\s+)?"
                rf"(?P<position>{NUMBER_PATTERN})(?:st|nd|rd|th)?\s*(?P<unit>{POSITION_UNIT_PATTERN})",
                re.IGNORECASE,
            ),
        ),
        (
            "point_value",
            re.compile(
                rf"\b(?:at|by)\s+(?:the\s+)?(?P<position>{NUMBER_PATTERN})(?:st|nd|rd|th)?\s*"
                rf"(?P<unit>{POSITION_UNIT_PATTERN})[^.]{0,80}?\b(?:value|level|reading)\s+(?:of\s+)?"
                rf"(?P<value>{NUMBER_PATTERN})",
                re.IGNORECASE,
            ),
        ),
    ]
    for kind, pattern in patterns:
        for match in pattern.finditer(description):
            raw_unit = match.groupdict().get("unit")
            if kind in {"peak", "trough"} and raw_unit is None:
                prefix_before_position = match.group(0).lower().rsplit(match.group("position").lower(), 1)[0]
                if not re.search(r"\b(?:at|by)\s+(?:the\s+)?$", prefix_before_position):
                    continue
            unit = raw_unit or ("point" if kind in {"peak", "trough"} else None)
            index = _position_to_index(match.group("position"), unit, len(series))
            if index is None:
                continue
            expected = float(match.group("value"))
            key = (kind, index, round(expected, 8))
            if key in seen:
                continue
            seen.add(key)
            observed = float(series[index])
            score = _value_alignment_score(observed, expected, series)
            evidence = (
                f"index={index}, unit={match.group('unit')}, claimed_position={match.group('position')}, "
                f"claimed_value={expected:.6f}, observed_value={observed:.6f}"
            )
            claims.append(
                _make_claim(
                    name=f"point_value_{index}",
                    score=score,
                    confidence=0.95,
                    evidence=evidence,
                    support_threshold=0.75,
                    contradict_threshold=0.4,
                )
            )
    return claims


def _explicit_range_bound_claims(description: str, series: np.ndarray) -> list[ClaimResult]:
    description = _normalize_unit_before_position(description)
    pattern = re.compile(
        rf"\bbetween\s+(?P<low>{NUMBER_PATTERN})\s+and\s+(?P<high>{NUMBER_PATTERN})\s+"
        rf"from\s+(?:the\s+)?(?P<start>{NUMBER_PATTERN})(?:st|nd|rd|th)?\s+to\s+"
        rf"(?:the\s+)?(?P<end>{NUMBER_PATTERN})(?:st|nd|rd|th)?\s*(?P<unit>{POSITION_UNIT_PATTERN})",
        re.IGNORECASE,
    )
    claims: list[ClaimResult] = []
    for match in pattern.finditer(description):
        start = _position_to_index(match.group("start"), match.group("unit"), len(series))
        end = _position_to_index(match.group("end"), match.group("unit"), len(series))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        lower = min(float(match.group("low")), float(match.group("high")))
        upper = max(float(match.group("low")), float(match.group("high")))
        segment = series[start : end + 1]
        if segment.size == 0:
            continue
        tolerance = max(0.02 * _series_scale(series), 0.1)
        violations = np.maximum(lower - segment, segment - upper)
        violations = np.maximum(violations, 0.0)
        inside_rate = float(np.mean(violations <= tolerance))
        magnitude_score = max(0.0, 1.0 - float(np.max(violations)) / tolerance)
        score = 0.65 * inside_rate + 0.35 * magnitude_score
        evidence = (
            f"window=({start},{end}), claimed_range=({lower:.6f},{upper:.6f}), "
            f"observed_min={float(np.min(segment)):.6f}, observed_max={float(np.max(segment)):.6f}, "
            f"inside_rate={inside_rate:.3f}"
        )
        claims.append(
            _make_claim(
                name=f"range_bounds_{start}_{end}",
                score=score,
                confidence=0.9,
                evidence=evidence,
                support_threshold=0.75,
                contradict_threshold=0.4,
            )
        )
    return claims


def _segment_direction_claims(description: str, series: np.ndarray) -> list[ClaimResult]:
    claims: list[ClaimResult] = []
    lowered = description.lower()
    range_pattern = re.compile(r"from timestep\s+(\d+)\s+to\s+(\d+)[^.]{0,80}", re.IGNORECASE)
    for match in range_pattern.finditer(description):
        start = int(match.group(1))
        end = int(match.group(2))
        if start >= end or end >= len(series):
            continue
        snippet = match.group(0).lower()
        seg = series[start : end + 1]
        delta = float(seg[-1] - seg[0])
        strength = abs(delta) / _series_scale(series)
        evidence = f"window=({start},{end}), endpoint_delta={delta:.6f}, normalized_strength={strength:.3f}"
        if any(token in snippet for token in ["down", "declin", "fall", "drop"]):
            score = 0.15
            if delta < 0:
                score = 0.6 + 0.4 * min(1.0, strength / 0.08)
            claims.append(_make_claim(name=f"segment_down_{start}_{end}", score=score, confidence=0.8, evidence=evidence))
        elif any(token in snippet for token in ["up", "rise", "increase", "climb"]):
            score = 0.15
            if delta > 0:
                score = 0.6 + 0.4 * min(1.0, strength / 0.08)
            claims.append(_make_claim(name=f"segment_up_{start}_{end}", score=score, confidence=0.8, evidence=evidence))
        elif any(token in snippet for token in ["flat", "stable", "constant", "plateau"]):
            score = 1.0 - min(1.0, strength / 0.04)
            claims.append(_make_claim(name=f"segment_flat_{start}_{end}", score=score, confidence=0.7, evidence=evidence))

    if not claims and any(token in lowered for token in ["final third", "initial", "middle"]):
        return claims
    return claims


def evaluate_description_series_consistency(description: str, series: np.ndarray | list[float]) -> dict[str, Any]:
    series_array = np.asarray(series, dtype=float)
    if series_array.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if series_array.size < 8:
        raise ValueError("series must contain at least 8 points")
    if not np.isfinite(series_array).all():
        raise ValueError("series must be finite")

    claim_results: list[ClaimResult] = []
    for builder in (_trend_claim, _volatility_claim, _change_point_claim):
        result = builder(description, series_array)
        if result is not None:
            claim_results.append(result)
    claim_results.extend(_explicit_point_value_claims(description, series_array))
    claim_results.extend(_explicit_range_bound_claims(description, series_array))
    claim_results.extend(_segment_direction_claims(description, series_array))
    claim_results.extend(_peak_and_trough_claims(description, series_array))

    if not claim_results:
        claim_results.append(_make_unverifiable("no_extractable_claims", "description did not contain measurable rules"))

    supported = [claim for claim in claim_results if claim.supported]
    contradicted = [claim for claim in claim_results if claim.contradicted]
    unverifiable = [claim for claim in claim_results if claim.unverifiable]
    usable = [claim for claim in claim_results if not claim.unverifiable]

    aspect_scores = {
        claim.name: round(claim.score, 4)
        for claim in claim_results
    }

    if usable:
        consistency_score = float(np.mean([claim.score for claim in usable]))
        evidence_strength = float(np.mean([abs(claim.score - 0.5) * 2.0 for claim in usable]))
        explicit_ratio = min(1.0, len(usable) / 4.0)
        confidence_score = 0.35 + 0.35 * explicit_ratio + 0.30 * evidence_strength
    else:
        consistency_score = 0.5
        confidence_score = 0.25

    consistency_score = _clip01(consistency_score)
    confidence_score = _clip01(confidence_score)
    caption_correctness_score, correctness_details = _caption_correctness_score(claim_results)

    verdict = _summary_verdict_from_claims(claim_results, caption_correctness_score)

    return {
        "summary_verdict": verdict,
        "consistency_score": round(consistency_score, 4),
        "caption_correctness_score": round(caption_correctness_score, 4),
        "correctness_details": {
            key: round(value, 4)
            for key, value in correctness_details.items()
        },
        "confidence_score": round(confidence_score, 4),
        "verifiable_claim_count": len(usable),
        "contradicted_claim_rate": round(float(len(contradicted) / len(usable)), 4) if usable else 0.0,
        "aspect_scores": aspect_scores,
        "extracted_claims": [claim.name for claim in claim_results],
        "matched_claims": [asdict(claim) for claim in supported],
        "mismatched_claims": [asdict(claim) for claim in contradicted],
        "unverifiable_claims": [asdict(claim) for claim in unverifiable],
        "claim_results": [asdict(claim) for claim in claim_results],
        "series_summary": {
            "length": int(series_array.size),
            "min": float(np.min(series_array)),
            "max": float(np.max(series_array)),
            "mean": float(np.mean(series_array)),
            "std": float(np.std(series_array)),
            "argmax": int(np.argmax(series_array)),
            "argmin": int(np.argmin(series_array)),
            "slope": _linear_slope(series_array),
        },
    }


def _load_series(path: Path, index: int | None) -> np.ndarray:
    if path.suffix == ".npy":
        payload = np.load(path)
        if payload.ndim == 1:
            return np.asarray(payload, dtype=float)
        if index is None:
            raise ValueError("index is required when loading a 2D .npy file")
        return np.asarray(payload[index], dtype=float)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "reference_series" in payload[0]:
        if index is None:
            raise ValueError("index is required when loading a case list")
        return np.asarray(payload[index]["reference_series"], dtype=float)
    if isinstance(payload, list):
        return np.asarray(payload, dtype=float)
    raise ValueError(f"Unsupported series input: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate description/series consistency with simple rule-based scores.")
    parser.add_argument("--description", help="Natural-language description to evaluate.")
    parser.add_argument("--description-file", type=Path, help="Path to a UTF-8 text file containing the description.")
    parser.add_argument("--series-file", type=Path, required=True, help="Path to a .npy file or JSON list of series values/cases.")
    parser.add_argument("--index", type=int, default=None, help="Series index for 2D .npy or case-list JSON inputs.")
    args = parser.parse_args()

    if args.description:
        description = args.description
    elif args.description_file is not None:
        description = args.description_file.read_text(encoding="utf-8").strip()
    else:
        raise ValueError("either --description or --description-file is required")

    series = _load_series(args.series_file, args.index)
    report = evaluate_description_series_consistency(description, series)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
