from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_CASES_DIR = REPO_ROOT / "data" / "controlled_gold_cases"
CASE_FILES = {
    "atomic": CONTROLLED_CASES_DIR / "single_claim_cases.json",
    "compositional": CONTROLLED_CASES_DIR / "multi_claim_cases.json",
    "adversarial": CONTROLLED_CASES_DIR / "robustness_cases.json",
}


def _point(timestep: int, value: float, tolerance: float = 1e-6) -> dict:
    return {"timestep": timestep, "value": value, "tolerance": tolerance}


def _const(start: int, end: int, value: float, tolerance: float = 1e-6) -> dict:
    return {"start": start, "end": end, "value": value, "tolerance": tolerance}


def _shift_plateau_signal_checks(
    *,
    length: int,
    base: float,
    shift: float,
    shift_timestep: int,
    plateau_start: int,
    plateau_end: int,
    plateau_value: float,
) -> dict:
    shifted = base + shift
    segments = [_const(0, shift_timestep - 1, base)]
    if shift_timestep <= plateau_start - 1:
        segments.append(_const(shift_timestep, plateau_start - 1, shifted))
    segments.append(_const(plateau_start, plateau_end, plateau_value))
    if plateau_end + 1 <= length - 1:
        segments.append(_const(plateau_end + 1, length - 1, shifted))
    return {"length": length, "constant_segments": segments}


def build_atomic_cases() -> list[dict]:
    cases: list[dict] = []

    flat_levels = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0]
    for idx, level in enumerate(flat_levels):
        cases.append(
            {
                "case_id": f"generated_atomic_flat_{idx:02d}",
                "category": "atomic",
                "description": f"The series remains flat at {level:.1f} across all 100 timesteps.",
                "length": 100,
                "expected_candidate_primitives": ["add_flat"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {
                            "primitive": "add_flat",
                            "effect_type": "overwrite",
                            "parameters": {"start_timestep": 0, "end_timestep": 99, "target_value": level},
                            "semantic_role": "global flat baseline",
                        }
                    ],
                },
                "expected_signal_checks": {
                    "length": 100,
                    "constant_segments": [_const(0, 99, level)],
                },
            }
        )

    ramp_specs = [
        (10, 30, 1.0, 4.0),
        (15, 35, 2.0, 6.0),
        (20, 40, -1.0, 3.0),
        (5, 25, 0.5, 2.5),
        (30, 60, 3.0, 7.0),
        (40, 80, 5.0, 1.0),
        (25, 55, 2.0, -1.0),
        (0, 20, 1.0, 5.0),
        (50, 90, 4.0, 8.0),
        (35, 65, 6.0, 2.0),
    ]
    for idx, (start, end, a, b) in enumerate(ramp_specs):
        cases.append(
            {
                "case_id": f"generated_atomic_ramp_{idx:02d}",
                "category": "atomic",
                "description": f"Between timestep {start} and {end}, the series moves linearly from {a:.1f} to {b:.1f}.",
                "length": 100,
                "expected_candidate_primitives": ["add_ramp", "add_trend"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {
                            "primitive": "add_ramp",
                            "effect_type": "overwrite",
                            "parameters": {
                                "start_timestep": start,
                                "end_timestep": end,
                                "start_value": a,
                                "end_value": b,
                            },
                            "semantic_role": "exact linear segment",
                        }
                    ],
                },
                "expected_signal_checks": {
                    "length": 100,
                    "point_values": [_point(start, a), _point(end, b)],
                },
            }
        )

    spike_specs = [
        (12, 4.0),
        (18, 5.0),
        (24, 3.5),
        (30, 6.0),
        (36, 4.5),
        (42, 5.5),
        (48, 3.0),
        (54, 7.0),
        (60, 5.0),
        (72, 4.0),
    ]
    for idx, (timestep, value) in enumerate(spike_specs):
        cases.append(
            {
                "case_id": f"generated_atomic_spike_{idx:02d}",
                "category": "atomic",
                "description": f"There is a sharp spike at timestep {timestep}, reaching {value:.1f}.",
                "length": 100,
                "expected_candidate_primitives": ["add_spike"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {
                            "primitive": "add_spike",
                            "effect_type": "additive",
                            "parameters": {"timestep": timestep, "target_value": value},
                            "semantic_role": "sharp positive point event",
                        }
                    ],
                },
                "expected_signal_checks": {"length": 100, "point_values": [_point(timestep, value)]},
            }
        )

    shift_specs = [
        (20, 1.0),
        (25, -1.5),
        (30, 2.0),
        (35, -2.0),
        (40, 1.5),
        (45, -1.0),
        (50, 2.5),
        (55, -2.5),
        (60, 1.0),
        (65, -1.5),
    ]
    for idx, (anchor, shift) in enumerate(shift_specs):
        direction = "upward" if shift > 0 else "downward"
        cases.append(
            {
                "case_id": f"generated_atomic_shift_{idx:02d}",
                "category": "atomic",
                "description": f"At timestep {anchor}, the series shifts {direction} by {abs(shift):.1f} and stays at that new level through the end.",
                "length": 100,
                "expected_candidate_primitives": ["add_level_shift"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {
                            "primitive": "add_level_shift",
                            "effect_type": "additive",
                            "parameters": {"anchor_timestep": anchor, "shift": shift},
                            "semantic_role": "sustained level shift",
                        }
                    ],
                },
                "expected_signal_checks": {
                    "length": 100,
                    "point_values": [_point(anchor - 1, 0.0), _point(anchor, shift), _point(99, shift)],
                },
            }
        )

    peak_specs = [
        (20, 5.0, 2.0, 15, 25),
        (25, 6.0, 2.5, 20, 30),
        (30, 7.0, 1.5, 26, 34),
        (35, 8.0, 2.0, 30, 40),
        (40, 5.5, 2.0, 35, 45),
        (45, 6.5, 2.5, 40, 50),
        (50, 7.5, 1.5, 46, 54),
        (55, 8.5, 2.0, 50, 60),
        (60, 6.0, 2.0, 55, 65),
        (65, 5.0, 2.5, 60, 70),
    ]
    for idx, (center, value, width, start, end) in enumerate(peak_specs):
        cases.append(
            {
                "case_id": f"generated_atomic_peak_{idx:02d}",
                "category": "atomic",
                "description": f"A rounded peak appears at timestep {center}, reaching {value:.1f} within the local window from {start} to {end}.",
                "length": 100,
                "expected_candidate_primitives": ["add_peak"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {
                            "primitive": "add_peak",
                            "effect_type": "additive",
                            "parameters": {
                                "timestep": center,
                                "target_value": value,
                                "width": width,
                                "start_timestep": start,
                                "end_timestep": end,
                            },
                            "semantic_role": "rounded local maximum",
                        }
                    ],
                },
                "expected_signal_checks": {
                    "length": 100,
                    "point_values": [_point(center, value)],
                },
            }
        )

    return cases[:50]


def build_compositional_cases() -> list[dict]:
    cases: list[dict] = []

    for idx in range(10):
        base = 2.0 + 0.5 * (idx % 4)
        spike_t = 20 + idx * 3
        spike_v = base + 3.0
        start_decline = 55 + idx
        cases.append(
            {
                "case_id": f"generated_combo_flat_spike_decline_{idx:02d}",
                "category": "compositional",
                "description": f"The series stays flat at {base:.1f} from timestep 0 to 99. A sharp spike to {spike_v:.1f} occurs at timestep {spike_t}. After timestep {start_decline}, the series declines with slope -0.05.",
                "length": 100,
                "expected_candidate_primitives": ["add_flat", "add_spike", "add_trend"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_flat", "effect_type": "overwrite", "parameters": {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "semantic_role": "baseline"},
                        {"primitive": "add_spike", "effect_type": "additive", "parameters": {"timestep": spike_t, "target_value": spike_v}, "semantic_role": "local spike"},
                        {"primitive": "add_trend", "effect_type": "additive", "parameters": {"start_timestep": start_decline, "end_timestep": 99, "slope": -0.05}, "semantic_role": "late decline"},
                    ],
                },
                "expected_signal_checks": {"length": 100, "point_values": [_point(spike_t, spike_v)]},
            }
        )

    for idx in range(10):
        base = 1.0 + 0.5 * idx
        jump = 1.0 + 0.5 * (idx % 5)
        plateau = base + jump + 0.5
        shift_t = 30 + idx * 2
        p_start = 70
        p_end = 80 + (idx % 5)
        cases.append(
            {
                "case_id": f"generated_combo_shift_plateau_{idx:02d}",
                "category": "compositional",
                "description": f"The first half is flat at {base:.1f}. At timestep {shift_t} the series jumps up by {jump:.1f} into a new level, and from timestep {p_start} to {p_end} it levels off into a plateau at {plateau:.1f}.",
                "length": 100,
                "expected_candidate_primitives": ["add_flat", "add_level_shift"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_flat", "effect_type": "overwrite", "parameters": {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "semantic_role": "persistent baseline", "semantic_layer": "global_scaffold"},
                        {"primitive": "add_level_shift", "effect_type": "additive", "parameters": {"anchor_timestep": shift_t, "shift": jump}, "semantic_role": "regime jump", "semantic_layer": "global_scaffold"},
                        {"primitive": "add_flat", "effect_type": "overwrite", "parameters": {"start_timestep": p_start, "end_timestep": p_end, "target_value": plateau}, "semantic_role": "late plateau", "semantic_layer": "segment_structure"},
                    ],
                },
                "expected_signal_checks": _shift_plateau_signal_checks(
                    length=100,
                    base=base,
                    shift=jump,
                    shift_timestep=shift_t,
                    plateau_start=p_start,
                    plateau_end=p_end,
                    plateau_value=plateau,
                ),
            }
        )

    for idx in range(10):
        base = 8.0 + idx * 0.5
        peak_t = 20 + idx * 2
        trough_t = 70 + idx
        peak_v = base + 2.0
        trough_v = base - 3.0
        cases.append(
            {
                "case_id": f"generated_combo_peak_trough_{idx:02d}",
                "category": "compositional",
                "description": f"The series is flat at {base:.1f}. A rounded peak reaches {peak_v:.1f} near timestep {peak_t}, and a rounded trough reaches {trough_v:.1f} near timestep {trough_t}.",
                "length": 100,
                "expected_candidate_primitives": ["add_flat", "add_peak", "add_trough"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_flat", "effect_type": "overwrite", "parameters": {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "semantic_role": "baseline"},
                        {"primitive": "add_peak", "effect_type": "additive", "parameters": {"timestep": peak_t, "target_value": peak_v, "width": 2.0, "start_timestep": peak_t - 5, "end_timestep": peak_t + 5}, "semantic_role": "local maximum"},
                        {"primitive": "add_trough", "effect_type": "additive", "parameters": {"timestep": trough_t, "target_value": trough_v, "width": 2.0, "start_timestep": trough_t - 5, "end_timestep": trough_t + 5}, "semantic_role": "local minimum"},
                    ],
                },
                "expected_signal_checks": {"length": 100, "point_values": [_point(peak_t, peak_v), _point(trough_t, trough_v)]},
            }
        )

    for idx in range(10):
        base = 4.0 + 0.2 * idx
        amp = 1.0 + 0.1 * idx
        period = 10.0 + (idx % 4) * 2.0
        start = 15 + idx
        end = 80 - idx
        cases.append(
            {
                "case_id": f"generated_combo_flat_seasonal_{idx:02d}",
                "category": "compositional",
                "description": f"The series starts flat at {base:.1f}, then from timestep {start} to {end} a periodic sine oscillation with amplitude {amp:.1f} and period {period:.1f} is added on top.",
                "length": 100,
                "expected_candidate_primitives": ["add_flat", "add_seasonality"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_flat", "effect_type": "overwrite", "parameters": {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "semantic_role": "baseline"},
                        {"primitive": "add_seasonality", "effect_type": "additive", "parameters": {"start_timestep": start, "end_timestep": end, "amplitude": amp, "period": period, "waveform": "sine"}, "semantic_role": "mid-series oscillation"},
                    ],
                },
                "expected_signal_checks": {"length": 100, "unchanged_windows": [_const(0, start - 1, base), _const(end + 1, 99, base)]},
            }
        )

    for idx in range(10):
        base = 5.0 + 0.5 * (idx % 3)
        noise_scale = 0.1 + 0.02 * idx
        vol_scale = 1.5 + 0.1 * idx
        start = 20
        mid = 40
        end = 60 + (idx % 5)
        seed = 11 + idx
        cases.append(
            {
                "case_id": f"generated_combo_noise_volatility_{idx:02d}",
                "category": "compositional",
                "description": f"The series is flat at {base:.1f}. Between timestep {start} and {end}, seeded Gaussian noise with scale {noise_scale:.2f} is added. Between timestep {mid} and {end}, the fluctuations become {vol_scale:.1f} times as large around the same baseline.",
                "length": 100,
                "expected_candidate_primitives": ["add_flat", "add_noise", "add_volatility"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_flat", "effect_type": "overwrite", "parameters": {"start_timestep": 0, "end_timestep": 99, "target_value": base}, "semantic_role": "baseline"},
                        {"primitive": "add_noise", "effect_type": "additive", "parameters": {"start_timestep": start, "end_timestep": end, "noise_scale": round(noise_scale, 2), "random_seed": seed}, "semantic_role": "noise"},
                        {"primitive": "add_volatility", "effect_type": "overwrite", "parameters": {"start_timestep": mid, "end_timestep": end, "volatility_scale": round(vol_scale, 1), "baseline": base}, "semantic_role": "scaled deviations"},
                    ],
                },
                "expected_signal_checks": {
                    "length": 100,
                    "constant_segments": [_const(0, start - 1, base), _const(end + 1, 99, base)],
                    "variance_comparison": {"reference_window": [start, mid - 1], "target_window": [mid, end], "expect": "target_greater"},
                },
            }
        )

    return cases[:50]


def build_adversarial_cases() -> list[dict]:
    cases: list[dict] = []

    neg_levels = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.5, 9.0]
    for idx, level in enumerate(neg_levels):
        cases.append(
            {
                "case_id": f"generated_adv_negated_extrema_{idx:02d}",
                "category": "adversarial",
                "description": f"The series is constant at {level:.1f} with no peaks or troughs anywhere in the sequence.",
                "length": 100,
                "expected_candidate_primitives": ["add_flat"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_flat", "effect_type": "overwrite", "parameters": {"start_timestep": 0, "end_timestep": 99, "target_value": level}, "semantic_role": "constant signal"}
                    ],
                },
                "assertions": {"must_not_retrieve": ["add_peak", "add_trough", "add_spike", "add_dip"]},
            }
        )

    for idx in range(10):
        cases.append(
            {
                "case_id": f"generated_adv_noise_only_{idx:02d}",
                "category": "adversarial",
                "description": f"The last 30 timesteps become noisy with random jitter of scale {0.1 + 0.02 * idx:.2f}, but there is no claim that the existing swings are being rescaled.",
                "length": 100,
                "expected_candidate_primitives": ["add_noise"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_noise", "effect_type": "additive", "parameters": {"start_timestep": 70, "end_timestep": 99, "noise_scale": round(0.1 + 0.02 * idx, 2), "random_seed": 20 + idx}, "semantic_role": "random jitter"}
                    ],
                },
                "assertions": {"must_not_retrieve": ["add_volatility"]},
            }
        )

    for idx in range(10):
        cases.append(
            {
                "case_id": f"generated_adv_volatility_only_{idx:02d}",
                "category": "adversarial",
                "description": f"The fluctuations become more volatile after timestep {30 + idx}, meaning the existing swings widen around the same baseline rather than introducing fresh random noise.",
                "length": 100,
                "expected_candidate_primitives": ["add_volatility"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_volatility", "effect_type": "overwrite", "parameters": {"start_timestep": 30 + idx, "end_timestep": 99, "volatility_scale": 2.0 + 0.1 * idx, "baseline": 0.0}, "semantic_role": "wider swings around same center"}
                    ],
                },
                "assertions": {"must_not_retrieve": ["add_noise"]},
            }
        )

    for idx in range(10):
        cases.append(
            {
                "case_id": f"generated_adv_oscillation_weak_{idx:02d}",
                "category": "adversarial",
                "description": f"The signal oscillates mildly around a stable level near {2.0 + 0.2 * idx:.1f}, but the description does not indicate any repeating period or seasonal cycle.",
                "length": 100,
                "expected_candidate_primitives": ["add_volatility", "add_noise"],
                "expected_plan_outline": {
                    "needs_library_extension": True,
                    "unresolved_semantics": [
                        {
                            "semantic": "mild_nonperiodic_oscillation",
                            "reason": "The description denies seasonality but does not provide enough stochastic parameters for a registered noise or volatility primitive.",
                        }
                    ],
                    "steps": [],
                },
                "assertions": {"must_not_retrieve": ["add_seasonality"], "expected_retrieval_status": "weak"},
            }
        )

    for idx in range(5):
        cases.append(
            {
                "case_id": f"generated_adv_mean_reversion_{idx:02d}",
                "category": "adversarial",
                "description": "After deviating upward, the series is pulled back toward its long-run mean in a genuine mean-reverting manner.",
                "length": 100,
                "expected_candidate_primitives": [],
                "expected_plan_outline": {
                    "needs_library_extension": True,
                    "unresolved_semantics": [{"semantic": "mean_reversion", "reason": "No existing primitive explicitly applies pullback toward a reference level."}],
                    "steps": [],
                },
                "assertions": {"must_set_needs_library_extension": True, "must_include_unresolved_semantics": ["mean_reversion"]},
            }
        )

    for idx in range(5):
        start = 20 + idx * 5
        end = start + 8
        cases.append(
            {
                "case_id": f"generated_adv_gap_{idx:02d}",
                "category": "adversarial",
                "description": f"From timestep {start} to {end}, the data is missing entirely rather than merely dipping to a lower observed value.",
                "length": 100,
                "expected_candidate_primitives": ["add_gap"],
                "expected_plan_outline": {
                    "needs_library_extension": False,
                    "unresolved_semantics": [],
                    "steps": [
                        {"primitive": "add_gap", "effect_type": "overwrite", "parameters": {"start_timestep": start, "end_timestep": end}, "semantic_role": "missing interval"}
                    ],
                },
                "assertions": {"must_not_retrieve": ["add_dip", "add_trough"]},
            }
        )

    return cases[:50]


def _load_existing_manual_cases(path: Path, category: str) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return [case for case in payload if case.get("category") == category and not str(case.get("case_id", "")).startswith("generated_")]


def main() -> None:
    generated_payloads = {
        "atomic": build_atomic_cases(),
        "compositional": build_compositional_cases(),
        "adversarial": build_adversarial_cases(),
    }
    for category, generated_cases in generated_payloads.items():
        path = CASE_FILES[category]
        manual_cases = _load_existing_manual_cases(path, category)
        payload = [*manual_cases, *generated_cases]
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {path} ({len(payload)} cases: {len(manual_cases)} manual + {len(generated_cases)} generated)")


if __name__ == "__main__":
    main()
