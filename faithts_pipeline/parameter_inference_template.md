You are inferring primitive call parameters for a time-series synthesis plan.

You must NOT change which primitives are selected.
You must output exactly one JSON object. No prose. No markdown.

Goal:
- keep the selected step list unchanged
- infer only the remaining value-like `parameters` for each primitive step
- use the full caption and the interaction between steps to choose arguments jointly across the whole plan
- identify any underlying baseline or background level implied by the caption and keep additive effects consistent with that baseline
- satisfy each primitive's required argument patterns and constraints
- keep the result deterministic for any stochastic primitive

Inputs:
- Series length: {series_length}
- Caption:
{caption}

- Selection plan:
{selection_plan}

- Primitive step specs:
{primitive_step_specs}

Hard rules:
1. Output exactly one JSON object with a top-level `steps` array.
2. Each returned item must contain exactly `id`, `primitive`, and `parameters`.
3. Return one item for every primitive step in the selection plan.
4. Do not add, remove, or reorder steps.
5. Do not change any `primitive` name.
6. `parameters` must satisfy the primitive's required argument patterns and mutually exclusive argument rules.
7. Respect the series length and numeric range when choosing timesteps and values.
8. If the caption is ambiguous, choose the minimum-assumption parameterization consistent with the selected primitive.
9. For stochastic primitives such as `add_noise`, always provide a deterministic `random_seed`.
10. Treat temporal scope chosen in the selection plan as fixed structure unless it is missing or clearly inconsistent. Prefer filling value-like parameters such as levels, amplitudes, slopes, shifts, periods, phases, and stochastic seeds in this stage.
11. Do not invent unsupported fields outside `parameters`.
12. Infer parameters for the entire selected plan jointly, not as isolated per-step slot filling.
13. Earlier steps are not frozen local interpretations. If later text changes how an earlier selected step should function in the overall composition, adjust that earlier step's parameters so the whole plan remains globally coherent.
14. When two or more steps interact, choose parameter ranges and values that keep their combined effect consistent with the full caption, even if that requires revising the coverage of an earlier step.
15. Before finalizing parameters, identify whether the caption implies a persistent baseline or background level. If a baseline is introduced and the caption does not explicitly end or replace it, and later text only describes additive effects on top of that baseline, infer the baseline as continuing until the end of the series or until a later explicit overwrite/transition changes it. Treat phrases like "added on top", "superimposed on", or "around that level" as evidence that the underlying baseline should remain present before, during, and after the local additive effect.

Required JSON shape:
{{
  "steps": [
    {{
      "id": "step_1",
      "primitive": "add_flat",
      "parameters": {{
        "start_timestep": 0,
        "end_timestep": 127,
        "target_value": 7.5
      }}
    }}
  ]
}}
