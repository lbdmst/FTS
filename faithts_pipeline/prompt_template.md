You are planning a time-series synthesis program.

You must NOT write Python code.
You must output exactly one JSON object that follows the required schema contract.

Goal:
- infer the structure of the caption
- represent it as a strict intermediate plan
- select which primitive functions should be used
- infer temporal scope in this stage and leave value-like primitive call parameters for a later parameter-inference step
- use only provided candidate primitives
- never invent a primitive
- mark unsupported semantics explicitly

Inputs:
- Series length: {series_length}
- Caption:
{caption}

- Candidate primitives:
{candidate_primitives}

- Primitive specs:
{primitive_specs}

Hard rules:
1. Output exactly one JSON object. No prose. No markdown.
2. Do not output Python code.
3. Use only primitives from the candidate set.
4. Do not output patch steps or free-form numpy edits.
5. Every executable step must be a `primitive` step.
5a. At this planning stage, use `parameters` to encode temporal scope when it is inferable from the caption, such as `start_timestep`, `end_timestep`, `anchor_timestep`, or `timestep`. Leave value-like parameters for the later parameter-inference step.
5b. Preserve caption indexing carefully: `timestep` means the executor's zero-based index, while `point`, `data point`, `observation`, `mark`, and bare decimal ordinal positions in real captions such as `25.0` mean one-based caption points, so `25.0` maps to index `24`.
6. If semantics are unsupported, emit an `unresolved` step and add a matching item to `unresolved_semantics`.
6a. Every item in `unresolved_semantics` must be an object with keys `id`, `description`, `source_text`, and `reason`. Never emit raw strings in `unresolved_semantics`.
7. Set `needs_library_extension` to `true` if and only if unresolved semantics remain.
8. Never use an additive primitive as if it sets an absolute level.
9. Keep step order chronological and composition-safe.
10. Use the minimum-assumption interpretation when the caption is ambiguous.
11. Use `semantic_layer` to distinguish `global_scaffold`, `segment_structure`, `local_event`, and `texture`.
12. Use `priority` for within-layer ordering when later steps must intentionally override earlier ones.
13. Use `protected_constraints` when a step establishes a property that later steps should preserve.
14. Use `add_ramp` for absolute linear segments of the realized series, such as "from 41.0 down to 39.82".
15. Use `add_trend` only for additive drift relative to an existing baseline; never use it to express absolute start/end series levels.
16. If you choose `add_trend`, provide a valid argument pattern: either `slope`, or both `start_offset` and `end_offset`. Avoid `start_value`/`end_value` unless required for backward compatibility, and never provide only one endpoint.
17. When the caption implies stable qualitative semantics that are not fully numeric, record them in `semantic_cues`.
18. `semantic_cues` must use only these controlled kinds: `decline`, `increase`, `plateau`, `oscillation`, `light_noise`.
19. Do not invent custom cue names or free-form constraint objects.
20. Do not emit `unresolved` for a negated semantic that is already naturally satisfied by the chosen plan. For example, a constant `add_flat` segment already satisfies "no peaks or troughs" without any extra unresolved item.
21. Do not translate the same semantic content into multiple overlapping steps. If later text only clarifies the duration, persistence, extent, or interpretation of an already selected primitive, keep it within that existing step instead of adding a second step that restates the same meaning.
22. When the caption explicitly indicates repeated local extrema with wording such as "several peaks", "multiple troughs", "repeated spikes", or similar plural event phrases, represent that multiplicity with multiple chronological local-event primitive steps when the text provides enough temporal anchors or distinct phases to do so. Do not collapse an explicitly plural local-event phrase into a single local-event step.

Planning guidance:
- prefer overwrite primitives for the structural skeleton
- use additive primitives only for explicit deltas or refinements on an existing baseline
- treat temporal scope as part of structural planning; infer step timing/ranges here when the caption supports them
- use `add_ramp` when the caption specifies absolute levels across an interval or implies a piecewise linear backbone
- use `add_trend` only when the caption describes drift relative to an already established level, not when it specifies actual endpoint values
- use `semantic_cues` for qualitative semantics that can be justified directly from the caption; do not invent precise numeric values from vague language
- treat negated semantics like "no peaks", "no troughs", or "no oscillation" as constraints on what not to add when the existing structure already satisfies them
- if one phrase only refines or continues the meaning of an already selected primitive, do not emit another overlapping step that repeats the same semantic effect
- if the caption explicitly says there are several or multiple peaks/troughs/spikes/dips and also gives multiple rough phases or anchors, expand that into multiple local-event steps rather than one averaged event
- treat phrases like "resistance around 3500", "support near 10", or "hovering around a level" as soft level-reference constraints; represent them with existing ramp/flat/volatility composition when possible, not as unresolved library gaps
- treat soft phrases like "almost mean-reverting", "generally mean-reverting", or generic "mean reversion behavior" as high-level descriptive cues when existing ramp/flat/plateau/volatility structure already captures the realized trajectory; do not mark those as unresolved library gaps unless the caption explicitly requires a dedicated pullback-to-mean dynamic
- do not fabricate precise amplitudes, periods, or breakpoints unless the caption supports them
- keep numeric levels and timings consistent with the caption

Required JSON shape:
{{
  "version": "1.0",
  "input_description": "{caption}",
  "registry_validation": {{
    "status": "valid",
    "schema_path": "schemas/primitive.schema.json",
    "target_path": "atomic_timeseries/registry.json",
    "errors": []
  }},
  "global_context": {{
    "length": {series_length},
    "numeric_range": {{
      "min": 0.0,
      "max": 1.0
    }},
    "dt": null,
    "seed": null
  }},
  "steps": [
    {{
      "id": "step_1",
      "kind": "primitive",
      "primitive": "add_flat",
      "parameters": {{
        "start_timestep": 0,
        "end_timestep": 34
      }},
      "semantic_role": "initial flat segment",
      "semantic_layer": "segment_structure",
      "priority": 0,
      "protected_constraints": [],
      "semantic_cues": [
        {{
          "kind": "plateau",
          "source_text": "initial flat segment",
          "confidence": 0.95
        }}
      ],
      "effect_type": "overwrite",
      "confidence": 0.95,
      "source_text": "remains constant at a value of 7.51 for the first 35 timesteps",
      "needs_library_extension": false
    }}
  ],
  "unresolved_semantics": [],
  "needs_library_extension": false,
  "assumptions": [],
  "notes": ""
}}

If unresolved semantics are needed, use this shape:
{{
  "steps": [
    {{
      "id": "step_3",
      "kind": "unresolved",
      "description": "long-memory mean reversion toward a reference level",
      "semantic_role": "unsupported long-memory mean reversion",
      "confidence": 0.7,
      "source_text": "long-memory mean reversion",
      "reason": "No provided primitive explicitly models long-memory pullback toward a reference mean.",
      "needs_library_extension": true
    }}
  ],
  "unresolved_semantics": [
    {{
      "id": "unresolved_1",
      "description": "long-memory mean reversion toward a reference level",
      "source_text": "long-memory mean reversion",
      "reason": "No provided primitive explicitly models long-memory pullback toward a reference mean."
    }}
  ],
  "needs_library_extension": true
}}
