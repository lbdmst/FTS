# FaithTS

FaithTS is a registry-grounded pipeline for translating natural-language
time-series descriptions into executable time-series programs. It constrains
generation to a fixed library of atomic primitives, validates structured plans
against JSON schemas, executes the resulting signal, and evaluates whether the
generated series satisfies the stated semantic claims.

The repository includes the public controlled benchmark cases and the
TSFragment-Eval split used in the paper. Manuscript files and external baseline
checkouts are intentionally excluded from version control.

## Repository Layout

- [`atomic_timeseries/`](atomic_timeseries): immutable numpy primitives such as
  `add_flat`, `add_ramp`, `add_spike`, and `add_seasonality`.
- [`faithts/`](faithts): semantic evaluation, rejection, repair, and failure
  taxonomy utilities.
- [`faithts_pipeline/`](faithts_pipeline): retrieval-aware planning,
  validation, compilation, execution, and reporting.
- [`data/controlled_gold_cases/`](data/controlled_gold_cases): controlled
  single-claim, multi-claim, and robustness benchmark cases.
- [`data/tsfragment_eval/`](data/tsfragment_eval): the 2500-case
  TSFragment-Eval split.
- [`scripts/`](scripts): data construction and split-generation scripts.
- [`evals/`](evals): benchmark runners, baseline adapters, result rendering, and
  analysis utilities.
- [`schemas/`](schemas): JSON schemas for primitive registries, plans, and
  evaluation reports.
- [`tests/`](tests): regression tests for primitives, schemas, evaluation, and
  pipeline behavior.

## Environment

Minimal editable install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install jsonschema pandas matplotlib scikit-learn
```

For the fuller research environment, use the conda file:

```bash
conda env create -f environment.yml
conda activate bidts
pip install -e .
```

External LLM runs use Gemini through `google-genai`:

```bash
pip install google-genai
export GOOGLE_API_KEY=...
```

Offline tests and saved-report inspection do not require an API key.

## Data

The checked-in public data lives under [`data/`](data):

- `data/controlled_gold_cases/single_claim_cases.json`
- `data/controlled_gold_cases/multi_claim_cases.json`
- `data/controlled_gold_cases/robustness_cases.json`
- `data/tsfragment_eval/tsfragment_eval_2500_cases.json`

The controlled cases are constructed deterministically:

```bash
python scripts/construct_controlled_gold_cases.py
python scripts/expand_controlled_gold_cases.py
```

`expand_controlled_gold_cases.py` also writes a local construction summary to
`data/controlled_gold_cases/construction_report.json`; that file is generated
and ignored.

The TSFragment-Eval split is already checked in. Reconstructing auxiliary random
audit splits requires the external TSFragment/VerbalTS records that are not
included in this public repository:

```bash
python scripts/construct_tsfragment_eval_split.py \
  --input-root evals/reports/t2s_native_case_inputs \
  --record-root baselines/VerbalTS/datasets
```

## Validation

Validate the primitive registry:

```bash
python schema_validator.py primitive atomic_timeseries/registry.json
```

Validate a generated plan or report:

```bash
python schema_validator.py plan path/to/plan.json
python schema_validator.py eval_report path/to/eval_report.json
```

Regenerate the primitive retrieval catalog and local retrieval audit artifacts:

```bash
python coverage_audit.py
```

This writes `primitive_catalog.json`, `retrieval_coverage_report.jsonl`, and
`retrieval_summary.md`. The latter two are ignored because they are local
generated artifacts.

## Running FaithTS

Run the retrieval-aware workflow on selected TSFragment-Eval examples:

```bash
python faithts_pipeline/workflow.py \
  --provider gemini \
  --indices 1 3 \
  --series-length 128 \
  --output faithts_pipeline/workflow_results.jsonl \
  --eval-report-output faithts_pipeline/workflow_eval_reports.jsonl
```

Run controlled-case checks:

```bash
python evals/run_controlled_gold_cases.py
```

Run the offline research pipeline from saved artifacts:

```bash
python evals/run_research_experiments.py --report-dir evals/reports
```

Generated evaluation outputs are written under `evals/reports/`, which is
ignored by Git. After generating the prerequisite reports, render paper-style
summary tables with:

```bash
python evals/render_research_tables.py \
  --baseline-json evals/reports/baseline_results.json \
  --ablation-json evals/reports/ablation_results.json \
  --t2s-json evals/reports/t2s_main_benchmark_comparison_2500.json \
  --markdown-output evals/reports/research_tables.md
```

## Tests

Run the main test suite:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Useful focused targets:

```bash
python -m unittest tests.test_atomic_timeseries
python -m unittest tests.test_schema_validator tests.test_primitive_catalog
python -m unittest tests.test_faithts_pipeline
python -m unittest tests.test_controlled_gold_cases
```

## Core Contract

FaithTS generation is constrained by three public contracts:

- Only registered primitives from [`atomic_timeseries/registry.json`](atomic_timeseries/registry.json)
  may be used in executable plans.
- Plans must validate against [`schemas/plan.schema.json`](schemas/plan.schema.json).
- Evaluation reports must validate against
  [`schemas/eval_report.schema.json`](schemas/eval_report.schema.json).

If a description cannot be represented by the registered primitives or their
declared compositions, the plan should mark the item as unresolved and set
`needs_library_extension: true` instead of inventing a primitive.
