# Controlled Gold Cases

This directory contains the controlled benchmark used in the paper.

- `single_claim_cases.json`: cases with one measurable temporal claim.
- `multi_claim_cases.json`: cases combining multiple ordered claims.
- `robustness_cases.json`: negation, near-miss wording, and unsupported-request cases.

Construction scripts:

- `scripts/construct_controlled_gold_cases.py`: builds the seed controlled cases.
- `scripts/expand_controlled_gold_cases.py`: expands the seed cases into the balanced 1000-case benchmark.

The expansion script also writes `construction_report.json` as a local generated
summary; it is ignored by Git.
