from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parent
SCHEMA_DIR = REPO_ROOT / "schemas"
SCHEMA_PATHS = {
    "primitive": SCHEMA_DIR / "primitive.schema.json",
    "registry": SCHEMA_DIR / "primitive.schema.json",
    "plan": SCHEMA_DIR / "plan.schema.json",
    "eval": SCHEMA_DIR / "eval_report.schema.json",
    "eval_report": SCHEMA_DIR / "eval_report.schema.json",
}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def load_schema(kind: str) -> tuple[str, Path, dict[str, Any]]:
    try:
        schema_path = SCHEMA_PATHS[kind]
    except KeyError as exc:
        supported = ", ".join(sorted(SCHEMA_PATHS))
        raise ValueError(f"Unsupported schema kind: {kind}. Expected one of: {supported}") from exc
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    return kind, schema_path, schema


def validate_payload(kind: str, instance: Any, target_path: str = "<memory>") -> dict[str, Any]:
    normalized_kind, schema_path, schema = load_schema(kind)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda err: list(err.absolute_path))
    return {
        "status": "valid" if not errors else "invalid",
        "schema_kind": normalized_kind,
        "schema_path": str(schema_path.relative_to(REPO_ROOT)),
        "target_path": str(target_path),
        "errors": [
            f"{'/'.join(str(part) for part in err.absolute_path) or '$'}: {err.message}"
            for err in errors
        ],
    }


def validate_instance(kind: str, instance_path: str | Path) -> dict[str, Any]:
    target_path = Path(instance_path)
    instance = load_json(target_path)
    return validate_payload(kind, instance, target_path=str(target_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a JSON artifact against a pipeline schema.")
    parser.add_argument(
        "kind",
        choices=sorted(SCHEMA_PATHS),
        help="Artifact kind to validate: primitive/registry, plan, or eval/eval_report.",
    )
    parser.add_argument("path", help="Path to the JSON artifact.")
    args = parser.parse_args()

    report = validate_instance(args.kind, args.path)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
