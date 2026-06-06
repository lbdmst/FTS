from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from faithts_pipeline.workflow import (
    GeneratedCodeExecutionError,
    UnsafeGeneratedCodeError,
    execute_generated_code,
    strip_code_fences,
)


DIRECT_ARRAY_PROMPT = """You are given a natural-language time-series description.
Generate the described time series and return only its values as a JSON array.
The array must contain exactly {length} numeric values.
Do not return any explanation, code, or markdown.

Description:
{description}
"""


DIRECT_CODE_PROMPT = """Write Python code that generates the described time series.
Return only code. The time series length is {length}. Store the result in a
top-level variable named `series`.

Description:
{description}
"""


STRONG_PROMPT_CODE_PROMPT = """Write a safe, reproducible Python program that generates a numpy array named series.
Hard requirements:
- import numpy as np
- series must be one-dimensional and length {length}
- use deterministic random seeds for any stochastic component
- do not hardcode the final array directly
- keep exact timestep windows and local events from the description
- return only Python code

Description:
{description}
"""


def build_prompt(case: dict[str, Any], mode: str) -> str:
    templates = {
        "llm_direct_array": DIRECT_ARRAY_PROMPT,
        "llm_direct_code": DIRECT_CODE_PROMPT,
        "llm_strong_prompt": STRONG_PROMPT_CODE_PROMPT,
        "llm_strong_prompt_code": STRONG_PROMPT_CODE_PROMPT,
    }
    if mode not in templates:
        raise ValueError(f"Unsupported LLM baseline mode: {mode}")
    return templates[mode].format(length=int(case["length"]), description=case["description"])


def invalid_direct_output_plan(case: dict[str, Any], mode: str, errors: list[str] | None = None) -> dict[str, Any]:
    notes = f"{mode} baseline output is evaluated as a direct signal, not as a registry-grounded primitive plan."
    if errors:
        notes += " Errors: " + "; ".join(errors)
    return {
        "version": "1.0",
        "input_description": case["description"],
        "registry_validation": {
            "status": "not_run",
            "schema_path": "schemas/primitive.schema.json",
            "target_path": "atomic_timeseries/registry.json",
            "errors": [],
        },
        "global_context": {
            "length": int(case["length"]),
            "numeric_range": case.get("numeric_range") or {"min": -1000.0, "max": 1000.0},
            "dt": None,
            "seed": None,
        },
        "steps": [],
        "unresolved_semantics": [],
        "needs_library_extension": False,
        "notes": notes,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = strip_code_fences(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM output JSON must be an object.")
    return payload


def parse_direct_array_output(text: str, length: int) -> np.ndarray:
    try:
        payload = extract_json_object(text)
        values = payload.get("series")
    except Exception:
        match = re.search(r"\[[\s\S]*\]", text)
        if match is None:
            raise
        values = json.loads(match.group(0))

    if not isinstance(values, list):
        raise ValueError("Direct array output must contain a list-valued `series`.")
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("Direct array output must be one-dimensional.")
    if len(series) != length:
        raise ValueError(f"Direct array output length {len(series)} does not match expected length {length}.")
    return series


_PERMISSIVE_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "print": lambda *args, **kwargs: None,
    "range": range,
}
_PERMISSIVE_FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "getattr",
    "delattr",
    "vars",
}
_PERMISSIVE_DISALLOWED_AST_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_PERMISSIVE_NUMPY_CALLS = {
    "array",
    "arange",
    "clip",
    "concatenate",
    "convolve",
    "cos",
    "exp",
    "full",
    "linspace",
    "max",
    "maximum",
    "mean",
    "min",
    "ones",
    "power",
    "sin",
    "std",
    "zeros",
}
_PERMISSIVE_NUMPY_RANDOM_CALLS = {
    "choice",
    "default_rng",
    "normal",
    "rand",
    "randn",
    "seed",
    "uniform",
}
_PERMISSIVE_NUMPY_ATTRIBUTES = {"nan", "pi"}


def _permissive_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: tuple[Any, ...] = (), level: int = 0) -> Any:
    del globals_, locals_
    if level != 0 or name != "numpy":
        raise UnsafeGeneratedCodeError(f"Only numpy imports are allowed, got: {name}", category="illegal_import")
    return __import__(name, {}, {}, fromlist, level)


_PERMISSIVE_SAFE_BUILTINS["__import__"] = _permissive_import


def _validate_permissive_direct_code(tree: ast.AST) -> set[str]:
    user_functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, _PERMISSIVE_DISALLOWED_AST_NODES):
            raise UnsafeGeneratedCodeError(
                f"Unsupported Python construct in generated code: {type(node).__name__}",
                category="contract_invalid",
            )
        if isinstance(node, ast.Name) and node.id in _PERMISSIVE_FORBIDDEN_NAMES:
            raise UnsafeGeneratedCodeError(
                f"Forbidden name used in generated code: {node.id}",
                category="contract_invalid",
            )
        if isinstance(node, ast.Import):
            if len(node.names) != 1 or node.names[0].name != "numpy" or node.names[0].asname != "np":
                raise UnsafeGeneratedCodeError("Only `import numpy as np` is allowed.", category="illegal_import")
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise UnsafeGeneratedCodeError("Decorators are not allowed in generated code.", category="contract_invalid")
            user_functions.add(node.name)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id == "np" and node.attr not in (_PERMISSIVE_NUMPY_CALLS | _PERMISSIVE_NUMPY_ATTRIBUTES | {"random"}):
                    raise UnsafeGeneratedCodeError(
                        f"Disallowed numpy API usage: np.{node.attr}",
                        category="unsupported_numpy_api",
                    )
            elif isinstance(node.value, ast.Attribute):
                base = node.value
                if isinstance(base.value, ast.Name) and base.value.id == "np" and base.attr == "random":
                    if node.attr not in _PERMISSIVE_NUMPY_RANDOM_CALLS:
                        raise UnsafeGeneratedCodeError(
                            f"Disallowed numpy random API usage: np.random.{node.attr}",
                            category="unsupported_numpy_api",
                        )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _PERMISSIVE_SAFE_BUILTINS and func.id not in user_functions:
                    raise UnsafeGeneratedCodeError(
                        f"Disallowed function call: {func.id}",
                        category="contract_invalid",
                    )
            elif isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "np":
                    if func.attr not in _PERMISSIVE_NUMPY_CALLS:
                        raise UnsafeGeneratedCodeError(
                            f"Disallowed numpy call: np.{func.attr}",
                            category="unsupported_numpy_api",
                        )
                elif (
                    isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "np"
                    and func.value.attr == "random"
                ):
                    if func.attr not in _PERMISSIVE_NUMPY_RANDOM_CALLS:
                        raise UnsafeGeneratedCodeError(
                            f"Disallowed numpy random call: np.random.{func.attr}",
                            category="unsupported_numpy_api",
                        )
                elif func.attr in _PERMISSIVE_NUMPY_RANDOM_CALLS:
                    continue
                else:
                    raise UnsafeGeneratedCodeError(
                        "Only whitelisted `np.*` calls are allowed.",
                        category="unsupported_numpy_api",
                    )
    return user_functions


def _execute_permissive_direct_code(text: str) -> np.ndarray:
    stripped = strip_code_fences(text)
    try:
        tree = ast.parse(stripped, mode="exec")
    except SyntaxError as exc:
        raise UnsafeGeneratedCodeError(
            f"Generated code is not valid Python: {exc}",
            category="contract_invalid",
        ) from exc

    user_functions = _validate_permissive_direct_code(tree)
    exec_globals = {
        "__builtins__": _PERMISSIVE_SAFE_BUILTINS,
        "np": np,
        "numpy": np,
    }
    exec_locals: dict[str, Any] = {}
    original_default_rng = np.random.default_rng

    def deterministic_default_rng(seed: Any = None) -> Any:
        return original_default_rng(0 if seed is None else seed)

    try:
        np.random.seed(0)
        np.random.default_rng = deterministic_default_rng
        exec(compile(tree, filename="<direct_llm_generated>", mode="exec"), exec_globals, exec_locals)
        if "series" not in exec_locals and "series" not in exec_globals:
            for fn_name in ("generate_series", "generate_time_series", "generate_stable_series"):
                fn = exec_locals.get(fn_name, exec_globals.get(fn_name))
                if callable(fn):
                    exec_locals["series"] = fn()
                    break
    except UnsafeGeneratedCodeError:
        raise
    except Exception as exc:
        raise GeneratedCodeExecutionError(
            f"Generated code failed during execution: {exc}",
            category="execution_failed",
        ) from exc
    finally:
        np.random.default_rng = original_default_rng

    series = exec_locals.get("series", exec_globals.get("series"))
    if series is None:
        available = ", ".join(sorted(user_functions)) if user_functions else "none"
        raise GeneratedCodeExecutionError(
            f"Generated code did not assign a `series` variable. Helper functions seen: {available}",
            category="execution_failed",
        )
    array = np.asarray(series, dtype=float)
    if array.ndim != 1:
        raise GeneratedCodeExecutionError("Generated `series` must be one-dimensional.", category="execution_failed")
    return array


def parse_direct_code_output(text: str, length: int, *, allowed_primitives: list[str] | None = None) -> np.ndarray:
    allowed = allowed_primitives or [
        "add_change_point",
        "add_decay",
        "add_dip",
        "add_flat",
        "add_gap",
        "add_growth",
        "add_level_shift",
        "add_noise",
        "add_outlier",
        "add_peak",
        "add_plateau",
        "add_ramp",
        "add_seasonality",
        "add_spike",
        "add_trend",
        "add_trough",
        "add_volatility",
    ]
    try:
        series = execute_generated_code(text, allowed)
    except (UnsafeGeneratedCodeError, GeneratedCodeExecutionError):
        try:
            series = _execute_permissive_direct_code(text)
        except (UnsafeGeneratedCodeError, GeneratedCodeExecutionError) as exc:
            raise ValueError(str(exc)) from exc
    if len(series) != length:
        raise ValueError(f"Direct code output length {len(series)} does not match expected length {length}.")
    return np.asarray(series, dtype=float)


def load_llm_output(output_dir: Path, case_id: str) -> str:
    path = output_dir / f"{case_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM output file: {path}")
    return path.read_text(encoding="utf-8")
