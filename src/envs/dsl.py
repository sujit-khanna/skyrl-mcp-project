"""Shared DSL helpers for SkyRL MCP environment.

These utilities mirror the dataset generation semantics for analysis requirements:
- extraction paths (aliases, dotted paths, list projections, map projections)
- compute/select expressions executed in a constrained namespace
- boolean acceptance checks and placeholder resolution

The goal is to keep dataset -> environment behaviour aligned so rollouts are
rewarded identically to the synthetic data execution traces.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractResult:
    key: str
    value: Any
    success: bool
    directive: str


class DSLExecutionError(RuntimeError):
    """Raised when a compute/select expression fails."""


_SAFE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _navigate_path(obj: Any, parts: List[str]) -> Any:
    for part in parts:
        if not isinstance(obj, Mapping):
            return None
        obj = obj.get(part)
        if obj is None:
            return None
    return obj


def _default_extract_key(path: str) -> str:
    base = path.split("{")[0]
    base = base.replace("[]", "")
    if "." in base:
        base = base.split(".")[-1]
    return base.strip()


def extract_path(result: Mapping[str, Any], path: str) -> Tuple[Any, bool]:
    """Replicates dataset extraction semantics.

    Returns (value, success flag).
    """
    try:
        if "[" not in path and "{" not in path and "." not in path:
            return result.get(path), (path in result)

        if "." in path and "[" not in path and "{" not in path:
            parts = path.split(".")
            val = _navigate_path(result, parts)
            return val, (val is not None)

        if path.endswith("[]"):
            key_path = path[:-2]
            if "." in key_path:
                parts = key_path.split(".")
                val = _navigate_path(result, parts)
                return val if isinstance(val, list) else [], (val is not None)
            return result.get(key_path, []), True

        if "[][" in path:
            base, field = path.split("[][", 1)
            field = field.rstrip("]")
            if "." in base:
                parts = base.split(".")
                items = _navigate_path(result, parts)
            else:
                items = result.get(base, [])
            if not isinstance(items, list):
                return [], True
            return [it.get(field) if isinstance(it, Mapping) else None for it in items], True

        if "{" in path and "->" in path:
            base = path.split("{")[0]
            mapping = path.split("{")[1].split("}")[0]
            key_field, val_field = mapping.split("->")
            if "." in base:
                parts = base.split(".")
                items = _navigate_path(result, parts)
            else:
                items = result.get(base, [])
            if not isinstance(items, list):
                return {}, True
            out: Dict[str, Any] = {}
            for item in items:
                if isinstance(item, Mapping) and key_field in item:
                    out[item[key_field]] = item.get(val_field)
            return out, True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("extract_path failed for %s: %s", path, exc)
        return None, False

    return None, False


def parse_extract_directive(result: Mapping[str, Any], directive: str) -> ExtractResult:
    if " = " in directive:
        alias, raw_path = [s.strip() for s in directive.split("=", 1)]
        key = alias
    else:
        raw_path = directive.strip()
        key = _default_extract_key(raw_path)
    value, success = extract_path(result, raw_path)
    return ExtractResult(key=key, value=value, success=success, directive=directive)


def _safe_namespace(state: Mapping[str, Any]) -> Dict[str, Any]:
    def pct_change_last_day(price_json):
        pct: Dict[str, float] = {}
        if not isinstance(price_json, Mapping):
            return pct
        for ticker, arr in price_json.items():
            if not isinstance(arr, list) or len(arr) < 2:
                continue
            prev, curr = arr[-2], arr[-1]
            try:
                prev_c = float(prev.get("close"))
                curr_c = float(curr.get("close"))
            except (AttributeError, TypeError, ValueError):
                continue
            if prev_c:
                pct[ticker] = (curr_c / prev_c) - 1.0
        return pct

    def topk(d: Mapping[str, Any], k: int):
        if not isinstance(d, Mapping):
            return []
        return [k_ for k_, _ in sorted(d.items(), key=lambda item: item[1], reverse=True)[:k]]

    def head(lst: List[Any], n: int):
        return lst[:n] if isinstance(lst, list) else []

    def unique(lst: List[Any]):
        if not isinstance(lst, list):
            return []
        seen = {}
        for item in lst:
            seen.setdefault(item, len(seen))
        return list(seen.keys())

    def concat(*iterables):
        out: List[Any] = []
        for it in iterables:
            if isinstance(it, list):
                out.extend(it)
        return out

    def count_keys(d: Mapping[str, Any]):
        return len(d) if isinstance(d, Mapping) else 0

    def regex_extract_all(pattern: str, text: str):
        try:
            return re.findall(pattern, text or "")
        except re.error as exc:
            raise DSLExecutionError(f"Invalid regex pattern: {pattern}") from exc

    safe_builtins = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "max": max,
        "min": min,
        "sum": sum,
        "sorted": sorted,
    }

    ns = dict(state)
    ns.update(
        {
            "pct_change_last_day": pct_change_last_day,
            "topk": topk,
            "head": head,
            "unique": unique,
            "concat": concat,
            "count_keys": count_keys,
            "regex_extract_all": regex_extract_all,
        }
    )
    ns["__builtins__"] = safe_builtins
    return ns


def compute_expression(expr: str, state: Mapping[str, Any]) -> Dict[str, Any]:
    if "=" not in expr:
        raise DSLExecutionError(f"Compute expression missing '=': {expr}")
    name, rhs = [s.strip() for s in expr.split("=", 1)]
    namespace = _safe_namespace(state)
    try:
        value = eval(rhs, {"__builtins__": namespace["__builtins__"]}, namespace)
    except Exception as exc:  # pragma: no cover - expression errors handled by caller
        raise DSLExecutionError(str(exc)) from exc
    return {name: value}


def check_condition(cond: str, state: Mapping[str, Any]) -> bool:
    namespace = _safe_namespace(state)
    try:
        if " ~= " in cond:
            lhs, pattern = [s.strip() for s in cond.split("~=", 1)]
            pattern = pattern.strip("'\"")
            value = eval(lhs, {"__builtins__": namespace["__builtins__"]}, namespace)
            return re.search(pattern, str(value)) is not None
        return bool(eval(cond, {"__builtins__": namespace["__builtins__"]}, namespace))
    except Exception as exc:
        logger.warning("Condition '%s' evaluation failed: %s", cond, exc)
        return False


def resolve_placeholders(obj: Any, state: Mapping[str, Any]) -> Any:
    if isinstance(obj, str):
        def repl(match: re.Match[str]) -> str:
            expr = match.group(1)
            namespace = _safe_namespace(state)
            try:
                value = eval(expr, {"__builtins__": namespace["__builtins__"]}, namespace)
            except Exception as exc:
                logger.warning("Failed to resolve placeholder %s: %s", expr, exc)
                return match.group(0)
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return str(value)

        return _PLACEHOLDER_PATTERN.sub(repl, obj)

    if isinstance(obj, list):
        return [resolve_placeholders(item, state) for item in obj]

    if isinstance(obj, Mapping):
        return {k: resolve_placeholders(v, state) for k, v in obj.items()}

    return obj


def extract_values(result: Mapping[str, Any], directives: Iterable[str]) -> Tuple[Dict[str, Any], List[str]]:
    updates: Dict[str, Any] = {}
    missing: List[str] = []
    for directive in directives or []:
        parsed = parse_extract_directive(result, directive)
        if parsed.success and parsed.value is not None:
            updates[parsed.key] = parsed.value
        else:
            missing.append(parsed.directive)
    return updates, missing

