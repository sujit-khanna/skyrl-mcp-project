"""Safe Python execution MCP server exposed over HTTP."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import math
import statistics
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

load_dotenv(override=True)

logger = logging.getLogger("mcp.python")

# Optional matplotlib support
try:  # pragma: no cover - optional dependency branch
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    matplotlib = None
    plt = None

SAFE_GLOBALS: Dict[str, Any] = {
    "__builtins__": {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "print": print,
        "__import__": __import__,
        "__name__": "__main__",
        "__doc__": None,
    },
    "pd": pd,
    "np": np,
    "json": json,
    "datetime": datetime,
    "timedelta": timedelta,
    "io": io,
    "base64": base64,
    "math": math,
    "statistics": statistics,
    "plt": plt,
    "matplotlib": matplotlib,
}

EXECUTION_CONTEXT: Dict[str, Any] = {}


def validate_code_safety(code: str) -> tuple[bool, str]:
    dangerous = [
        "exec",
        "eval",
        "subprocess",
        "socket",
        "open(",
        "__import__('os')",
        "http",
    ]
    lowered = code.lower()
    for token in dangerous:
        if token in lowered:
            return False, f"Disallowed token detected: {token}"
    return True, ""


def serialize_result(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "value": value.to_dict(orient="records"),
            "columns": list(value.columns),
        }
    if isinstance(value, (pd.Series, np.ndarray)):
        return {
            "type": "array",
            "value": value.tolist() if hasattr(value, "tolist") else list(value),
        }
    if isinstance(value, (list, dict, str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def safe_execute(code: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    context = context or {}
    ok, message = validate_code_safety(code)
    if not ok:
        return {
            "ok": False,
            "output": "",
            "result": None,
            "error": message,
        }

    env = dict(SAFE_GLOBALS)
    env.update(context)

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            try:
                result = eval(code, env)  # noqa: S307 - restricted env
            except SyntaxError:
                exec(code, env)  # noqa: S102 - restricted env
                result = None
    except Exception as exc:  # pragma: no cover - runtime errors vary
        logger.exception("Python execution failure")
        return {
            "ok": False,
            "output": stdout_capture.getvalue(),
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    if result is None:
        for candidate in ("result", "output", "data", "df"):
            if candidate in env and candidate not in SAFE_GLOBALS:
                result = env[candidate]
                break

    return {
        "ok": True,
        "output": stdout_capture.getvalue(),
        "result": serialize_result(result),
    }


async def execute_python(request: Request) -> Response:
    payload = await request.json()
    args = payload.get("arguments", {})
    code = args["code"]

    start = asyncio.get_running_loop().time()
    result = await asyncio.to_thread(safe_execute, code)
    latency = int((asyncio.get_running_loop().time() - start) * 1000)
    result["latency_ms"] = latency
    return JSONResponse(result)


async def process_mcp_data(request: Request) -> Response:
    payload = await request.json()
    args = payload.get("arguments", {})
    data_json = args["data"]
    processing_code = args["processing_code"]

    try:
        data = json.loads(data_json)
    except json.JSONDecodeError as exc:
        return JSONResponse({"ok": False, "error": f"Invalid JSON: {exc}"}, status_code=400)

    context = {"mcp_data": data, "data": data}
    start = asyncio.get_running_loop().time()
    result = await asyncio.to_thread(safe_execute, processing_code, context)
    latency = int((asyncio.get_running_loop().time() - start) * 1000)
    result["latency_ms"] = latency
    return JSONResponse(result)


async def store_execution_result(request: Request) -> Response:
    payload = await request.json()
    args = payload.get("arguments", {})
    key = args["context_key"]
    data_json = args["data"]

    try:
        data = json.loads(data_json)
    except json.JSONDecodeError as exc:
        return JSONResponse({"ok": False, "error": f"Invalid JSON: {exc}"}, status_code=400)

    EXECUTION_CONTEXT[key] = data
    return JSONResponse({"ok": True, "message": f"Stored context '{key}'"})


async def get_execution_context(request: Request) -> Response:
    payload = await request.json()
    args = payload.get("arguments", {})
    key = args.get("context_key")

    if not key:
        return JSONResponse({"ok": True, "data": list(EXECUTION_CONTEXT.keys())})
    if key not in EXECUTION_CONTEXT:
        return JSONResponse({"ok": False, "error": "Key not found"}, status_code=404)
    return JSONResponse({"ok": True, "data": EXECUTION_CONTEXT[key]})


async def readiness(_: Request) -> Response:
    return JSONResponse({"ok": True})


routes = [
    Route("/health", readiness, methods=["GET"]),
    Route("/tools/execute_python", execute_python, methods=["POST"]),
    Route("/tools/process_mcp_data", process_mcp_data, methods=["POST"]),
    Route("/tools/store_execution_result", store_execution_result, methods=["POST"]),
    Route("/tools/get_execution_context", get_execution_context, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)

__all__ = ["app", "EXECUTION_CONTEXT"]
