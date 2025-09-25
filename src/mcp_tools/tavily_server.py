"""Tavily MCP tool server providing web search over the Tavily API."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from tavily import TavilyClient

load_dotenv(override=True)

logger = logging.getLogger("mcp.tavily")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

_client: TavilyClient | None = None


def get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not configured")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


async def tavily_search(request: Request) -> Response:
    payload = await request.json()
    args: Dict[str, Any] = payload.get("arguments", {})

    query = args["query"]
    search_depth = args.get("search_depth", "basic")
    max_results = int(args.get("max_results", 5))
    include_images = bool(args.get("include_images", False))

    client = get_client()
    start = asyncio.get_running_loop().time()
    try:
        result = await asyncio.to_thread(
            client.search,
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_images=include_images,
        )
        ok = True
        data: Dict[str, Any] | None = result
        error: str | None = None
    except Exception as exc:  # pragma: no cover - depends on Tavily plan
        logger.exception("Tavily search failed")
        ok = False
        data = None
        error = str(exc)

    latency = int((asyncio.get_running_loop().time() - start) * 1000)
    payload: Dict[str, Any] = {"ok": ok, "latency_ms": latency}
    if data is not None:
        payload["data"] = data
    if error is not None:
        payload["error"] = error
    return JSONResponse(payload, status_code=200 if ok else 502)


async def tavily_extract(request: Request) -> Response:
    payload = await request.json()
    args: Dict[str, Any] = payload.get("arguments", {})
    url = args["url"]

    client = get_client()
    start = asyncio.get_running_loop().time()
    try:
        extract_result = await asyncio.to_thread(client.extract, [url])
        if isinstance(extract_result, dict) and "results" in extract_result:
            data = extract_result["results"][0] if extract_result["results"] else extract_result
        elif isinstance(extract_result, list):
            data = extract_result[0] if extract_result else {}
        else:
            data = extract_result
        ok = True
        error: str | None = None
    except TypeError:
        try:
            data = await asyncio.to_thread(client.extract, url)
            ok = True
            error = None
        except Exception as inner_exc:  # pragma: no cover
            logger.exception("Tavily extract failed with legacy fallback")
            ok = False
            data = None
            error = str(inner_exc)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tavily extract failed")
        ok = False
        data = None
        error = str(exc)
    latency = int((asyncio.get_running_loop().time() - start) * 1000)
    payload: Dict[str, Any] = {"ok": ok, "latency_ms": latency}
    if data is not None:
        payload["data"] = data
    if error is not None:
        payload["error"] = error
    return JSONResponse(payload, status_code=200 if ok else 502)


async def readiness(_: Request) -> Response:
    try:
        get_client()
        return JSONResponse({"ok": True})
    except Exception as exc:  # pragma: no cover
        logger.exception("Tavily readiness failure")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


routes = [
    Route("/health", readiness, methods=["GET"]),
    Route("/tools/tavily_search", tavily_search, methods=["POST"]),
    Route("/tools/tavily_extract", tavily_extract, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)

__all__ = ["app"]
