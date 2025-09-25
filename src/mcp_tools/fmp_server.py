"""Financial Modeling Prep MCP tool server via Starlette."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

load_dotenv(override=True)

logger = logging.getLogger("mcp.fmp")

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/api"


class _Cache:
    def __init__(self, ttl: int = 300):
        self._ttl = ttl
        self._store: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)


CACHE = _Cache()


def _require_key() -> None:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is not configured")


async def _fmp_request(endpoint: str, params: Dict[str, Any] | None = None) -> Any:
    _require_key()
    params = params.copy() if params else {}
    params["apikey"] = FMP_API_KEY
    cache_key = f"{endpoint}:{sorted(params.items())}"

    cached = CACHE.get(cache_key)
    if cached is not None:
        return cached

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{FMP_BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        data = response.json()
        CACHE.set(cache_key, data)
        return data


async def fmp_get_quote(request: Request) -> Response:
    payload = await request.json()
    symbol = payload["arguments"]["symbol"]

    start = asyncio.get_running_loop().time()
    data = await _fmp_request("/v3/quote/{symbol}".format(symbol=symbol))
    latency = int((asyncio.get_running_loop().time() - start) * 1000)

    if not data:
        return JSONResponse({"ok": False, "error": "No quote data"}, status_code=404)

    return JSONResponse({"ok": True, "data": data[0], "latency_ms": latency})


async def fmp_get_company_profile(request: Request) -> Response:
    payload = await request.json()
    symbol = payload["arguments"]["symbol"]

    start = asyncio.get_running_loop().time()
    data = await _fmp_request("/v3/profile/{symbol}".format(symbol=symbol))
    latency = int((asyncio.get_running_loop().time() - start) * 1000)

    if not data:
        return JSONResponse({"ok": False, "error": "No profile data"}, status_code=404)

    return JSONResponse({"ok": True, "data": data[0], "latency_ms": latency})


async def fmp_get_income_statement(request: Request) -> Response:
    payload = await request.json()
    args = payload["arguments"]
    symbol = args["symbol"]
    period = args.get("period", "annual")
    limit = int(args.get("limit", 2))

    data = await _fmp_request(
        f"/v3/income-statement/{symbol}", {"period": period, "limit": limit}
    )
    return JSONResponse({"ok": True, "data": data})


async def fmp_get_market_gainers(_: Request) -> Response:
    data = await _fmp_request("/v3/stock_market/gainers")
    return JSONResponse({"ok": True, "data": data})


async def fmp_get_market_losers(_: Request) -> Response:
    data = await _fmp_request("/v3/stock_market/losers")
    return JSONResponse({"ok": True, "data": data})


async def fmp_search_ticker(request: Request) -> Response:
    payload = await request.json()
    args = payload["arguments"]
    query = args["query"]
    limit = int(args.get("limit", 10))

    data = await _fmp_request("/v3/search", {"query": query, "limit": limit})
    return JSONResponse({"ok": True, "data": data})


async def fmp_get_stock_news(request: Request) -> Response:
    payload = await request.json()
    args = payload["arguments"]
    tickers = args.get("tickers")
    limit = int(args.get("limit", 10))

    params: Dict[str, Any] = {"limit": limit}
    if tickers:
        params["tickers"] = tickers

    data = await _fmp_request("/v3/stock_news", params)
    return JSONResponse({"ok": True, "data": data})


async def fmp_get_change_percent(request: Request) -> Response:
    payload = await request.json()
    symbol = payload["arguments"]["symbol"]
    data = await _fmp_request("/v3/quote/{symbol}".format(symbol=symbol))
    if not data:
        return JSONResponse({"ok": False, "error": "No data"}, status_code=404)
    quote = data[0]
    change_percent = quote.get("changesPercentage")
    return JSONResponse({"ok": True, "data": {"changesPercentage": change_percent}})


async def fmp_get_previous_close(request: Request) -> Response:
    payload = await request.json()
    symbol = payload["arguments"]["symbol"]
    data = await _fmp_request("/v3/quote-short/{symbol}".format(symbol=symbol))
    if not data:
        return JSONResponse({"ok": False, "error": "No data"}, status_code=404)
    return JSONResponse({"ok": True, "data": data[0]})


async def readiness(_: Request) -> Response:
    try:
        _require_key()
        return JSONResponse({"ok": True})
    except Exception as exc:  # pragma: no cover
        logger.exception("FMP readiness failure")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


routes = [
    Route("/health", readiness, methods=["GET"]),
    Route("/tools/fmp_get_quote", fmp_get_quote, methods=["POST"]),
    Route("/tools/fmp_get_company_profile", fmp_get_company_profile, methods=["POST"]),
    Route("/tools/fmp_get_income_statement", fmp_get_income_statement, methods=["POST"]),
    Route("/tools/fmp_get_market_gainers", fmp_get_market_gainers, methods=["POST"]),
    Route("/tools/fmp_get_market_losers", fmp_get_market_losers, methods=["POST"]),
    Route("/tools/fmp_search_ticker", fmp_search_ticker, methods=["POST"]),
    Route("/tools/fmp_get_stock_news", fmp_get_stock_news, methods=["POST"]),
    Route("/tools/fmp_get_change_percent", fmp_get_change_percent, methods=["POST"]),
    Route("/tools/fmp_get_previous_close", fmp_get_previous_close, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)

__all__ = ["app"]
