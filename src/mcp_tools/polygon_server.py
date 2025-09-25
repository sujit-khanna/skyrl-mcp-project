"""Polygon MCP tool server implemented as a Starlette application."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from polygon import RESTClient as PolygonClient
from polygon.rest.models import Agg
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

# Ensure environment variables are loaded when running standalone
load_dotenv(override=True)

logger = logging.getLogger("mcp.polygon")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

_client: Optional[PolygonClient] = None


def get_client() -> PolygonClient:
    global _client
    if _client is None:
        if not POLYGON_API_KEY:
            raise RuntimeError("POLYGON_API_KEY is not configured")
        _client = PolygonClient(api_key=POLYGON_API_KEY)
    return _client


def _fetch_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    frequency: str = "daily",
) -> pd.DataFrame:
    client = get_client()

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if frequency.lower() == "daily":
        timespan = "day"
        multiplier = 1
    elif frequency.lower() == "hourly":
        timespan = "hour"
        multiplier = 1
    elif frequency.lower() == "minute":
        timespan = "minute"
        multiplier = 15
    else:
        raise ValueError("frequency must be one of daily, hourly, minute")

    aggs: List[Agg] = []
    for agg in client.list_aggs(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=start_dt.strftime("%Y-%m-%d"),
        to=end_dt.strftime("%Y-%m-%d"),
        limit=5_000,
    ):
        if isinstance(agg, Agg):
            aggs.append(agg)

    df = pd.DataFrame(
        {
            "timestamp": [datetime.fromtimestamp(a.timestamp / 1000) for a in aggs],
            "open": [a.open for a in aggs],
            "high": [a.high for a in aggs],
            "low": [a.low for a in aggs],
            "close": [a.close for a in aggs],
            "volume": [a.volume for a in aggs],
        }
    )

    if df.empty:
        return df

    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

    if frequency.lower() == "hourly":
        df = df.between_time("09:30", "16:00")

    return df


def _fetch_news(ticker: str, limit: int = 10) -> Dict[str, Any]:
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is not configured")

    params = {"ticker": ticker, "limit": limit, "apiKey": POLYGON_API_KEY}
    resp = requests.get(
        "https://api.polygon.io/v2/reference/news", params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_ticker_details(ticker: str) -> Dict[str, Any]:
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is not configured")

    resp = requests.get(
        f"https://api.polygon.io/v3/reference/tickers/{ticker}",
        params={"apikey": POLYGON_API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_market_status() -> Dict[str, Any]:
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is not configured")

    resp = requests.get(
        "https://api.polygon.io/v1/marketstatus/now",
        params={"apikey": POLYGON_API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_previous_close(ticker: str) -> Dict[str, Any]:
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is not configured")

    resp = requests.get(
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev",
        params={"apikey": POLYGON_API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


async def polygon_get_aggs(request: Request) -> Response:
    payload = await request.json()
    args = payload.get("arguments", {})
    ticker = args["ticker"]
    start_date = args["start_date"]
    end_date = args["end_date"]
    frequency = args.get("frequency", "daily")

    start = asyncio.get_running_loop().time()
    df = await asyncio.to_thread(
        _fetch_price_data, ticker, start_date, end_date, frequency
    )
    latency = int((asyncio.get_running_loop().time() - start) * 1000)

    if df.empty:
        return JSONResponse(
            {"ok": False, "error": f"No price data for {ticker}", "latency_ms": latency},
            status_code=404,
        )

    summary = {
        "records": len(df),
        "high": df["high"].max(),
        "low": df["low"].min(),
        "avg_volume": float(df["volume"].mean()),
    }
    recent = [
        {
            "timestamp": idx.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
        }
        for idx, row in df.tail(5).iterrows()
    ]

    return JSONResponse(
        {
            "ok": True,
            "data": {
                "ticker": ticker,
                "frequency": frequency,
                "summary": summary,
                "recent": recent,
            },
            "latency_ms": latency,
        }
    )


async def polygon_get_news(request: Request) -> Response:
    payload = await request.json()
    args = payload.get("arguments", {})
    ticker = args["ticker"]
    limit = int(args.get("limit", 5))

    start = asyncio.get_running_loop().time()
    data = await asyncio.to_thread(_fetch_news, ticker, limit)
    latency = int((asyncio.get_running_loop().time() - start) * 1000)

    results = data.get("results", [])
    articles = [
        {
            "title": item.get("title"),
            "published_utc": item.get("published_utc"),
            "description": item.get("description"),
            "article_url": item.get("article_url"),
        }
        for item in results
    ]

    return JSONResponse({"ok": True, "data": articles, "latency_ms": latency})


async def polygon_get_ticker_details(request: Request) -> Response:
    payload = await request.json()
    args = payload.get("arguments", {})
    ticker = args["ticker"]

    start = asyncio.get_running_loop().time()
    data = await asyncio.to_thread(_fetch_ticker_details, ticker)
    latency = int((asyncio.get_running_loop().time() - start) * 1000)

    return JSONResponse({"ok": True, "data": data, "latency_ms": latency})


async def polygon_get_market_status(request: Request) -> Response:
    start = asyncio.get_running_loop().time()
    data = await asyncio.to_thread(_fetch_market_status)
    latency = int((asyncio.get_running_loop().time() - start) * 1000)
    return JSONResponse({"ok": True, "data": data, "latency_ms": latency})


async def polygon_get_previous_close(request: Request) -> Response:
    payload = await request.json()
    args = payload.get("arguments", {})
    ticker = args["ticker"]

    start = asyncio.get_running_loop().time()
    data = await asyncio.to_thread(_fetch_previous_close, ticker)
    latency = int((asyncio.get_running_loop().time() - start) * 1000)

    return JSONResponse({"ok": True, "data": data, "latency_ms": latency})


async def readiness(_: Request) -> Response:
    try:
        get_client()
        return JSONResponse({"ok": True})
    except Exception as exc:  # pragma: no cover - transports error to client
        logger.exception("Polygon readiness failure")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


routes = [
    Route("/health", readiness, methods=["GET"]),
    Route("/tools/polygon_get_aggs", polygon_get_aggs, methods=["POST"]),
    Route("/tools/polygon_get_news", polygon_get_news, methods=["POST"]),
    Route("/tools/polygon_get_ticker_details", polygon_get_ticker_details, methods=["POST"]),
    Route("/tools/polygon_get_market_status", polygon_get_market_status, methods=["POST"]),
    Route("/tools/polygon_get_previous_close", polygon_get_previous_close, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)

__all__ = ["app"]
