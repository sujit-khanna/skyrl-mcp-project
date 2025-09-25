from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta, datetime, timezone

import httpx
import pytest
from dotenv import load_dotenv

from src.mcp_tools import polygon_server, fmp_server, tavily_server, python_execution_server, slack_server

load_dotenv(override=True)


def require_env(var: str) -> None:
    if not os.getenv(var):
        pytest.skip(f"{var} not configured")


@pytest.mark.asyncio
async def test_polygon_market_status() -> None:
    require_env("POLYGON_API_KEY")
    transport = httpx.ASGITransport(app=polygon_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/tools/polygon_get_market_status", json={"arguments": {}})
        payload = response.json()
        assert payload["ok"] is True
        assert "data" in payload


@pytest.mark.asyncio
async def test_polygon_aggregates() -> None:
    require_env("POLYGON_API_KEY")
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=5)
    body = {
        "arguments": {
            "ticker": "AAPL",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "frequency": "daily",
        }
    }
    transport = httpx.ASGITransport(app=polygon_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/tools/polygon_get_aggs", json=body)
        payload = response.json()
        assert payload["ok"] is True
        assert payload["data"]["summary"]["records"] > 0


@pytest.mark.asyncio
async def test_fmp_quote() -> None:
    require_env("FMP_API_KEY")
    transport = httpx.ASGITransport(app=fmp_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/tools/fmp_get_quote", json={"arguments": {"symbol": "AAPL"}})
        payload = response.json()
        assert payload["ok"] is True
        assert payload["data"]["symbol"].upper() == "AAPL"


@pytest.mark.asyncio
async def test_tavily_search() -> None:
    require_env("TAVILY_API_KEY")
    transport = httpx.ASGITransport(app=tavily_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/tools/tavily_search",
            json={"arguments": {"query": "OpenAI latest news", "max_results": 2}},
        )
        payload = response.json()
        if not payload["ok"]:
            pytest.skip(f"Tavily search unavailable: {payload.get('error')}")
        assert "results" in payload["data"]


@pytest.mark.asyncio
async def test_python_execute() -> None:
    transport = httpx.ASGITransport(app=python_execution_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/tools/execute_python",
            json={"arguments": {"code": "result = sum(range(10))"}},
        )
        payload = response.json()
        assert payload["ok"] is True
        assert payload["result"] == 45


@pytest.mark.asyncio
async def test_slack_auth_test() -> None:
    require_env("SLACK_BOT_TOKEN")
    transport = httpx.ASGITransport(app=slack_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/tools/slack_auth_test", json={"arguments": {}})
        payload = response.json()
        assert payload["ok"] is True
        assert payload["data"]["ok"] is True


@pytest.mark.asyncio
async def test_slack_send_and_history() -> None:
    require_env("SLACK_BOT_TOKEN")
    channel_id = os.getenv("SLACK_TEST_CHANNEL_ID")
    if not channel_id:
        pytest.skip("SLACK_TEST_CHANNEL_ID not configured")

    message = f"MCP test message {datetime.now(timezone.utc).isoformat()}"
    transport = httpx.ASGITransport(app=slack_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        send_resp = await client.post(
            "/tools/send_slack_message",
            json={"arguments": {"channel": channel_id, "message": message}},
        )
        send_payload = send_resp.json()
        assert send_payload["ok"] is True

        history_resp = await client.post(
            "/tools/get_slack_channel_history",
            json={"arguments": {"channel_id": channel_id, "limit": 20}},
        )
        history_payload = history_resp.json()
        assert history_payload["ok"] is True
        messages = history_payload.get("data", [])
        assert any(msg.get("text") == message for msg in messages)
