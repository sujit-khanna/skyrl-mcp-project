"""Smoke tests for MCP servers using the HTTP ToolManager."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta, timezone, datetime
from pathlib import Path
import sys
import os
import json
import uuid
from typing import Any, Awaitable, Callable, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

LOG_DIR = PROJECT_ROOT / "scripts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"mcp_tool_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.log"


def log(message: str) -> None:
    print(message)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")

from src.utils.tool_manager import ToolManager


class SkipToolTest(Exception):
    """Raised when a tool test should be skipped (e.g., missing env)."""


SharedState = Dict[str, Any]


def require_env(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise SkipToolTest(f"Missing required environment variables: {', '.join(missing)}")


async def assert_ok_response(tool_name: str, response: Dict[str, Any]) -> Dict[str, Any]:
    if not response.get("ok"):
        error = response.get("error", "Unknown error")
        raise AssertionError(f"{tool_name} returned error: {error}")
    pretty = json.dumps(response, indent=2, default=str)
    log(f"{tool_name} response:\n{pretty}")
    return response


ToolTest = Callable[[ToolManager, SharedState], Awaitable[None]]


async def test_polygon_get_aggs(manager: ToolManager, _: SharedState) -> None:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    response = await manager.execute_tool(
        "polygon_get_aggs",
        {
            "ticker": "AAPL",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "frequency": "daily",
        },
    )
    payload = await assert_ok_response("polygon_get_aggs", response)
    assert payload["data"]["summary"]["records"] > 0


async def test_polygon_get_news(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool(
        "polygon_get_news",
        {"ticker": "AAPL", "limit": 3},
    )
    payload = await assert_ok_response("polygon_get_news", response)
    assert payload["data"], "Polygon news returned empty results"


async def test_polygon_get_ticker_details(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool(
        "polygon_get_ticker_details",
        {"ticker": "AAPL"},
    )
    payload = await assert_ok_response("polygon_get_ticker_details", response)
    assert payload["data"].get("results", {}).get("ticker") == "AAPL"


async def test_polygon_get_market_status(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("polygon_get_market_status", {})
    payload = await assert_ok_response("polygon_get_market_status", response)
    assert "market" in payload["data"]


async def test_polygon_get_previous_close(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("polygon_get_previous_close", {"ticker": "AAPL"})
    payload = await assert_ok_response("polygon_get_previous_close", response)
    assert payload["data"].get("results")


async def test_fmp_get_quote(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("fmp_get_quote", {"symbol": "AAPL"})
    payload = await assert_ok_response("fmp_get_quote", response)
    assert payload["data"]["symbol"].upper() == "AAPL"


async def test_fmp_get_company_profile(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("fmp_get_company_profile", {"symbol": "AAPL"})
    payload = await assert_ok_response("fmp_get_company_profile", response)
    assert payload["data"].get("companyName")


async def test_fmp_get_income_statement(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool(
        "fmp_get_income_statement",
        {"symbol": "AAPL", "period": "annual", "limit": 1},
    )
    payload = await assert_ok_response("fmp_get_income_statement", response)
    assert payload["data"], "Income statement returned empty payload"


async def test_fmp_get_market_gainers(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("fmp_get_market_gainers", {})
    payload = await assert_ok_response("fmp_get_market_gainers", response)
    assert payload["data"], "No market gainers returned"


async def test_fmp_get_market_losers(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("fmp_get_market_losers", {})
    payload = await assert_ok_response("fmp_get_market_losers", response)
    assert payload["data"], "No market losers returned"


async def test_fmp_search_ticker(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool(
        "fmp_search_ticker",
        {"query": "Apple", "limit": 5},
    )
    payload = await assert_ok_response("fmp_search_ticker", response)
    assert payload["data"], "Ticker search returned no results"


async def test_fmp_get_stock_news(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool(
        "fmp_get_stock_news",
        {"tickers": "AAPL", "limit": 3},
    )
    payload = await assert_ok_response("fmp_get_stock_news", response)
    assert payload["data"], "Stock news returned empty payload"


async def test_fmp_get_change_percent(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("fmp_get_change_percent", {"symbol": "AAPL"})
    payload = await assert_ok_response("fmp_get_change_percent", response)
    assert "changesPercentage" in payload["data"]


async def test_fmp_get_previous_close(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("fmp_get_previous_close", {"symbol": "AAPL"})
    payload = await assert_ok_response("fmp_get_previous_close", response)
    assert "price" in payload["data"]


async def test_tavily_search(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("tavily_search", {"query": "latest NVIDIA earnings", "max_results": 3})
    if not response.get("ok") and "usage limit" in response.get("error", "").lower():
        raise SkipToolTest("Tavily usage limit reached; skipping search test")
    payload = await assert_ok_response("tavily_search", response)
    assert payload["data"].get("results"), "Tavily search returned no results"


async def test_tavily_extract(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("tavily_extract", {"url": "https://openai.com"})
    error_text = response.get("error", "").lower()
    if not response.get("ok"):
        if "usage limit" in error_text or "502 bad gateway" in error_text:
            raise SkipToolTest("Tavily extract unavailable (plan limit or gateway error)")
    payload = await assert_ok_response("tavily_extract", response)
    data = payload.get("data", {})
    has_content = False
    if isinstance(data, dict):
        has_content = any(
            key in data and data[key]
            for key in ("content", "raw_content", "results")
        )
    elif isinstance(data, list):
        has_content = bool(data)
    assert has_content, "Tavily extract returned no content"


async def test_execute_python(manager: ToolManager, _: SharedState) -> None:
    response = await manager.execute_tool("execute_python", {"code": "result = 40 + 2"})
    payload = await assert_ok_response("execute_python", response)
    assert payload["result"] == 42


async def test_process_mcp_data(manager: ToolManager, _: SharedState) -> None:
    data = json.dumps({"values": [1, 2, 3]})
    response = await manager.execute_tool(
        "process_mcp_data",
        {
            "data": data,
            "processing_code": "result = sum(mcp_data['values'])",
        },
    )
    payload = await assert_ok_response("process_mcp_data", response)
    assert payload["result"] == 6


async def test_store_execution_result(manager: ToolManager, state: SharedState) -> None:
    context_key = f"mcp_test_{uuid.uuid4().hex}"
    state["python_context_key"] = context_key
    response = await manager.execute_tool(
        "store_execution_result",
        {
            "context_key": context_key,
            "data": json.dumps({"status": "stored"}),
        },
    )
    await assert_ok_response("store_execution_result", response)


async def test_get_execution_context(manager: ToolManager, state: SharedState) -> None:
    context_key = state.get("python_context_key")
    if not context_key:
        await test_store_execution_result(manager, state)
        context_key = state["python_context_key"]
    response = await manager.execute_tool(
        "get_execution_context",
        {"context_key": context_key},
    )
    payload = await assert_ok_response("get_execution_context", response)
    assert payload["data"].get("status") == "stored"


async def test_list_slack_channels(manager: ToolManager, _: SharedState) -> None:
    require_env("SLACK_BOT_TOKEN")
    response = await manager.execute_tool("list_slack_channels", {})
    payload = await assert_ok_response("list_slack_channels", response)
    assert payload["data"], "No Slack channels returned"


async def test_slack_auth_test(manager: ToolManager, _: SharedState) -> None:
    require_env("SLACK_BOT_TOKEN")
    response = await manager.execute_tool("slack_auth_test", {})
    payload = await assert_ok_response("slack_auth_test", response)
    assert payload["data"].get("ok") is True


async def test_send_slack_message(manager: ToolManager, state: SharedState) -> None:
    require_env("SLACK_BOT_TOKEN")
    channel_id = os.getenv("SLACK_TEST_CHANNEL_ID") or os.getenv("SLACK_CHANNEL_ID")
    if not channel_id:
        raise SkipToolTest("SLACK_TEST_CHANNEL_ID or SLACK_CHANNEL_ID not set")
    message = f"SkyRL MCP comprehensive test at {datetime.now(timezone.utc).isoformat()}"
    state["slack_last_message"] = message
    response = await manager.execute_tool(
        "send_slack_message",
        {"channel": channel_id, "message": message},
    )
    await assert_ok_response("send_slack_message", response)


async def test_get_slack_channel_history(manager: ToolManager, state: SharedState) -> None:
    require_env("SLACK_BOT_TOKEN")
    channel_id = os.getenv("SLACK_TEST_CHANNEL_ID") or os.getenv("SLACK_CHANNEL_ID")
    if not channel_id:
        raise SkipToolTest("SLACK_TEST_CHANNEL_ID or SLACK_CHANNEL_ID not set")
    if "slack_last_message" not in state:
        await test_send_slack_message(manager, state)
    response = await manager.execute_tool(
        "get_slack_channel_history",
        {"channel_id": channel_id, "limit": 20},
    )
    payload = await assert_ok_response("get_slack_channel_history", response)
    messages = payload.get("data", [])
    assert any(msg.get("text") == state["slack_last_message"] for msg in messages)


TOOL_TESTS: Dict[str, ToolTest] = {
    "polygon_get_aggs": test_polygon_get_aggs,
    "polygon_get_news": test_polygon_get_news,
    "polygon_get_ticker_details": test_polygon_get_ticker_details,
    "polygon_get_market_status": test_polygon_get_market_status,
    "polygon_get_previous_close": test_polygon_get_previous_close,
    "fmp_get_quote": test_fmp_get_quote,
    "fmp_get_company_profile": test_fmp_get_company_profile,
    "fmp_get_income_statement": test_fmp_get_income_statement,
    "fmp_get_market_gainers": test_fmp_get_market_gainers,
    "fmp_get_market_losers": test_fmp_get_market_losers,
    "fmp_search_ticker": test_fmp_search_ticker,
    "fmp_get_stock_news": test_fmp_get_stock_news,
    "fmp_get_change_percent": test_fmp_get_change_percent,
    "fmp_get_previous_close": test_fmp_get_previous_close,
    "tavily_search": test_tavily_search,
    "tavily_extract": test_tavily_extract,
    "execute_python": test_execute_python,
    "process_mcp_data": test_process_mcp_data,
    "store_execution_result": test_store_execution_result,
    "get_execution_context": test_get_execution_context,
    "list_slack_channels": test_list_slack_channels,
    "slack_auth_test": test_slack_auth_test,
    "send_slack_message": test_send_slack_message,
    "get_slack_channel_history": test_get_slack_channel_history,
}


async def main() -> None:
    async with ToolManager.from_config_dir("mcp_servers/configs") as manager:
        shared_state: SharedState = {}
        config_dir = PROJECT_ROOT / "mcp_servers" / "configs"
        tested_tools = set()

        for config_file in sorted(config_dir.glob("*.json")):
            with config_file.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            service = config.get("service", config_file.stem)
            log(f"\nValidating tools for service: {service}")
            for tool_name in config.get("tools", {}).keys():
                tested_tools.add(tool_name)
                test_fn = TOOL_TESTS.get(tool_name)
                if test_fn is None:
                    raise SystemExit(f"No test registered for tool '{tool_name}'")
                try:
                    await test_fn(manager, shared_state)
                except SkipToolTest as exc:
                    log(f"- {tool_name}: skipped ({exc})")
                except AssertionError as exc:
                    raise SystemExit(f"- {tool_name}: validation failed ({exc})")
                else:
                    log(f"- {tool_name}: passed")

        missing_tests = set(TOOL_TESTS.keys()) - tested_tools
        if missing_tests:
            log(f"\nWarning: the following tool tests were registered but not present in configs: {sorted(missing_tests)}")

    log("All MCP tool smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
