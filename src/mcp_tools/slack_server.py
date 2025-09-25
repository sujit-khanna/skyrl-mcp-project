"""Slack MCP tool server built on top of Slack's Web API."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

load_dotenv(override=True)

logger = logging.getLogger("mcp.slack")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

_client: AsyncWebClient | None = None


def get_client() -> AsyncWebClient:
    global _client
    if _client is None:
        if not SLACK_BOT_TOKEN:
            raise RuntimeError("SLACK_BOT_TOKEN is not configured")
        _client = AsyncWebClient(token=SLACK_BOT_TOKEN)
    return _client


async def list_channels(_: Request) -> Response:
    client = get_client()
    try:
        response = await client.conversations_list(limit=200, exclude_archived=True)
    except SlackApiError as exc:  # pragma: no cover - requires network failure
        logger.exception("Slack conversations_list failed")
        return JSONResponse({"ok": False, "error": exc.response.get("error")}, status_code=500)

    payload = response.data if hasattr(response, "data") else response

    channels = [
        {
            "id": channel.get("id"),
            "name": channel.get("name"),
            "is_channel": channel.get("is_channel"),
            "num_members": channel.get("num_members"),
        }
        for channel in payload.get("channels", [])
        if channel.get("is_channel")
    ]
    return JSONResponse({"ok": True, "data": channels})


async def send_message(request: Request) -> Response:
    client = get_client()
    args = (await request.json()).get("arguments", {})
    channel = args["channel"]
    message = args["message"]
    if not channel.startswith("C") and not channel.startswith("#"):
        channel = f"#{channel}"

    try:
        response = await client.chat_postMessage(channel=channel, text=message)
    except SlackApiError as exc:  # pragma: no cover - token/channel errors
        logger.exception("Slack chat_postMessage failed")
        return JSONResponse({"ok": False, "error": exc.response.get("error")}, status_code=500)

    payload = response.data if hasattr(response, "data") else response
    return JSONResponse({"ok": payload.get("ok", False), "data": payload})


async def channel_history(request: Request) -> Response:
    client = get_client()
    args = (await request.json()).get("arguments", {})
    channel_id = args["channel_id"]
    limit = int(args.get("limit", 10))

    try:
        response = await client.conversations_history(channel=channel_id, limit=limit)
    except SlackApiError as exc:  # pragma: no cover
        logger.exception("Slack conversations_history failed")
        return JSONResponse({"ok": False, "error": exc.response.get("error")}, status_code=500)

    payload = response.data if hasattr(response, "data") else response

    messages = [
        {
            "ts": message.get("ts"),
            "user": message.get("user"),
            "text": message.get("text"),
            "datetime": datetime.fromtimestamp(float(message.get("ts", 0))).isoformat(),
        }
        for message in payload.get("messages", [])
    ]
    return JSONResponse({"ok": True, "data": messages})


async def auth_test(_: Request) -> Response:
    client = get_client()
    try:
        response = await client.auth_test()
        payload = response.data if hasattr(response, "data") else response
        return JSONResponse({"ok": True, "data": payload})
    except SlackApiError as exc:  # pragma: no cover
        logger.exception("Slack auth.test failed")
        return JSONResponse({"ok": False, "error": exc.response.get("error")}, status_code=500)


async def readiness(_: Request) -> Response:
    try:
        get_client()
        return JSONResponse({"ok": True})
    except Exception as exc:  # pragma: no cover
        logger.exception("Slack readiness failure")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


routes = [
    Route("/health", readiness, methods=["GET"]),
    Route("/tools/slack_auth_test", auth_test, methods=["POST"]),
    Route("/tools/list_slack_channels", list_channels, methods=["POST"]),
    Route("/tools/send_slack_message", send_message, methods=["POST"]),
    Route("/tools/get_slack_channel_history", channel_history, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)

__all__ = ["app"]
