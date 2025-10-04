"""Robust parsing for tool call / final answer directives."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

JSON_FENCE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
# Fallback block should be non-greedy to avoid swallowing neighbouring JSON snippets
JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)
TOOL_BLOCK = re.compile(r"<tool>(?P<body>.*?)</tool>", re.DOTALL | re.IGNORECASE)
NAMED_BLOCK = re.compile(r"<(?P<name>[A-Za-z0-9_:-]+)>(?P<body>.*?)</(?P=name)>", re.DOTALL)


@dataclass
class ToolCall:
    server: str
    tool: str
    params: Dict[str, Any]
    raw: Dict[str, Any] | None = None


@dataclass
class ParsedAction:
    tool_call: Optional[ToolCall]
    final_answer: Optional[str]
    raw: Dict[str, Any] | None
    had_json: bool


def _load_json(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_json_segment(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("{"):
        direct = _load_json(stripped)
        if direct is not None:
            return direct
    for line in text.splitlines():
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            direct = _load_json(candidate)
            if direct is not None:
                return direct
    fence = JSON_FENCE.search(text)
    if fence:
        payload = _load_json(fence.group("body"))
        if payload is not None:
            return payload
    matches = list(JSON_BLOCK.finditer(text))
    if matches:
        for match in sorted(matches, key=lambda m: len(m.group(0)), reverse=True):
            payload = _load_json(match.group(0))
            if payload is not None:
                return payload
    return None


def _parse_tool_from_dict(doc: Dict[str, Any]) -> Optional[ToolCall]:
    # Preferred schema
    if "tool_call" in doc and isinstance(doc["tool_call"], dict):
        tc = doc["tool_call"]
        server = tc.get("server")
        tool = tc.get("tool")
        params = tc.get("params", {})
        if isinstance(server, str) and isinstance(tool, str) and isinstance(params, dict):
            return ToolCall(server=server, tool=tool, params=params, raw=tc)

    # Legacy schema {"tool": "name", "arguments": {...}}
    if "tool" in doc and isinstance(doc["tool"], str):
        server = doc.get("server")
        tool = doc["tool"]
        params = doc.get("arguments", {})
        if server is None and "." in tool:
            server, tool = tool.split(".", 1)
        if isinstance(server, str) and isinstance(params, dict):
            return ToolCall(server=server, tool=tool, params=params, raw=doc)

    return None


def _parse_final_from_dict(doc: Dict[str, Any]) -> Optional[str]:
    if "final_answer" in doc and isinstance(doc["final_answer"], str):
        return doc["final_answer"].strip()
    if "answer" in doc and isinstance(doc["answer"], str):  # lenient fallback
        return doc["answer"].strip()
    return None


def parse_action(text: str) -> ParsedAction:
    # First attempt JSON extraction
    payload = _extract_json_segment(text)
    tool_call = None
    final_answer = None
    had_json = payload is not None

    if payload:
        tool_call = _parse_tool_from_dict(payload)
        final_answer = _parse_final_from_dict(payload)

    # If still no directives try XML-style tool format
    if tool_call is None:
        tool_match = TOOL_BLOCK.search(text)
        if tool_match:
            inner = tool_match.group("body")
            name_match = NAMED_BLOCK.search(inner)
            params: Dict[str, Any] = {}
            server: Optional[str] = None
            tool: Optional[str] = None
            while name_match:
                tag = name_match.group("name").lower()
                body = name_match.group("body").strip()
                if tag in {"server", "service"}:
                    server = body
                elif tag == "tool":
                    tool = body
                elif tag in {"arguments", "params", "parameters"}:
                    params = _load_json(body) or {"raw": body}
                inner = inner.replace(name_match.group(0), "", 1)
                name_match = NAMED_BLOCK.search(inner)
            if not tool:
                tool = inner.strip()
            if tool and server:
                if not isinstance(params, dict):
                    params = {"raw": params}
                tool_call = ToolCall(server=server, tool=tool, params=params)

    return ParsedAction(
        tool_call=tool_call,
        final_answer=final_answer,
        raw=payload,
        had_json=had_json,
    )


__all__ = ["ToolCall", "ParsedAction", "parse_action"]
