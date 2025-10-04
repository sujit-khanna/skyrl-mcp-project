"""Reward shaping helpers for MCPToolEnv."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .acceptance import AcceptanceResult


@dataclass
class StepRewardWeights:
    extract: float = 0.2
    accept: float = 0.4
    valid_tool: float = 0.1
    invalid_tool: float = -0.25
    tool_error: float = -0.4
    repeated_tool: float = -0.1
    parse_failure: float = -0.1


@dataclass
class TerminalRewardWeights:
    heuristic: float = 0.4
    judge: float = 0.6
    success_bonus: float = 0.05
    missing_success_penalty: float = -0.1


DEFAULT_STEP_WEIGHTS = StepRewardWeights()
DEFAULT_TERMINAL_WEIGHTS = TerminalRewardWeights()


@dataclass
class JudgeClient:
    """Thin wrapper over an async callable for LLM-as-a-Judge scoring."""

    model_name: str = "gpt-judge"
    schema: Mapping[str, Any] | None = None
    transport: Optional[Any] = None
    fallback_total: float = 0.0

    async def score(self, payload: Mapping[str, Any]) -> Dict[str, float]:
        if self.transport is None:
            return {
                "coverage": self.fallback_total,
                "grounding": self.fallback_total,
                "clarity": self.fallback_total,
                "safety": self.fallback_total,
                "total": self.fallback_total,
            }
        return await self.transport(
            model=self.model_name,
            payload=payload,
            schema=self.schema,
        )

    @property
    def is_configured(self) -> bool:
        return self.transport is not None


def step_reward(
    *,
    weights: StepRewardWeights,
    extract_directives,
    updates: Mapping[str, Any],
    accept_result: AcceptanceResult,
    tool_allowed: bool,
    tool_error: bool,
    repeated: bool,
    had_parse_failure: bool,
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}
    realized = 0
    for directive in extract_directives or []:
        key = directive.split("=")[0].strip() if "=" in directive else directive
        key = key.split("[")[0].split("{")[0].split(".")[-1]
        if key in updates and updates[key] is not None:
            realized += 1
    breakdown["r_extract"] = weights.extract * realized
    breakdown["r_accept"] = weights.accept if accept_result.passed and (extract_directives or updates) else 0.0
    breakdown["r_tool_allowed"] = weights.valid_tool if tool_allowed else weights.invalid_tool
    breakdown["r_tool_error"] = weights.tool_error if tool_error else 0.0
    breakdown["r_repeated"] = weights.repeated_tool if repeated else 0.0
    breakdown["r_parse_failure"] = weights.parse_failure if had_parse_failure else 0.0

    total = sum(breakdown.values())
    return total, breakdown


def final_heuristic(
    final_text: str,
    final_answer_requirements: Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    far = final_answer_requirements or {}
    must_include = far.get("must_include", [])
    grounded_from = far.get("grounded_from", [])

    def _hit_ratio(targets) -> float:
        if not targets:
            return 1.0
        hits = 0
        for item in targets:
            value = facts.get(item)
            if value is None:
                continue
            if _value_in_text(value, final_text):
                hits += 1
        return hits / max(1, len(targets))

    coverage = _hit_ratio(must_include)
    grounding = _hit_ratio(grounded_from)

    length_range = far.get("length_range") or far.get("target_length_range")
    tokens = len(final_text.split())
    if isinstance(length_range, (list, tuple)) and len(length_range) == 2:
        length_ok = 1.0 if length_range[0] <= tokens <= length_range[1] else 0.0
    else:
        length_ok = 1.0

    breakdown = {
        "coverage": coverage,
        "grounding": grounding,
        "length": length_ok,
    }
    heur_weights = weights or {}
    w_cov = heur_weights.get("heur_cov", 0.5)
    w_grd = heur_weights.get("heur_grd", 0.4)
    w_len = heur_weights.get("heur_len", 0.1)
    heur_score = w_cov * coverage + w_grd * grounding + w_len * length_ok
    breakdown["heuristic"] = heur_score
    return heur_score, breakdown


def aggregate_terminal(
    heuristic_score: float,
    judge_total: float,
    success_called: bool,
    *,
    weights: TerminalRewardWeights,
) -> Tuple[float, Dict[str, float]]:
    breakdown = {
        "heuristic_component": heuristic_score * weights.heuristic,
        "judge_component": judge_total * weights.judge,
        "success_bonus": weights.success_bonus if success_called else weights.missing_success_penalty,
    }
    return sum(breakdown.values()), breakdown


def _value_in_text(value: Any, text: str) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return f"{value}" in text or f"{value:,.2f}" in text or f"{value:,.1f}" in text
    if isinstance(value, (list, tuple, set)):
        return all(_value_in_text(v, text) for v in value)
    if isinstance(value, Mapping):
        return all(_value_in_text(v, text) for v in value.values())
    return str(value) in text


__all__ = [
    "StepRewardWeights",
    "TerminalRewardWeights",
    "DEFAULT_STEP_WEIGHTS",
    "DEFAULT_TERMINAL_WEIGHTS",
    "JudgeClient",
    "step_reward",
    "final_heuristic",
    "aggregate_terminal",
]
