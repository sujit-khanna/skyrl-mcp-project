"""SkyRL environment for multi-tool, multi-turn MCP tasks."""
from __future__ import annotations

import asyncio
import json
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency in dev environments
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
except ImportError:  # pragma: no cover - lightweight fallback for unit tests
    from typing import TypedDict

    class BaseTextEnvStepOutput(TypedDict):
        observations: List[Dict[str, str]]
        reward: float
        done: bool
        metadata: Dict[str, Any]
        postprocessed_action: Optional[str]

    class BaseTextEnv:  # minimal stub replicating interface used here
        def __init__(self) -> None:
            self.turns = 0
            self.max_turns = 1

        def init(self, prompt: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
            return prompt, {}

        def close(self) -> None:  # pragma: no cover - no-op fallback
            return None

from .acceptance import evaluate_accept_if
from .dsl import (
    compute_expression,
    extract_values,
    resolve_placeholders,
    DSLExecutionError,
)
from .parsers import ParsedAction, ToolCall, parse_action
from .reward import (
    JudgeClient,
    StepRewardWeights,
    TerminalRewardWeights,
    aggregate_terminal,
    final_heuristic,
    step_reward,
)
from .state import EpisodeState
from .tool_group import MCPToolGroup, ToolCallResult


@dataclass
class EnvironmentConfig:
    step_weights: StepRewardWeights = field(default_factory=lambda: StepRewardWeights())
    terminal_weights: TerminalRewardWeights = field(default_factory=lambda: TerminalRewardWeights())
    invalid_action_penalty: float = -0.05
    limit_violation_penalty: float = -0.3
    tool_timeout: float = 30.0
    allowlist: Optional[Sequence[str]] = None
    auto_close_tool_group: bool = True
    heuristic_weights: Optional[Dict[str, float]] = None


class MCPToolEnv(BaseTextEnv):
    """Environment that mirrors the synthetic dataset semantics."""

    def __init__(
        self,
        task: Dict[str, Any],
        *,
        tool_group: MCPToolGroup | None = None,
        judge_client: JudgeClient | None = None,
        config: EnvironmentConfig | None = None,
        tool_executor: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.sample = task
        self.prompt = task.get("prompt") or []
        if len(self.prompt) < 2:
            raise ValueError("Task prompt must include system and user messages")
        reward_spec = task.get("reward_spec") or {}
        self.ground = reward_spec.get("ground_truth") or {}
        if not self.ground:
            raise ValueError("Task missing reward_spec.ground_truth")

        self.config = config or EnvironmentConfig()
        self.tool_group = tool_group
        self.tool_executor = tool_executor
        self.judge = judge_client or JudgeClient(
            model_name=(self.ground.get("judge_rubric", {}).get("model") or "gpt-judge"),
            schema=self.ground.get("judge_rubric", {}).get("schema"),
        )

        self.max_turns = int(self.ground.get("max_turns") or 6)
        self.limits = self.ground.get("limits", {})
        self.must_call_tool = (self.ground.get("success") or {}).get("must_call_tool")

        plan_steps = self.ground.get("tool_sequence", [])
        self.plan_steps = list(plan_steps)

        # Build analysis requirements from rubric first to ensure every step has directives
        self.analysis_steps: Dict[int, Dict[str, Any]] = {}
        analysis_rubric = (self.ground.get("analysis_rubric") or {}).get("steps", [])
        for entry in analysis_rubric:
            step_id = entry.get("step") if isinstance(entry, dict) else None
            if step_id is None:
                continue
            ar = entry.get("analysis_requirements") if isinstance(entry, dict) else None
            if isinstance(ar, dict):
                self.analysis_steps[step_id] = ar
            elif isinstance(entry, dict):
                # fallback: entry itself is the analysis requirements block
                self.analysis_steps[step_id] = entry
            else:
                self.analysis_steps[step_id] = {}

        # Allow tool_sequence to override if it carries explicit analysis requirements
        for step_info in self.plan_steps:
            step_id = step_info.get("step")
            if step_id is None:
                continue
            ar = step_info.get("analysis_requirements")
            if isinstance(ar, dict) and ar:
                self.analysis_steps[step_id] = ar

        self.step_lookup_by_tool = {}
        for step in self.plan_steps:
            fqdn = f"{step.get('server')}.{step.get('tool')}"
            self.step_lookup_by_tool.setdefault(fqdn, []).append(step.get("step"))

        allowlist = self.config.allowlist
        if allowlist:
            self.tool_allowlist = {tool for tool in allowlist}
        else:
            explicit = self.ground.get("limits", {}).get("allow_list", [])
            self.tool_allowlist = {tool for tool in explicit} if explicit else None

        self.messages: List[Dict[str, str]] = []
        self.state = EpisodeState()
        self.completed_steps: set[int] = set()
        self._terminated = False
        self._closed = False

    # ------------------------------------------------------------------
    # BaseTextEnv overrides
    # ------------------------------------------------------------------
    def init(self, prompt: List[Dict[str, str]] | None = None) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        self.messages = [msg.copy() for msg in (prompt or self.prompt)]
        self.state = EpisodeState()
        self.completed_steps.clear()
        self._terminated = False
        metadata = {"task_id": self.ground.get("task_id")}
        return self._observation(), metadata

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self._terminated:
            raise RuntimeError("Environment already terminated; call init() to start a new episode")

        turn = self.state.increment_turn()
        self.messages.append({"role": "assistant", "content": action})
        parsed = parse_action(action)
        reward = 0.0
        done = False
        metadata: Dict[str, Any] = {
            "turn": turn,
            "tool_called": None,
            "parse_had_json": parsed.had_json,
        }

        if parsed.tool_call is not None:
            tool_reward, tool_done, tool_meta = self._handle_tool_call(parsed.tool_call, parsed, turn)
            reward += tool_reward
            metadata.update(tool_meta)
            done = tool_done
        elif parsed.final_answer is not None:
            final_reward, final_meta = self._handle_final_answer(parsed.final_answer)
            reward += final_reward
            metadata.update(final_meta)
            done = True
        else:
            reward += self.config.invalid_action_penalty
            metadata["error"] = "invalid_action"

        if not done and turn >= self.max_turns:
            done = True
            metadata.setdefault("termination_reason", []).append("max_turns")
            reward += self.config.limit_violation_penalty

        if done:
            self._terminated = True

        return BaseTextEnvStepOutput(
            observations=self._observation(),
            reward=float(reward),
            done=done,
            metadata=metadata,
            postprocessed_action=None,
        )

    def close(self) -> None:
        if self.config.auto_close_tool_group and self.tool_group and not self._closed:
            try:
                asyncio.run(self.tool_group.aclose())
            except RuntimeError:  # pragma: no cover - already running loop
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.tool_group.aclose())
        self._closed = True

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _observation(self) -> List[Dict[str, str]]:
        return [msg.copy() for msg in self.messages]

    def _handle_tool_call(self, call: ToolCall, parsed: ParsedAction, turn: int) -> Tuple[float, bool, Dict[str, Any]]:
        fqdn = f"{call.server}.{call.tool}"
        metadata: Dict[str, Any] = {
            "tool_called": fqdn,
            "tool_params": call.params,
        }

        if self.tool_allowlist is not None and fqdn not in self.tool_allowlist:
            metadata["error"] = "tool_not_allowed"
            reward, breakdown = step_reward(
                weights=self.config.step_weights,
                extract_directives=[],
                updates={},
                accept_result=evaluate_accept_if([], {}),
                tool_allowed=False,
                tool_error=True,
                repeated=False,
                had_parse_failure=not parsed.had_json,
            )
            metadata["reward_breakdown"] = breakdown
            return reward, False, metadata

        limit_reason = self._check_limits(call.server)
        if limit_reason:
            metadata["limit_violation"] = limit_reason
            reward = self.config.limit_violation_penalty
            metadata.setdefault("reward_breakdown", {})["limit_violation"] = reward
            return reward, True, metadata

        result_payload = self._execute_tool(call.server, call.tool, call.params)
        metadata["tool_response"] = result_payload

        ok = bool(result_payload.get("ok", True))
        normalized = result_payload.get("data") if isinstance(result_payload.get("data"), dict) else result_payload

        step_index = self._match_step(fqdn)
        analysis = self.analysis_steps.get(step_index, {})
        extract_directives = analysis.get("extract", [])
        updates, missing = extract_values(normalized or {}, extract_directives)
        local_state = {**self.state.facts, **updates}

        compute_failures: List[str] = []
        for expr in analysis.get("compute", []) or []:
            try:
                computed = compute_expression(expr, local_state)
                updates.update(computed)
                local_state.update(computed)
            except DSLExecutionError as exc:
                compute_failures.append(f"{expr}: {exc}")
        for expr in analysis.get("select", []) or []:
            try:
                selected = compute_expression(expr, local_state)
                updates.update(selected)
                local_state.update(selected)
            except DSLExecutionError as exc:
                compute_failures.append(f"{expr}: {exc}")

        acceptance = evaluate_accept_if(analysis.get("accept_if", []), local_state)

        repeated = self.state.has_called_with_params(fqdn, call.params)
        reward, breakdown = step_reward(
            weights=self.config.step_weights,
            extract_directives=extract_directives,
            updates=updates,
            accept_result=acceptance,
            tool_allowed=True,
            tool_error=not ok,
            repeated=repeated,
            had_parse_failure=not parsed.had_json,
        )

        metadata["reward_breakdown"] = breakdown
        metadata["missing_extracts"] = missing
        metadata["compute_failures"] = compute_failures
        metadata["accept_passed"] = acceptance.passed
        metadata["accept_failed_conditions"] = acceptance.failed_conditions
        metadata["step_index"] = step_index

        self.state.record_tool_call(
            call.server,
            call.tool,
            result_payload,
            params=call.params,
        )
        self.state.update_facts(updates)
        if step_index is not None:
            self.completed_steps.add(step_index)

        self.messages.append({"role": "tool", "content": json.dumps(result_payload)})

        return reward, False, metadata

    def _handle_final_answer(self, final_text: str) -> Tuple[float, Dict[str, Any]]:
        final_text = final_text.strip()
        far = (self.ground.get("analysis_rubric") or {}).get("final_answer_requirements") or {}
        heuristic_score, heuristic_breakdown = final_heuristic(
            final_text,
            far,
            self.state.facts,
            weights=self.config.heuristic_weights,
        )

        judge_payload = {
            "messages": self._observation(),
            "final_answer": final_text,
            "facts": self.state.facts,
            "rubric": self.ground.get("judge_rubric", {}),
        }
        judge_scores = self._run_async(self.judge.score(judge_payload))
        judge_total = float(judge_scores.get("total", 0.0))

        success_called = True
        if self.must_call_tool:
            success_called = self.must_call_tool in self.state.called_tools

        total_reward, terminal_breakdown = aggregate_terminal(
            heuristic_score,
            judge_total,
            success_called,
            weights=self.config.terminal_weights,
        )

        reward = total_reward
        metadata = {
            "final_answer": final_text,
            "heuristic": heuristic_breakdown,
            "judge_scores": judge_scores,
            "terminal_breakdown": terminal_breakdown,
            "success_called": success_called,
        }

        return reward, metadata

    def _check_limits(self, server: str) -> Optional[str]:
        max_tools = self.limits.get("max_tools")
        if max_tools is not None and self.state.total_tool_calls >= max_tools:
            return "max_tools"
        max_servers = self.limits.get("max_servers")
        if max_servers is not None:
            unique_servers = self.state.unique_servers
            if server not in unique_servers and len(unique_servers) >= max_servers:
                return "max_servers"
        return None

    def _execute_tool(self, server: str, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        resolved_params = resolve_placeholders(params, self.state.facts)
        if self.tool_executor is not None:
            result = self.tool_executor(server, tool, resolved_params)
            if inspect.iscoroutine(result):
                result = self._run_async(result)
            if not isinstance(result, dict):
                raise TypeError("Tool executor must return a dict payload")
            return result
        if not self.tool_group:
            raise RuntimeError("Tool group not configured for MCPToolEnv")
        call_result: ToolCallResult = self.tool_group.call_sync(
            server,
            tool,
            resolved_params,
            timeout=self.config.tool_timeout,
        )
        return call_result.payload

    def _match_step(self, fqdn: str) -> Optional[int]:
        candidates = self.step_lookup_by_tool.get(fqdn)
        if not candidates:
            return None
        for step_idx in candidates:
            if step_idx not in self.completed_steps:
                return step_idx
        return candidates[-1]

    def _run_async(self, coro: Any) -> Any:
        if asyncio.iscoroutine(coro):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            else:  # pragma: no cover - seldom used during synchronous tests
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result()
        return coro
