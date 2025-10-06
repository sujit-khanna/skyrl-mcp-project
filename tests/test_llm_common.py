import pytest

import json
from pathlib import Path

from src.dataset.llm.common import (
    load_curriculum_prompts,
    normalize_complexity,
    to_skyrl_sample,
)


def _sample_task():
    return {
        "task_id": "eq-123",
        "user_prompt": "Summarize SPY performance and notify the team.",
        "complexity": "moderate",
        "max_turns": 6,
        "tools_available": ["polygon_get_aggs", "send_slack_message"],
        "limits": {"max_tools": 3, "max_servers": 2},
        "tool_sequence": [
            {
                "step": 1,
                "server": "polygon",
                "tool": "polygon_get_aggs",
                "params": {"ticker": "SPY", "start_date": "2025-01-01", "end_date": "2025-01-05"},
            },
            {
                "step": 2,
                "server": "slack",
                "tool": "send_slack_message",
                "params": {"channel": "general", "message": "SPY summary"},
            },
        ],
        "evaluation_rubric": {
            "rubric_name": "Daily summary",
            "criteria": [
                {
                    "name": "Insight",
                    "description": "Mentions key price changes",
                    "scoring": "1 if correct, 0 otherwise",
                },
                {
                    "name": "Notification",
                    "description": "Slack message sent",
                    "scoring": "1 if channel updated",
                },
            ],
        },
    }


def test_to_skyrl_sample_builds_reward_spec_with_rubric():
    task = _sample_task()
    sample = to_skyrl_sample(task, env_class="MCPToolEnv", data_source="synthetic/llm")

    assert sample.prompt[0]["role"] == "system"
    assert "Available tools" in sample.prompt[0]["content"]
    assert sample.prompt[1]["role"] == "user"

    reward = sample.reward_spec
    assert reward["method"] == "rule"
    assert reward["ground_truth"]["tool_sequence"][0]["server"] == "polygon"
    assert reward["ground_truth"]["success"]["must_call_tool"] == "polygon.polygon_get_aggs"
    assert reward["ground_truth"]["analysis_rubric"]["steps"][0]["step"] == 1


def test_to_skyrl_sample_requires_user_prompt():
    task = _sample_task()
    task.pop("user_prompt")
    with pytest.raises(ValueError):
        to_skyrl_sample(task, env_class="MCPToolEnv", data_source="synthetic/llm")


def test_normalize_complexity_maps_curriculum_labels():
    assert normalize_complexity("easy") == "simple"
    assert normalize_complexity("medium") == "moderate"
    assert normalize_complexity("difficult") == "complex"
    assert normalize_complexity("Complex") == "complex"
    with pytest.raises(ValueError):
        normalize_complexity("unknown")


def test_load_curriculum_prompts(tmp_path: Path):
    data = [
        {"id": "E01", "user_prompt": "Hello", "complexity": "easy", "extra": 1},
        {"user_prompt": "World", "complexity": "complex"},
    ]
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(data))

    prompts = load_curriculum_prompts(prompt_file)
    assert len(prompts) == 2
    assert prompts[0]["complexity"] == "simple"
    assert prompts[0]["prompt_id"] == "E01"
    assert prompts[0]["metadata"]["extra"] == 1
    assert prompts[0]["metadata"]["original_complexity"] == "easy"
    assert prompts[1]["complexity"] == "complex"


def test_to_skyrl_sample_requires_tool_sequence():
    task = _sample_task()
    task["tool_sequence"] = []
    with pytest.raises(ValueError):
        to_skyrl_sample(task, env_class="MCPToolEnv", data_source="synthetic/llm")
