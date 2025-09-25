import pytest

from src.dataset.llm.common import to_skyrl_sample


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
    assert reward["ground_truth"]["success"]["must_call_tool"] == "polygon_get_aggs"
    assert reward["evaluation"]["rubric"]["criteria"][0]["name"] == "Insight"


def test_to_skyrl_sample_requires_user_prompt():
    task = _sample_task()
    task.pop("user_prompt")
    with pytest.raises(ValueError):
        to_skyrl_sample(task, env_class="MCPToolEnv", data_source="synthetic/llm")


def test_to_skyrl_sample_requires_tool_sequence():
    task = _sample_task()
    task["tool_sequence"] = []
    with pytest.raises(ValueError):
        to_skyrl_sample(task, env_class="MCPToolEnv", data_source="synthetic/llm")
