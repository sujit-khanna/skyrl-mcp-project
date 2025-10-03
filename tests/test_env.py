import json

from src.envs.mcp_tool_env import MCPToolEnv, EnvironmentConfig


TASK_SAMPLE = {
    "prompt": [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt requesting stock info"},
    ],
    "reward_spec": {
        "ground_truth": {
            "task_id": "unit-test-task",
            "max_turns": 3,
            "success": {"must_call_tool": "fmp.fmp_get_quote"},
            "limits": {"max_tools": 3, "max_servers": 2},
            "tool_sequence": [
                {
                    "step": 1,
                    "server": "fmp",
                    "tool": "fmp_get_quote",
                    "params": {"symbol": "AAPL"},
                    "analysis_requirements": {
                        "extract": ["price", "volume"],
                        "compute": [],
                        "select": [],
                        "accept_if": ["price is not None"],
                    },
                }
            ],
            "analysis_rubric": {
                "steps": [
                    {
                        "step": 1,
                        "extract": ["price", "volume"],
                        "compute": [],
                        "select": [],
                        "accept_if": ["price is not None"],
                    }
                ],
                "final_answer_requirements": {
                    "format": "text",
                    "must_include": ["price", "volume"],
                    "grounded_from": ["price", "volume"],
                    "quality_criteria": [],
                },
            },
            "judge_rubric": {
                "weights": {
                    "coverage": 0.4,
                    "grounding": 0.4,
                    "clarity": 0.2,
                    "safety": 0.0,
                },
                "schema": {
                    "type": "object",
                    "properties": {
                        "coverage": {"type": "number", "minimum": 0, "maximum": 1},
                        "grounding": {"type": "number", "minimum": 0, "maximum": 1},
                        "clarity": {"type": "number", "minimum": 0, "maximum": 1},
                        "safety": {"type": "number", "minimum": 0, "maximum": 1},
                        "total": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["coverage", "grounding", "clarity", "safety", "total"],
                    "additionalProperties": False,
                },
            },
        }
    },
}


def stub_tool_executor(server: str, tool: str, params: dict) -> dict:
    assert server == "fmp"
    assert tool == "fmp_get_quote"
    assert params["symbol"] == "AAPL"
    return {
        "ok": True,
        "data": {"price": 101.23, "volume": 123456},
        "latency_ms": 12,
    }


def test_env_tool_step_and_final_answer():
    env = MCPToolEnv(TASK_SAMPLE, tool_executor=stub_tool_executor, config=EnvironmentConfig())
    observation, info = env.init()
    assert observation[0]["role"] == "system"
    assert info["task_id"] == "unit-test-task"

    tool_action = json.dumps(
        {
            "tool_call": {
                "server": "fmp",
                "tool": "fmp_get_quote",
                "params": {"symbol": "AAPL"},
            }
        }
    )
    tool_step = env.step(tool_action)
    assert not tool_step["done"]
    assert tool_step["metadata"]["tool_called"] == "fmp.fmp_get_quote"
    assert tool_step["metadata"]["reward_breakdown"]["r_extract"] >= 0

    final_action = json.dumps({"final_answer": "The price is 101.23 and volume 123456."})
    final_step = env.step(final_action)
    assert final_step["done"]
    assert final_step["metadata"]["final_answer"].startswith("The price")
    assert "terminal_breakdown" in final_step["metadata"]
