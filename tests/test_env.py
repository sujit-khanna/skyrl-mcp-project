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
            "max_turns": 5,
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
    symbol = params["symbol"]
    price_map = {"AAPL": 101.23, "MSFT": 222.45}
    return {
        "ok": True,
        "data": {"price": price_map.get(symbol, 50.0), "volume": 123456},
        "latency_ms": 12,
    }


def test_env_tool_step_and_final_answer():
    config = EnvironmentConfig(heuristic_weights={"heur_cov": 1.0, "heur_grd": 0.0, "heur_len": 0.0})
    env = MCPToolEnv(TASK_SAMPLE, tool_executor=stub_tool_executor, config=config)
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
    tool_step_1 = env.step(tool_action)
    assert not tool_step_1["done"]
    assert tool_step_1["metadata"]["tool_called"] == "fmp.fmp_get_quote"
    assert tool_step_1["metadata"]["reward_breakdown"]["r_extract"] >= 0
    assert tool_step_1["metadata"]["reward_breakdown"].get("r_repeated", 0.0) == 0.0

    msft_action = json.dumps(
        {
            "tool_call": {
                "server": "fmp",
                "tool": "fmp_get_quote",
                "params": {"symbol": "MSFT"},
            }
        }
    )
    tool_step_2 = env.step(msft_action)
    assert not tool_step_2["done"]
    assert tool_step_2["metadata"]["reward_breakdown"].get("r_repeated", 0.0) == 0.0

    tool_step_3 = env.step(tool_action)
    assert not tool_step_3["done"]
    assert tool_step_3["metadata"]["reward_breakdown"]["r_repeated"] < 0

    final_action = json.dumps({"final_answer": "The price is 101.23 and volume 123456."})
    final_step = env.step(final_action)
    assert final_step["done"]
    assert final_step["metadata"]["final_answer"].startswith("The price")
    assert "terminal_breakdown" in final_step["metadata"]
    assert final_step["metadata"]["heuristic"]["heuristic"] == final_step["metadata"]["heuristic"]["coverage"]
