def test_llm_skyrl_sample_shape():
    from src.dataset.llm.common import to_skyrl_sample
    task = {
        "task_id": "t1",
        "complexity": "moderate",
        "user_prompt": "Get SPY close and DM me.",
        "max_turns": 6,
        "tool_sequence": [{"step":1,"server":"yahoo_finance","tool":"get_yfinance_price_history","params":{}}],
        "tools_available": ["yahoo_finance","slack"]
    }
    s = to_skyrl_sample(task, env_class="MCPToolEnv", data_source="synthetic/llm")
    assert s.prompt[0]["role"] == "system" and s.prompt[1]["role"] == "user"
    assert s.reward_spec["method"] == "rule"
    assert s.reward_spec["ground_truth"]["tool_sequence"][0]["tool"] == "get_yfinance_price_history"
