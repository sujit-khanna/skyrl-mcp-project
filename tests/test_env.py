from src.envs.mcp_tool_env import MCPToolEnv
def test_env_init_step():
    env = MCPToolEnv()
    obs, info = env.init()
    assert isinstance(obs, list) and isinstance(info, dict)
    out = env.step('{"tool":"polygon","arguments":{}}')
    assert isinstance(out["reward"], float)
