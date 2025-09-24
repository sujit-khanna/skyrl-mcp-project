#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
python - <<'PY'
from src.envs.mcp_tool_env import MCPToolEnv
env = MCPToolEnv()
obs, info = env.init()
print("init ->", type(obs), info)
o = env.step('{"tool":"polygon","arguments":{"market":"US"}}')
print("step ->", o["reward"], o["done"], list(o["metadata"].keys()))
PY
