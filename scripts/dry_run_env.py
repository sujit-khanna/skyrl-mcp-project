from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from src.envs.mcp_tool_env import EnvironmentConfig, MCPToolEnv
from src.envs.tool_group import MCPToolGroup
from src.utils.tool_manager import ToolManager
CONFIG_DIR = ROOT / "mcp_servers" / "configs"
DATA_PATH = ROOT / "data" / "processed" / "test_phase1_single.json"

SERVERS: List[Tuple[str, str, int]] = [
    ("polygon", "src.mcp_tools.polygon_server", 7001),
    ("fmp", "src.mcp_tools.fmp_server", 7002),
    ("tavily", "src.mcp_tools.tavily_server", 7003),
    ("python", "src.mcp_tools.python_execution_server", 7004),
    ("slack", "src.mcp_tools.slack_server", 7005),
]


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def start_servers() -> List[subprocess.Popen]:
    processes: List[subprocess.Popen] = []
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    for name, module, port in SERVERS:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            f"{module}:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(proc)
    # Allow servers to warm up
    time.sleep(3.0)
    return processes


def stop_servers(processes: List[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def to_openai_messages(observation):
    messages = []
    for msg in observation:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "tool":
            # Represent tool output as a user message prefixed for clarity
            content = f"[tool-response]\n{content}"
            role = "user"
        messages.append({"role": role, "content": content})
    guidance = (
        "When you want to call a tool, respond with a single JSON object of the form "
        "{\"tool_call\": {\"server\": \"<server-name>\", \"tool\": \"<tool-name>\", \"params\": {...}}}. "
        "Use one tool call per response. When you are ready to answer, respond with {\"final_answer\": \"...\"} "
        "and no additional text."
    )
    messages.append({"role": "system", "content": guidance})
    return messages


def run_episode(client: OpenAI, env: MCPToolEnv, max_turns: int = 8) -> Dict[str, Any]:
    observation, info = env.init()
    print("Episode task:", info.get("task_id"))
    done = False
    turn = 0
    total_reward = 0.0
    steps: List[Dict[str, Any]] = []
    final_metadata: Dict[str, Any] | None = None

    while not done and turn < max_turns:
        messages = to_openai_messages(observation)
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
        )
        action = response.choices[0].message.content.strip()
        print(f"\n=== Turn {turn + 1} Model Action ===\n{action}\n")

        step = env.step(action)
        reward = step["reward"]
        total_reward += reward
        done = step["done"]
        metadata = step["metadata"]
        observation = step["observations"]

        print(f"Reward: {reward:.4f}, cumulative: {total_reward:.4f}")
        if metadata.get("tool_called"):
            print("Tool invoked:", metadata["tool_called"])
        if metadata.get("error"):
            print("Error:", metadata["error"])
        if done:
            print("Episode finished. Final metadata:")
            print(json.dumps(metadata, indent=2))

        steps.append(
            {
                "turn": turn + 1,
                "messages": messages,
                "action": action,
                "reward": reward,
                "cumulative_reward": total_reward,
                "metadata": metadata,
            }
        )
        final_metadata = metadata
        turn += 1

    return {
        "task_id": info.get("task_id"),
        "max_turns": max_turns,
        "total_reward": total_reward,
        "terminated": done,
        "num_steps": len(steps),
        "steps": steps,
        "final_metadata": final_metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a dry-run episode through MCPToolEnv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results. Defaults to runs/dry_runs/dry_run_<timestamp>.json",
    )
    args = parser.parse_args()

    load_env()
    client = OpenAI()

    with DATA_PATH.open("r", encoding="utf-8") as fh:
        samples = json.load(fh)
    sample = samples[0]

    processes = start_servers()
    try:
        manager = ToolManager.from_config_dir(str(CONFIG_DIR))
        tool_group = MCPToolGroup(manager)
        env = MCPToolEnv(sample, tool_group=tool_group, config=EnvironmentConfig())
        try:
            episode_result = run_episode(
                client,
                env,
                max_turns=sample["reward_spec"]["ground_truth"].get("max_turns", 8),
            )
            output_path = args.output
            if output_path is None:
                stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                output_path = ROOT / "runs" / "dry_runs" / f"dry_run_{stamp}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(episode_result, indent=2), encoding="utf-8")
            print(f"\nSaved dry-run transcript to {output_path}")
        finally:
            env.close()
    finally:
        stop_servers(processes)


if __name__ == "__main__":
    main()
