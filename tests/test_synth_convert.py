from pathlib import Path
import json
from src.dataset.synth.from_dependent_specs import convert_spec_to_skyrl_samples

def test_convert_spec_to_skyrl_samples(tmp_path: Path):
    spec = {
        "dependent_dataset": [{
            "scenario": "x", "turns": 2, "limits": {},
            "tasks": [{
                "task_id": "x1", "complexity": "simple",
                "conversation": [{"role":"user","message":"Hello"}],
                "solution": {"tool_sequence":[{"step":1,"server":"yahoo_finance","tool":"get_yfinance_price_history","params":{}}]}
            }]
        }]
    }
    out = convert_spec_to_skyrl_samples(spec, env_class="MCPToolEnv")
    assert len(out) == 1
    s = out[0]
    assert s.env_class == "MCPToolEnv"
    assert s.prompt[0]["role"] == "system" and s.prompt[1]["role"] == "user"
    assert s.reward_spec["method"] == "rule"
