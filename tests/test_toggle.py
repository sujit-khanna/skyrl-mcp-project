import yaml
from pathlib import Path
def test_toggle_config(tmp_path: Path):
    cfg = {"model":{"name":"Qwen/Qwen2.5-0.5B-Instruct","dtype":"bfloat16","tokenizer":"Qwen/Qwen2.5-0.5B-Instruct"},"finetune":{"mode":"full"}}
    p = tmp_path / "m.yaml"; p.write_text(yaml.safe_dump(cfg))
    y = yaml.safe_load(p.read_text())
    assert y["finetune"]["mode"] == "full"
