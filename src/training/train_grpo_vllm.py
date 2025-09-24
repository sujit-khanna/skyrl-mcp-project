#!/usr/bin/env python3
from __future__ import annotations

import os, json, argparse, yaml
from pathlib import Path
import torch
import wandb

from src.utils.registry import get_trainer_classes, get_grpo_classes, get_value_model_class, set_global_seed
from src.envs.mcp_tool_env import MCPToolEnv, MCPEnvConfig
from src.training.policies.vllm_policy import VLLMPolicy
from src.training.policies.hf_policy import HFPolicy
from src.training.callbacks.ppo_health_callback import PPOHealthCallback

def load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--algo-config", type=Path, required=True)
    ap.add_argument("--rollout-config", type=Path, required=True)
    ap.add_argument("--train-data", type=Path, default=Path("data/processed/train.json"))
    ap.add_argument("--eval-data", type=Path, default=Path("data/processed/validation.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints"))
    ap.add_argument("--wandb-project", type=str, default="skyrl-mcp-training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", action="store_true", help="Actually run training (requires SkyRL installed)")
    args = ap.parse_args()

    set_global_seed(args.seed)

    model_cfg = load_yaml(args.model_config)
    algo_cfg = load_yaml(args.algo_config)
    roll_cfg = load_yaml(args.rollout_config)

    finetune_mode = model_cfg.get("finetune", {}).get("mode", "lora").lower()
    backend = roll_cfg.get("rollout", {}).get("backend", "auto").lower()
    if backend == "auto":
        backend = "vllm" if finetune_mode == "lora" else "hf"

    if os.getenv("WANDB_API_KEY"):
        wandb.init(project=args.wandb_project, config={"model": model_cfg, "algo": algo_cfg, "rollout": roll_cfg}, name=f"{finetune_mode}-{backend}-{args.seed}")

    # Import SkyRL classes through the shim
    Trainer, TrainerConfig = get_trainer_classes()
    GRPO, GRPOConfig = get_grpo_classes()
    ValueModel = get_value_model_class()

    # Create value-head model
    dtype = getattr(torch, model_cfg["model"].get("dtype", "bfloat16"))
    model_name = model_cfg["model"]["name"]
    model = ValueModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=model_cfg["model"].get("trust_remote_code", True))

    # LoRA vs Full toggle
    if finetune_mode == "lora" and model_cfg.get("lora", {}).get("enabled", True):
        from peft import LoraConfig, get_peft_model
        lconf = model_cfg["lora"]
        lora_cfg = LoraConfig(r=lconf["r"], lora_alpha=lconf["alpha"], lora_dropout=lconf["dropout"], target_modules=lconf["target_modules"])
        model = get_peft_model(model, lora_cfg)
        print("🔧 Finetune mode: LoRA (PEFT) enabled.")
    else:
        for p in model.parameters(): p.requires_grad = True
        print("🔧 Finetune mode: FULL fine-tune (all params).")

    # Select rollout policy backend
    if backend == "vllm":
        policy = VLLMPolicy(model=model, vllm_config=model_cfg.get("vllm", {}), tokenizer_name=model_cfg["model"]["tokenizer"])
        print("🚀 Rollout backend: vLLM")
    else:
        policy = HFPolicy(model=model, tokenizer_name=model_cfg["model"]["tokenizer"])
        print("🚀 Rollout backend: HuggingFace (HF)")

    # Algorithm config
    grpo_conf = GRPOConfig(
        gamma=algo_cfg["grpo"]["gamma"],
        lambda_gae=algo_cfg["grpo"]["lambda_gae"],
        clip_ratio=algo_cfg["grpo"]["clip_ratio"],
        value_loss_coef=algo_cfg["grpo"]["value_loss_coef"],
        kl_coef=algo_cfg["grpo"]["kl_coef"],
        target_kl=algo_cfg["grpo"]["target_kl"],
        normalize_advantages=algo_cfg["grpo"]["normalize_advantages"],
    )
    algo = GRPO(policy=policy, config=grpo_conf, optimizer_config={"lr": algo_cfg["optimization"]["learning_rate"], "gradient_clip": algo_cfg["optimization"]["gradient_clip"]})

    # Trainer config
    def env_fn():
        return MCPToolEnv(MCPEnvConfig(max_turns=roll_cfg["rollout"]["max_steps_per_episode"], tools=roll_cfg["rollout"]["tools"]["allow_list"]))

    trainer_conf = TrainerConfig(env_fn=env_fn, num_envs=roll_cfg["rollout"]["num_envs"], max_epochs=algo_cfg["training"]["max_epochs"], steps_per_epoch=algo_cfg["training"]["steps_per_epoch"], save_frequency=algo_cfg["training"]["save_frequency"], output_dir=str(args.output_dir))
    trainer = Trainer(algorithm=algo, config=trainer_conf, callbacks=[PPOHealthCallback()])

    train_data = json.loads(args.train_data.read_text()) if args.train_data.exists() else None

    if not args.run:
        print(f"ℹ️ Scaffold dry-run: mode={finetune_mode}, backend={backend}. Install SkyRL and pass --run to start training.")
    else:
        print(f"🚀 Starting SkyRL GRPO training (mode={finetune_mode}, backend={backend}) ...")
        trainer.train(train_data)
        print(f"✅ Training complete → {args.output_dir}")

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
