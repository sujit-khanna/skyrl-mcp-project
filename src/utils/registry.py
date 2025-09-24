from __future__ import annotations
from typing import Any, Tuple

def get_trainer_classes():
    attempts = [
        ("skyrl_train.trainer", "Trainer", "TrainerConfig"),
        ("skyrl_train.train", "Trainer", "TrainerConfig"),
        ("skyrl.trainer", "Trainer", "TrainerConfig"),
    ]
    last_err = None
    for mod, T, C in attempts:
        try:
            m = __import__(mod, fromlist=[T, C])
            return getattr(m, T), getattr(m, C)
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import Trainer from skyrl_train. Last error: {last_err}")

def get_grpo_classes():
    attempts = [
        ("skyrl_train.algorithms", "GRPO", "GRPOConfig"),
        ("skyrl_train.algo", "GRPO", "GRPOConfig"),
        ("skyrl.algorithms", "GRPO", "GRPOConfig"),
    ]
    last_err = None
    for mod, A, C in attempts:
        try:
            m = __import__(mod, fromlist=[A, C])
            return getattr(m, A), getattr(m, C)
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import GRPO from skyrl_train. Last error: {last_err}")

def get_value_model_class():
    attempts = [
        ("skyrl_train.models", "AutoModelForCausalLMWithValueHead"),
        ("skyrl.models", "AutoModelForCausalLMWithValueHead"),
    ]
    last_err = None
    for mod, Cls in attempts:
        try:
            m = __import__(mod, fromlist=[Cls])
            return getattr(m, Cls)
        except Exception as e:
            last_err = e
    try:
        from trl import AutoModelForCausalLMWithValueHead  # fallback
        return AutoModelForCausalLMWithValueHead
    except Exception as e:
        raise ImportError(f"Could not resolve value-head model: {last_err} / {e}")

def set_global_seed(seed: int):
    try:
        import random, numpy as np, torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
