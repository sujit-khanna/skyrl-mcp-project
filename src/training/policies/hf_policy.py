from __future__ import annotations
from typing import Dict, Any
import torch
from transformers import AutoTokenizer

class HFPolicy:
    def __init__(self, model: Any, tokenizer_name: str):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95) -> Dict[str, Any]:
        device = next(self.model.parameters()).device
        enc = self.tokenizer(prompt, return_tensors="pt").to(device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True
        )
        gen_ids_full = out.sequences[0]
        prompt_len = enc.input_ids.shape[1]
        gen_ids_only = gen_ids_full[prompt_len:]
        scores = out.scores
        token_logprobs = []
        for step_idx, logits in enumerate(scores):
            logprobs = torch.log_softmax(logits, dim=-1)
            tok_id = gen_ids_only[step_idx].item()
            token_logprobs.append(logprobs[tok_id].item())
        text = self.tokenizer.decode(gen_ids_only, skip_special_tokens=True)
        return {
            "text": text,
            "token_ids": [t.item() for t in gen_ids_only],
            "token_logprobs": token_logprobs,
            "logprob_sum": float(sum(token_logprobs)),
        }
