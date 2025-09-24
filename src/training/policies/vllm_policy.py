from __future__ import annotations
from typing import Any, Dict

class VLLMPolicy:
    def __init__(self, model: Any, vllm_config: Dict[str, Any], tokenizer_name: str):
        self.model = model
        self.vllm_config = vllm_config or {}
        self.tokenizer_name = tokenizer_name
        self._lora_req = None
        self._init_vllm()

    def _init_vllm(self):
        try:
            import vllm
            self.vllm = vllm.LLM(
                model=self.vllm_config.get("model_name", None) or getattr(self.model, "name_or_path", None),
                dtype=self.vllm_config.get("dtype", "bfloat16"),
                trust_remote_code=self.vllm_config.get("trust_remote_code", True),
                max_seq_len=self.vllm_config.get("max_model_len", 4096),
                gpu_memory_utilization=self.vllm_config.get("gpu_memory_utilization", 0.30),
            )
            from vllm import SamplingParams
            self.SamplingParams = SamplingParams
        except Exception as e:
            self.vllm = None
            self.SamplingParams = None
            self._init_error = e

    def refresh_lora(self, adapter_path: str, int_id: int):
        try:
            from vllm import LoraRequest
            self._lora_req = LoraRequest(
                lora_name=f"adapter_{int_id}",
                lora_int_id=int_id,
                lora_path=adapter_path
            )
        except Exception as e:
            raise RuntimeError(f"vLLM LoRA not available: {e}")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95) -> Dict[str, Any]:
        if self.vllm is None:
            raise RuntimeError(f"vLLM is not available: {getattr(self, '_init_error', 'unknown error')}")
        sp = self.SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, logprobs=1, detokenize=True)
        outs = self.vllm.generate([prompt], sp, lora_request=self._lora_req)
        o = outs[0].outputs[0]
        token_ids = o.token_ids
        token_logprobs = [t.logprob for t in o.logprobs]
        return {"text": o.text, "token_ids": token_ids, "token_logprobs": token_logprobs, "logprob_sum": float(sum(token_logprobs))}
