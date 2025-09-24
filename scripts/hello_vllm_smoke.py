import os
try:
    from vllm import LLM, SamplingParams
except Exception as e:
    print("[vLLM] not installed or GPU not available:", e)
    raise SystemExit(0)
model = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
print("[vLLM] initializing:", model)
llm = LLM(model=model, trust_remote_code=True, max_seq_len=int(os.environ.get("VLLM_MAX_MODEL_LEN","4096")), gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION","0.30")))
sp = SamplingParams(max_tokens=16, temperature=0.7, top_p=0.95, logprobs=1)
out = llm.generate(["Hello from vLLM!"], sp)
o = out[0].outputs[0]
print("[vLLM] text:", o.text.strip())
print("[vLLM] tokens:", len(o.token_ids), "logprobs:", len(o.logprobs))
print("✅ vLLM smoke OK")
