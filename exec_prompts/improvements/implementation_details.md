Below is a **single, comprehensive implementation document** you can paste into your repo as
`docs/IMPLEMENTATION_GUIDE_LONG_HORIZON.md`. It is written for an AI coding agent (Cursor/Copilot/Codex) and for human developers. It covers:

* **What to build and why (math + RL intent)**
* **Exact changes to `src/dataset/llm/generate_with_llm.py`** so the generator:

  1. asks an LLM to propose a multi‑turn plan,
  2. **executes** that plan over MCP tools,
  3. performs **step‑wise analysis** (extract/compute/select/accept_if),
  4. composes a **reference final answer**, and
  5. writes a **SkyRL‑compatible dataset item** including `final_reference` and `judge_rubric`.
* **A SkyRL environment** that:

  1. parses tool calls and the final text answer,
  2. computes dense per‑turn **heuristic rewards** for tools + analysis,
  3. computes a terminal **LLM‑as‑a‑Judge (LAJ)** reward for the final text,
  4. returns `BaseTextEnvStepOutput`.
* **Configs, tests, and a runbook**.

> **Note:** Earlier attachments may have expired. If you want me to produce line‑exact patches against your current files, please re‑upload the latest `generate_with_llm.py` and your env file (or confirm its path), and I’ll generate a ready‑to‑apply patch.

---

# Long‑Horizon Multi‑Tool RL: Implementation Guide

**Repository:** `https://github.com/sujit-khanna/skyrl-mcp-project`
**Primary files touched:**

* `src/dataset/llm/generate_with_llm.py` (**major**)
* `src/envs/mcp_tool_env.py` (**new or major**)
* `src/envs/reward_functions.py` (**new**)
* `src/utils/tool_manager.py` (wire MCP) (**minor**)
* `src/training/configs/{env,rollout,algo}.yaml` (**minor**)
* `tests/` (**new** validators & smoke tests)

---

## 0) Why these changes (short math)

We want a policy (\pi_\theta) that **uses tools correctly** across multiple turns and **writes a high‑quality final text answer** grounded in tool outputs. We assign rewards:

* **Turn‑level rewards** (r_t) for correct tool choice, argument binding, extraction, compute/select, and accept checks.
* **Terminal reward** (r_T = \lambda_h,r_{\text{heur}}(y_T) + \lambda_j,r_{\text{LAJ}}(y_T, y^*))
  where (y_T) is the policy’s final text and (y^*) is a **reference** answer built by the data generator after executing the plan.
  LAJ returns a numeric score; PPO/GRPO maximizes (\mathbb{E}_\pi[\sum_t r_t]), learning both tools and writing.

This is fully compatible with SkyRL’s multi‑turn `BaseTextEnv` and dataset contract.

---

## 1) Data Format (Target)

Each dataset **item** (one task) looks like:

```json
{
  "data_source": "synthetic/llm",
  "env_class": "MCPToolEnv",
  "prompt": [
    {"role":"system","content":"... tool usage policy ..."},
    {"role":"user","content":"... natural language task ..."}
  ],
  "reward_spec": {
    "method": "rule",
    "ground_truth": {
      "task_id": "string",
      "complexity": "simple|moderate|complex",
      "max_turns": 8,
      "limits": {"max_servers":3,"max_tools":10},

      "tool_sequence": [
        {
          "step": 1,
          "server": "DuckDuckGo",
          "tool": "search",
          "params": {"query": "NASDAQ-100 components", "max_results": 1},
          "analysis_requirements": {
            "extract": ["tickers_url"],
            "compute": [],
            "select": [],
            "accept_if": ["tickers_url ~= '^https?://.*'"],
            "next_args_from": "tickers_url"
          }
        }
        // ... more steps ...
      ],

      "analysis_rubric": {
        "steps": [ /* same length as tool_sequence, see above */ ],
        "final_answer_requirements": {
          "format": "markdown",
          "must_include": ["top3","neg_titles"],
          "grounded_from": ["top3","title_sentiment_map"],
          "quality_criteria": ["no hallucinations","concise (<=120 words)"]
        }
      },

      "final_reference": {
        "answer_text": "Top-3: NVDA, AMD, META. Negative headlines: ...",
        "facts": {"top3":["NVDA","AMD","META"], "neg_titles":["...","..."]},
        "citations": {"top3":[4], "neg_titles":[9]}
      },

      "judge_rubric": {
        "weights": { "coverage":0.35, "grounding":0.4, "clarity":0.15, "safety":0.1 },
        "target_length_range": [40,140],
        "schema": {
          "type":"object",
          "properties":{
            "coverage":{"type":"number","minimum":0,"maximum":1},
            "grounding":{"type":"number","minimum":0,"maximum":1},
            "clarity":{"type":"number","minimum":0,"maximum":1},
            "safety":{"type":"number","minimum":0,"maximum":1},
            "total":{"type":"number","minimum":0,"maximum":1}
          },
          "required":["coverage","grounding","clarity","safety","total"]
        }
      }
    }
  },
  "extra_info": {"scenario":{"scenario":"long_horizon_news","turns":10}}
}
```

**Invariant:** keep `prompt` **compact** (system + first user). All multi‑turn supervision lives in `reward_spec.ground_truth` for the **environment**.

---

## 2) Generator: `src/dataset/llm/generate_with_llm.py`

### 2.1 Responsibilities

1. Ask an LLM to propose a **multi‑turn plan** (`tool_sequence`) with **per‑step analysis** (`analysis_requirements`).
2. **Verify/repair** plan (chaining, step counts).
3. **Execute** the plan over MCP tools with a **safe DSL**:
   `extract`, `compute`, `select`, `accept_if`, `next_args_from`.
4. Compose a **reference final answer** from the **derived state**.
5. Write the **dataset item** with `final_reference` and `judge_rubric`.

### 2.2 Code: New Helpers & Data Types

Create a small execution record and a **safe DSL**.
**Add these near the top**, after your existing imports:

```python
# --- New imports ---
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from copy import deepcopy
import json, os, re

# Reuse your MCP ToolManager if present, otherwise stub
try:
    from src.utils.tool_manager import ToolManager
    MCP_AVAILABLE = True
except Exception:
    MCP_AVAILABLE = False
    ToolManager = None

@dataclass
class ExecStep:
    step: int
    tool_fqn: str
    args: dict
    result_summary: dict
    accept_pass: bool
    checks: dict

@dataclass
class ExecOut:
    state: dict
    steps: List[ExecStep]
```

**Safe DSL** (deterministic; no arbitrary `eval`):

```python
def _extract_path(result: Any, path: str) -> Tuple[Optional[Any], bool]:
    try:
        if path.endswith("[]"):
            key = path[:-2]; return result.get(key, []), True
        if "[][title]" in path:
            key = path.split("[]")[0]
            items = result.get(key, []); return [it.get("title") for it in items if isinstance(it, dict)], True
        if "{title->score}" in path:
            base = path.split("{")[0]; items = result.get(base, [])
            return {it["title"]: it.get("score", 0.0) for it in items if isinstance(it, dict) and "title" in it}, True
        return result.get(path), path in result
    except Exception:
        return None, False

def _compute(expr: str, state: dict) -> dict:
    out = {}
    name, rhs = [s.strip() for s in expr.split("=", 1)]

    def pct_change_last_day(price_json):
        pct = {}
        for k, arr in price_json.items():
            if len(arr) >= 2 and "close" in arr[-1] and "close" in arr[-2]:
                a, b = float(arr[-1]["close"]), float(arr[-2]["close"])
                if b != 0: pct[k] = a / b - 1.0
        return pct
    def topk(d: dict, k: int): return [k_ for k_, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]]
    def head(lst: list, n: int): return lst[:n]
    def unique(lst: list): return list(dict.fromkeys(lst))
    def concat(*lsts): out=[]; [out.extend(_l) for _l in lsts]; return out
    def count_keys(d: dict): return len(d) if isinstance(d, dict) else 0
    def regex_extract_all(pattern: str, text: str):
        return re.findall(pattern, text or "")

    safe_ns = {
        **deepcopy(state),
        "pct_change_last_day": pct_change_last_day,
        "topk": topk, "head": head, "unique": unique, "concat": concat,
        "count_keys": count_keys, "regex_extract_all": regex_extract_all
    }
    value = eval(rhs, {"__builtins__": {}}, safe_ns)  # controlled env
    out[name] = value
    return out

def _check(cond: str, state: dict) -> bool:
    try:
        if " ~=" in cond:
            lhs, pattern = [s.strip() for s in cond.split("~=", 1)]
            return re.search(pattern.strip("'\""), str(eval(lhs, {"__builtins__": {}}, state))) is not None
        return bool(eval(cond, {"__builtins__": {}}, state))
    except Exception:
        return False

def _resolve_placeholders(obj: Any, state: dict) -> Any:
    if isinstance(obj, str):
        def repl(m):
            key = m.group(1)
            try: return str(eval(key, {"__builtins__": {}}, state))
            except Exception: return m.group(0)
        return re.sub(r"\$\{([^}]+)\}", repl, obj)
    if isinstance(obj, dict): return {k: _resolve_placeholders(v, state) for k, v in obj.items()}
    if isinstance(obj, list): return [_resolve_placeholders(v, state) for v in obj]
    return obj
```

### 2.3 Call Planner LLM (unchanged logic, **stronger schema**)

Augment your **task schema** to **require**:

* `analysis_requirements` per step (`extract|compute|select|accept_if|next_args_from`)
* `final_answer_requirements`
* `judge_rubric`

*(Omitted here for brevity—use the schema block you and I finalized; copy it into your file.)*

Also **tighten the prompt** you send: enforce step counts by complexity (simple: 2–4, moderate: 4–8, complex: 8–12), forbid variables not derived earlier, demand `final_answer_requirements` and a numeric `judge_rubric`.

### 2.4 Verify/Repair + Execute Plan

Add:

```python
async def simulate_plan_and_collect(task: dict, tm: Optional[ToolManager]) -> ExecOut:
    state: dict = {}
    exec_steps: List[ExecStep] = []
    for step_obj in task["tool_sequence"]:
        step = int(step_obj["step"])
        tool_fqn = f'{step_obj["server"]}.{step_obj["tool"]}'
        params = _resolve_placeholders(step_obj.get("params", {}), state)

        if tm is None:
            result = {"ok": True, "echo": params}  # offline stub
        else:
            result = await tm.execute_tool(tool_fqn, params, timeout=20.0)

        ar = step_obj.get("analysis_requirements", {})
        updates, missing, accept = {}, [], True

        for need in ar.get("extract", []):
            val, ok = _extract_path(result, need)
            if ok:
                key = need.split("[")[0].split("{")[0]
                updates[key] = val
            else:
                missing.append(need); accept = False

        for expr in ar.get("compute", []):
            try: updates.update(_compute(expr, {**state, **updates}))
            except Exception: accept = False
        for expr in ar.get("select", []):
            try: updates.update(_compute(expr, {**state, **updates}))
            except Exception: accept = False

        for cond in ar.get("accept_if", []):
            if not _check(cond, {**state, **updates}): accept = False

        state.update(updates)
        exec_steps.append(ExecStep(
            step=step, tool_fqn=tool_fqn, args=params,
            result_summary={"keys": list(result)[:10]},
            accept_pass=accept,
            checks={"missing": missing, "updated": list(updates.keys())}
        ))
    return ExecOut(state=state, steps=exec_steps)
```

### 2.5 Compose the Reference Final Answer

Either **template** or **LLM**; keep it grounded in `state`:

```python
async def compose_reference_answer(task: dict, exec_out: ExecOut, client) -> dict:
    far = task["final_answer_requirements"]
    facts = {name: exec_out.state.get(name) for name in far.get("grounded_from", [])}

    # Template option (deterministic)
    if far["format"] in ("text","markdown"):
        answer_text = ""
        if "top3" in facts: answer_text += f"Top-3: {', '.join(facts['top3'])}. "
        if "neg_titles" in facts: answer_text += f"Negative headlines: {', '.join(facts['neg_titles'])}."
    else:
        answer_text = json.dumps(facts)

    # Map fact -> last step that updated it (citations)
    name_to_step = {}
    for s in reversed(exec_out.steps):
        for k in exec_out.state:
            if k not in name_to_step and k in s.checks.get("updated", []):
                name_to_step[k] = s.step
    citations = {k: [v] for k, v in name_to_step.items() if k in facts}

    return {"answer_text": answer_text, "facts": facts, "citations": citations}
```

(If you prefer an LLM to phrase the final answer: call your Responses/Chat model with the **facts** and **must_include**, and force structured output or a short text band; it must remain **grounded**.)

### 2.6 Thread it into Your Pipeline

Inside your `_one_task(...)` or equivalent **per‑item** generation routine:

```python
task = await _call_llm_with_schema(..., TASK_SCHEMA, ...)  # planner
task = _verify_and_repair(task)                             # ensure chaining & counts

tm = ToolManager() if MCP_AVAILABLE else None
exec_out = await simulate_plan_and_collect(task, tm)

final_ref = await compose_reference_answer(task, exec_out, client=None)  # or pass LLM client
task["_final_reference"] = final_ref    # keep small; don't store huge tool results
task["_exec_out"] = {"steps": [asdict(s) for s in exec_out.steps]}  # optional breadcrumbs
```

### 2.7 Write the SkyRL Dataset Item

```python
def to_skyrl_sample(task: dict, system_prompt: str, env_class: str="MCPToolEnv") -> dict:
    return {
      "data_source": "synthetic/llm",
      "env_class": env_class,
      "prompt": [
        {"role":"system","content": system_prompt},
        {"role":"user",  "content": task["user_prompt"]}
      ],
      "reward_spec": {
        "method": "rule",
        "ground_truth": {
          "task_id": task["task_id"],
          "complexity": task["complexity"],
          "max_turns": task["max_turns"],
          "limits": task.get("limits", {}),
          "tool_sequence": task["tool_sequence"],
          "analysis_rubric": {
            "steps": [
              {"step": s["step"], **s["analysis_requirements"]}
              for s in task["tool_sequence"]
            ],
            "final_answer_requirements": task["final_answer_requirements"]
          },
          "final_reference": task["_final_reference"],
          "judge_rubric": task["judge_rubric"]
        }
      },
      "extra_info": {"gen": {"backend": task.get("_backend"), "model": task.get("_model")}}
    }
```

---

## 3) Environment: `src/envs/mcp_tool_env.py`

### 3.1 Responsibilities

* Subclass **`BaseTextEnv`**.
* Accept **actions** as:

  * **Tool call** (JSON): `{"tool":"server.tool","arguments":{...}}`
  * **Final answer** (JSON): `{"final_answer":"…text…"}`
  * *(Optional)* Allow tag forms (`<tool>...`, `<answer>...>`) like SkyRL’s Search example by adding stop strings in configs.
* For tool steps: call tool, update **state** via DSL, compute **per‑turn reward**.
* For final: compute **heuristic** + **LAJ** reward, return `done=True`.

### 3.2 Code Skeleton

```python
# src/envs/mcp_tool_env.py
from typing import Any, Dict, Tuple, List
import json, re, os, hashlib
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from openai import AsyncOpenAI

from src.utils.tool_manager import ToolManager  # your MCP bridge
from src.envs.reward_functions import (
    extract_path, compute_dsl, check_cond, resolve_placeholders,
    score_tool_step_heuristic,
    score_final_answer_heuristic,
    score_final_answer_laj
)

class MCPToolEnv(BaseTextEnv):
    def __init__(self, env_config: Dict[str,Any] = None):
        super().__init__()
        cfg = env_config or {}
        self.max_turns = int(cfg.get("max_turns", 8))
        self.weights   = cfg.get("reward_weights", {
            "tool_name": 0.2, "param_binding": 0.15, "extract": 0.15,
            "compute": 0.15, "accept_if": 0.1, "penalty": -0.1,
            "final_heur": 0.6, "final_laj": 0.4
        })
        self.tm = ToolManager()
        self.judge = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.turn = 0
        self.gt = None
        self.state = {}
        self._laj_cache = {}

    # ---------- BaseTextEnv API ----------
    def init(self, prompt, ground_truth=None) -> Tuple[list, dict]:
        self.turn = 0
        self.state = {}
        self.gt = ground_truth or {}
        return prompt, {"task_id": self.gt.get("task_id","unknown")}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turn += 1
        kind, payload = self._parse_action(action)

        if kind == "tool":
            tool_fqn = payload["name"]
            args     = payload.get("arguments", {})
            # Execute tool (no placeholder resolution here; model must provide args already resolved)
            try:
                group, tool = tool_fqn.split(".", 1)
                result = self._execute_tool(group, tool, [args])
            except Exception as e:
                return BaseTextEnvStepOutput(
                    observations=[{"role":"user","content": json.dumps({"error":str(e)})}],
                    reward=self.weights["penalty"],
                    done=(self.turn>=self.max_turns),
                    metadata={"error":"tool_exec", "exception":str(e)}
                )

            # Score tool step (updates self.state via DSL)
            step_idx = self._match_step(tool_fqn)
            r_step, breakdown = score_tool_step_heuristic(
                step_idx=step_idx, tool_fqn=tool_fqn, args=args, result=result,
                gt=self.gt, state=self.state, weights=self.weights
            )
            return BaseTextEnvStepOutput(
                observations=[{"role":"user","content": json.dumps(result)[:2048]}],
                reward=r_step,
                done=(self.turn>=self.max_turns),
                metadata={"step": step_idx, **breakdown}
            )

        # Final text branch
        final_text = payload
        # Heuristic
        r_heur, heur_det = score_final_answer_heuristic(
            text=final_text, gt=self.gt, state=self.state
        )
        # LAJ (cached)
        key = hashlib.md5((self.gt["task_id"] + final_text).encode()).hexdigest()
        if key in self._laj_cache:
            r_laj, laj_det = self._laj_cache[key]
        else:
            r_laj, laj_det = score_final_answer_laj(
                client=self.judge, text=final_text, gt=self.gt, state=self.state
            )
            self._laj_cache[key] = (r_laj, laj_det)

        total = self.weights["final_heur"]*r_heur + self.weights["final_laj"]*r_laj
        return BaseTextEnvStepOutput(
            observations=[], reward=float(total), done=True,
            metadata={"final":{"heur":heur_det,"laj":laj_det}}
        )

    # ---------- Helpers ----------
    def _match_step(self, tool_fqn: str) -> int:
        for s in self.gt.get("tool_sequence", []):
            if f'{s["server"]}.{s["tool"]}' == tool_fqn: return int(s["step"])
        return self.turn

    def _parse_action(self, text: str):
        try:
            obj = json.loads(text)
            if "tool" in obj: return "tool", {"name": obj["tool"], "arguments": obj.get("arguments", {})}
            if "final_answer" in obj: return "final", obj["final_answer"]
        except Exception:
            pass
        if "<answer>" in text and "</answer>" in text:
            return "final", text.split("<answer>")[1].split("</answer>")[0].strip()
        if "<tool>" in text and "</tool>" in text:
            inner = text.split("<tool>")[1].split("</tool>")[0]
            m = re.search(r"<([a-zA-Z0-9_.-]+)>(.*)</\\1>", inner, re.DOTALL)
            if m:
                name, args_text = m.group(1), m.group(2).strip()
                try: args = json.loads(args_text)
                except Exception: args = {"raw": args_text}
                return "tool", {"name": name, "arguments": args}
        return "final", text.strip()
```

### 3.3 Reward Functions: `src/envs/reward_functions.py`

Centralize reusable scoring/DSL here so both **generator** and **env** stay in sync.

```python
# src/envs/reward_functions.py
from typing import Tuple, Dict, Any
import json, re

# ---- Re-export DSL helpers if you want a single source of truth ----
from src.dataset.llm.generate_with_llm import (
    _extract_path as extract_path,
    _compute as compute_dsl,
    _check as check_cond,
    _resolve_placeholders as resolve_placeholders
)

def score_tool_step_heuristic(step_idx: int, tool_fqn: str, args: dict, result: dict,
                              gt: dict, state: dict, weights: dict) -> Tuple[float, Dict[str,Any]]:
    reward = 0.0
    breakdown = {"tool": tool_fqn, "args": args, "accept_if": []}

    # Expected tool for this step?
    expected_fqn = None
    for st in gt["tool_sequence"]:
        if int(st["step"]) == step_idx:
            expected_fqn = f'{st["server"]}.{st["tool"]}'; break
    if expected_fqn == tool_fqn:
        reward += weights["tool_name"]

    # Rubric for this step
    ar = None
    for s in gt["analysis_rubric"]["steps"]:
        if int(s["step"]) == step_idx: ar = s; break
    if not ar: return reward, {"warn": "no_rubric"}

    # Param binding (did the args include the required symbol?)
    naf = ar.get("next_args_from")
    if naf and naf in json.dumps(args):
        reward += weights["param_binding"]

    # Extract → update state
    ext_ok = True
    for need in ar.get("extract", []):
        val, ok = extract_path(result, need)
        if not ok: ext_ok = False
        else:
            key = need.split("[")[0].split("{")[0]
            state[key] = val
    if ext_ok:
        reward += weights["extract"]

    # Compute & Select → update state
    try:
        for expr in ar.get("compute", []): state.update(compute_dsl(expr, state))
        for expr in ar.get("select",  []): state.update(compute_dsl(expr, state))
        reward += weights["compute"]
    except Exception:
        pass

    # accept_if
    pass_all = True
    for cond in ar.get("accept_if", []):
        ok = check_cond(cond, state); pass_all = pass_all and ok
        breakdown["accept_if"].append({"cond": cond, "ok": ok})
    if pass_all:
        reward += weights["accept_if"]

    return float(reward), breakdown

def score_final_answer_heuristic(text: str, gt: dict, state: dict) -> Tuple[float, Dict[str,Any]]:
    far = gt["analysis_rubric"]["final_answer_requirements"]
    jr  = gt["judge_rubric"]
    weights = jr.get("weights", {})
    lo, hi = (jr.get("target_length_range") or [0, 10**9])

    # coverage
    must = far.get("must_include", [])
    cov_hits = sum(1 for k in must if k in text or k in json.dumps(state))
    coverage = cov_hits / max(1, len(must))

    # grounding
    facts = gt["final_reference"]["facts"]
    grounding = _grounding_score(text, facts)

    # clarity
    words = len(text.split())
    clarity = 1.0 if lo <= words <= hi else 0.5 if (0.7*lo)<=words<=(1.5*hi) else 0.0

    # safety
    safety = 1.0 if not re.search(r"\b(SSN|password|api_key)\b", text, re.I) else 0.0

    total = (weights.get("coverage",0)*coverage +
             weights.get("grounding",0)*grounding +
             weights.get("clarity",0)*clarity +
             weights.get("safety",0)*safety)
    return float(total), {"coverage":coverage,"grounding":grounding,"clarity":clarity,"safety":safety}

def _grounding_score(text: str, facts: dict) -> float:
    if "top3" in facts:
        m = set(re.findall(r"\b[A-Z]{1,5}\b", text))
        t = set(facts["top3"])
        if not m: return 0.5
        return 1.0 if all(x in t for x in m if x.isupper()) else 0.0
    return 0.5

async def score_final_answer_laj(client, text: str, gt: dict, state: dict) -> Tuple[float, Dict[str,Any]]:
    schema = gt["judge_rubric"]["schema"]
    payload = {
        "instructions": ["Evaluate FINAL vs reference & facts. Return ONLY JSON per schema."],
        "facts": gt["final_reference"]["facts"],
        "reference": gt["final_reference"]["answer_text"],
        "final": text
    }
    resp = await client.chat.completions.create(
        model=os.getenv("OPENAI_JUDGE_MODEL","gpt-4o-mini"),
        response_format={"type":"json_schema", "json_schema":{"name":"judge_schema","schema":schema}},
        messages=[{"role":"system","content":"You are a strict evaluator."},
                  {"role":"user","content": json.dumps(payload)}],
        temperature=0.0
    )
    data = json.loads(resp.choices[0].message.content)
    return float(data.get("total", 0.0)), data
```

---

## 4) Config & Training Integration

* If you adopt **tag‑style** actions (optional), add sampler stop strings:

  ```yaml
  generator:
    sampling_params:
      stop: ["</tool>", "</answer>"]
  ```
* Set `max_turns` to `ground_truth.max_turns`.
* Start training with **heuristics‑heavy** final reward (`final_heur` weight high, e.g., 0.7) and **LAJ** lower (0.3). Increase LAJ as behavior stabilizes to emphasize textual quality.
* Ensure rollouts **capture logprobs** (needed by PPO/GRPO).
* Optionally refresh LoRA / update policy weights visible to vLLM after optimizer steps.

---

## 5) Tests (CI‑friendly)

1. **Dataset Validator** (`tests/test_dataset_schema.py`)

   * Check each item has `env_class`, `prompt[0..1]`, `reward_spec.method`, `ground_truth.tool_sequence`, `analysis_rubric.steps` length matches `tool_sequence`, and `final_reference`.
   * Verify `${var}` placeholders in `params` are introduced in previous steps via extract/compute/select.

2. **DSL Unit Tests** (`tests/test_dsl.py`)

   * `extract_path` on list, map, array‑of‑objects.
   * `compute_dsl` for `pct_change_last_day`, `topk`, `unique`, `regex_extract_all`.
   * `check_cond` for regex and arithmetic.

3. **Env Step Tests** (`tests/test_env_step.py`)

   * Provide a minimal `gt` and a fake ToolGroup that returns canned results.
   * Assert reward components change when args omit `next_args_from` vs include it.

4. **Final Answer Scoring** (`tests/test_final_scoring.py`)

   * With `facts = {"top3":["NVDA","AMD","META"]}`, test coverage/grounding behavior for a few strings.

5. **Judge Mock** (`tests/test_laj_mock.py`)

   * Monkeypatch `AsyncOpenAI` to return a fixed JSON (e.g., total=0.8); verify env combines heur+LAJ by weights.

---

## 6) Runbook

**Generate updated dataset (with final_reference):**

```bash
# ensure your .env has MCP keys + OPENAI_API_KEY
bash scripts/generate_llm_dataset.sh
# output to data/processed/train_llm.json
```

**Quick validation:**

```bash
python -m scripts.verify_setup.py  # add checks for dataset schema + env import
pytest -q
```

**Train (GRPO/PPO):**

```bash
bash scripts/train_grpo.sh
# watch W&B: policy_loss, value_loss, KL, r_tool_step, r_final_heur, r_final_laj
```

---

## 7) Observability & Debugging

* Log per‑turn `metadata`:

  * expected vs used tool, `next_args_from` symbol, which `extract` keys were found, which `accept_if` failed.
* Log final breakdown:

  * heuristic components (coverage/grounding/clarity/safety),
  * LAJ JSON,
  * total reward composition.
* Add a `trace_id` combining `task_id` and episode number for cross‑referencing dataset and rollout logs.

---

## 8) Frequently Asked Questions

**Q: Can the reference leak into observations?**
A: No. Do **not** put `final_reference` into env observations; it is reward‑only.

**Q: Cost control for LAJ?**
A: Cache LAJ by `(task_id, hash(final_text))`. Use heuristics‑only for early training; enable LAJ on a subset of episodes or every N updates.

**Q: Does this break SkyRL’s dataset contract?**
A: No. The dataset remains **compact prompts** with `reward_spec`. All supervision metadata is under `ground_truth`, which the env uses for rewards—standard practice.

---

## 9) Definition of Done

* ✅ Generator executes plans over MCP tools, applies the DSL, and writes `final_reference` & `judge_rubric`.
* ✅ Environment scores **every step** and the **final answer** (heuristics + LAJ), returns `BaseTextEnvStepOutput`.
* ✅ Tests pass: schema validator, DSL, env tool step, final scoring, judge mock.
* ✅ Training can run end‑to‑end on a small shard, W&B shows non‑zero dense rewards and a final reward distribution.

---

## 10) Copy‑Paste Tasks for Cursor/Codex

**Task A — Update Generator**

1. Open `src/dataset/llm/generate_with_llm.py`.
2. Insert the data classes `ExecStep`, `ExecOut` and DSL helpers.
3. Strengthen `TASK_SCHEMA` to include `analysis_requirements`, `final_answer_requirements`, `judge_rubric`.
4. Add `simulate_plan_and_collect()` and `compose_reference_answer()`.
5. In your main generation loop, after planning, call both functions and store `_final_reference` in `task`.
6. Update `to_skyrl_sample()` to emit `final_reference` and `judge_rubric`.

**Task B — Add Reward Functions**

1. Create `src/envs/reward_functions.py` with the code above.
2. Ensure it re‑exports DSL helpers or references the same implementations.

**Task C — Create/Update Env**

1. Create/adjust `src/envs/mcp_tool_env.py` with the `MCPToolEnv` class.
2. Wire `ToolManager` to your MCP servers.
3. Ensure `step()` differentiates tool vs final and computes rewards.

**Task D — Tests**

1. Add validators in `tests/` as described.
2. Add a fake ToolGroup to return canned outputs for fast unit tests.

**Task E — Config**

1. If using tags, add sampler `stop` strings in your generator config.
2. Start with `final_heur: 0.7, final_laj: 0.3`.

---

### Appendix A — Two sample dataset items (ready to paste)

**NAS100 news triage (complex)** and **S3 errors (complex)** were already provided in prior messages; they conform to the exact structure above. If you want, I can regenerate them with your **actual tool names** and **params** once you confirm the MCP server contracts exposed by `ToolManager`.

---

If you’d like me to turn this into a **ready‑to‑apply PR** (with new files, imports, and unit tests), re‑upload your latest `generate_with_llm.py` and any env stubs you already have.
