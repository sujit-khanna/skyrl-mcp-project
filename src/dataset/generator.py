from __future__ import annotations
import json, random, argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
from .task_templates import EASY_TASKS, MEDIUM_TASKS, HARD_TASKS

@dataclass
class TrainingExample:
    task_id: str
    complexity: str
    system_prompt: str
    user_prompt: str
    available_tools: List[str]
    max_turns: int
    success_criteria: Dict[str, Any]
    metadata: Dict[str, Any]

class DatasetGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _pick(self, bank): return self.rng.choice(bank)

    def generate_examples(self, n: int) -> List[TrainingExample]:
        out: List[TrainingExample] = []
        for i in range(n):
            c = self.rng.choices(["easy","medium","hard"], weights=[0.3,0.5,0.2])[0]
            bank = {"easy": EASY_TASKS, "medium": MEDIUM_TASKS, "hard": HARD_TASKS}[c]
            t = self._pick(bank)
            out.append(TrainingExample(
                task_id=f"{c[0]}{i:04d}",
                complexity=c,
                system_prompt="You are a helpful assistant with access to tools.",
                user_prompt=t["user_prompt"],
                available_tools=t.get("available_tools", []),
                max_turns=t.get("max_turns", 8),
                success_criteria=t.get("success_criteria", {}),
                metadata={"source":"v4"}
            ))
        return out

    def save(self, samples: List[TrainingExample], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(s) for s in samples], indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", type=Path, default=Path("data/processed/train.json"))
    ap.add_argument("--out-val", type=Path, default=Path("data/processed/validation.json"))
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gen = DatasetGenerator(seed=args.seed)
    gen.save(gen.generate_examples(args.n_train), args.out_train)
    gen.save(gen.generate_examples(args.n_val), args.out_val)
    print(f"✅ Wrote {args.out_train} and {args.out_val}")

if __name__ == "__main__":
    main()
