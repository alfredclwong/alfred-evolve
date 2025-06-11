import random
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluatorConfig:
    pass


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg

    def evaluate(self, program_content: str) -> dict[str, float]:
        return {"score": random.uniform(0, 1), "complexity": len(program_content) / 10000}
