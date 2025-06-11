from dataclasses import dataclass
import random

from alfred_evolve.primitive import Program, Result


@dataclass(frozen=True)
class EvaluatorConfig:
    pass


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg

    def evaluate(self, program: Program) -> Result:
        return Result({"score": random.uniform(0, 1)})
