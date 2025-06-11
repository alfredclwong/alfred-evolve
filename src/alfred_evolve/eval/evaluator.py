from dataclasses import dataclass

from alfred_evolve.primitive import Program, Result


@dataclass(frozen=True)
class EvaluatorConfig:
    pass


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg

    def evaluate(self, program: Program) -> Result:
        return Result()
