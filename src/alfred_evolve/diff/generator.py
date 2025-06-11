from dataclasses import dataclass

from alfred_evolve.primitive import Diff, Prompt


@dataclass(frozen=True)
class DiffGeneratorConfig:
    pass


class DiffGenerator:
    def __init__(self, cfg: DiffGeneratorConfig):
        self.cfg = cfg

    def generate(self, prompt: Prompt) -> Diff:
        return Diff()
