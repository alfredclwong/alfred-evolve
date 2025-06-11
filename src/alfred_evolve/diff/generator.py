from dataclasses import dataclass
import random

from alfred_evolve.primitive import Diff, Prompt


MOCK_DIFFS = [
    Diff("diff1\n"),
    Diff("diff2\n"),
    Diff("diff3\n"),
    Diff("diff4\n"),
    Diff("diff5\n"),
]


@dataclass(frozen=True)
class DiffGeneratorConfig:
    pass


class DiffGenerator:
    def __init__(self, cfg: DiffGeneratorConfig):
        self.cfg = cfg

    def generate(self, prompt: Prompt) -> Diff:
        return random.choice(MOCK_DIFFS)
