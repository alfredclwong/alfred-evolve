import random
from dataclasses import dataclass

MOCK_DIFFS = [
    "diff1\n",
    "diff2\n",
    "diff3\n",
    "diff4\n",
    "diff5\n",
]


@dataclass(frozen=True)
class DiffGeneratorConfig:
    pass


class DiffGenerator:
    def __init__(self, cfg: DiffGeneratorConfig):
        self.cfg = cfg

    def generate(self, prompt: str) -> str:
        return random.choice(MOCK_DIFFS)
