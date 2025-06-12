import random
from dataclasses import dataclass

from alfred_evolve.database.base import Program
from alfred_evolve.prompt.template import PREMABLE, EPILOGUE, VARIATIONS, ISLAND_VARIATIONS


@dataclass(frozen=True)
class PromptSamplerConfig:
    task: str


class PromptSampler:
    def __init__(self, cfg: PromptSamplerConfig):
        self.cfg = cfg

    def build(self, parent: Program, inspirations: list[Program]) -> str:
        # In the future, this may use meta prompting to generate a prompt based on the task and inspirations.
        # This could be expensive, so we provide the scaffolding for it by treating it as a ray remote function.
        inspiration_components = {}
        for insp in inspirations:
            inspiration_components[f"inspiration_{insp.id}"] = insp.content
            inspiration_components[f"inspiration_{insp.id}_scores"] = "\n".join(map(str, insp.scores))
            inspiration_components[f"inspiration_{insp.id}_reasoning"] = insp.reasoning
        variation = random.choice(ISLAND_VARIATIONS[parent.island_id % len(ISLAND_VARIATIONS)])
        components = {
            "preamble": PREMABLE,
            "task": self.cfg.task,
            "parent": parent.content,
            "parent_scores": "\n".join(map(str, parent.scores)),
            "parent_reasoning": parent.reasoning,
            **inspiration_components,
            "variation": VARIATIONS[variation],
            "epilogue": EPILOGUE,
        }
        content = "\n".join(
            f"<{key.upper()}>\n{value}\n</{key.upper()}>"
            for key, value in components.items()
        )
        return content
