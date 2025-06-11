from dataclasses import dataclass

from alfred_evolve.database.base import Program


@dataclass(frozen=True)
class PromptSamplerConfig:
    task: str


class PromptSampler:
    def __init__(self, cfg: PromptSamplerConfig):
        self.cfg = cfg

    def build(self, parent: Program, inspirations: list[Program]) -> str:
        # In the future, this may use meta prompting to generate a prompt based on the task and inspirations.
        # This could be expensive, so we provide the scaffolding for it by treating it as a ray remote function.
        components = {
            "task": self.cfg.task,
            "parent": parent.content,
            "parent_scores": "\n".join(map(str, parent.scores)),
        }
        for insp in inspirations:
            components[f"inspiration_{insp.id}"] = insp.content
            components[f"inspiration_{insp.id}_scores"] = "\n".join(map(str, insp.scores))
        content = "\n".join(
            f"<{key.upper()}>\n{value}\n</{key.upper()}>"
            for key, value in components.items()
        )
        return content
