from dataclasses import dataclass

from alfred_evolve.primitive import Program, Prompt


@dataclass(frozen=True)
class PromptSamplerConfig:
    task: str


class PromptSampler:
    def __init__(self, cfg: PromptSamplerConfig):
        self.cfg = cfg

    def build(self, parent_program: Program, inspirations: list[Program]) -> Prompt:
        # In the future, this may use meta prompting to generate a prompt based on the task and inspirations.
        # This could be expensive, so we provide the scaffolding for it by treating it as a ray remote function.
        components = {
            "task": self.cfg.task,
            "parent": parent_program.content,
            **{f"inspiration_{insp.program_id}": insp.content for insp in inspirations},
        }
        content = "\n".join(
            f"<{key.upper()}>\n{value}\n</{key.upper()}>"
            for key, value in components.items()
        )
        return Prompt(content)
