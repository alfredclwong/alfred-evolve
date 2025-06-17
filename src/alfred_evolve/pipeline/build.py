import random
from dataclasses import dataclass

from alfred_evolve.models.data_models import Program


@dataclass(frozen=True)
class PromptBuilderConfig:
    preamble: str
    task: str
    epilogue: str
    variations: dict[str, str]


def build_prompt(
    parent: Program,
    inspirations: list[Program],
    cfg: PromptBuilderConfig,
) -> str:
    parent_components = {}
    parent_components["parent"] = parent.content
    parent_components["parent_reasoning"] = parent.reasoning
    parent_components["parent_scores"] = str(parent.scores)
    inspiration_components = {}
    for insp in inspirations:
        inspiration_components[f"inspiration_{insp.id}"] = insp.content
        inspiration_components[f"inspiration_{insp.id}_reasoning"] = insp.reasoning
        inspiration_components[f"inspiration_{insp.id}_scores"] = str(insp.scores)
    variation = random.choice(list(cfg.variations.values()))
    components = {
        "preamble": cfg.preamble,
        "task": cfg.task,
        **parent_components,
        **inspiration_components,
        "variation": variation,
        "epilogue": cfg.epilogue,
    }
    prompt = "\n".join(
        f"<{key.upper()}>\n{value}\n</{key.upper()}>"
        for key, value in components.items()
    )
    return prompt
