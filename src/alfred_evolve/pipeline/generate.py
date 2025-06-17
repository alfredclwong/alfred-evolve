from dataclasses import dataclass
from alfred_evolve.utils.llm import get_llm_response
from alfred_evolve.utils.str import extract_tagged_text


@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    temperature: float
    max_tokens: int
    cost_in: float = 0.0
    cost_out: float = 0.0


def generate_diff_and_reasoning(
    prompt: str, api_key: str, cfg: LLMConfig
) -> tuple[str, str]:
    llm_output = get_llm_response(
        prompt,
        api_key,
        cfg.model_name,
        cfg.temperature,
        cfg.max_tokens,
        cfg.cost_in,
        cfg.cost_out,
    )
    reasoning = extract_tagged_text(llm_output, "REASONING")
    diff = extract_tagged_text(llm_output, "DIFF")
    return diff, reasoning
