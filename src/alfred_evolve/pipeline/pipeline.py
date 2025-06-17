from dataclasses import dataclass

from alfred_evolve.models.data_models import Program
from alfred_evolve.pipeline.build import PromptBuilderConfig, build_prompt
from alfred_evolve.pipeline.evaluate import (
    ProgramEvaluatorConfig,
    apply_diff,
    evaluate_program,
)
from alfred_evolve.pipeline.generate import LLMConfig, generate_diff_and_reasoning


@dataclass(frozen=True)
class PipelineConfig:
    prompt_builder_cfg: PromptBuilderConfig
    llm_cfg: LLMConfig
    program_evaluator_cfg: ProgramEvaluatorConfig


def run_pipeline(
    parent: Program,
    inspirations: list[Program],
    cfg: PipelineConfig,
    api_key: str,
) -> Program | Exception:
    try:
        prompt = build_prompt(parent, inspirations, cfg.prompt_builder_cfg)
        diff, reasoning = generate_diff_and_reasoning(prompt, api_key, cfg.llm_cfg)
        child_content = apply_diff(parent.content, diff)
        scores, artifacts = evaluate_program(child_content, cfg.program_evaluator_cfg)
        child = Program(
            id=None,
            island_id=parent.island_id,
            generation=parent.generation + 1,
            content=child_content,
            parent_id=parent.id,
            inspired_by_ids=[insp.id for insp in inspirations if insp.id is not None],
            prompt=prompt,
            reasoning=reasoning,
            diff=diff,
            scores=scores,
            artifacts=artifacts,
        )
        return child
    except Exception as e:
        return e
