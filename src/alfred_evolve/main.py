from pathlib import Path

from alfred_evolve.evolve import AlfredEvolve, AlfredEvolveConfig
from alfred_evolve.island import IslandConfig, SampleConfig, SampleScope, SampleStrategy
from alfred_evolve.pipeline.build import PromptBuilderConfig
from alfred_evolve.pipeline.evaluate import (
    DockerConfig,
    GoogleCloudEvaluatorConfig,
    evaluate_program,
)
from alfred_evolve.pipeline.generate import LLMConfig
from alfred_evolve.pipeline.pipeline import PipelineConfig
from alfred_evolve.prompt_template import EPILOGUE, PREMABLE, VARIATIONS


def main():
    eval_timeout = 600
    n_eval_runs = 1
    task = """\
You are an expert mathematician specializing in circle packing problems and computational geometry. \
Your task is to improve a constructor function that produces a specific arrangement of 26 circles \
in a unit square, such that none of them overlap. The function `pack_26()` should return a numpy \
array with 26 (x, y, r) rows, where (x, y) is the center of a circle and r is its radius. \
The score will be the sum of the radii of all circles, which you should maximise. """
    if n_eval_runs > 1:
        task += f"The function will be evaluated {n_eval_runs} times, and the best score will be used."
    task += """\
Invalid packings, where circles overlap or extend beyond the unit square, will score 0. \
Functions which take more than {eval_timeout} seconds to run will time out and score 0. \
The code for checking overlaps and bounds works with a numerical tolerance of 1e-9. \
You will not have access to many built-ins, including: eval, exec, compile, input, print, open. \
The Python global scope is pre-populated with the following imports: numpy as np, scipy. \
"""

    initial_program_path = Path("src/examples/circle_packing/initial_program.py")
    # initial_program_path = Path("src/examples/circle_packing/example_program.py")
    initial_program_content = initial_program_path.read_text()

    program_evaluator_cfg = GoogleCloudEvaluatorConfig(
        base_name="circle-packing-eval",
        region="europe-west2",
        project_id="alfred-evolve",
        image="gcr.io/alfred-evolve/circle-packing",
        cpu_limit="4",
        memory_limit="2Gi",
        timeout=eval_timeout,
        n_eval_runs=n_eval_runs,
    )
    # program_evaluator_cfg = DockerConfig(
    #     base_name="circle_packing",
    #     image="gcr.io/alfred-evolve/circle-packing:latest",
    #     cpu_limit="1",
    #     memory_limit="1g",
    #     timeout=eval_timeout,
    #     n_eval_runs=n_eval_runs,
    # )
    initial_program_scores, initial_program_artifacts = evaluate_program(
        initial_program_content, program_evaluator_cfg
    )

    n_islands = 8

    cfg = AlfredEvolveConfig(
        n=n_islands * 200,
        db_url="data/programs.db",
        api_key_path=Path("secret.txt"),
        pipeline_cfg=PipelineConfig(
            prompt_builder_cfg=PromptBuilderConfig(
                preamble=PREMABLE,
                task=task,
                epilogue=EPILOGUE,
                variations=VARIATIONS,
            ),
            llm_cfg=LLMConfig(
                model_name="google/gemini-2.5-flash-preview-05-20",
                temperature=0.7,
                max_tokens=8192,
                cost_in=0.15e-6,
                cost_out=0.60e-6,
            ),
            program_evaluator_cfg=program_evaluator_cfg,
        ),
        island_cfgs=[
            IslandConfig(
                initial_program_content=initial_program_content,
                parent_sample_config=SampleConfig(
                    n=1,
                    scope=SampleScope.ISLAND,
                    strategy=SampleStrategy.BEST,
                ),
                inspiration_sample_configs=[
                    SampleConfig(
                        n=4,
                        scope=SampleScope.ISLAND,
                        strategy=SampleStrategy.BEST,
                    ),
                    SampleConfig(
                        n=1,
                        scope=SampleScope.ISLAND,
                        strategy=SampleStrategy.PREV,
                    ),
                    SampleConfig(
                        n=1,
                        scope=SampleScope.ISLAND,
                        strategy=SampleStrategy.RAND,
                    ),
                ],
                population_size=100,
                score_key="SCORE",
                migration_k=1,
                migration_frequency=25,
                max_parallel_tasks=3,
                initial_program_scores=initial_program_scores,
                initial_program_artifacts=initial_program_artifacts,
            )
            for _ in range(n_islands)
        ],
    )
    alfred_evolve = AlfredEvolve(cfg)
    alfred_evolve.evolve()


if __name__ == "__main__":
    main()
