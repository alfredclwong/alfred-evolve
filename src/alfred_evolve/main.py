from pathlib import Path

from alfred_evolve.database.database import ProgramDatabaseConfig
from alfred_evolve.diff.generator import DiffGeneratorConfig
from alfred_evolve.eval.evaluator import EvaluatorConfig
from alfred_evolve.evolve import AlfredEvolve, Config
from alfred_evolve.prompt.sampler import PromptSamplerConfig


def main():
    eval_timeout = 600
    task = f"""\
You are an expert mathematician specializing in circle packing problems and \
computational geometry. Your task is to improve a constructor function that \
produces a specific arrangement of 26 circles in a unit square, such that none \
of them overlap. The function `pack_26()` should return a numpy array with 26 \
(x, y, r) rows, where (x, y) is the center of a circle and r is its radius. \
The score will be the sum of the radii of all circles, which you should maximise. \
Invalid packings, where circles overlap or extend beyond the unit square, will score 0. \
Functions which take more than {eval_timeout} seconds to run will time out and score 0. \
The code for checking overlaps and bounds works with a numerical tolerance of 1e-9. \
This is a difficult problem so hard-coded solutions will not work well. \
The current best score found by other researchers is 2.635. \
The Python environment has the following libraries available: numpy, scipy.\
"""
    initial_program_path = Path("src") / "examples" / "circle_packing" / "initial_program.py"
    initial_program_content = initial_program_path.read_text()

    config = Config(
        max_concurrent_builds=1,
        max_concurrent_generates=3,
        max_concurrent_evaluates=8,
        max_pending_generates=1,
        max_pending_evaluates=6,
        prompt_sampler_config=PromptSamplerConfig(task=task),
        diff_generator_config=DiffGeneratorConfig(
            api_key_path=Path("secret.txt"),
            # model_name="google/gemma-3-27b-it:free",
            model_name="google/gemini-2.5-flash-preview-05-20",
            temperature=0.7,
            max_tokens=8192,
            providers=["google-ai-studio"],
        ),
        evaluator_pool_config=EvaluatorConfig(
            container_name="evaluator_container",
            image="circle-packing:latest",
            eval_file=Path("src") / "examples" / "circle_packing" / "eval.py",
            cpu_limit="1",
            memory_limit="1g",
            timeout=eval_timeout,
        ),
        program_database_config=ProgramDatabaseConfig(
            url="sqlite:///data/programs.db",
            n_islands=3,
            initial_program_content=initial_program_content,
            n_inspirations_best=3,
            n_inspirations_prev=1,
            n_inspirations_rand=1,
            migration_k=1,
            migration_frequency=20,
        ),
    )

    alfred_evolve = AlfredEvolve(config)
    completed_iterations, programs = alfred_evolve.run(num_iterations=100)
    print(f"Completed {completed_iterations} iterations.")

    print("Programs:")
    for program in programs:
        print(program)


if __name__ == "__main__":
    main()
