from alfred_evolve.database.database import ProgramDatabaseConfig
from alfred_evolve.diff.generator import DiffGeneratorConfig
from alfred_evolve.eval.evaluator import EvaluatorConfig
from alfred_evolve.evolve import AlfredEvolve, Config
from alfred_evolve.prompt.sampler import PromptSamplerConfig


def main():
    config = Config(
        max_concurrent_builds=1,
        max_concurrent_generates=5,
        max_concurrent_evaluates=5,
        prompt_sampler_config=PromptSamplerConfig(task="example_task"),
        diff_generator_config=DiffGeneratorConfig(),
        evaluator_pool_config=EvaluatorConfig(),
        program_database_config=ProgramDatabaseConfig(
            url="sqlite:///data/programs.db",
            n_islands=3,
            initial_program_content="",
            n_inspirations=2,
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
