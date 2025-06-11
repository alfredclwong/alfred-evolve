from alfred_evolve.database.database import ProgramDatabaseConfig
from alfred_evolve.diff.generator import DiffGeneratorConfig
from alfred_evolve.eval.evaluator import EvaluatorConfig
from alfred_evolve.evolve import AlfredEvolve, Config
from alfred_evolve.primitive import Program
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
            n_islands=3,
            initial_program=Program(island_id=0),
            n_inspirations=2,
        ),
    )

    alfred_evolve = AlfredEvolve(config)
    completed_iterations, programs, results = alfred_evolve.run(num_iterations=10)
    print(f"Completed {completed_iterations} iterations.")

    print("Programs:")
    for island_id, island_programs in programs.items():
        print(f"Island {island_id}:")
        for program in island_programs:
            print(f"  {program}")


if __name__ == "__main__":
    main()
