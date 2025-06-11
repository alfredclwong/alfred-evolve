import random
import time
from dataclasses import dataclass

import ray
import ray.util.queue

from alfred_evolve.database.database import ProgramDatabase, ProgramDatabaseConfig
from alfred_evolve.diff.generator import DiffGenerator, DiffGeneratorConfig
from alfred_evolve.eval.evaluator import Evaluator, EvaluatorConfig
from alfred_evolve.primitive import Diff, Program, Prompt, Result
from alfred_evolve.prompt.sampler import PromptSampler, PromptSamplerConfig


@ray.remote
class ProgramDatabaseActor:
    def __init__(self, program_database: ProgramDatabase):
        self.program_database = program_database

    def sample(self):
        return self.program_database.sample()

    def add(self, program: Program, result):
        self.program_database.add(program, result)

    def get_programs(self):
        return self.program_database.programs

    def get_results(self):
        return self.program_database.results


@ray.remote
def build_prompt(
    prompt_sampler: PromptSampler, parent: Program, inspirations: list[Program]
) -> Prompt:
    time.sleep(random.uniform(0, .1))
    return prompt_sampler.build(parent, inspirations)


@ray.remote
def generate_diff(diff_generator: DiffGenerator, prompt: Prompt) -> Diff:
    time.sleep(random.uniform(0, .5))
    return diff_generator.generate(prompt)


@ray.remote
def evaluate_program(evaluator: Evaluator, program: Program) -> Result:
    time.sleep(random.uniform(0, .5))
    return evaluator.evaluate(program)


@dataclass(frozen=True)
class Config:
    max_concurrent_builds: int
    max_concurrent_generates: int
    max_concurrent_evaluates: int
    prompt_sampler_config: PromptSamplerConfig
    diff_generator_config: DiffGeneratorConfig
    evaluator_pool_config: EvaluatorConfig
    program_database_config: ProgramDatabaseConfig


class AlfredEvolve:
    """
    AlfredEvolve orchestrates the evolution of programs through a series of tasks:
    1. Building prompts based on sampled programs.
    2. Generating diffs from those prompts.
    3. Evaluating the generated programs.
    4. Storing the results in a program database.
    It manages the concurrency of these tasks using Ray's remote functions and queues."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # The program database is wrapped in a Ray actor to ensure single-threaded access
        program_database = ProgramDatabase(cfg.program_database_config)
        self.program_database_actor = ProgramDatabaseActor.remote(program_database)

        # The other components will be passed to Ray remote functions for parallel processing
        self.prompt_sampler = PromptSampler(cfg.prompt_sampler_config)
        self.diff_generator = DiffGenerator(cfg.diff_generator_config)
        self.evaluator = Evaluator(cfg.evaluator_pool_config)

        # Initialize queues and task tracking dictionaries
        self.build_tasks = {}
        self.generate_tasks = {}
        self.evaluate_tasks = {}
        self.build_queue = ray.util.queue.Queue()
        self.generate_queue = ray.util.queue.Queue()
        self.evaluate_queue = ray.util.queue.Queue()

    def run(self, num_iterations: int):
        """Evolve the programs through a series of iterations.

        The pipeline consists of:
        1. Building prompts based on sampled programs.
        2. Generating diffs from those prompts.
        3. Evaluating the generated programs.
        4. Storing the results in a program database.

        It starts with a fixed number of build tasks and then runs a loop to process completions at each stage.
        Tasks are tracked in dictionaries, and queues are used to manage the flow of data between stages.
        When a task is completed, the result is passed to the next stage via a queue.
        New tasks are started to process the queue items until the specified number of iterations is reached.

        Args:
            num_iterations (int): The number of iterations to run the evolution process.
        Returns:
            tuple: A tuple containing the number of completed iterations, the programs, and the results.
        """
        completed_iterations = 0

        for _ in range(min(num_iterations, self.cfg.max_concurrent_builds)):
            self._start_build_task()

        while completed_iterations < num_iterations:
            self._process_build_completions()
            self._process_generate_completions()
            completed_iterations += self._process_evaluate_completions()
            self._start_new_tasks(num_iterations, completed_iterations)
            time.sleep(0.01)  # Prevent busy waiting

        self._wait_for_remaining_tasks()

        programs = ray.get(self.program_database_actor.get_programs.remote())
        results = ray.get(self.program_database_actor.get_results.remote())

        return completed_iterations, programs, results

    def _start_build_task(self):
        if len(self.build_tasks) < self.cfg.max_concurrent_builds:
            sample_future = self.program_database_actor.sample.remote()
            parent, inspirations = ray.get(sample_future)
            build_future = build_prompt.remote(
                self.prompt_sampler, parent, inspirations
            )
            task_id = id(build_future)
            self.build_tasks[task_id] = {
                "future": build_future,
                "parent": parent,
                "inspirations": inspirations,
            }

    def _process_build_completions(self):
        completed_tasks = []

        for task_id, task_info in self.build_tasks.items():
            future = task_info["future"]
            ready, _ = ray.wait([future], timeout=0)

            if ready:
                prompt = ray.get(future)
                parent = task_info["parent"]
                self.generate_queue.put(
                    {
                        "prompt": prompt,
                        "parent": parent,
                    }
                )
                completed_tasks.append(task_id)

        for task_id in completed_tasks:
            del self.build_tasks[task_id]

    def _process_generate_completions(self):
        completed_tasks = []

        for task_id, task_info in self.generate_tasks.items():
            future = task_info["future"]
            ready, _ = ray.wait([future], timeout=0)

            if ready:
                diff: Diff = ray.get(future)
                parent = task_info["parent"]
                child = Program(parent.island_id, parent, diff)
                self.evaluate_queue.put({"child": child})
                completed_tasks.append(task_id)

        for task_id in completed_tasks:
            del self.generate_tasks[task_id]

    def _process_evaluate_completions(self):
        completed_tasks = []
        completed_count = 0

        for task_id, task_info in self.evaluate_tasks.items():
            future = task_info["future"]
            ready, _ = ray.wait([future], timeout=0)

            if ready:
                result = ray.get(future)
                child = task_info["child"]
                ray.get(self.program_database_actor.add.remote(child, result))
                completed_tasks.append(task_id)
                completed_count += 1

        for task_id in completed_tasks:
            del self.evaluate_tasks[task_id]

        return completed_count

    def _start_new_tasks(self, num_iterations: int, completed_iterations: int):
        remaining_iterations = num_iterations - completed_iterations
        total_in_progress = (
            len(self.build_tasks) + len(self.generate_tasks) + len(self.evaluate_tasks)
        )

        while (
            len(self.build_tasks) < self.cfg.max_concurrent_builds
            and total_in_progress < remaining_iterations
        ):
            self._start_build_task()
            total_in_progress += 1

        while (
            len(self.generate_tasks) < self.cfg.max_concurrent_generates
            and self.generate_queue
        ):
            item = self.generate_queue.get()
            generate_future = generate_diff.remote(self.diff_generator, item["prompt"])
            task_id = id(generate_future)
            self.generate_tasks[task_id] = {
                "future": generate_future,
                "parent": item["parent"],
            }

        while (
            len(self.evaluate_tasks) < self.cfg.max_concurrent_evaluates
            and self.evaluate_queue
        ):
            item = self.evaluate_queue.get()
            evaluate_future = evaluate_program.remote(self.evaluator, item["child"])
            task_id = id(evaluate_future)
            self.evaluate_tasks[task_id] = {
                "future": evaluate_future,
                "child": item["child"],
            }

    def _wait_for_remaining_tasks(self):
        all_futures = []

        for task_info in self.build_tasks.values():
            all_futures.append(task_info["future"])
        for task_info in self.generate_tasks.values():
            all_futures.append(task_info["future"])
        for task_info in self.evaluate_tasks.values():
            all_futures.append(task_info["future"])

        if all_futures:
            ray.wait(all_futures, num_returns=len(all_futures))

            self._process_build_completions()
            self._process_generate_completions()
            self._process_evaluate_completions()
