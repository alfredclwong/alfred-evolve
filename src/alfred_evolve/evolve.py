import random
import time
from dataclasses import dataclass

import ray
import ray.util.queue

from alfred_evolve.database.base import Program
from alfred_evolve.database.database import ProgramDatabase, ProgramDatabaseConfig
from alfred_evolve.diff.generator import DiffGenerator, DiffGeneratorConfig
from alfred_evolve.eval.evaluator import Evaluator, EvaluatorConfig
from alfred_evolve.prompt.sampler import PromptSampler, PromptSamplerConfig
from alfred_evolve.util import apply_diff


@ray.remote
class ProgramDatabaseActor:
    def __init__(self, program_database_config: ProgramDatabaseConfig):
        self.program_database = ProgramDatabase(program_database_config)

    def sample(self):
        return self.program_database.sample()

    def add_program(
        self,
        parent: Program,
        inspiration_ids: list[int],
        prompt: str,
        diff: str,
        score_dict: dict[str, float],
    ):
        self.program_database.add_program(
            parent, inspiration_ids, prompt, diff, score_dict
        )

    def get_programs(self):
        return self.program_database.get_programs()


@ray.remote
def build_prompt(
    prompt_sampler: PromptSampler, parent: Program, inspirations: list[Program]
) -> str:
    time.sleep(random.uniform(0, 0.1))
    return prompt_sampler.build(parent, inspirations)


@ray.remote
def generate_diff(diff_generator: DiffGenerator, prompt: str) -> str:
    time.sleep(random.uniform(0, 0.5))
    return diff_generator.generate(prompt)


@ray.remote
def evaluate_program(evaluator: Evaluator, program_content: str) -> dict[str, float]:
    time.sleep(random.uniform(0, 0.5))
    return evaluator.evaluate(program_content)


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
        self.program_database_actor = ProgramDatabaseActor.remote(
            cfg.program_database_config
        )

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

        return completed_iterations, programs

    def _start_build_task(self):
        if len(self.build_tasks) < self.cfg.max_concurrent_builds:
            sample_future = self.program_database_actor.sample.remote()
            parent, inspirations = ray.get(sample_future)
            build_future = build_prompt.remote(
                self.prompt_sampler, parent, inspirations
            )
            inspiration_ids = [insp.id for insp in inspirations]
            task_id = id(build_future)
            self.build_tasks[task_id] = {
                "future": build_future,
                "parent": parent,
                "inspiration_ids": inspiration_ids,
            }

    def _process_build_completions(self):
        completed_tasks = []

        for task_id, task_info in self.build_tasks.items():
            future = task_info["future"]
            ready, _ = ray.wait([future], timeout=0)

            if ready:
                prompt: str = ray.get(future)
                del task_info["future"]
                self.generate_queue.put({"prompt": prompt, **task_info})
                completed_tasks.append(task_id)

        for task_id in completed_tasks:
            del self.build_tasks[task_id]

    def _process_generate_completions(self):
        completed_tasks = []

        for task_id, task_info in self.generate_tasks.items():
            future = task_info["future"]
            ready, _ = ray.wait([future], timeout=0)

            if ready:
                diff: str = ray.get(future)
                del task_info["future"]
                self.evaluate_queue.put({"diff": diff, **task_info})
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
                score_dict: dict[str, float] = ray.get(future)
                ray.get(
                    self.program_database_actor.add_program.remote(
                        task_info["parent"],
                        task_info["inspiration_ids"],
                        task_info["prompt"],
                        task_info["diff"],
                        score_dict,
                    )
                )
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
                **item,
            }

        while (
            len(self.evaluate_tasks) < self.cfg.max_concurrent_evaluates
            and self.evaluate_queue
        ):
            item = self.evaluate_queue.get()
            child_content = apply_diff(item["parent"].content, item["diff"])
            evaluate_future = evaluate_program.remote(self.evaluator, child_content)
            task_id = id(evaluate_future)
            self.evaluate_tasks[task_id] = {
                "future": evaluate_future,
                **item,
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
