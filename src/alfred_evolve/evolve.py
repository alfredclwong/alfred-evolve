import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ray

from alfred_evolve.database import Database
from alfred_evolve.island import (
    IslandConfig,
    ProgramIsland,
)
from alfred_evolve.models.data_models import Program
from alfred_evolve.pipeline.pipeline import PipelineConfig
from alfred_evolve.ray.tasks import run_pipeline_task
from alfred_evolve.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AlfredEvolveConfig:
    n: int  # Total number of programs to evolve
    db_url: str
    api_key_path: Path
    pipeline_cfg: PipelineConfig
    island_cfgs: list[IslandConfig]


class AlfredEvolve:
    def __init__(self, cfg: AlfredEvolveConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key_path.read_text().strip()

        self.db = Database(cfg.db_url)
        self.islands = [
            # ProgramIslandActor.remote(i, self.db, island_cfg)
            ProgramIsland(i, self.db, island_cfg)
            for i, island_cfg in enumerate(cfg.island_cfgs)
        ]
        self.n_islands = len(self.islands)
        self.prev_migration = {
            i: 0 for i in range(self.n_islands)
        }  # TODO improve resuming logic

        # Each island can evolve in parallel
        # For each island, we store a dict {task_id: task_future} of running tasks
        self.island_tasks = defaultdict(dict)

    def evolve(self):
        logger.info("Starting evolution loop...")
        logger.info(self.cfg)

        while True:
            # TODO handle task failures
            self._process_task_completions()
            done = self._start_new_tasks()
            if done:
                break
            self._migrate()
            time.sleep(0.1)  # Sleep to avoid busy-waiting

        logger.info(f"Waiting for {self.n_running_tasks} running tasks to finish...")
        self._wait_for_all_tasks()

    def _migrate(self):
        # Simple migration strategy
        # For each island, every n programs, we migrate its best k programs to the neigbouring islands
        if len(self.islands) < 2:
            return
        for island in self.islands:
            n_generated = island.n_generated
            if (
                n_generated % island.cfg.migration_frequency == 0
            ) and n_generated > self.prev_migration[island.island_id]:
                self.prev_migration[island.island_id] = n_generated
                logger.info(
                    f"Migrating {island.cfg.migration_k} programs from island {island.island_id} to neighbours."
                )
                best_programs = island.sample_elites(island.cfg.migration_k)
                neighbor_ids = [island.island_id - 1, island.island_id + 1]
                neighbor_ids = list(set(i % len(self.islands) for i in neighbor_ids))
                for neighbor_id in neighbor_ids:
                    neighbor = self.islands[neighbor_id]
                    neighbor.receive_programs(best_programs)

    def _process_task_completions(self):
        # Store child programs
        completed_tasks = []

        for island_id, tasks in self.island_tasks.items():
            for task_id, future in list(tasks.items()):
                ready, _ = ray.wait([future], timeout=0)
                if not ready:
                    continue
                completed_tasks.append((island_id, task_id))
                result: Program | Exception = ray.get(future)
                if isinstance(result, Exception):
                    logger.warning(f"Task {task_id} on island {island_id} failed")
                    logger.debug(f"Error: {result}")
                elif isinstance(result, Program):
                    self.islands[island_id].add_program(result)  # TODO batch insert

        for island_id, task_id in completed_tasks:
            logger.info(f"Closed task {task_id} on island {island_id}.")
            del self.island_tasks[island_id][task_id]

    def _start_new_tasks(self) -> bool:
        # If we have reached the target number of programs, don't start any new tasks
        if self.db.get_program_count() - len(self.islands) >= self.cfg.n:
            logger.info("Reached target number of programs, stopping task creation.")
            return True

        # Sample (parent, inspirations)
        # Run pipeline
        for island_id, island in enumerate(self.islands):
            if len(self.island_tasks[island_id]) >= island.cfg.max_parallel_tasks:
                # Skip this island if it has reached the max number of running tasks
                continue

            parent, inspirations = island.sample()
            task = run_pipeline_task.remote(
                parent,
                inspirations,
                self.cfg.pipeline_cfg,
                self.api_key,
            )
            task_id = id(task)
            self.island_tasks[island_id][task_id] = task
            logger.info(f"Started task {task_id} on island {island_id}.")

        return False

    def _wait_for_all_tasks(self):
        while self.n_running_tasks > 0:
            self._process_task_completions()
            time.sleep(0.1)

    @property
    def n_running_tasks(self):
        return sum(len(tasks) for tasks in self.island_tasks.values())
