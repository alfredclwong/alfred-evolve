import random
from bisect import insort_left
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from alfred_evolve.database import Database
from alfred_evolve.models.data_models import Program
from alfred_evolve.pipeline.evaluate import ProgramEvaluatorConfig, evaluate_program
from alfred_evolve.utils.logging import get_logger

logger = get_logger(__name__)


class SampleScope(Enum):
    ISLAND = auto()
    GLOBAL = auto()


class SampleStrategy(Enum):
    BEST = auto()
    PREV = auto()
    RAND = auto()


@dataclass(frozen=True)
class SampleConfig:
    n: int
    scope: SampleScope
    strategy: SampleStrategy


@dataclass(frozen=True)
class IslandConfig:
    initial_program_content: str
    parent_sample_config: SampleConfig
    inspiration_sample_configs: list[SampleConfig]
    population_size: int
    score_key: str
    migration_k: int
    migration_frequency: int
    max_parallel_tasks: int
    initial_program_scores: Optional[dict[str, float]] = None
    initial_program_artifacts: Optional[dict[str, str]] = None


class ProgramIsland:
    def __init__(
        self,
        island_id: int,
        db: Database,
        cfg: IslandConfig,
        program_evaluator_cfg: Optional[ProgramEvaluatorConfig] = None,
    ):
        self.island_id = island_id
        self.db = db
        self.cfg = cfg
        self.programs_desc: list[Program] = self._load_programs()
        if not self.programs_desc:
            logger.info(
                f"Island {self.island_id} is empty. Initializing with an initial program."
            )
            if cfg.initial_program_scores is not None and cfg.initial_program_artifacts is not None:
                initial_program_scores = cfg.initial_program_scores
                initial_program_artifacts = cfg.initial_program_artifacts
            elif program_evaluator_cfg is not None:
                initial_program_scores, initial_program_artifacts = evaluate_program(
                    cfg.initial_program_content, program_evaluator_cfg
                )
            else:
                initial_program_scores = None
                initial_program_artifacts = None
            initial_program = Program(
                id=None,
                island_id=self.island_id,
                generation=0,
                content=self.cfg.initial_program_content,
                reasoning="Initial program",
                parent_id=None,
                scores=initial_program_scores,
                artifacts=initial_program_artifacts,
            )
            self.add_program(initial_program)

    def receive_programs(self, programs: list[Program]):
        for program in programs:
            copy_program = Program(
                id=None,
                island_id=self.island_id,
                generation=program.generation,
                content=program.content,
                reasoning=program.reasoning,
                parent_id=program.id,
                scores=program.scores,
            )
            self.add_program(copy_program)

    def _load_programs(self) -> list[Program]:
        return self.db.get_topk_programs(
            k=self.cfg.population_size,
            score_key=self.cfg.score_key,
            island_id=self.island_id,
        )

    def add_program(self, program: Program):
        assert program.scores is not None and self.cfg.score_key in program.scores
        assert all(p.scores is not None for p in self.programs_desc)
        program_id = self.db.add_program(program)
        program.id = program_id
        insort_left(
            self.programs_desc, program, key=lambda p: -p.scores[self.cfg.score_key]
        )
        if len(self.programs_desc) > self.cfg.population_size:
            removed_program = self.programs_desc.pop()
            logger.info(
                f"Island {self.island_id}: Removed program {removed_program.id} to maintain population size."
            )

    def sample(self) -> tuple[Program, list[Program]]:
        """Sample step called at the beginning of each evolution loop.

        The parent is sampled based on the parent_sample_config, and inspirations are sampled
        based on the inspiration_sample_configs. The parent and inspirations are guaranteed to be
        different programs.

        Returns:
            A tuple containing the parent program and a list of inspiration programs.
        """
        parent = self._sample_programs(self.cfg.parent_sample_config)[0]
        inspirations = []
        for sample_config in self.cfg.inspiration_sample_configs:
            inspirations.extend(self._sample_programs(sample_config))
        inspirations = {insp.id: insp for insp in inspirations if insp.id != parent.id}
        inspirations = list(inspirations.values())
        logger.info(
            f"Island {self.island_id}: Sampled parent {parent.id} and inspirations {[insp.id for insp in inspirations]}."
        )
        return parent, inspirations

    def sample_elites(self, n: int) -> list[Program]:
        return self._sample_programs(
            SampleConfig(
                n=n,
                scope=SampleScope.ISLAND,
                strategy=SampleStrategy.BEST,
            )
        )

    def _sample_programs(
        self, sample_config: SampleConfig, score_key: Optional[str] = None
    ) -> list[Program]:
        if score_key is None:
            score_key = self.cfg.score_key
        if sample_config.scope == SampleScope.ISLAND:
            if sample_config.strategy == SampleStrategy.BEST:
                return self.programs_desc[: sample_config.n]
            elif sample_config.strategy == SampleStrategy.RAND:
                return random.sample(self.programs_desc, sample_config.n)
            elif sample_config.strategy == SampleStrategy.PREV:
                return sorted(
                    self.programs_desc,
                    key=lambda p: p.id if p.id is not None else 0,
                    reverse=True,
                )[: sample_config.n]
        elif sample_config.scope == SampleScope.GLOBAL:
            if sample_config.strategy == SampleStrategy.BEST:
                return self.db.get_topk_programs(k=sample_config.n, score_key=score_key)
            elif sample_config.strategy == SampleStrategy.RAND:
                return self.db.get_random_programs(
                    n=sample_config.n, island_id=self.island_id
                )
            elif sample_config.strategy == SampleStrategy.PREV:
                return self.db.get_previous_programs(
                    n=sample_config.n, island_id=self.island_id
                )
        raise ValueError(f"Invalid sample configuration: {sample_config}")

    @property
    def population_size(self) -> int:
        return len(self.programs_desc)

    @property
    def n_generated(self) -> int:
        return self.db.get_program_count(island_id=self.island_id)
