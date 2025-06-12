import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from alfred_evolve.database.base import Program, Score
from alfred_evolve.database.sql import SQLDatabase
from alfred_evolve.util import apply_diff


@dataclass(frozen=True)
class ProgramDatabaseConfig:
    url: str
    n_islands: int
    initial_program_content: str
    n_inspirations: int
    migration_k: int
    migration_frequency: float


class SampleMode(Enum):
    ISLAND_ALL = auto()
    GLOBAL_INSPIRATIONS = auto()
    GLOBAL_PARENT = auto()
    SCORE_WEIGHTED = auto()


class ProgramDatabase(SQLDatabase):
    def __init__(self, cfg: ProgramDatabaseConfig):
        super().__init__(cfg.url)

        if not self.get_n(Program):
            # Initialize the database with empty islands and initial program
            print(
                f"Initializing database with {cfg.n_islands} islands and initial program content."
            )
            for i in range(cfg.n_islands):
                program_id = self.add(
                    Program(
                        island_id=i,
                        content=cfg.initial_program_content,
                        reasoning="Initial program",
                    )
                )
                self.add(Score(name="SCORE", value=0.0, program_id=program_id))

        self.cfg = cfg
        self.current_island = 0

    def _next_island(self):
        self.current_island = (self.current_island + 1) % self.cfg.n_islands

    def sample(
        self, sample_mode: SampleMode = SampleMode.GLOBAL_INSPIRATIONS
    ) -> tuple[Program, list[Program]]:
        if random.random() < self.cfg.migration_frequency:
            # Instead of sampling from this island, migrate and sample from the next island
            self.migrate(k=self.cfg.migration_k)

        top_island_programs = self.get_topk_programs(
            k=1 + self.cfg.n_inspirations,
            score_name="SCORE",
            island_id=self.current_island,
        )
        if not top_island_programs:
            return self.get(Program, filter_by={"island_id": self.current_island}), []
        if sample_mode == SampleMode.ISLAND_ALL:
            parent = top_island_programs[0]
            inspirations = top_island_programs[1:]
        elif sample_mode == SampleMode.GLOBAL_INSPIRATIONS:
            parent = top_island_programs[0]
            inspirations = self.get_topk_programs(
                k=1 + self.cfg.n_inspirations,
                score_name="SCORE",
                island_id=None,
            )
            inspirations = [insp for insp in inspirations if insp.id != parent.id][
                : self.cfg.n_inspirations
            ]
        elif sample_mode == SampleMode.GLOBAL_PARENT:
            parent = self.get_topk_programs(k=1, score_name="SCORE", island_id=None)[0]
            if parent == top_island_programs[0]:
                inspirations = top_island_programs[1:]
            else:
                inspirations = top_island_programs[:-1]
        elif sample_mode == SampleMode.SCORE_WEIGHTED:
            raise NotImplementedError("Score-weighted sampling is not implemented yet.")
        else:
            raise ValueError(f"Unknown sample mode: {sample_mode}")

        self._next_island()
        return parent, inspirations

    def get_topk_programs(
        self,
        k: Optional[int] = None,
        score_name: str = "SCORE",
        island_id: Optional[int] = None,
    ) -> list[Program]:
        with self.get_session() as session:
            query = session.query(Program).join(Score).filter(Score.name == score_name)
            if island_id is not None:
                query = query.filter(Program.island_id == island_id)
            query = query.order_by(Score.value.desc())
            if k is not None:
                query = query.limit(k)
            return query.all()

    def migrate(self, k: int):
        """Migrate the topk programs to the next island."""
        topk_programs = self.get_topk_programs(k=k, island_id=self.current_island)
        if not topk_programs:
            return

        for program in topk_programs:
            copy_program = Program(
                island_id=(self.current_island + 1) % self.cfg.n_islands,
                content=program.content,
                prompt=program.prompt,
                diff=program.diff,
                parent_id=program.id,
                reasoning=program.reasoning,
            )
            self.add(copy_program)

        print(
            f"Migrated {len(topk_programs)} programs to island {self.current_island}."
        )
        self._next_island()

    def add_program(
        self,
        parent: Program,
        inspiration_ids: list[int],
        prompt: str,
        diff: str,
        reasoning: str,
        score_dict: dict[str, float],
    ):
        child = Program(
            island_id=parent.island_id,
            content=apply_diff(parent.content, diff),
            prompt=prompt,
            diff=diff,
            reasoning=reasoning,
            parent_id=parent.id,
        )
        for inspiration_id in inspiration_ids:
            inspiration = self.get(Program, filter_by={"id": inspiration_id})
            child.inspirations.append(inspiration)
        child_id = self.add(child)
        for name, value in score_dict.items():
            score = Score(name=name, value=value, program_id=child_id)
            self.add(score)

    def get_programs(self) -> list[Program]:
        return self.get_n(Program)
