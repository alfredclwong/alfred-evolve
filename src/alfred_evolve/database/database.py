import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from sqlalchemy import func

from alfred_evolve.database.base import Program, Score
from alfred_evolve.database.sql import SQLDatabase
from alfred_evolve.util import apply_diff


@dataclass(frozen=True)
class ProgramDatabaseConfig:
    url: str
    n_islands: int
    initial_program_content: str
    n_inspirations_best: int
    n_inspirations_prev: int
    n_inspirations_rand: int
    migration_k: int
    migration_frequency: int


class SampleScope(Enum):
    ISLAND = auto()
    GLOBAL = auto()


class SampleStrategy(Enum):
    BEST = auto()
    PREV = auto()
    RAND = auto()


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
                        generation_id=0,
                        content=cfg.initial_program_content,
                        reasoning="Initial program",
                    )
                )
                self.add(Score(name="SCORE", value=0.0, program_id=program_id))

        self.cfg = cfg
        self.current_island = 0

    def _next_island(self):
        self.current_island = (self.current_island + 1) % self.cfg.n_islands

    def _get_island_generation(self, island_id: int) -> int:
        with self.get_session() as session:
            return (
                session.query(func.max(Program.generation_id))
                .filter(Program.island_id == island_id)
                .scalar()
                or 0
            )

    def sample(self) -> tuple[Program, list[Program]]:
        self.migrate()

        parent = self._sample(
            k=1, scope=SampleScope.ISLAND, strategy=SampleStrategy.BEST, parent=None
        )[0]
        inspirations = sum(
            [
                self._sample(k=k, scope=scope, strategy=strategy, parent=parent)
                for k, scope, strategy in [
                    (
                        self.cfg.n_inspirations_best,
                        SampleScope.ISLAND,
                        SampleStrategy.BEST,
                    ),
                    (
                        self.cfg.n_inspirations_prev,
                        SampleScope.ISLAND,
                        SampleStrategy.PREV,
                    ),
                    (
                        self.cfg.n_inspirations_rand,
                        SampleScope.ISLAND,
                        SampleStrategy.RAND,
                    ),
                ]
            ],
            [],
        )
        # Make sure we don't have duplicate inspirations
        inspirations = list({insp.id: insp for insp in inspirations}.values())

        self._next_island()
        return parent, inspirations

    def _sample(
        self,
        k: int,
        scope: SampleScope,
        strategy: SampleStrategy,
        parent: Optional[Program],
    ) -> list[Program]:
        if scope == SampleScope.ISLAND:
            programs = self.get_topk_programs(k=k + 1, island_id=self.current_island)
        elif scope == SampleScope.GLOBAL:
            programs = self.get_topk_programs(k=k + 1)
        else:
            raise ValueError(f"Unknown sample scope: {scope}")
        if parent is not None:
            programs = [p for p in programs if p.id != parent.id][:k]
        if strategy == SampleStrategy.BEST:
            return programs[:k]
        elif strategy == SampleStrategy.RAND:
            # weights = [
            #     self.get(Score, filter_by={"program_id": p.id, "name": "SCORE"}).value
            #     for p in programs
            # ]
            return random.sample(programs, k) if len(programs) > k else programs
        elif strategy == SampleStrategy.PREV:
            return sorted(programs, key=lambda p: p.id, reverse=True)[:k]
        else:
            raise ValueError(f"Unknown sample strategy: {strategy}")

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

    def migrate(self):
        """If the current island is due for migration, migrate elites to the neighbouring islands."""
        if self.cfg.n_islands < 2 or self.cfg.migration_frequency <= 0:
            print("Migration is disabled or not applicable. Skipping...")
            return
        current_generation = self._get_island_generation(self.current_island)
        if current_generation > 0 and current_generation % self.cfg.migration_frequency == 0:
            prev_island = (self.current_island - 1) % self.cfg.n_islands
            next_island = (self.current_island + 1) % self.cfg.n_islands
            elites = self.get_topk_programs(
                k=self.cfg.migration_k, island_id=self.current_island
            )
            if not elites:
                print("No elites to migrate. Skipping...")
                return
            print(
                f"\033[92mMigrating {len(elites)} elites from island {self.current_island} "
                f"to islands {prev_island} and {next_island}.\033[0m"
            )
            for program in elites:
                try:
                    self._migrate_program(program.id, prev_island)
                    self._migrate_program(program.id, next_island)
                except ValueError as e:
                    print(f"Migration error for program {program.id}: {e}")
                except Exception as e:
                    print(f"Unexpected error during migration: {e}")

    def _migrate_program(self, program_id: int, target_island_id: int):
        program = self.get(Program, filter_by={"id": program_id})
        if program is None:
            raise ValueError(f"Program with ID {program_id} not found.")
        if program.island_id == target_island_id:
            raise ValueError(
                f"Program with ID {program_id} is already in island {target_island_id}."
            )
        target_island_generation = self._get_island_generation(target_island_id)
        # Create a copy of the program for the target island
        copy_program = Program(
            island_id=target_island_id,
            generation_id=target_island_generation + 1,
            content=program.content,
            prompt=program.prompt,
            diff=program.diff,
            parent_id=program.id,
            reasoning=program.reasoning,
        )
        copy_scores = [
            Score(name=score.name, value=score.value, program_id=copy_program.id)
            for score in self.get_n(Score, filter_by={"program_id": program.id})
        ]
        # Add the copied program and its scores to the database
        try:
            copy_program_id = self.add(copy_program)
            for score in copy_scores:
                score.program_id = copy_program_id
                self.add(score)
            print(
                f"Program {program_id} copy-migrated from island {program.island_id}->{target_island_id}."
            )
        except Exception as e:
            print(f"Error migrating program {program_id}: {e}")
            raise

    def add_program(
        self,
        parent: Program,
        inspiration_ids: list[int],
        prompt: str,
        diff: str,
        reasoning: str,
        score_dict: dict[str, float],
        artifacts: Optional[str] = None,
    ):
        child = Program(
            island_id=parent.island_id,
            generation_id=parent.generation_id + 1,
            content=apply_diff(parent.content, diff),
            prompt=prompt,
            diff=diff,
            reasoning=reasoning,
            parent_id=parent.id,
            artifacts=artifacts,
        )
        for inspiration_id in inspiration_ids:
            inspiration = self.get(Program, filter_by={"id": inspiration_id})
            child.inspirations.append(inspiration)
        try:
            child_id = self.add(child)
        except Exception as e:
            print(f"Error adding program: {e}")
            print(child)
            raise
        for name, value in score_dict.items():
            score = Score(name=name, value=value, program_id=child_id)
            self.add(score)

    def get_programs(self) -> list[Program]:
        return self.get_n(Program)
