from dataclasses import dataclass

from alfred_evolve.database.base import Program, Score
from alfred_evolve.database.sql import SQLDatabase
from alfred_evolve.util import apply_diff


@dataclass(frozen=True)
class ProgramDatabaseConfig:
    url: str
    n_islands: int
    initial_program_content: str
    n_inspirations: int


class ProgramDatabase(SQLDatabase):
    def __init__(self, cfg: ProgramDatabaseConfig):
        super().__init__(cfg.url)

        for i in range(cfg.n_islands):
            self.add(Program(island_id=i))

        self.cfg = cfg
        self.current_island = 0

    def sample(self) -> tuple[Program, list[Program]]:
        parent = self._sample_parent()
        inspirations = self._sample_inspirations(parent)
        return parent, inspirations

    def _sample_parent(self) -> Program:
        filters = {"island_id": self.current_island}
        parent = self.get(Program, filter_by=filters, order_by="id desc")
        return parent

    def _sample_inspirations(self, parent: Program) -> list[Program]:
        inspirations = self.get_n(
            Program,
            n=self.cfg.n_inspirations,
            filter_by={"island_id": self.current_island},
            order_by="id desc",
        )
        return inspirations

    def get_program(self, program_id: int) -> tuple[Program, list[Score]]:
        program = self.get(Program, filter_by={"id": program_id})
        scores = self._get_scores(program)
        return program, scores

    def _get_scores(self, program) -> list[Score]:
        return self.get_n(Score, filter_by={"program_id": program.id})

    def add_program(
        self,
        parent: Program,
        inspiration_ids: list[int],
        prompt: str,
        diff: str,
        score_dict: dict[str, float],
    ):
        child = Program(
            island_id=parent.island_id,
            content=apply_diff(parent.content, diff),
            prompt=prompt,
            diff=diff,
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
