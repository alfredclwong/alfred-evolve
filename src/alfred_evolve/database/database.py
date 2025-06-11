import random
from dataclasses import dataclass
from collections import defaultdict

from alfred_evolve.primitive import Program, Result


@dataclass(frozen=True)
class ProgramDatabaseConfig:
    n_islands: int
    initial_program: Program
    n_inspirations: int


class ProgramDatabase:
    def __init__(self, cfg: ProgramDatabaseConfig):
        self.cfg = cfg
        self.current_island = 0
        self.programs: dict[int, list[Program]] = defaultdict(list)
        self.results: dict[int, list[Result]] = defaultdict(list)

    def sample(self) -> tuple[Program, list[Program]]:
        parent = self._sample_parent()
        inspirations = self._sample_inspirations(parent)
        return parent, inspirations

    def _sample_parent(self) -> Program:
        programs = self.programs[self.current_island]
        return random.choice(programs) if programs else self.cfg.initial_program

    def _sample_inspirations(self, parent: Program) -> list[Program]:
        programs = self.programs[parent.island_id]
        if len(programs) > self.cfg.n_inspirations:
            return random.sample(programs, self.cfg.n_inspirations)
        return programs

    def add(self, program: Program, result: Result):
        self.programs[program.island_id].append(program)
        self.results[program.island_id].append(result)
